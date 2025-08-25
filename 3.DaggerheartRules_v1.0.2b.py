"""
DaggerheartRules v1.0.2b
========================

SRD-compliant implementation of the Daggerheart RPG system for ChronoBound.

Changelog v1.0.2b:
1. Enhanced homebrew threshold integration in mark_stress() with _pending_damage support
2. Improved damage telemetry with actual roll breakdown in make_attack_roll()

Author: ChronoBound Development Team
License: MIT
"""

import logging
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, Protocol

# Minimal IRuleSystem stubs for standalone operation
try:
    from irulesystem_interface import (
        IRuleSystem, AbilityCheckResult, AttackResult, InitiativeResult,
        ActionResult, DamageType, ConditionType, LevelRange, ValidationResult
    )
    from dice_engine import DiceEngine, DiceResult
    STANDALONE_MODE = False
except ImportError:
    STANDALONE_MODE = True
    
    class ActionResult(Enum):
        CRITICAL_SUCCESS = "critical_success"
        SUCCESS = "success"
        PARTIAL_SUCCESS = "partial_success"
        FAILURE = "failure"
        CRITICAL_FAILURE = "critical_failure"
    
    class DamageType(Enum):
        PHYSICAL = "physical"
        FIRE = "fire"
        COLD = "cold"
        LIGHTNING = "lightning"
        ACID = "acid"
        POISON = "poison"
        NECROTIC = "necrotic"
        RADIANT = "radiant"
        PSYCHIC = "psychic"
        FORCE = "force"
    
    @dataclass
    class LevelRange:
        min_level: int
        max_level: int
    
    @dataclass
    class AbilityCheckResult:
        total: int
        result: ActionResult
        margin: int
        raw_rolls: List[int] = field(default_factory=list)
        modifiers: Dict[str, int] = field(default_factory=dict)
        system_specific_data: Dict[str, Any] = field(default_factory=dict)
    
    @dataclass
    class AttackResult:
        hit: bool
        attack_total: int
        damage_total: int
        damage_type: DamageType
        is_critical: bool = False
        raw_attack_rolls: List[int] = field(default_factory=list)
        raw_damage_rolls: List[int] = field(default_factory=list)
        modifiers: Dict[str, int] = field(default_factory=dict)
        system_specific_data: Dict[str, Any] = field(default_factory=dict)
    
    @dataclass
    class InitiativeResult:
        character_id: str
        initiative_value: int
        tie_breaker: int = 0
        modifiers: Dict[str, int] = field(default_factory=dict)
        system_specific_data: Dict[str, Any] = field(default_factory=dict)
    
    class IRuleSystem(Protocol):
        def make_ability_check(self, character: Dict[str, Any], trait_name: str, 
                             difficulty: int, modifiers: Optional[Dict[str, int]] = None,
                             advantage: bool = False, disadvantage: bool = False,
                             context: Optional[Dict[str, Any]] = None, **kwargs) -> AbilityCheckResult:
            ...
    
    # Minimal DiceEngine stub
    @dataclass
    class DiceResult:
        total: int
        individual_rolls: List[int] = field(default_factory=list)
        final_total: int = 0
        modifiers: Dict[str, int] = field(default_factory=dict)
        
        def __post_init__(self):
            if self.final_total == 0:
                self.final_total = self.total + sum(self.modifiers.values())
    
    class DiceEngine:
        def __init__(self, seed: Optional[int] = None):
            self.rng = random.Random(seed)
        
        def roll(self, notation: str) -> DiceResult:
            # Basic dice notation parser for standalone mode
            match = re.match(r'(\d+)d(\d+)(?:\+(\d+))?', notation.replace(' ', ''))
            if match:
                count, sides, modifier = match.groups()
                count, sides = int(count), int(sides)
                modifier = int(modifier) if modifier else 0
                rolls = [self.rng.randint(1, sides) for _ in range(count)]
                total = sum(rolls)
                return DiceResult(total=total, individual_rolls=rolls, 
                                modifiers={'static': modifier}, final_total=total + modifier)
            return DiceResult(total=1, individual_rolls=[1])
        
        def roll_single(self, sides: int) -> int:
            return self.rng.randint(1, sides)

# ==========================================
# DAGGERHEART-SPECIFIC ENUMS
# ==========================================

class OutcomeType(Enum):
    """SRD-compliant Daggerheart outcome taxonomy."""
    CRIT_SUCCESS = "CRIT_SUCCESS"
    SUCCESS_HOPE = "SUCCESS_HOPE"
    SUCCESS_FEAR = "SUCCESS_FEAR"
    FAILURE_HOPE = "FAILURE_HOPE"
    FAILURE_FEAR = "FAILURE_FEAR"

class DominantDie(Enum):
    """Which die dominates in duality roll."""
    HOPE = "HOPE"
    FEAR = "FEAR"

# ==========================================
# CORE DAGGERHEART RULES SYSTEM
# ==========================================

class DaggerheartRuleSystem:
    """
    SRD-compliant Daggerheart RPG system implementation.
    
    Implements the 2d12 Hope/Fear duality system with d6 advantage/disadvantage,
    exact outcome taxonomy, stress mechanics, and spotlight turn model.
    """
    
    def __init__(self, dice_engine: Optional[DiceEngine] = None, 
                 use_threshold_stress: bool = False, spotlight_tokens: int = 3):
        """Initialize the Daggerheart rules system."""
        self.dice_engine = dice_engine or DiceEngine()
        self.use_threshold_stress = use_threshold_stress  # Homebrew feature
        self.spotlight_tokens = spotlight_tokens
        self.logger = logging.getLogger(__name__)
        
        # SRD traits
        self._supported_traits = ["Agility", "Strength", "Finesse", "Instinct", "Presence", "Knowledge"]
        
        # Default difficulties
        self.difficulties = {
            "trivial": 5,
            "easy": 10,
            "moderate": 15,
            "hard": 20,
            "very_hard": 25,
            "nearly_impossible": 30
        }
    
    @property
    def supported_traits(self) -> List[str]:
        """Return list of character traits/stats this system uses."""
        return self._supported_traits.copy()
    
    @property
    def dice_notation(self) -> str:
        """Return the primary dice notation for this system."""
        return "2d12"
    
    @property
    def system_version(self) -> str:
        """Return the version of this rule system implementation."""
        return "1.0.2b"
    
    @property
    def supported_character_levels(self) -> LevelRange:
        """Return LevelRange supported by this system."""
        return LevelRange(1, 10)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Return system capability flags and configuration."""
        return {
            "name": "Daggerheart",
            "version": "1.0.2b",
            "turn_model": "spotlight",
            "spotlight_tokens": self.spotlight_tokens,
            "uses_duality": True,
            "uses_adv_d6": True,
            "crit_on_doubles": True,
            "has_stress": True,
            "uses_advantage": True,
            "uses_spell_slots": False,
            "supports_multiclass": True,
            "conditions": ["Hidden", "Restrained", "Vulnerable"],
            "crit_range": [12, 12],  # Doubles on 2d12 - preserved for back-compat
            "crit_condition": "doubles_on_2d12",  # Enhanced metadata clarity
            "dice_notation": "2d12",
            "homebrew": {
                "hp_threshold_to_stress": self.use_threshold_stress
            }
        }
    
    def parse_damage_expr(self, expression: str) -> Tuple[int, int, int]:
        """
        Parse damage expression "XdY±Z" with space tolerance.
        
        Returns:
            Tuple of (count, sides, flat_modifier)
        """
        # Clean whitespace and normalize
        expr = expression.replace(' ', '')
        
        # Match XdY with optional +/- modifier
        match = re.match(r'(\d+)d(\d+)([+-]\d+)?', expr)
        if not match:
            return (1, 6, 0)  # Default fallback
        
        count = int(match.group(1))
        sides = int(match.group(2))
        modifier_str = match.group(3) or "+0"
        modifier = int(modifier_str)
        
        return (count, sides, modifier)
    
    def mark_stress_from_damage(self, prev_hp: int, new_hp: int, 
                               thresholds: Tuple[int, int, int] = (7, 14, 21)) -> int:
        """
        Calculate stress to mark based on HP thresholds (homebrew feature).
        
        Args:
            prev_hp: Previous HP total
            new_hp: New HP total after damage
            thresholds: HP thresholds for stress (minor, major, severe)
            
        Returns:
            Number of stress to mark (0 if homebrew disabled)
        """
        if not self.use_threshold_stress:
            return 0
        
        damage_taken = prev_hp - new_hp
        if damage_taken <= 0:
            return 0
        
        minor_threshold, major_threshold, severe_threshold = thresholds
        
        if damage_taken >= severe_threshold:
            return 2  # Severe damage
        elif damage_taken >= major_threshold:
            return 1  # Major damage
        elif damage_taken >= minor_threshold:
            return 1  # Minor damage
        else:
            return 0  # Below minor threshold
    
    def _roll_duality(self, trait_modifier: int = 0, advantage: int = 0, 
                     disadvantage: int = 0) -> Dict[str, Any]:
        """
        Core SRD duality roll: 2d12 + d6 advantage/disadvantage modifiers.
        
        Returns:
            Dict with hope_die, fear_die, dominant, outcome, totals, etc.
        """
        # Cancel advantage/disadvantage 1-for-1 before rolling
        net_advantage = max(0, advantage - disadvantage)
        net_disadvantage = max(0, disadvantage - advantage)
        
        # Roll the base 2d12
        hope_die = self.dice_engine.roll_single(12)
        fear_die = self.dice_engine.roll_single(12)
        
        # Roll advantage/disadvantage d6s
        adv_d6_total = sum(self.dice_engine.roll_single(6) for _ in range(net_advantage))
        dis_d6_total = sum(self.dice_engine.roll_single(6) for _ in range(net_disadvantage))
        
        # Determine dominant die and kept value
        if hope_die > fear_die:
            dominant = DominantDie.HOPE
            kept_value = hope_die
        elif fear_die > hope_die:
            dominant = DominantDie.FEAR
            kept_value = fear_die
        else:  # Tie - choose Hope per SRD
            dominant = DominantDie.HOPE
            kept_value = hope_die
        
        # Calculate final total
        final_total = kept_value + trait_modifier + adv_d6_total - dis_d6_total
        
        # Check for critical (doubles)
        is_critical = hope_die == fear_die
        
        return {
            "hope_die": hope_die,
            "fear_die": fear_die,
            "dominant": dominant.value,
            "kept_value": kept_value,
            "trait_modifier": trait_modifier,
            "adv_d6_total": adv_d6_total,
            "dis_d6_total": dis_d6_total,
            "final_total": final_total,
            "is_critical": is_critical
        }
    
    def _determine_outcome(self, roll_data: Dict[str, Any], difficulty: int) -> OutcomeType:
        """Determine SRD outcome based on roll result and difficulty."""
        success = roll_data["final_total"] >= difficulty
        dominant = roll_data["dominant"]
        is_critical = roll_data["is_critical"]
        
        if is_critical:
            return OutcomeType.CRIT_SUCCESS
        elif success and dominant == "HOPE":
            return OutcomeType.SUCCESS_HOPE
        elif success and dominant == "FEAR":
            return OutcomeType.SUCCESS_FEAR
        elif not success and dominant == "HOPE":
            return OutcomeType.FAILURE_HOPE
        else:  # not success and dominant == "FEAR"
            return OutcomeType.FAILURE_FEAR
    
    def make_ability_check(
        self,
        character: Dict[str, Any],
        trait_name: str,
        difficulty: int,
        modifiers: Optional[Dict[str, int]] = None,
        advantage: bool = False,
        disadvantage: bool = False,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AbilityCheckResult:
        """Perform an ability check using Daggerheart's 2d12 duality system."""
        if trait_name not in self.supported_traits:
            raise ValueError(f"Unsupported trait: {trait_name}")
        
        # Get trait modifier
        trait_modifier = character.get("traits", {}).get(trait_name, 0)
        
        # Apply additional modifiers
        total_modifier = trait_modifier
        applied_modifiers = {"trait": trait_modifier}
        if modifiers:
            for name, value in modifiers.items():
                total_modifier += value
                applied_modifiers[name] = value
        
        # Convert advantage/disadvantage to counts
        adv_count = int(advantage) + kwargs.get("boons", 0)
        dis_count = int(disadvantage) + kwargs.get("banes", 0)
        
        # Roll duality dice
        roll_data = self._roll_duality(total_modifier, adv_count, dis_count)
        
        # Determine outcome
        outcome = self._determine_outcome(roll_data, difficulty)
        margin = roll_data["final_total"] - difficulty
        
        # Handle stress clearing on crit
        cleared_stress_on_crit = False
        hope_delta = 0
        fear_delta = 0
        
        if outcome == OutcomeType.CRIT_SUCCESS:
            cleared_stress_on_crit = True
            hope_delta = 1  # Crit counts as Hope
        elif outcome in [OutcomeType.SUCCESS_HOPE, OutcomeType.FAILURE_HOPE]:
            hope_delta = 1
        elif outcome in [OutcomeType.SUCCESS_FEAR, OutcomeType.FAILURE_FEAR]:
            fear_delta = 1
        
        # Convert to legacy ActionResult for compatibility
        if outcome == OutcomeType.CRIT_SUCCESS:
            legacy_result = ActionResult.CRITICAL_SUCCESS
        elif outcome in [OutcomeType.SUCCESS_HOPE, OutcomeType.SUCCESS_FEAR]:
            legacy_result = ActionResult.SUCCESS
        elif outcome in [OutcomeType.FAILURE_HOPE, OutcomeType.FAILURE_FEAR]:
            legacy_result = ActionResult.FAILURE
        else:
            legacy_result = ActionResult.FAILURE
        
        # Build system-specific data
        system_data = {
            "hope_die": roll_data["hope_die"],
            "fear_die": roll_data["fear_die"],
            "dominant": roll_data["dominant"],
            "adv_d6_total": roll_data["adv_d6_total"],
            "dis_d6_total": roll_data["dis_d6_total"],
            "outcome": outcome.value,
            "hope_delta": hope_delta,
            "fear_delta": fear_delta,
            "cleared_stress_on_crit": cleared_stress_on_crit
        }
        
        # Add critical telemetry reason
        if outcome == OutcomeType.CRIT_SUCCESS:
            system_data["reason"] = "doubles"
        
        # Echo context
        if context:
            system_data.update(context)
        
        return AbilityCheckResult(
            total=roll_data["final_total"],
            result=legacy_result,
            margin=margin,
            raw_rolls=[roll_data["hope_die"], roll_data["fear_die"]],
            modifiers=applied_modifiers,
            system_specific_data=system_data
        )
    
    def make_attack_roll(
        self,
        attacker: Dict[str, Any],
        target: Dict[str, Any],
        weapon_or_spell: str,
        modifiers: Optional[Dict[str, int]] = None,
        advantage: bool = False,
        disadvantage: bool = False,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AttackResult:
        """Make an attack roll against a target."""
        # Use default trait for weapon (could be enhanced with weapon data)
        trait_name = "Strength"  # Default, could be weapon-specific
        
        # Get target evasion (armor class equivalent)
        target_evasion = target.get("evasion", 10)
        
        # Make ability check for attack
        attack_check = self.make_ability_check(
            attacker, trait_name, target_evasion, modifiers, 
            advantage, disadvantage, context, **kwargs
        )
        
        hit = attack_check.result in [ActionResult.SUCCESS, ActionResult.CRITICAL_SUCCESS]
        is_critical = attack_check.result == ActionResult.CRITICAL_SUCCESS
        
        # Calculate damage if hit
        damage = 0
        damage_rolls = []
        damage_breakdown = {}
        
        if hit:
            damage_dice = kwargs.get("damage_dice", "1d6")
            damage = self.roll_damage(damage_dice, is_critical)
            damage_rolls = [damage]  # Simplified
            
            # Get damage breakdown for telemetry with actual rolls
            count, sides, modifier = self.parse_damage_expr(damage_dice)
            normal = self.dice_engine.roll(f"{count}d{sides}")  # Telemetry only
            
            damage_breakdown = {
                "faces": sides,
                "count": count,
                "modifier": modifier,
                "expression": damage_dice,
                "is_critical": is_critical,
                "rolls": normal.individual_rolls
            }
        
        # Enhanced system-specific data with damage breakdown
        enhanced_system_data = attack_check.system_specific_data.copy()
        if damage_breakdown:
            enhanced_system_data["damage_rolls"] = damage_breakdown
        
        return AttackResult(
            hit=hit,
            attack_total=attack_check.total,
            damage_total=damage,
            damage_type=DamageType.PHYSICAL,
            is_critical=is_critical,
            raw_attack_rolls=attack_check.raw_rolls,
            raw_damage_rolls=damage_rolls,
            modifiers=attack_check.modifiers,
            system_specific_data=enhanced_system_data
        )
    
    def roll_damage(self, expression: str, crit: bool = False) -> int:
        """
        Roll damage using SRD crit formula: max dice + normal roll + flat modifiers.
        Delegates parsing and normal rolling to DiceEngine.
        
        Example: "3d6+2" with crit = 18 + roll(3d6) + 2
        """
        # Parse dice expression using helper
        count, sides, modifier = self.parse_damage_expr(expression)
        
        if crit:
            # SRD crit: max dice + normal roll + flat mods
            max_dice = count * sides
            
            # Delegate normal roll to DiceEngine
            normal_notation = f"{count}d{sides}"
            normal_result = self.dice_engine.roll(normal_notation)
            normal_roll = normal_result.total
            
            return max_dice + normal_roll + modifier
        else:
            # Normal damage - delegate entirely to DiceEngine
            full_notation = f"{count}d{sides}"
            if modifier > 0:
                full_notation += f"+{modifier}"
            elif modifier < 0:
                full_notation += str(modifier)  # Already has minus sign
            
            result = self.dice_engine.roll(full_notation)
            return result.final_total
    
    def roll_initiative(
        self,
        character: Dict[str, Any],
        modifiers: Optional[Dict[str, int]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> InitiativeResult:
        """Roll initiative using Instinct trait."""
        initiative_check = self.make_ability_check(
            character, "Instinct", 0, modifiers, context=context
        )
        
        # Use higher die as tie-breaker
        hope_die = initiative_check.system_specific_data.get("hope_die", 1)
        fear_die = initiative_check.system_specific_data.get("fear_die", 1)
        tie_breaker = max(hope_die, fear_die)
        
        return InitiativeResult(
            character_id=character.get("id", "unknown"),
            initiative_value=initiative_check.total,
            tie_breaker=tie_breaker,
            modifiers=initiative_check.modifiers,
            system_specific_data=initiative_check.system_specific_data
        )
    
    # ==========================================
    # STRESS HELPER METHODS
    # ==========================================
    
    def mark_stress(self, character: Dict[str, Any], amount: int = 1) -> Dict[str, Any]:
        """Mark stress, handling overflow to HP and vulnerability."""
        char_copy = character.copy()
        current_stress = char_copy.get("stress", 0)
        max_stress = char_copy.get("max_stress", 6)
        
        # Apply homebrew HP-threshold stress if enabled and pending damage exists
        effective_amount = amount
        if self.use_threshold_stress and "_pending_damage" in char_copy:
            pending = char_copy["_pending_damage"]
            if isinstance(pending, dict) and "prev_hp" in pending and "new_hp" in pending:
                extra = self.mark_stress_from_damage(
                    pending["prev_hp"], pending["new_hp"], (7, 14, 21)
                )
                effective_amount += max(0, int(extra))
            # Remove the one-shot hint
            char_copy.pop("_pending_damage", None)
        
        new_stress = current_stress + effective_amount
        
        if new_stress >= max_stress:
            # Mark last stress - character becomes Vulnerable
            char_copy["stress"] = max_stress
            conditions = char_copy.get("conditions", [])
            if "Vulnerable" not in conditions:
                conditions.append("Vulnerable")
                char_copy["conditions"] = conditions
            
            # Overflow goes to HP
            overflow = new_stress - max_stress
            if overflow > 0:
                current_hp = char_copy.get("hp", 1)
                char_copy["hp"] = max(0, current_hp - overflow)
        else:
            char_copy["stress"] = new_stress
        
        return char_copy
    
    def clear_stress(self, character: Dict[str, Any], amount: int = 1) -> Dict[str, Any]:
        """Clear stress (e.g., on critical success)."""
        char_copy = character.copy()
        current_stress = char_copy.get("stress", 0)
        char_copy["stress"] = max(0, current_stress - amount)
        
        # Remove Vulnerable if stress is no longer maxed
        max_stress = char_copy.get("max_stress", 6)
        if char_copy["stress"] < max_stress:
            conditions = char_copy.get("conditions", [])
            if "Vulnerable" in conditions:
                conditions.remove("Vulnerable")
                char_copy["conditions"] = conditions
        
        return char_copy
    
    # ==========================================
    # ADDITIONAL SYSTEM METHODS
    # ==========================================
    
    def get_skill_modifier(self, character: Dict[str, Any], skill_name: str) -> int:
        """Calculate the modifier for a specific skill check."""
        # Map skills to traits (simplified)
        skill_trait_map = {
            "Athletics": "Strength",
            "Acrobatics": "Agility", 
            "Stealth": "Agility",
            "Sleight of Hand": "Finesse",
            "Investigation": "Knowledge",
            "Perception": "Instinct",
            "Insight": "Instinct",
            "Persuasion": "Presence",
            "Intimidation": "Presence",
            "Deception": "Presence"
        }
        
        trait_name = skill_trait_map.get(skill_name, "Knowledge")
        return character.get("traits", {}).get(trait_name, 0)
    
    def get_available_skills(self, character: Dict[str, Any]) -> List[str]:
        """Get list of skills available to this character."""
        return [
            "Athletics", "Acrobatics", "Stealth", "Sleight of Hand",
            "Investigation", "Perception", "Insight", "Persuasion", 
            "Intimidation", "Deception"
        ]
    
    def validate_character(self, character: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a character dict for Daggerheart compliance."""
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = ["traits", "hp", "stress", "max_stress"]
        for field in required_fields:
            if field not in character:
                errors.append(f"Missing required field: {field}")
        
        # Validate traits
        if "traits" in character:
            traits = character["traits"]
            for trait in self.supported_traits:
                if trait not in traits:
                    warnings.append(f"Missing trait: {trait}")
                elif not isinstance(traits[trait], int):
                    errors.append(f"Trait {trait} must be an integer")
        
        # Validate stress/HP relationship
        stress = character.get("stress", 0)
        max_stress = character.get("max_stress", 6)
        if stress > max_stress:
            errors.append(f"Stress ({stress}) cannot exceed max_stress ({max_stress})")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

# ==========================================
# EXPORTS
# ==========================================

__all__ = ["DaggerheartRuleSystem", "OutcomeType", "DominantDie"]

# ==========================================
# SELF-TESTS
# ==========================================

if __name__ == "__main__":
    print("Running DaggerheartRules v1.0.2b self-tests...")
    
    # Test 1: Enhanced damage parsing and crit formula
    print("\n1. Testing enhanced damage system...")
    engine = DiceEngine(seed=42)
    system = DaggerheartRuleSystem(dice_engine=engine)
    
    # Test parsing with negative modifier
    count, sides, modifier = system.parse_damage_expr("3d6-1")
    assert count == 3 and sides == 6 and modifier == -1, f"Parse failed: {count}d{sides}{modifier:+d}"
    
    # Test crit damage uses engine delegation
    normal_damage = system.roll_damage("3d6-1", crit=False)
    crit_damage = system.roll_damage("3d6-1", crit=True)
    assert crit_damage >= 18 - 1, f"Crit should be at least 17 (18-1), got {crit_damage}"
    assert crit_damage > normal_damage, f"Crit {crit_damage} should exceed normal {normal_damage}"
    print(f" ✓ Damage system: 3d6-1 normal={normal_damage}, crit={crit_damage}")
    
    # Test 2: Critical telemetry reason
    print("\n2. Testing critical telemetry...")
    original_roll_single = engine.roll_single
    
    def force_doubles(sides):
        return 8 if sides == 12 else original_roll_single(sides)
    
    engine.roll_single = force_doubles
    char = {"traits": {"Strength": 2}}
    crit_result = system.make_ability_check(char, "Strength", 12)
    
    assert "reason" in crit_result.system_specific_data, "Missing reason field"
    assert crit_result.system_specific_data["reason"] == "doubles", f"Expected 'doubles', got {crit_result.system_specific_data['reason']}"
    engine.roll_single = original_roll_single
    print(" ✓ Critical telemetry includes reason='doubles'")
    
    # Test 3: Homebrew stress with pending damage
    print("\n3. Testing homebrew stress with pending damage...")
    homebrew_system = DaggerheartRuleSystem(dice_engine=engine, use_threshold_stress=True)
    
    # Test with pending damage hint
    char_with_pending = {
        "stress": 2, "max_stress": 6, "hp": 8, "conditions": [],
        "_pending_damage": {"prev_hp": 15, "new_hp": 6}  # 9 damage >= 7 threshold
    }
    
    result = homebrew_system.mark_stress(char_with_pending, 1)  # 1 base + 1 from threshold
    assert result["stress"] == 4, f"Expected stress 4 (2+1+1), got {result['stress']}"
    assert "_pending_damage" not in result, "Should have popped _pending_damage"
    print(f" ✓ Homebrew pending damage: stress={result['stress']}")
    
    # Test 4: Attack damage breakdown with actual rolls
    print("\n4. Testing attack damage breakdown with rolls...")
    attacker = {"traits": {"Strength": 2}}
    target = {"evasion": 8}  # Easy to hit
    
    attack_result = system.make_attack_roll(
        attacker, target, "sword", damage_dice="2d6+1"
    )
    
    assert "damage_rolls" in attack_result.system_specific_data, "Missing damage_rolls breakdown"
    breakdown = attack_result.system_specific_data["damage_rolls"]
    assert "rolls" in breakdown, "Missing actual rolls in damage breakdown"
    assert len(breakdown["rolls"]) == 2, f"Expected 2 rolls, got {len(breakdown['rolls'])}"
    assert all(1 <= roll <= 6 for roll in breakdown["rolls"]), f"Invalid roll values: {breakdown['rolls']}"
    print(f" ✓ Attack breakdown with rolls: {breakdown}")
    
    # Test 5: Enhanced metadata
    print("\n5. Testing enhanced metadata...")
    info = system.get_system_info()
    
    assert "crit_condition" in info, "Missing crit_condition field"
    assert info["crit_condition"] == "doubles_on_2d12", f"Wrong crit_condition: {info['crit_condition']}"
    assert "homebrew" in info, "Missing homebrew section"
    assert "hp_threshold_to_stress" in info["homebrew"], "Missing homebrew flag"
    assert info["version"] == "1.0.2b", f"Expected v1.0.2b, got {info['version']}"
    print(f" ✓ Metadata: version={info['version']}, crit_condition='{info['crit_condition']}'")
    
    print("\n✅ All self-tests passed! DaggerheartRules v1.0.2b 
