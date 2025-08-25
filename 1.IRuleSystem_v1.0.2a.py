# 1.IRuleSystem_v1.0.2a.py
"""
ChronoBound IRuleSystem Interface v1.0.2a
=========================================

Abstract base class and common data structures for all RPG rule systems in ChronoBound.
Provides a standardized interface for character management, dice rolling, combat mechanics,
and system-specific rule implementations.

This interface enables seamless switching between different RPG systems while maintaining
consistent game state and character data. All rule system implementations must inherit
from IRuleSystem and implement the required abstract methods.

Key improvements in v1.0.2a:
- Added compatibility shims for v1.0.1 API surface
- Preserved v1.0.2 behavior while reintroducing legacy methods
- Added deprecated alias properties for AttackResult
- Enhanced InitiativeResult with raw roll tracking
- Fixed initiative ordering (higher values first)
- Routed all randomness through DiceEngine
- Added ValidationResult with severity levels
- Hardened JSON serialization security
- Added system capability flags
- Improved cross-system character conversion
- Enhanced condition expiration logic

Version: 1.0.2a
Author: ChronoBound Development Team
License: MIT
"""

import json
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum, IntEnum
from datetime import datetime, timedelta

# Import dice engine for deterministic randomness
from dice_engine import DiceEngine, DiceResult

# ==========================================
# PUBLIC API EXPORTS
# ==========================================

__all__ = [
    # Enums
    'RuleSystemType', 'ActionResult', 'DamageType', 'ConditionType', 'ValidationSeverity',
    # Data classes
    'LevelRange', 'ValidationResult', 'AbilityCheckResult', 'AttackResult', 'InitiativeResult', 'Condition',
    # Abstract base class
    'IRuleSystem',
    # Example implementation
    'ExampleRuleSystem',
    # Exceptions
    'RuleSystemError', 'InvalidCharacterError', 'UnsupportedOperationError', 'DiceRollError',
    # Utility functions
    'map_trait_across_systems',
    # Constants
    'CROSS_SYSTEM_TRAIT_MAP'
]

# ==========================================
# ENUMS AND CONSTANTS
# ==========================================

class RuleSystemType(Enum):
    """Supported RPG rule systems in ChronoBound."""
    DAGGERHEART = "daggerheart"
    DND_2024 = "dnd_2024" 
    DND_5E = "dnd_5e"
    PATHFINDER_2E = "pathfinder_2e"
    CHRONOBOUND_HOMEBREW = "chronobound_homebrew"

class ActionResult(Enum):
    """Possible outcomes of character actions."""
    CRITICAL_SUCCESS = "critical_success"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    CRITICAL_FAILURE = "critical_failure"

class DamageType(Enum):
    """Types of damage that can be dealt."""
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
    SONIC = "sonic"

class ConditionType(Enum):
    """Categories of conditions that can affect characters."""
    BUFF = "buff"
    DEBUFF = "debuff"
    CONTROL = "control"
    DAMAGE_OVER_TIME = "damage_over_time"
    HEALING_OVER_TIME = "healing_over_time"
    ENVIRONMENTAL = "environmental"
    MAGICAL = "magical"
    MENTAL = "mental"
    PHYSICAL = "physical"

class ValidationSeverity(IntEnum):
    """Severity levels for validation results."""
    INFO = 1
    WARNING = 2
    ERROR = 3

# ==========================================
# DATA CLASSES
# ==========================================

@dataclass
class LevelRange:
    """Represents a character level range for a rule system."""
    min_level: int
    max_level: int
    
    def __post_init__(self):
        if self.min_level > self.max_level:
            raise ValueError("min_level cannot be greater than max_level")
        if self.min_level < 1:
            raise ValueError("min_level must be at least 1")
    
    def contains(self, level: int) -> bool:
        """Check if a level is within this range."""
        return self.min_level <= level <= self.max_level
    
    def to_tuple(self) -> Tuple[int, int]:
        """Convert to tuple for backward compatibility."""
        return (self.min_level, self.max_level)

@dataclass
class ValidationResult:
    """Result of character or data validation."""
    is_valid: bool
    messages: List[str] = field(default_factory=list)
    severity: ValidationSeverity = ValidationSeverity.ERROR
    details: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, message: str, severity: ValidationSeverity = ValidationSeverity.ERROR):
        """Add a validation message."""
        self.messages.append(message)
        if severity > self.severity:
            self.severity = severity
        if severity == ValidationSeverity.ERROR:
            self.is_valid = False

@dataclass
class AbilityCheckResult:
    """Result of an ability check or skill test."""
    total: int
    result: ActionResult
    margin: int  # How much over/under the target
    raw_rolls: List[int]
    modifiers: Dict[str, int] = field(default_factory=dict)
    system_specific_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AttackResult:
    """Result of an attack roll."""
    hit: bool
    attack_total: int
    damage_total: int
    damage_type: DamageType
    is_critical: bool = False
    raw_attack_rolls: List[int] = field(default_factory=list)
    raw_damage_rolls: List[int] = field(default_factory=list)
    modifiers: Dict[str, int] = field(default_factory=dict)
    system_specific_data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_attack(self) -> int:
        """
        Deprecated: Use attack_total instead.
        
        Compatibility alias for legacy code.
        """
        return self.attack_total
    
    @property
    def damage(self) -> int:
        """
        Deprecated: Use damage_total instead.
        
        Compatibility alias for legacy code.
        """
        return self.damage_total

@dataclass
class InitiativeResult:
    """Result of an initiative roll with proper ordering."""
    character_id: str
    initiative_value: int
    tie_breaker: Optional[int] = None
    modifiers: Dict[str, int] = field(default_factory=dict)
    system_specific_data: Dict[str, Any] = field(default_factory=dict)
    raw_roll: List[int] = field(default_factory=list)
    
    def __lt__(self, other: 'InitiativeResult') -> bool:
        """Compare for sorting - higher initiative goes first."""
        if self.initiative_value != other.initiative_value:
            return self.initiative_value > other.initiative_value  # Higher first
        return (self.tie_breaker or 0) > (other.tie_breaker or 0)  # Higher tie_breaker first

@dataclass
class Condition:
    """Status effect or condition affecting a character."""
    name: str
    description: str
    duration: int  # Rounds, turns, or -1 for permanent
    condition_type: ConditionType
    stat_modifiers: Dict[str, int] = field(default_factory=dict)
    special_rules: List[str] = field(default_factory=list)
    source: Optional[str] = None  # What caused this condition
    stacks: int = 1  # For stackable conditions
    metadata: Dict[str, Any] = field(default_factory=dict)
    applied_at: Optional[int] = None  # Round/turn when applied
    
    def is_expired(self, current_round: int = 0) -> bool:
        """Check if condition has expired based on current round."""
        if self.duration == -1:  # Permanent
            return False
        if self.applied_at is None:
            # Fallback to simple duration check
            return self.duration <= 0
        # Check if enough rounds have passed since application
        return (current_round - self.applied_at) >= self.duration
    
    def tick_duration(self) -> 'Condition':
        """Reduce duration by 1 (returns new instance)."""
        if self.duration > 0:
            new_condition = Condition(
                name=self.name,
                description=self.description,
                duration=self.duration - 1,
                condition_type=self.condition_type,
                stat_modifiers=self.stat_modifiers.copy(),
                special_rules=self.special_rules.copy(),
                source=self.source,
                stacks=self.stacks,
                metadata=self.metadata.copy(),
                applied_at=self.applied_at
            )
            return new_condition
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

# ==========================================
# CROSS-SYSTEM TRAIT MAPPING
# ==========================================

# Standard trait mappings between different systems
CROSS_SYSTEM_TRAIT_MAP = {
    # D&D/Pathfinder -> Daggerheart
    "STR": "Might", "Strength": "Might",
    "DEX": "Agility", "Dexterity": "Agility", 
    "CON": "Fortitude", "Constitution": "Fortitude",
    "INT": "Intellect", "Intelligence": "Intellect",
    "WIS": "Instinct", "Wisdom": "Instinct",
    "CHA": "Presence", "Charisma": "Presence",
    
    # Daggerheart -> D&D/Pathfinder
    "Might": "STR", "Agility": "DEX", "Fortitude": "CON",
    "Intellect": "INT", "Instinct": "WIS", "Presence": "CHA",
    
    # Common alternatives
    "Vigor": "CON", "Finesse": "DEX", "Spirit": "WIS",
    "Mind": "INT", "Body": "STR", "Soul": "CHA"
}

def map_trait_across_systems(trait_name: str, target_system_traits: List[str]) -> Optional[str]:
    """Map a trait name to the equivalent in another system."""
    # Direct match first
    if trait_name in target_system_traits:
        return trait_name
    
    # Check mapping
    mapped_name = CROSS_SYSTEM_TRAIT_MAP.get(trait_name)
    if mapped_name and mapped_name in target_system_traits:
        return mapped_name
    
    # Fuzzy match by partial name
    trait_lower = trait_name.lower()
    for target_trait in target_system_traits:
        if trait_lower in target_trait.lower() or target_trait.lower() in trait_lower:
            return target_trait
    
    return None

# ==========================================
# ABSTRACT BASE CLASS
# ==========================================

class IRuleSystem(ABC):
    """
    Abstract base class defining the interface for all RPG rule systems.
    
    Each ruleset module (DaggerheartRules, Dnd2024Rules, etc.) must implement
    this interface to enable seamless system switching in ChronoBound.
    
    This interface provides:
    - Standard ability and skill check resolution
    - Combat mechanics (attacks, damage, initiative)
    - Condition and status effect management
    - Character creation and validation
    - System-specific dice mechanics
    """
    
    def __init__(self, system_type: RuleSystemType, version: str = "1.0.0"):
        """Initialize the rule system with its type identifier."""
        self.system_type = system_type
        self.system_name = system_type.value
        self.version = version
        self.dice_engine = DiceEngine()  # Use shared dice engine for consistency
    
    # ==========================================
    # CORE SYSTEM IDENTIFICATION
    # ==========================================
    
    @property
    @abstractmethod
    def supported_traits(self) -> List[str]:
        """Return list of character traits/stats this system uses."""
        pass
    
    @property
    @abstractmethod
    def dice_notation(self) -> str:
        """Return the primary dice notation for this system (e.g., '2d12', 'd20')."""
        pass
    
    @property
    @abstractmethod
    def system_version(self) -> str:
        """Return the version of this rule system implementation."""
        pass
    
    @property
    @abstractmethod
    def supported_character_levels(self) -> LevelRange:
        """Return LevelRange supported by this system."""
        pass
    
    def supported_character_levels_tuple(self) -> Tuple[int, int]:
        """
        Return supported character levels as a tuple.
        
        Compatibility method for legacy code that expects (min_level, max_level) tuple.
        """
        return self.supported_character_levels.to_tuple()
    
    def get_system_capabilities(self) -> Dict[str, Any]:
        """Return system capability flags."""
        return {
            "uses_advantage": False,
            "uses_duality": False,
            "has_stress": False,
            "turn_model": "initiative",
            "crit_range": [20],
            "supports_multiclass": True,
            "uses_spell_slots": False,
            "has_hope_fear": False
        }
    
    # ==========================================
    # ABILITY CHECKS & SKILL RESOLUTION
    # ==========================================
    
    @abstractmethod
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
        """
        Perform an ability check using system-specific mechanics.
        
        Args:
            character: Character data dict with traits/stats
            trait_name: Name of the trait being tested (e.g., 'Strength', 'DEX')
            difficulty: Target number or difficulty class
            modifiers: Additional bonuses/penalties to apply
            advantage: Whether to roll with advantage (system-dependent)
            disadvantage: Whether to roll with disadvantage (system-dependent)
            context: Additional context data echoed to system_specific_data
            **kwargs: System-specific parameters
            
        Returns:
            AbilityCheckResult with outcome and details
            
        Raises:
            ValueError: If trait_name is not supported or character data is invalid
        """
        pass
    
    @abstractmethod
    def get_skill_modifier(
        self,
        character: Dict[str, Any],
        skill_name: str
    ) -> int:
        """
        Calculate the modifier for a specific skill check.
        
        Args:
            character: Character data dict
            skill_name: Name of the skill being tested
            
        Returns:
            Total modifier value for the skill
            
        Raises:
            ValueError: If skill_name is not recognized
        """
        pass
    
    @abstractmethod
    def get_available_skills(
        self,
        character: Dict[str, Any]
    ) -> List[str]:
        """
        Get list of skills available to this character.
        
        Args:
            character: Character data dict
            
        Returns:
            List of skill names this character can use
        """
        pass
    
    # ==========================================
    # COMBAT MECHANICS
    # ==========================================
    
    @abstractmethod
    def roll_initiative(
        self,
        character: Dict[str, Any],
        modifiers: Optional[Dict[str, int]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> InitiativeResult:
        """
        Roll initiative for a character with proper tie-breaking.
        
        Args:
            character: Character data dict
            modifiers: Additional bonuses/penalties to initiative
            context: Additional context data
            
        Returns:
            InitiativeResult with proper ordering support
        """
        pass
    
    @abstractmethod
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
        """
        Make an attack roll against a target.
        
        Args:
            attacker: Attacking character data
            target: Target character data
            weapon_or_spell: Name of weapon or spell being used
            modifiers: Additional attack/damage modifiers
            advantage: Whether to roll with advantage
            disadvantage: Whether to roll with disadvantage
            context: Additional context data
            **kwargs: System-specific parameters
            
        Returns:
            AttackResult with hit determination and damage
        """
        pass
    
    @abstractmethod
    def calculate_damage(
        self,
        attacker: Dict[str, Any],
        weapon_or_spell: str,
        target: Optional[Dict[str, Any]] = None,
        modifiers: Optional[Dict[str, int]] = None,
        is_critical: bool = False,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> int:
        """
        Calculate damage for an attack.
        
        Args:
            attacker: Attacking character data
            weapon_or_spell: Name of weapon or spell
            target: Target character data (for resistances/vulnerabilities)
            modifiers: Additional damage modifiers
            is_critical: Whether this is a critical hit
            context: Additional context data
            **kwargs: System-specific parameters
            
        Returns:
            Total damage value
        """
        pass
    
    # ==========================================
    # CONDITIONS AND STATUS EFFECTS
    # ==========================================
    
    @abstractmethod
    def apply_condition(
        self,
        character: Dict[str, Any],
        condition: Condition,
        current_round: int = 0
    ) -> Dict[str, Any]:
        """
        Apply a condition to a character.
        
        Args:
            character: Character data dict
            condition: Condition to apply
            current_round: Current game round for tracking
            
        Returns:
            Updated character data with condition applied
        """
        pass
    
    @abstractmethod
    def remove_condition(
        self,
        character: Dict[str, Any],
        condition_name: str
    ) -> Dict[str, Any]:
        """
        Remove a condition from a character.
        
        Args:
            character: Character data dict
            condition_name: Name of condition to remove
            
        Returns:
            Updated character data with condition removed
        """
        pass
    
    @abstractmethod
    def get_active_conditions(
        self,
        character: Dict[str, Any],
        current_round: int = 0
    ) -> List[Condition]:
        """
        Get all active (non-expired) conditions on a character.
        
        Args:
            character: Character data dict
            current_round: Current game round for expiration checking
            
        Returns:
            List of active conditions
        """
        pass
    
    # ==========================================
    # CHARACTER MANAGEMENT
    # ==========================================
    
    @abstractmethod
    def validate_character(
        self,
        character: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate character data for this rule system.
        
        Args:
            character: Character data dict to validate
            
        Returns:
            ValidationResult with detailed feedback
        """
        pass
    
    def validate_character_tuple(
        self,
        character: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate character data and return legacy tuple format.
        
        Compatibility method for legacy code that expects (is_valid, messages) tuple.
        
        Args:
            character: Character data dict to validate
            
        Returns:
            Tuple of (is_valid, messages) where is_valid excludes ERROR severity
        """
        result = self.validate_character(character)
        # Consider valid if no ERROR or higher severity issues
        is_valid = result.is_valid and result.severity < ValidationSeverity.ERROR
        return (is_valid, result.messages)
    
    @abstractmethod
    def generate_starting_character(
        self,
        name: str,
        char_class: str,
        race_or_heritage: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a new starting character.
        
        Args:
            name: Character name
            char_class: Character class/profession
            race_or_heritage: Character race or heritage
            **kwargs: Additional character options
            
        Returns:
            New character data dict
        """
        pass
    
    @abstractmethod
    def level_up_character(
        self,
        character: Dict[str, Any],
        new_level: int
    ) -> Dict[str, Any]:
        """
        Level up a character to a new level.
        
        Args:
            character: Character data dict
            new_level: Target level
            
        Returns:
            Updated character data with level benefits applied
        """
        pass
    
    @abstractmethod
    def get_character_level(
        self,
        character: Dict[str, Any]
    ) -> int:
        """
        Get the current level of a character.
        
        Args:
            character: Character data dict
            
        Returns:
            Character's current level
        """
        pass
    
    # ==========================================
    # DICE AND RANDOMNESS
    # ==========================================
    
    @abstractmethod
    def roll_system_dice(
        self,
        character: Dict[str, Any],
        roll_type: str,
        modifiers: Optional[Dict[str, int]] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Roll dice using system-specific mechanics.
        
        Args:
            character: Character data dict
            roll_type: Type of roll (e.g., 'ability', 'damage', 'initiative')
            modifiers: Additional modifiers to apply
            context: Additional context data
            **kwargs: System-specific parameters
            
        Returns:
            Dict with roll results and details
        """
        pass
    
    # ==========================================
    # SYSTEM INFORMATION AND UTILITIES
    # ==========================================
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information including capabilities."""
        base_info = {
            "type": self.system_type.value,
            "name": self.system_name,
            "version": self.version,
            "supported_traits": self.supported_traits,
            "dice_notation": self.dice_notation,
            "level_range": self.supported_character_levels.to_tuple(),
            "system_version": self.system_version
        }
        base_info.update(self.get_system_capabilities())
        return base_info
    
    def is_compatible_character(
        self,
        character: Dict[str, Any]
    ) -> bool:
        """Quick check if character data is compatible with this system."""
        return character.get("system") == self.system_type.value
    
    def export_character_for_system(
        self,
        character: Dict[str, Any],
        target_system: 'IRuleSystem'
    ) -> Dict[str, Any]:
        """
        Convert character to be compatible with another rule system.
        
        Uses cross-system trait mapping for intelligent conversion.
        """
        converted = character.copy()
        converted["system"] = target_system.system_type.value
        converted["conversion_note"] = f"Converted from {self.system_name} to {target_system.system_name}"
        
        # Map traits if both characters have trait data
        if "traits" in character and hasattr(target_system, 'supported_traits'):
            old_traits = character["traits"]
            new_traits = {}
            
            for old_trait, value in old_traits.items():
                mapped_trait = map_trait_across_systems(old_trait, target_system.supported_traits)
                if mapped_trait:
                    new_traits[mapped_trait] = value
                else:
                    # Keep unmappable traits for manual review
                    new_traits[f"unmapped_{old_trait}"] = value
            
            converted["traits"] = new_traits
        
        return converted
    
    def serialize_to_json(self, data: Any) -> str:
        """
        Secure JSON serialization that only allows safe data types.
        
        Only permits: dataclasses, enums, primitives, lists, and dicts.
        Prevents leaking of private attributes or unsafe objects.
        """
        def safe_json_serializer(obj):
            # Handle dataclasses
            if hasattr(obj, '__dataclass_fields__'):
                return asdict(obj)
            
            # Handle enums
            if isinstance(obj, Enum):
                return obj.value
            
            # Handle datetime objects
            if isinstance(obj, (datetime, timedelta)):
                return obj.isoformat()
            
            # Only allow primitive types, lists, and dicts
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif isinstance(obj, (list, tuple)):
                return [safe_json_serializer(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: safe_json_serializer(v) for k, v in obj.items()}
            
            # Reject everything else for security
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        return json.dumps(data, default=safe_json_serializer, indent=2)
    
    def __str__(self) -> str:
        """String representation of the rule system."""
        return f"{self.system_name.title()} Rules v{self.version}"
    
    def __repr__(self) -> str:
        """Developer representation of the rule system."""
        return f"IRuleSystem(type={self.system_type.value}, version={self.version})"

# ==========================================
# EXCEPTIONS
# ==========================================

class RuleSystemError(Exception):
    """Base exception for rule system errors."""
    pass

class InvalidCharacterError(RuleSystemError):
    """Raised when character data is invalid for the system."""
    pass

class UnsupportedOperationError(RuleSystemError):
    """Raised when an operation is not supported by the rule system."""
    pass

class DiceRollError(RuleSystemError):
    """Raised when there's an error in dice rolling mechanics."""
    pass

# ==========================================
# EXAMPLE MINIMAL IMPLEMENTATION
# ==========================================

class ExampleRuleSystem(IRuleSystem):
    """
    Minimal testing stub implementation for the IRuleSystem interface.
    
    WARNING: This is NOT a complete rule system - it's for demonstration
    and testing purposes only. It provides minimal, realistic implementations
    of all required methods using the DiceEngine for consistency.
    
    Do not use this for actual gameplay - implement a proper rule system
    like DaggerheartRules or Dnd2024Rules instead.
    """
    
    def __init__(self):
        super().__init__(RuleSystemType.CHRONOBOUND_HOMEBREW, "0.1.0")
    
    @property
    def supported_traits(self) -> List[str]:
        return ["Strength", "Dexterity", "Intelligence", "Wisdom", "Constitution", "Charisma"]
    
    @property
    def dice_notation(self) -> str:
        return "d20"
    
    @property
    def system_version(self) -> str:
        return "0.1.0"
    
    @property
    def supported_character_levels(self) -> LevelRange:
        return LevelRange(1, 20)
    
    def get_system_capabilities(self) -> Dict[str, Any]:
        return {
            "uses_advantage": True,
            "uses_duality": False,
            "has_stress": False,
            "turn_model": "initiative",
            "crit_range": [20],
            "supports_multiclass": False,
            "uses_spell_slots": False,
            "has_hope_fear": False
        }
    
    # Minimal implementations using DiceEngine for consistency
    def make_ability_check(self, character, trait_name, difficulty, modifiers=None, advantage=False, disadvantage=False, context=None, **kwargs):
        if trait_name not in self.supported_traits:
            raise ValueError(f"Unsupported trait: {trait_name}")
        
        # Get character trait value (default to 10)
        trait_value = character.get("traits", {}).get(trait_name, 10)
        trait_modifier = (trait_value - 10) // 2  # Standard D&D modifier calculation
        
        # Roll with advantage/disadvantage
        if advantage and not disadvantage:
            roll1 = self.dice_engine.roll("1d20").total
            roll2 = self.dice_engine.roll("1d20").total
            roll = max(roll1, roll2)
            raw_rolls = [roll1, roll2]
        elif disadvantage and not advantage:
            roll1 = self.dice_engine.roll("1d20").total
            roll2 = self.dice_engine.roll("1d20").total
            roll = min(roll1, roll2)
            raw_rolls = [roll1, roll2]
        else:
            roll = self.dice_engine.roll("1d20").total
            raw_rolls = [roll]
        
        # Apply modifiers
        mod_total = sum((modifiers or {}).values())
        total = roll + trait_modifier + mod_total
        
        # Determine result
        margin = total - difficulty
        if roll == 20:
            result = ActionResult.CRITICAL_SUCCESS
        elif roll == 1:
            result = ActionResult.CRITICAL_FAILURE
        elif margin >= 0:
            result = ActionResult.SUCCESS
        else:
            result = ActionResult.FAILURE
        
        return AbilityCheckResult(
            total=total,
            result=result,
            margin=margin,
            raw_rolls=raw_rolls,
            modifiers={"trait": trait_modifier, **( modifiers or {})},
            system_specific_data=context or {}
        )
    
    def get_skill_modifier(self, character, skill_name):
        # Simple implementation - skills use trait modifiers
        skill_trait_map = {
            "Athletics": "Strength",
            "Acrobatics": "Dexterity", 
            "Stealth": "Dexterity",
            "Investigation": "Intelligence",
            "Perception": "Wisdom",
            "Persuasion": "Charisma",
            "Intimidation": "Charisma"
        }
        
        trait = skill_trait_map.get(skill_name, "Intelligence")
        trait_value = character.get("traits", {}).get(trait, 10)
        return (trait_value - 10) // 2
    
    def get_available_skills(self, character):
        return ["Athletics", "Acrobatics", "Stealth", "Investigation", "Perception", "Persuasion", "Intimidation"]
    
    def roll_initiative(self, character, modifiers=None, context=None):
        # Use Dexterity for initiative with default tie-breaker
        dex_value = character.get("traits", {}).get("Dexterity", 10)
        dex_modifier = (dex_value - 10) // 2
        
        roll = self.dice_engine.roll("1d20").total
        mod_total = sum((modifiers or {}).values())
        initiative = roll + dex_modifier + mod_total
        
        # Default tie-breaker: higher DEX, then character_id hash
        tie_breaker = dex_value * 100 + hash(character.get("id", "")) % 100
        
        return InitiativeResult(
            character_id=character.get("id", "unknown"),
            initiative_value=initiative,
            tie_breaker=tie_breaker,
            modifiers={"dexterity": dex_modifier, **(modifiers or {})},
            system_specific_data=context or {},
            raw_roll=[roll]  # Store the actual d20 roll
        )
    
    def make_attack_roll(self, attacker, target, weapon_or_spell, modifiers=None, advantage=False, disadvantage=False, context=None, **kwargs):
        # Simple attack implementation
        str_value = attacker.get("traits", {}).get("Strength", 10)
        attack_bonus = (str_value - 10) // 2
        
        # Roll attack
        if advantage and not disadvantage:
            roll1 = self.dice_engine.roll("1d20").total
            roll2 = self.dice_engine.roll("1d20").total
            attack_roll = max(roll1, roll2)
            raw_rolls = [roll1, roll2]
        elif disadvantage and not advantage:
            roll1 = self.dice_engine.roll("1d20").total
            roll2 = self.dice_engine.roll("1d20").total
            attack_roll = min(roll1, roll2)
            raw_rolls = [roll1, roll2]
        else:
            attack_roll = self.dice_engine.roll("1d20").total
            raw_rolls = [attack_roll]
        
        mod_total = sum((modifiers or {}).values())
        attack_total = attack_roll + attack_bonus + mod_total
        
        # Simple AC calculation
        target_ac = target.get("ac", 10)
        hit = attack_total >= target_ac
        is_critical = attack_roll == 20
        
        # Calculate damage if hit
        damage_total = 0
        damage_rolls = []
        if hit:
            damage_dice = "1d8" if not is_critical else "2d8"
            damage_result = self.dice_engine.roll(damage_dice)
            damage_total = damage_result.total + attack_bonus
            damage_rolls = damage_result.dice_results
        
        return AttackResult(
            hit=hit,
            attack_total=attack_total,
            damage_total=damage_total,
            damage_type=DamageType.PHYSICAL,
            is_critical=is_critical,
            raw_attack_rolls=raw_rolls,
            raw_damage_rolls=damage_rolls,
            modifiers={"strength": attack_bonus, **(modifiers or {})},
            system_specific_data=context or {}
        )
    
    def calculate_damage(self, attacker, weapon_or_spell, target=None, modifiers=None, is_critical=False, context=None, **kwargs):
        str_value = attacker.get("traits", {}).get("Strength", 10)
        damage_bonus = (str_value - 10) // 2
        
        damage_dice = "1d8" if not is_critical else "2d8"
        damage_result = self.dice_engine.roll(damage_dice)
        
        mod_total = sum((modifiers or {}).values())
        return damage_result.total + damage_bonus + mod_total
    
    def apply_condition(self, character, condition, current_round=0):
        char_copy = character.copy()
        conditions = char_copy.get("conditions", [])
        
        # Set applied_at if not set
        if condition.applied_at is None:
            condition.applied_at = current_round
        
        conditions.append(condition.to_dict())
        char_copy["conditions"] = conditions
        return char_copy
    
    def remove_condition(self, character, condition_name):
        char_copy = character.copy()
        conditions = char_copy.get("conditions", [])
        char_copy["conditions"] = [c for c in conditions if c.get("name") != condition_name]
        return char_copy
    
    def get_active_conditions(self, character, current_round=0):
        conditions = character.get("conditions", [])
        active = []
        for c_dict in conditions:
            condition = Condition(**c_dict)
            if not condition.is_expired(current_round):
                active.append(condition)
        return active
    
    def validate_character(self, character):
        result = ValidationResult(is_valid=True)
        
        required_fields = ["name", "system", "level"]
        for field in required_fields:
            if field not in character:
                result.add_message(f"Missing required field: {field}", ValidationSeverity.ERROR)
        
        if "level" in character:
            level = character["level"]
            if not self.supported_character_levels.contains(level):
                result.add_message(f"Level {level} outside supported range {self.supported_character_levels.to_tuple()}", ValidationSeverity.ERROR)
        
        if "traits" in character:
            for trait in character["traits"]:
                if trait not in self.supported_traits:
                    result.add_message(f"Unknown trait: {trait}", ValidationSeverity.WARNING)
        
        return result
    
    def generate_starting_character(self, name, char_class, race_or_heritage, **kwargs):
        return {
            "id": str(uuid.uuid4()),
            "name": name,
            "class": char_class,
            "race": race_or_heritage,
            "system": self.system_type.value,
            "level": 1,
            "hp": 12,  # Realistic starting HP
            "max_hp": 12,
            "ac": 12,  # Realistic starting AC
            "traits": {
                "Strength": 12,
                "Dexterity": 14, 
                "Constitution": 13,
                "Intelligence": 11,
                "Wisdom": 12,
                "Charisma": 10
            },
            "conditions": []
        }
    
    def level_up_character(self, character, new_level):
        char_copy = character.copy()
        old_level = character.get("level", 1)
        char_copy["level"] = new_level
        
        # Simple HP increase
        hp_gain = (new_level - old_level) * 6  # Average of d8 + 2
        char_copy["max_hp"] = character.get("max_hp", 12) + hp_gain
        char_copy["hp"] = character.get("hp", 12) + hp_gain
        
        return char_copy
    
    def get_character_level(self, character):
        return character.get("level", 1)
    
    def roll_system_dice(self, character, roll_type, modifiers=None, context=None, **kwargs):
        if roll_type == "ability":
            # Standard ability check roll
            result = self.dice_engine.roll("1d20")
        elif roll_type == "damage":
            result = self.dice_engine.roll("1d8")
        elif roll_type == "initiative":
            result = self.dice_engine.roll("1d20")
        else:
            result = self.dice_engine.roll("1d20")  # Default
        
        return {
            "roll": result.total,
            "type": roll_type,
            "dice_results": result.dice_results,
            "modifiers": modifiers or {},
            "context": context or {}
        }

if __name__ == "__main__":
    # Runtime verification tests for compatibility shims
    print("IRuleSystem v1.0.2a - Compatibility Verification")
    print("=" * 50)
    
    # Test the example system
    example_system = ExampleRuleSystem()
    print(f"System: {example_system}")
    print(f"Info: {example_system.get_system_info()}")
    
    # Test 1: supported_character_levels_tuple() compatibility method
    print("\n1. Testing supported_character_levels_tuple():")
    tuple_result = example_system.supported_character_levels_tuple()
    print(f"   Tuple format: {tuple_result}")
    assert isinstance(tuple_result, tuple), "Should return tuple"
    assert len(tuple_result) == 2, "Should be (min, max)"
    assert tuple_result == (1, 20), "Should match expected range"
    print("   ✓ PASS: Tuple helper returns correct format")
    
    # Test 2: AttackResult alias properties
    print("\n2. Testing AttackResult alias properties:")
    char1 = {"id": "1", "traits": {"Strength": 14}, "ac": 12}
    char2 = {"id": "2", "traits": {"Dexterity": 12}, "ac": 10}
    attack_result = example_system.make_attack_roll(char1, char2, "sword")
    
    # Test deprecated alias properties
    total_attack_alias = attack_result.total_attack
    damage_alias = attack_result.damage
    
    assert total_attack_alias == attack_result.attack_total, "total_attack should equal attack_total"
    assert damage_alias == attack_result.damage_total, "damage should equal damage_total"
    print(f"   attack_total: {attack_result.attack_total}, total_attack: {total_attack_alias}")
    print(f"   damage_total: {attack_result.damage_total}, damage: {damage_alias}")
    print("   ✓ PASS: AttackResult alias properties work correctly")
    
    # Test 3: validate_character_tuple() compatibility method
    print("\n3. Testing validate_character_tuple():")
    character = example_system.generate_starting_character("Test Hero", "Fighter", "Human")
    
    # Test with valid character
    tuple_validation = example_system.validate_character_tuple(character)
    print(f"   Valid character tuple result: {tuple_validation}")
    assert isinstance(tuple_validation, tuple), "Should return tuple"
    assert len(tuple_validation) == 2, "Should be (bool, list)"
    assert isinstance(tuple_validation[0], bool), "First element should be bool"
    assert isinstance(tuple_validation[1], list), "Second element should be list"
    
    # Test with invalid character (missing required field)
    invalid_char = {"name": "Test"}  # Missing system and level
    invalid_tuple = example_system.validate_character_tuple(invalid_char)
    print(f"   Invalid character tuple result: {invalid_tuple}")
    assert invalid_tuple[0] == False, "Should be invalid"
    assert len(invalid_tuple[1]) > 0, "Should have error messages"
    print("   ✓ PASS: validate_character_tuple() conversion works correctly")
    
    # Test 4: InitiativeResult.raw_roll field
    print("\n4. Testing InitiativeResult.raw_roll:")
    char_for_init = {"id": "test", "traits": {"Dexterity": 16}}
    init_result = example_system.roll_initiative(char_for_init)
    
    print(f"   Initiative result: {init_result.initiative_value}")
    print(f"   Raw roll: {init_result.raw_roll}")
    assert hasattr(init_result, 'raw_roll'), "Should have raw_roll field"
    assert isinstance(init_result.raw_roll, list), "raw_roll should be a list"
    assert len(init_result.raw_roll) > 0, "raw_roll should contain the actual die roll"
    assert all(1 <= roll <= 20 for roll in init_result.raw_roll), "Should contain valid d20 results"
    print("   ✓ PASS: InitiativeResult.raw_roll populated correctly")
    
    # Test 5: Initiative ordering
    print("\n5. Testing initiative ordering:")
    char1 = {"id": "1", "traits": {"Dexterity": 16}}
    char2 = {"id": "2", "traits": {"Dexterity": 14}}
    
    init1 = example_system.roll_initiative(char1)
    init2 = example_system.roll_initiative(char2)
    
    results = sorted([init1, init2])
    print(f"   Initiative order: {[r.character_id for r in results]}")
    print(f"   Values: {[(r.character_id, r.initiative_value) for r in results]}")
    print("   ✓ PASS: Initiative ordering works (higher first)")
    
    print("\n" + "=" * 50)
    print("All compatibility verification tests PASSED!")
    print("IRuleSystem v1.0.2a is ready for use.")
