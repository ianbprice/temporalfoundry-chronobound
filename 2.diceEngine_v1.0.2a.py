"""
ChronoBound DiceEngine v1.0.2a
=============================

Comprehensive dice rolling engine supporting multiple RPG systems.
Handles basic rolls, advantage/disadvantage, exploding dice, and system-specific mechanics.

Changelog v1.0.2a:
- Added deprecation warning for roll_daggerheart_duality (use rules modules)
- Fixed unsafe finally-return pattern in duality rolls
- Added _fmt_mod helper to eliminate dangling '+' in notation strings
- Made DiceResult.from_dict a proper @classmethod with round-trip support
- Enhanced critical hit policy documentation (single-die only by default)
- Added comprehensive runtime self-checks

This module provides:
- Standard dice rolling (d4, d6, d8, d10, d12, d20, d100)
- Advantage/disadvantage mechanics (D&D style)
- Exploding dice mechanics with custom thresholds
- Basic Daggerheart 2d12 Hope/Fear duality system (deprecated - use rules modules)
- Dice notation parsing with keep/drop operations
- Comprehensive result tracking with history management
- Per-instance RNG isolation
- Context preservation and percentile rolling

Version: 1.0.2a
Author: ChronoBound Development Team
License: MIT
"""

import random
import re
import logging
import warnings
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json

# ==========================================
# ENUMS AND CONSTANTS
# ==========================================

class DiceType(Enum):
    """Standard dice types supported by the engine."""
    D4 = 4
    D6 = 6
    D8 = 8
    D10 = 10
    D12 = 12
    D20 = 20
    D100 = 100

class RollType(Enum):
    """Different types of dice rolling mechanics."""
    NORMAL = "normal"
    ADVANTAGE = "advantage"
    DISADVANTAGE = "disadvantage"
    EXPLODING = "exploding"
    DAGGERHEART_DUALITY = "daggerheart_duality"
    KEEP_HIGHEST = "keep_highest"
    KEEP_LOWEST = "keep_lowest"
    DROP_HIGHEST = "drop_highest"
    DROP_LOWEST = "drop_lowest"

class CriticalType(Enum):
    """Types of critical results."""
    NONE = "none"
    SUCCESS = "critical_success"
    FAILURE = "critical_failure"
    BOTH = "mixed_critical"

# ==========================================
# EXCEPTIONS
# ==========================================

class DiceEngineError(Exception):
    """Base exception for dice engine errors."""
    pass

class InvalidDiceNotationError(DiceEngineError):
    """Raised when dice notation is invalid."""
    pass

class InvalidDiceTypeError(DiceEngineError):
    """Raised when an invalid die type is used."""
    pass

class ExplodingDiceError(DiceEngineError):
    """Raised when exploding dice go infinite."""
    pass

# ==========================================
# DATA CLASSES
# ==========================================

@dataclass
class DiceExpression:
    """Structured representation of a dice roll expression."""
    num_dice: int
    sides: int
    modifier: int = 0
    operation: Optional[str] = None  # keep_highest, keep_lowest, drop_highest, drop_lowest
    count: Optional[int] = None
    exploding: bool = False
    explode_at: Optional[int] = None
    
    def __post_init__(self):
        """Validate the dice expression after initialization."""
        if self.sides < 2:
            raise InvalidDiceTypeError(f"Die must have at least 2 sides, got {self.sides}")
        if self.num_dice < 1:
            raise InvalidDiceNotationError(f"Must roll at least 1 die, got {self.num_dice}")

@dataclass
class DiceResult:
    """
    Comprehensive result of a dice roll operation.
    
    Contains all information needed for different RPG systems to interpret
    the outcome and apply appropriate rules.
    
    Critical Hit Policy:
    - is_critical_success/is_critical_failure are only set for single-die rolls by default
    - Multi-die pools (2d12, 4d6kh3, etc.) leave critical determination to rules modules
    - This ensures system-agnostic behavior while allowing rules modules to implement
      their own critical hit mechanics for complex dice pools
    """
    total: int
    individual_rolls: List[int]
    modifiers: Dict[str, int]
    final_total: int
    roll_type: RollType
    dice_notation: str
    is_critical_success: bool = False
    is_critical_failure: bool = False
    hope_die: Optional[int] = None
    fear_die: Optional[int] = None
    dominant_outcome: Optional[str] = None  # "hope", "fear", or "balanced"
    discarded_rolls: List[int] = field(default_factory=list)
    special_effects: List[str] = field(default_factory=list)
    explosion_count: int = 0
    exploded_dice: List[int] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')
    system_context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary for serialization."""
        return {
            "total": self.total,
            "individual_rolls": self.individual_rolls,
            "modifiers": self.modifiers,
            "final_total": self.final_total,
            "roll_type": self.roll_type.value,
            "dice_notation": self.dice_notation,
            "is_critical_success": self.is_critical_success,
            "is_critical_failure": self.is_critical_failure,
            "hope_die": self.hope_die,
            "fear_die": self.fear_die,
            "dominant_outcome": self.dominant_outcome,
            "discarded_rolls": self.discarded_rolls,
            "special_effects": self.special_effects,
            "explosion_count": self.explosion_count,
            "exploded_dice": self.exploded_dice,
            "timestamp": self.timestamp,
            "system_context": self.system_context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DiceResult':
        """Create a DiceResult from a dictionary."""
        # Convert roll_type back to enum
        data = data.copy()  # Don't mutate input
        data["roll_type"] = RollType(data["roll_type"])
        return cls(**data)
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        result_str = f"{self.dice_notation} → {self.final_total}"
        
        if self.is_critical_success:
            result_str += " (CRITICAL SUCCESS!)"
        elif self.is_critical_failure:
            result_str += " (CRITICAL FAILURE!)"
        
        if self.dominant_outcome:
            result_str += f" [Dominant: {self.dominant_outcome.title()}]"
        
        return result_str
    
    def __repr__(self) -> str:
        """Developer representation."""
        return f"DiceResult(total={self.total}, final={self.final_total}, notation='{self.dice_notation}')"

# ==========================================
# MAIN DICE ENGINE CLASS
# ==========================================

class DiceEngine:
    """
    Core dice rolling engine for ChronoBound.
    
    Provides standardized dice rolling across all supported RPG systems
    with system-specific mechanics and comprehensive result tracking.
    
    Critical Hit Policy:
    The engine only sets critical success/failure flags for single-die rolls by default.
    Multi-die pools (like 4d6kh3, 2d12 duality) leave critical determination to
    rules modules to implement system-specific critical mechanics.
    """
    
    def __init__(self, seed: Optional[int] = None, max_explosions: int = 100, 
                 max_history: int = 1000, record_history: bool = True):
        """
        Initialize the dice engine.
        
        Args:
            seed: Optional random seed for reproducible results (useful for testing)
            max_explosions: Maximum number of dice explosions to prevent infinite loops
            max_history: Maximum number of rolls to keep in history (ring buffer)
            record_history: Whether to record roll history
        """
        self.rng = random.Random(seed)
        
        self.max_explosions = max_explosions
        self.max_history = max_history
        self.record_history = record_history
        self.roll_history: List[DiceResult] = []
        
        # Dice notation regex patterns
        self.dice_pattern = re.compile(r'(\d*)d(\d+)([!](?:\d+)?)?([+-]\d+)?$')
        
        # Advanced notation pattern for keep/drop operations
        self.advanced_pattern = re.compile(r'(\d*)d(\d+)(?:(k[hl])(\d+)|(d[hl])(\d+))?([!](?:\d+)?)?([+-]\d+)?$')
        
        # Standard dice types for validation
        self.standard_dice = [4, 6, 8, 10, 12, 20, 100]
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
    
    def _fmt_mod(self, modifier: int) -> str:
        """
        Format a modifier for dice notation strings.
        
        Args:
            modifier: The numeric modifier
            
        Returns:
            Formatted string (e.g., '+3', '-2', or '' for 0)
        """
        if modifier > 0:
            return f"+{modifier}"
        elif modifier < 0:
            return str(modifier)
        else:
            return ""
    
    def _add_to_history(self, result: DiceResult):
        """Add a result to the roll history with ring buffer management."""
        if not self.record_history:
            return
            
        self.roll_history.append(result)
        
        # Implement ring buffer - remove oldest entries if over limit
        if len(self.roll_history) > self.max_history:
            self.roll_history = self.roll_history[-self.max_history:]
    
    def validate_notation(self, notation: str) -> bool:
        """
        Validate if a dice notation string is supported.
        
        Args:
            notation: Dice notation string to validate
            
        Returns:
            True if notation is valid and supported
        """
        try:
            notation = notation.strip().lower().replace(" ", "")
            
            # Try basic notation first
            if self.dice_pattern.fullmatch(notation):
                return True
            
            # Try advanced notation
            if self.advanced_pattern.fullmatch(notation):
                return True
                
            return False
        except Exception:
            return False
    
    def is_supported_die(self, sides: int) -> bool:
        """
        Check if a die type is supported.
        
        Args:
            sides: Number of sides on the die
            
        Returns:
            True if the die type is supported
        """
        return sides in self.standard_dice or sides > 1
    
    # ==========================================
    # BASIC DICE ROLLING
    # ==========================================
    
    def roll_single(self, sides: int) -> int:
        """
        Roll a single die with specified number of sides.
        
        Args:
            sides: Number of sides on the die
            
        Returns:
            Random integer from 1 to sides (inclusive)
            
        Raises:
            InvalidDiceTypeError: If sides is less than 2
        """
        if sides < 2:
            raise InvalidDiceTypeError(f"Die must have at least 2 sides, got {sides}")
        
        return self.rng.randint(1, sides)
    
    def roll_basic(
        self,
        num_dice: int,
        sides: int,
        modifier: int = 0,
        exploding: bool = False,
        exploding_threshold: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> DiceResult:
        """
        Roll multiple dice with optional modifier and exploding dice.
        
        Critical Hit Policy:
        Only sets critical success/failure flags for single-die rolls (num_dice=1).
        Multi-die rolls leave critical determination to rules modules.
        
        Args:
            num_dice: Number of dice to roll
            sides: Number of sides per die
            modifier: Static modifier to add to the total
            exploding: Whether dice should explode on maximum roll
            exploding_threshold: Custom threshold for exploding (default: maximum value)
            context: Optional context dictionary to store with result
            
        Returns:
            DiceResult with complete information about the roll
            
        Raises:
            InvalidDiceTypeError: If sides is less than 2
            ExplodingDiceError: If exploding dice exceed max_explosions
        """
        try:
            if sides < 2:
                raise InvalidDiceTypeError(f"Die must have at least 2 sides, got {sides}")
            if num_dice < 1:
                raise InvalidDiceNotationError(f"Must roll at least 1 die, got {num_dice}")
            
            individual_rolls = []
            explosion_count = 0
            exploded_dice = []
            
            exploding_threshold = exploding_threshold or sides
            for _ in range(num_dice):
                die_roll = self.roll_single(sides)
                individual_rolls.append(die_roll)
                
                # Handle exploding dice  
                if exploding:
                    current_roll = die_roll
                    explosions_this_die = 0
                    
                    while current_roll >= exploding_threshold and explosions_this_die < self.max_explosions:
                        explosion_roll = self.roll_single(sides)
                        individual_rolls.append(explosion_roll)
                        exploded_dice.append(explosion_roll)
                        current_roll = explosion_roll
                        explosions_this_die += 1
                        explosion_count += 1
                    
                    if explosions_this_die >= self.max_explosions:
                        raise ExplodingDiceError(f"Dice exploded {self.max_explosions} times, stopping to prevent infinite loop")
            
            total = sum(individual_rolls)
            final_total = total + modifier
            
            # Only set critical flags for single-die rolls by default
            is_critical_success = num_dice == 1 and any(die == sides for die in individual_rolls)
            is_critical_failure = num_dice == 1 and any(die == 1 for die in individual_rolls)
            
            # Build notation
            notation = f"{num_dice}d{sides}"
            if exploding:
                if exploding_threshold != sides:
                    notation += f"!{exploding_threshold}"
                else:
                    notation += "!"
            notation += self._fmt_mod(modifier)
            
            result = DiceResult(
                total=total,
                individual_rolls=individual_rolls,
                modifiers={"static": modifier},
                final_total=final_total,
                roll_type=RollType.EXPLODING if exploding else RollType.NORMAL,
                dice_notation=notation,
                is_critical_success=is_critical_success,
                is_critical_failure=is_critical_failure,
                explosion_count=explosion_count,
                exploded_dice=exploded_dice,
                system_context=context
            )
            
            self._add_to_history(result)
            return result
        
        except Exception as e:
            self.logger.error(f"Error in roll_basic: {e}")
            raise
    
    def roll_advantage(
        self,
        sides: int,
        modifier: int = 0,
        context: Optional[Dict[str, Any]] = None
    ) -> DiceResult:
        """
        Roll with advantage (roll twice, take higher).
        
        Critical Hit Policy:
        Sets critical flags based on the kept die result (single-die semantics).
        
        Args:
            sides: Number of sides on each die
            modifier: Static modifier to add
            context: Optional context dictionary
            
        Returns:
            DiceResult with advantage mechanics applied
        """
        if sides < 2:
            raise InvalidDiceTypeError(f"Die must have at least 2 sides, got {sides}")
        
        rolls = []
        
        for _ in range(2):
            roll = self.rng.randint(1, sides)
            rolls.append(roll)
        
        kept_roll = max(rolls)
        final_total = kept_roll + modifier
        
        result = DiceResult(
            total=kept_roll,
            individual_rolls=[kept_roll],
            modifiers={"static": modifier},
            final_total=final_total,
            roll_type=RollType.ADVANTAGE,
            dice_notation=f"d{sides} advantage{self._fmt_mod(modifier)}",
            is_critical_success=kept_roll == sides,
            is_critical_failure=kept_roll == 1,
            discarded_rolls=[min(rolls)],
            special_effects=["advantage"],
            system_context=context
        )
        
        self._add_to_history(result)
        return result
    
    def roll_disadvantage(
        self,
        sides: int,
        modifier: int = 0,
        context: Optional[Dict[str, Any]] = None
    ) -> DiceResult:
        """
        Roll with disadvantage (roll twice, take lower).
        
        Critical Hit Policy:
        Sets critical flags based on the kept die result (single-die semantics).
        
        Args:
            sides: Number of sides on each die
            modifier: Static modifier to add
            context: Optional context dictionary
            
        Returns:
            DiceResult with disadvantage mechanics applied
        """
        if sides < 2:
            raise InvalidDiceTypeError(f"Die must have at least 2 sides, got {sides}")
        
        rolls = []
        
        for _ in range(2):
            roll = self.rng.randint(1, sides)
            rolls.append(roll)
        
        kept_roll = min(rolls)
        final_total = kept_roll + modifier
        
        result = DiceResult(
            total=kept_roll,
            individual_rolls=[kept_roll],
            modifiers={"static": modifier},
            final_total=final_total,
            roll_type=RollType.DISADVANTAGE,
            dice_notation=f"d{sides} disadvantage{self._fmt_mod(modifier)}",
            is_critical_success=kept_roll == sides,
            is_critical_failure=kept_roll == 1,
            discarded_rolls=[max(rolls)],
            special_effects=["disadvantage"],
            system_context=context
        )
        
        self._add_to_history(result)
        return result
    
    # ==========================================
    # DAGGERHEART DUALITY SYSTEM (DEPRECATED)
    # ==========================================
    
    def roll_daggerheart_duality(
        self,
        trait_modifier: int = 0,
        boons: int = 0,
        banes: int = 0,
        context: Optional[Dict[str, Any]] = None
    ) -> DiceResult:
        """
        Roll Daggerheart duality dice (2d12 Hope/Fear system) - DEPRECATED.
        
        DEPRECATION WARNING: This method provides only basic SRD-compliant mechanics.
        For full Daggerheart support including advantage/disadvantage, boon/bane mechanics,
        and advanced duality rules, use the DaggerheartRules module instead.
        
        Basic Implementation:
        - Rolls 2d12 (hope and fear dice)
        - Determines dominant outcome: "hope", "fear", or "balanced" (tie)
        - Sets critical success only when both dice show the same number (doubles)
        - Does NOT implement advantage/disadvantage or complex boon/bane mechanics
        
        Args:
            trait_modifier: Character trait modifier
            boons: Number of boon dice (not fully implemented - use rules module)
            banes: Number of bane dice (not fully implemented - use rules module)
            context: Optional context dictionary
            
        Returns:
            DiceResult with basic Hope/Fear duality information
        """
        # Emit deprecation warning
        warnings.warn(
            "roll_daggerheart_duality is deprecated and provides only basic SRD mechanics. "
            "Use DaggerheartRules module for full system support including advantage/disadvantage "
            "and complex boon/bane mechanics.",
            DeprecationWarning,
            stacklevel=2
        )
        
        try:
            self.logger.debug(f"Rolling basic Daggerheart duality: trait_modifier={trait_modifier}")
            
            # Roll the base 2d12
            hope_die = self.roll_single(12)
            fear_die = self.roll_single(12)
            
            # Determine dominant outcome
            if hope_die > fear_die:
                dominant_outcome = "hope"
                kept_value = hope_die
            elif fear_die > hope_die:
                dominant_outcome = "fear"
                kept_value = fear_die
            else:
                dominant_outcome = "balanced"
                kept_value = hope_die  # Rules say choose Hope on tie
            
            total = kept_value
            final_total = total + trait_modifier
            
            # Check for critical results (doubles)
            is_critical_success = hope_die == fear_die
            is_critical_failure = False  # Basic implementation doesn't set crit failure
            
            special_effects = [dominant_outcome]
            if boons > 0:
                special_effects.append(f"boons:{boons}")
            if banes > 0:
                special_effects.append(f"banes:{banes}")
            
            # Build notation string
            notation = "2d12"
            if boons > 0:
                notation += f"+{boons}boons"
            if banes > 0:
                notation += f"+{banes}banes"
            notation += self._fmt_mod(trait_modifier)
            
            result = DiceResult(
                total=total,
                individual_rolls=[kept_value],
                modifiers={"trait": trait_modifier, "boons": boons, "banes": banes},
                final_total=final_total,
                roll_type=RollType.DAGGERHEART_DUALITY,
                dice_notation=notation,
                is_critical_success=is_critical_success,
                is_critical_failure=is_critical_failure,
                hope_die=hope_die,
                fear_die=fear_die,
                dominant_outcome=dominant_outcome,
                discarded_rolls=[],
                special_effects=special_effects,
                system_context=context
            )
            
            self._add_to_history(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Error in roll_daggerheart_duality: {e}")
            raise
    
    # ==========================================
    # ADVANCED DICE OPERATIONS
    # ==========================================
    
    def roll_notation(
        self,
        notation: str,
        advantage: bool = False,
        disadvantage: bool = False,
        exploding: bool = False,
        context: Optional[Dict[str, Any]] = None
    ) -> DiceResult:
        """
        Roll dice from standard notation (e.g., "3d6+2", "d20", "2d12-1").
        
        Args:
            notation: Dice notation (e.g., "3d6+2", "d20", "2d12-1")
            advantage: Roll with advantage (if single die)
            disadvantage: Roll with disadvantage (if single die)
            exploding: Use exploding dice mechanics
            context: Optional context dictionary
            
        Returns:
            DiceResult with complete roll information
        """
        try:
            # Try advanced notation first
            expr = self.parse_advanced_notation(notation)
            return self.roll_with_operations(expr, context=context)
        except InvalidDiceNotationError:
            # Fall back to basic notation
            expr = self.parse_dice_notation(notation)
            
            # Handle advantage/disadvantage for single die rolls
            if expr.num_dice == 1 and advantage and not disadvantage:
                return self.roll_advantage(expr.sides, expr.modifier, context)
            elif expr.num_dice == 1 and disadvantage and not advantage:
                return self.roll_disadvantage(expr.sides, expr.modifier, context)
            else:
                return self.roll_basic(expr.num_dice, expr.sides, expr.modifier, exploding or expr.exploding, expr.explode_at, context)
    
    def roll_with_operations(
        self,
        expr: DiceExpression,
        context: Optional[Dict[str, Any]] = None
    ) -> DiceResult:
        """
        Roll dice with keep/drop operations.
        
        Critical Hit Policy:
        Only sets critical flags when exactly one die is kept after operations.
        Multi-die results leave critical determination to rules modules.
        
        Args:
            expr: DiceExpression containing all roll parameters
            context: Optional context dictionary
            
        Returns:
            DiceResult with operation applied
        """
        # Roll all the dice first
        all_rolls = []
        explosion_count = 0
        exploded_dice = []
        
        for _ in range(expr.num_dice):
            die_roll = self.roll_single(expr.sides)
            all_rolls.append(die_roll)
            
            # Handle exploding dice
            if expr.exploding:
                exploding_threshold = expr.explode_at or expr.sides
                current_roll = die_roll
                explosions_this_die = 0
                
                while current_roll >= exploding_threshold and explosions_this_die < self.max_explosions:
                    explosion_roll = self.roll_single(expr.sides)
                    all_rolls.append(explosion_roll)
                    exploded_dice.append(explosion_roll)
                    current_roll = explosion_roll
                    explosions_this_die += 1
                    explosion_count += 1
        
        # Apply operation to determine kept dice
        if expr.operation and expr.count:
            if expr.operation == 'keep_highest':
                kept_dice = sorted(all_rolls, reverse=True)[:expr.count]
                discarded = sorted(all_rolls, reverse=True)[expr.count:]
                roll_type = RollType.KEEP_HIGHEST
            elif expr.operation == 'keep_lowest':
                kept_dice = sorted(all_rolls)[:expr.count]
                discarded = sorted(all_rolls)[expr.count:]
                roll_type = RollType.KEEP_LOWEST
            elif expr.operation == 'drop_highest':
                kept_dice = sorted(all_rolls)[:-expr.count]
                discarded = sorted(all_rolls, reverse=True)[:expr.count]
                roll_type = RollType.DROP_HIGHEST
            elif expr.operation == 'drop_lowest':
                kept_dice = sorted(all_rolls, reverse=True)[:-expr.count]
                discarded = sorted(all_rolls)[:expr.count]
                roll_type = RollType.DROP_LOWEST
            else:
                raise InvalidDiceNotationError(f"Unknown operation: {expr.operation}")
        else:
            kept_dice = all_rolls
            discarded = []
            roll_type = RollType.EXPLODING if expr.exploding else RollType.NORMAL
        
        total = sum(kept_dice)
        final_total = total + expr.modifier
        
        # Only set critical flags for single-die results by default
        is_critical_success = len(kept_dice) == 1 and any(die == expr.sides for die in kept_dice)
        is_critical_failure = len(kept_dice) == 1 and kept_dice[0] == 1
        
        # Build notation string
        notation = f"{expr.num_dice}d{expr.sides}"
        if expr.operation and expr.count:
            op_short = {"keep_highest": "kh", "keep_lowest": "kl", 
                       "drop_highest": "dh", "drop_lowest": "dl"}[expr.operation]
            notation += f"{op_short}{expr.count}"
        if expr.exploding:
            if expr.explode_at and expr.explode_at != expr.sides:
                notation += f"!{expr.explode_at}"
            else:
                notation += "!"
        notation += self._fmt_mod(expr.modifier)
        
        result = DiceResult(
            total=total,
            individual_rolls=kept_dice,
            modifiers={"static": expr.modifier},
            final_total=final_total,
            roll_type=roll_type,
            dice_notation=notation,
            is_critical_success=is_critical_success,
            is_critical_failure=is_critical_failure,
            discarded_rolls=discarded,
            explosion_count=explosion_count,
            exploded_dice=exploded_dice,
            system_context=context
        )
        
        self._add_to_history(result)
        return result
    
    def roll_percentile(self, mode: str = "d100", context: Optional[Dict[str, Any]] = None) -> Tuple[int, int, int]:
        """
        Roll percentile dice with different modes.
        
        Args:
            mode: "d100" for single d100, "2d10" for separate tens/ones dice
            context: Optional context dictionary
            
        Returns:
            Tuple of (tens, ones, total) where total is 1-100
        """
        if mode == "d100":
            total = self.rng.randint(1, 100)
            tens = (total - 1) // 10
            ones = total % 10
            if ones == 0:
                ones = 10
        elif mode == "2d10":
            tens_die = self.rng.randint(0, 9)  # 0-9 for tens
            ones_die = self.rng.randint(1, 10)  # 1-10 for ones (10 = 0)
            tens = tens_die
            ones = ones_die if ones_die != 10 else 0
            total = tens * 10 + ones
            if total == 0:
                total = 100
        else:
            raise InvalidDiceNotationError(f"Unknown percentile mode: {mode}")
        
        # Record as a regular dice result
        result = DiceResult(
            total=total,
            individual_rolls=[total] if mode == "d100" else [tens * 10, ones],
            modifiers={},
            final_total=total,
            roll_type=RollType.NORMAL,
            dice_notation=f"d100({mode})",
            is_critical_success=total == 100,
            is_critical_failure=total == 1,
            system_context=context
        )
        
        self._add_to_history(result)
        return tens, ones, total
    
    # ==========================================
    # DICE NOTATION PARSING
    # ==========================================
    
    def parse_dice_notation(self, notation: str) -> DiceExpression:
        """
        Parse standard dice notation (e.g., "3d6+2", "d20-1").
        
        Args:
            notation: Dice notation string
            
        Returns:
            DiceExpression object
            
        Raises:
            InvalidDiceNotationError: If notation is invalid
        """
        notation = notation.strip().lower().replace(" ", "")
        
        match = self.dice_pattern.fullmatch(notation)
        if not match:
            raise InvalidDiceNotationError(f"Invalid dice notation: {notation}")
        
        num_dice_str, sides_str, exploding_str, modifier_str = match.groups()
        
        num_dice = int(num_dice_str) if num_dice_str else 1
        sides = int(sides_str)
        modifier = int(modifier_str) if modifier_str else 0
        
        # Parse exploding notation
        exploding = bool(exploding_str)
        explode_at = None
        if exploding_str and exploding_str.startswith('!') and len(exploding_str) > 1:
            try:
                explode_at = int(exploding_str[1:])
            except ValueError:
                raise InvalidDiceNotationError(f"Invalid exploding threshold: {exploding_str}")
        
        return DiceExpression(
            num_dice=num_dice,
            sides=sides,
            modifier=modifier,
            exploding=exploding,
            explode_at=explode_at
        )
    
    def parse_advanced_notation(self, notation: str) -> DiceExpression:
        """
        Parse advanced dice notation with keep/drop modifiers.
        
        Args:
            notation: Advanced dice notation (e.g., "4d6kh3", "5d10dl2+1")
            
        Returns:
            DiceExpression object
            
        Raises:
            InvalidDiceNotationError: If notation is invalid
        """
        notation = notation.strip().lower().replace(" ", "")
        
        match = self.advanced_pattern.fullmatch(notation)
        if not match:
            raise InvalidDiceNotationError(f"Invalid advanced dice notation: {notation}")
        
        num_dice_str, sides_str, keep_op, keep_count, drop_op, drop_count, exploding_str, modifier_str = match.groups()
        
        num_dice = int(num_dice_str) if num_dice_str else 1
        sides = int(sides_str)
        modifier = int(modifier_str) if modifier_str else 0
        
        # Parse exploding notation
        exploding = bool(exploding_str)
        explode_at = None
        if exploding_str and exploding_str.startswith('!') and len(exploding_str) > 1:
            try:
                explode_at = int(exploding_str[1:])
            except ValueError:
                raise InvalidDiceNotationError(f"Invalid exploding threshold: {exploding_str}")
        
        # Parse operation
        operation = None
        count = None
        
        if keep_op and keep_count:
            operation = "keep_highest" if keep_op == "kh" else "keep_lowest"
            count = int(keep_count)
        elif drop_op and drop_count:
            operation = "drop_highest" if drop_op == "dh" else "drop_lowest"
            count = int(drop_count)
        
        return DiceExpression(
            num_dice=num_dice,
            sides=sides,
            modifier=modifier,
            operation=operation,
            count=count,
            exploding=exploding,
            explode_at=explode_at
        )
    
    # ==========================================
    # UTILITY METHODS
    # ==========================================
    
    def clear_history(self):
        """Clear all roll history."""
        self.roll_history = []
    
    def export_history_json(self, path: Optional[str] = None) -> str:
        """Export roll history as JSON string."""
        json_data = json.dumps([result.to_dict() for result in self.roll_history], indent=2)
        
        if path:
            with open(path, 'w') as f:
                f.write(json_data)
        
        return json_data
    
    def __str__(self) -> str:
        """String representation of the dice engine."""
        return f"DiceEngine(rolls_made={len(self.roll_history)}, max_explosions={self.max_explosions})"
    
    def __repr__(self) -> str:
        """Developer representation of the dice engine."""
        return f"DiceEngine(max_explosions={self.max_explosions}, roll_history={len(self.roll_history)} rolls)"

# ==========================================
# RUNTIME SELF-CHECKS
# ==========================================

if __name__ == "__main__":
    print("=== ChronoBound DiceEngine v1.0.2a Self-Tests ===\n")
    
    # Test 1: No dangling '+' in notations for mod=0
    print("1. Testing notation formatting (no dangling '+')...")
    engine = DiceEngine(seed=42)
    
    basic_result = engine.roll_basic(3, 6, 0)
    assert "+" not in basic_result.dice_notation, f"Basic roll has dangling '+': {basic_result.dice_notation}"
    
    adv_result = engine.roll_advantage(20, 0)
    assert not adv_result.dice_notation.endswith("+"), f"Advantage roll has dangling '+': {adv_result.dice_notation}"
    
    disadv_result = engine.roll_disadvantage(20, 0)
    assert not disadv_result.dice_notation.endswith("+"), f"Disadvantage roll has dangling '+': {disadv_result.dice_notation}"
    
    print("   ✓ All notation strings clean for mod=0")
    
    # Test 2: Daggerheart duality shim
    print("\n2. Testing Daggerheart duality shim...")
    
    # Capture deprecation warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dh_result = engine.roll_daggerheart_duality(trait_modifier=2)
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message)
    
    # Check basic duality mechanics
    assert dh_result.hope_die is not None
    assert dh_result.fear_die is not None
    assert dh_result.dominant_outcome in ["hope", "fear", "balanced"]
    assert dh_result.roll_type == RollType.DAGGERHEART_DUALITY
    
    # Test critical on doubles
    engine_crit = DiceEngine(seed=12345)
    for _ in range(20):  # Try multiple times to potentially get doubles
        crit_result = engine_crit.roll_daggerheart_duality()
        if crit_result.hope_die == crit_result.fear_die:
            assert crit_result.is_critical_success, "Doubles should set critical success"
            break
    
    # Check history was appended
    history_len_before = len(engine.roll_history)
    engine.roll_daggerheart_duality()
    assert len(engine.roll_history) == history_len_before + 1, "Duality roll not added to history"
    
    print("   ✓ Duality shim works correctly with deprecation warning")
    
    # Test 3: DiceResult round-trip serialization
    print("\n3. Testing DiceResult round-trip serialization...")
    
    original = engine.roll_basic(4, 6, 3, exploding=True)
    data_dict = original.to_dict()
    reconstructed = DiceResult.from_dict(data_dict)
    
    # Check all important fields match
    assert original.total == reconstructed.total
    assert original.individual_rolls == reconstructed.individual_rolls
    assert original.modifiers == reconstructed.modifiers
    assert original.final_total == reconstructed.final_total
    assert original.roll_type == reconstructed.roll_type
    assert original.dice_notation == reconstructed.dice_notation
    assert original.is_critical_success == reconstructed.is_critical_success
    assert original.is_critical_failure == reconstructed.is_critical_failure
    assert original.timestamp == reconstructed.timestamp
    
    print("   ✓ DiceResult round-trip serialization works")
    
    # Test 4: Critical flag restriction
    print("\n4. Testing critical flag restriction...")
    
    # Single die should set critical flags
    single_engine = DiceEngine(seed=999)
    
    # Force a max roll for critical success test
    import unittest.mock
    with unittest.mock.patch.object(single_engine, 'roll_single', return_value=20):
        single_result = single_engine.roll_basic(1, 20)
        assert single_result.is_critical_success, "Single d20 max roll should be critical success"
    
    # Force a min roll for critical failure test
    with unittest.mock.patch.object(single_engine, 'roll_single', return_value=1):
        single_result = single_engine.roll_basic(1, 20)
        assert single_result.is_critical_failure, "Single d20 min roll should be critical failure"
    
    # Multiple dice should not set critical flags by default
    multi_result = single_engine.roll_basic(3, 20)
    assert not multi_result.is_critical_success, "Multi-die rolls should not set critical success by default"
    assert not multi_result.is_critical_failure, "Multi-die rolls should not set critical failure by default"
    
    print("   ✓ Critical flags only set for single-die rolls")
    
    # Test 5: RNG isolation
    print("\n5. Testing RNG isolation...")
    
    engine1 = DiceEngine(seed=123)
    engine2 = DiceEngine(seed=456)
    
    results1 = [engine1.roll_single(20) for _ in range(5)]
    results2 = [engine2.roll_single(20) for _ in range(5)]
    
    # Different seeds should produce different sequences
    assert results1 != results2, "Different seeds should produce different results"
    
    # Same seed should produce identical sequences
    engine1_repeat = DiceEngine(seed=123)
    results1_repeat = [engine1_repeat.roll_single(20) for _ in range(5)]
    assert results1 == results1_repeat, "Same seed should produce identical results"
    
    print("   ✓ RNG isolation working correctly")
    
    # Test 6: Context preservation
    print("\n6. Testing context preservation...")
    
    test_context = {"player": "TestPlayer", "action": "test_roll"}
    context_result = engine.roll_basic(2, 6, 1, context=test_context)
    assert context_result.system_context == test_context, "Context not preserved in result"
    
    print("   ✓ Context preservation working")
    
    # Test 7: Notation validation
    print("\n7. Testing notation validation...")
    
    # Valid notations
    assert engine.validate_notation("3d6")
    assert engine.validate_notation("d20+5")
    assert engine.validate_notation("4d6kh3")
    assert engine.validate_notation("2d12!")
    
    # Invalid notations (should reject trailing garbage)
    assert not engine.validate_notation("3d6 extra")
    assert not engine.validate_notation("d20+5x")
    assert not engine.validate_notation("invalid")
    
    print("   ✓ Notation validation working correctly")
    
    print("\n=== All Self-Tests Passed! ===")
    print(f"Total rolls in test history: {len(engine.roll_history)}")
    print("DiceEngine v1.0.2a is ready for use.")
