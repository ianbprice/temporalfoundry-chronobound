"""
ChronoBound CombatEngine v1.0.2a
================================

Enhanced combat management with system-driven profiles, deterministic initiative,
thread safety, action idempotency, and robust event handling.

Key improvements in v1.0.2a:
- System-driven combat profiles with spotlight/initiative turn models
- Deterministic initiative sorting with persistent tie-breakers  
- Null-safe environment handling with default CombatEnvironment
- State precondition guards with custom exception handling
- Engine-level RLock for thread-safe mutations
- Action idempotency via processed_action_ids set
- Robust event emission with listener exception isolation
- Enhanced export with lightweight mode, statistics, and state hashing
- Deterministic tie-breakers; removed global randomness

Version: 1.0.2a
Author: ChronoBound Development Team
License: MIT
"""

import json
import logging
import threading
import time
import hashlib
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import copy
import uuid
from queue import Queue, Empty

# Import our interfaces and modules
from irulesystem_interface import IRuleSystem, RuleSystemType, ActionResult, AttackResult, Condition
from character_manager import CharacterManager

# ==========================================
# ENUMS AND CONSTANTS
# ==========================================

class CombatState(Enum):
    """Current state of combat encounter."""
    NOT_STARTED = "not_started"
    INITIATIVE = "initiative"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    ENDED = "ended"
    ERROR = "error"

class ActionType(Enum):
    """Types of combat actions available."""
    ATTACK = "attack"
    SPELL = "spell"
    MOVE = "move"
    DEFEND = "defend"
    DASH = "dash"
    HELP = "help"
    HIDE = "hide"
    READY = "ready"
    SEARCH = "search"
    USE_ITEM = "use_item"
    CUSTOM = "custom"
    FULL_DEFENSE = "full_defense"
    AID_ANOTHER = "aid_another"
    CHARGE = "charge"
    WITHDRAW = "withdraw"
    DELAY = "delay"
    SYSTEM_ACTION = "system_action"

class CombatResult(Enum):
    """Possible combat outcomes."""
    ONGOING = "ongoing"
    PARTY_VICTORY = "party_victory"
    PARTY_DEFEAT = "party_defeat"
    RETREAT = "retreat"
    DRAW = "draw"
    INTERRUPTED = "interrupted"

class TurnPhase(Enum):
    """Phases within a combat turn."""
    START_TURN = "start_turn"
    MOVEMENT = "movement"
    ACTION = "action"
    BONUS_ACTION = "bonus_action"
    REACTION = "reaction"
    END_TURN = "end_turn"

class CombatEvent(Enum):
    """Types of combat events for listeners."""
    COMBAT_START = "combat_start"
    COMBAT_END = "combat_end"
    ROUND_START = "round_start"
    ROUND_END = "round_end"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    ACTION_TAKEN = "action_taken"
    DAMAGE_DEALT = "damage_dealt"
    CONDITION_APPLIED = "condition_applied"
    CHARACTER_DOWN = "character_down"
    CHARACTER_REVIVED = "character_revived"

# ==========================================
# DATA CLASSES
# ==========================================

@dataclass
class CombatProfile:
    """System-specific combat configuration."""
    turn_model: str = "spotlight"  # "spotlight" or "initiative"
    phases: List[str] = field(default_factory=lambda: ["action"])
    actions_allowed: List[str] = field(default_factory=lambda: ["attack", "spell", "move", "defend"])
    reactions: Optional[str] = "immediate"  # "immediate", "end_of_turn", or None

@dataclass
class CombatParticipant:
    """Represents a character participating in combat with enhanced tracking."""
    character_id: str
    character_data: Dict[str, Any]
    initiative: int
    tie_breaker: int
    is_pc: bool = True
    is_conscious: bool = True
    is_alive: bool = True
    
    # Action tracking
    actions_taken: List[str] = field(default_factory=list)
    actions_remaining: Dict[str, int] = field(default_factory=dict)
    
    # Turn state
    has_acted_this_turn: bool = False
    has_moved_this_turn: bool = False
    has_bonus_action: bool = True
    has_reaction: bool = True
    
    # Status tracking
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    temporary_modifiers: Dict[str, int] = field(default_factory=dict)
    
    def get_hp(self) -> Tuple[int, int]:
        """Get current and max HP."""
        current_hp = self.character_data.get("current_hp", 0)
        max_hp = self.character_data.get("max_hp", 0)
        return current_hp, max_hp
    
    def reset_turn_state(self):
        """Reset turn-specific state."""
        self.has_acted_this_turn = False
        self.has_moved_this_turn = False
        self.has_bonus_action = True
        self.has_reaction = True
        self.temporary_modifiers.clear()

@dataclass
class CombatEnvironment:
    """Environmental factors affecting combat."""
    name: str = "Default"
    terrain: str = "normal"  # normal, difficult, hazardous
    weather: str = "clear"   # clear, rain, fog, storm
    lighting: str = "bright" # bright, dim, dark
    temperature: str = "temperate" # cold, cool, temperate, warm, hot
    visibility_range: int = 100  # feet/meters
    movement_modifiers: Dict[str, float] = field(default_factory=dict)
    action_modifiers: Dict[str, int] = field(default_factory=dict)
    
    def get_movement_modifier(self, character_id: str) -> float:
        """Get movement modifier for a specific character."""
        base_modifier = 1.0
        
        if self.terrain == "difficult":
            base_modifier *= 0.5
        elif self.terrain == "hazardous":
            base_modifier *= 0.25
            
        return self.movement_modifiers.get(character_id, base_modifier)
    
    def get_action_modifier(self, action_type: ActionType) -> int:
        """Get action modifier for specific action types."""
        base_modifier = 0
        
        if self.lighting == "dim":
            base_modifier -= 1
        elif self.lighting == "dark":
            base_modifier -= 3
            
        return self.action_modifiers.get(action_type.value, base_modifier)

@dataclass
class CombatAction:
    """Represents a combat action taken by a participant."""
    action_id: str
    actor_id: str
    action_type: ActionType
    target_ids: List[str]
    details: Dict[str, Any]
    timestamp: str
    round_number: int
    turn_number: int
    turn_phase: TurnPhase
    result: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "action_id": self.action_id,
            "actor_id": self.actor_id,
            "action_type": self.action_type.value,
            "target_ids": self.target_ids,
            "details": self.details,
            "timestamp": self.timestamp,
            "round_number": self.round_number,
            "turn_number": self.turn_number,
            "turn_phase": self.turn_phase.value,
            "result": self.result
        }

# ==========================================
# EXCEPTIONS
# ==========================================

class CombatEngineError(Exception):
    """Base exception for combat engine errors."""
    pass

class InvalidCombatStateError(CombatEngineError):
    """Raised when combat is in an invalid state for the operation."""
    pass

class ParticipantNotFoundError(CombatEngineError):
    """Raised when a participant is not found in combat."""
    pass

class InvalidActionError(CombatEngineError):
    """Raised when an invalid action is attempted."""
    pass

class CombatResolutionError(CombatEngineError):
    """Raised when combat resolution fails."""
    pass

# ==========================================
# MAIN COMBAT ENGINE CLASS
# ==========================================

class CombatEngine:
    """
    Enhanced combat engine for ChronoBound with system-driven profiles,
    thread safety, idempotency, and robust event handling.
    """
    
    def __init__(
        self,
        rule_system: IRuleSystem,
        character_manager: CharacterManager,
        max_rounds: int = 100,
        auto_process_ai: bool = True
    ):
        """
        Initialize the CombatEngine.
        
        Args:
            rule_system: Active RPG rule system
            character_manager: Character data manager
            max_rounds: Maximum rounds before forced draw
            auto_process_ai: Whether to automatically process AI turns
        """
        self.rule_system = rule_system
        self.character_manager = character_manager
        self.max_rounds = max_rounds
        self.auto_process_ai = auto_process_ai
        
        # Thread safety - engine-level lock
        self._lock = threading.RLock()
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Get combat profile from rule system
        self.combat_profile = self._get_combat_profile()
        
        # Combat state
        self.state = CombatState.NOT_STARTED
        self.participants: List[CombatParticipant] = []
        self.turn_order: List[str] = []  # Character IDs in initiative order
        self.current_turn_index = 0
        self.round_number = 0
        self.turn_number = 0
        self.current_phase = TurnPhase.START_TURN
        
        # Environment (null-safe default)
        self.environment = CombatEnvironment()
        
        # Action processing
        self.action_queue = Queue()
        self.pending_reactions: List[Dict[str, Any]] = []
        self.delayed_actions: List[CombatAction] = []
        self.processed_action_ids: Set[str] = set()  # For idempotency
        
        # Event system with robust listener management
        self.event_listeners: Dict[CombatEvent, List[Callable]] = {}
        
        # Combat log and statistics
        self.combat_log: List[CombatAction] = []
        self.combat_start_time: Optional[str] = None
        self.combat_end_time: Optional[str] = None
        self.combat_result: Optional[CombatResult] = None
        
        # Enhanced tracking
        self.damage_dealt: Dict[str, int] = {}
        self.damage_taken: Dict[str, int] = {}
        self.healing_done: Dict[str, int] = {}
        self.actions_count: Dict[str, Dict[str, int]] = {}
        self.conditions_applied: Dict[str, List[str]] = {}
        
        # AI processing
        self.ai_processor_thread = None
        self.ai_processing_enabled = auto_process_ai
        
        # Performance metrics
        self.performance_stats = {
            "actions_processed": 0,
            "average_action_time": 0.0,
            "total_processing_time": 0.0
        }
        
        # Engine version for export
        self.engine_version = "1.0.2a"
    
    def _get_combat_profile(self) -> CombatProfile:
        """Get combat profile from rule system with fallback."""
        try:
            if hasattr(self.rule_system, 'get_combat_profile'):
                profile_data = self.rule_system.get_combat_profile()
                return CombatProfile(**profile_data)
        except Exception as e:
            self.logger.warning(f"Failed to get combat profile from rule system: {e}")
        
        # Fallback to Daggerheart-style spotlight
        return CombatProfile(
            turn_model="spotlight",
            phases=["action"],
            actions_allowed=["attack", "spell", "move", "defend", "help"],
            reactions="immediate"
        )
    
    # ==========================================
    # STATE GUARD METHODS
    # ==========================================
    
    def _require_state(self, *valid_states: CombatState):
        """Ensure combat is in one of the valid states."""
        if self.state not in valid_states:
            raise InvalidCombatStateError(
                f"Cannot perform action in state {self.state}. "
                f"Required states: {', '.join(s.value for s in valid_states)}"
            )
    
    def _require_participants(self):
        """Ensure combat has participants."""
        if not self.participants:
            raise CombatEngineError("No participants in combat")
    
    def _require_participant(self, character_id: str) -> CombatParticipant:
        """Get participant or raise error if not found."""
        participant = self.get_participant(character_id)
        if not participant:
            raise ParticipantNotFoundError(f"Participant {character_id} not found")
        return participant
    
    # ==========================================
    # COMBAT LIFECYCLE
    # ==========================================
    
    def start_combat(
        self,
        participants: List[Dict[str, Any]],
        environment: Optional[CombatEnvironment] = None,
        **kwargs
    ) -> bool:
        """
        Start a new combat encounter.
        
        Args:
            participants: List of character data dicts
            environment: Optional environmental factors
            **kwargs: Additional combat parameters
            
        Returns:
            True if combat started successfully
            
        Raises:
            InvalidCombatStateError: If combat is already active
            CombatEngineError: If participants are invalid
        """
        self._require_state(CombatState.NOT_STARTED)
        
        with self._lock:
            try:
                # Reset all combat state
                self._reset_combat_state()
                
                # Set environment (null-safe)
                self.environment = environment or CombatEnvironment()
                
                # Add participants
                for participant_data in participants:
                    self._add_participant(participant_data)
                
                # Validate we have participants
                if not self.participants:
                    raise CombatEngineError("Cannot start combat with no participants")
                
                # Set initial state
                self.state = CombatState.INITIATIVE
                self.combat_start_time = datetime.now().isoformat()
                
                # Initialize tracking dictionaries
                for participant in self.participants:
                    char_id = participant.character_id
                    self.damage_dealt[char_id] = 0
                    self.damage_taken[char_id] = 0
                    self.healing_done[char_id] = 0
                    self.actions_count[char_id] = {}
                    self.conditions_applied[char_id] = []
                
                # Fire combat start event
                self._fire_event(CombatEvent.COMBAT_START, {
                    "participant_count": len(self.participants),
                    "environment": self.environment.name,
                    "combat_profile": {
                        "turn_model": self.combat_profile.turn_model,
                        "phases": self.combat_profile.phases
                    }
                })
                
                self.logger.info(f"Combat started with {len(self.participants)} participants")
                return True
                
            except Exception as e:
                self.state = CombatState.ERROR
                self.logger.error(f"Failed to start combat: {e}")
                raise CombatEngineError(f"Combat start failed: {e}")
    
    def _add_participant(self, participant_data: Dict[str, Any]):
        """Add a participant to combat with deterministic tie-breaker generation."""
        char_id = participant_data.get("id")
        if not char_id:
            raise CombatEngineError("All participants must have an 'id' field")
        
        # Check for duplicates
        if any(p.character_id == char_id for p in self.participants):
            raise CombatEngineError(f"Participant {char_id} already in combat")
        
        # Determine tie-breaker: use existing or compute deterministic value
        if "_tie_breaker" in participant_data:
            tie_breaker = participant_data["_tie_breaker"]
        else:
            # Stable, bounded 0..999
            tie_breaker = int(hashlib.md5(char_id.encode()).hexdigest()[:6], 16) % 1000
            participant_data["_tie_breaker"] = tie_breaker
        
        is_pc = participant_data.get("type", "PC") == "PC"
        
        participant = CombatParticipant(
            character_id=char_id,
            character_data=copy.deepcopy(participant_data),
            initiative=0,  # Will be rolled later
            tie_breaker=tie_breaker,
            is_pc=is_pc,
            is_conscious=True,
            is_alive=True
        )
        
        self.participants.append(participant)
        self.logger.debug(f"Added participant: {char_id} (tie_breaker: {tie_breaker})")
    
    def roll_initiative(self) -> List[Tuple[str, int]]:
        """
        Roll initiative for all participants with deterministic sorting.
        
        Returns:
            List of (character_id, initiative_value) tuples in order
            
        Raises:
            InvalidCombatStateError: If not in initiative state
        """
        self._require_state(CombatState.INITIATIVE)
        
        with self._lock:
            initiative_results = []
            
            for participant in self.participants:
                try:
                    # Use rule system to roll initiative
                    init_result = self.rule_system.roll_initiative(
                        participant.character_data,
                        modifiers=self.environment.action_modifiers
                    )
                    
                    participant.initiative = init_result.initiative_value
                    initiative_results.append((participant.character_id, init_result.initiative_value))
                    
                    # Log initiative roll
                    self._log_action(
                        actor_id=participant.character_id,
                        action_type=ActionType.SYSTEM_ACTION,
                        target_ids=[],
                        details={"action": "initiative_roll", "initiative": init_result.initiative_value},
                        result={"initiative": init_result.initiative_value, "tie_breaker": participant.tie_breaker}
                    )
                    
                except Exception as e:
                    self.logger.error(f"Failed to roll initiative for {participant.character_id}: {e}")
                    participant.initiative = 10  # Default initiative
                    initiative_results.append((participant.character_id, 10))
            
            # Deterministic sorting: (initiative_total, tie_breaker, character_id) descending
            self.participants.sort(
                key=lambda p: (p.initiative, p.tie_breaker, p.character_id), 
                reverse=True
            )
            self.turn_order = [p.character_id for p in self.participants]
            
            # Start first round
            self.state = CombatState.IN_PROGRESS
            self.round_number = 1
            self.turn_number = 0
            self.current_turn_index = 0
            
            self._start_new_round()
            
            return initiative_results
    
    def end_combat(self, result: CombatResult = CombatResult.INTERRUPTED) -> bool:
        """
        End the current combat encounter.
        
        Args:
            result: How the combat ended
            
        Returns:
            True if combat ended successfully
        """
        self._require_state(CombatState.IN_PROGRESS, CombatState.PAUSED)
        
        with self._lock:
            try:
                self.state = CombatState.ENDED
                self.combat_end_time = datetime.now().isoformat()
                self.combat_result = result
                
                # Fire combat end event
                self._fire_event(CombatEvent.COMBAT_END, {
                    "result": result.value,
                    "duration": self._get_combat_duration(),
                    "rounds": self.round_number,
                    "turns": self.turn_number
                })
                
                self.logger.info(f"Combat ended: {result.value}")
                return True
                
            except Exception as e:
                self.logger.error(f"Error ending combat: {e}")
                return False
    
    # ==========================================
    # TURN MANAGEMENT
    # ==========================================
    
    def next_turn(self) -> Optional[str]:
        """
        Advance to the next turn in combat.
        
        Returns:
            Character ID of the next actor, or None if combat ended
            
        Raises:
            InvalidCombatStateError: If not in progress
        """
        self._require_state(CombatState.IN_PROGRESS)
        
        with self._lock:
            # End current turn
            if self.participants:
                current_actor = self.get_current_actor()
                if current_actor:
                    self._fire_event(CombatEvent.TURN_END, {
                        "character_id": current_actor,
                        "round": self.round_number,
                        "turn": self.turn_number
                    })
            
            # Advance to next participant
            self.current_turn_index += 1
            self.turn_number += 1
            
            # Check for end of round
            if self.current_turn_index >= len(self.participants):
                self._end_round()
                return self.next_turn()  # Start new round
            
            # Start next turn
            current_actor = self.get_current_actor()
            if current_actor:
                self._start_turn(current_actor)
                return current_actor
            
            return None
    
    def _start_new_round(self):
        """Start a new combat round."""
        with self._lock:
            self.current_turn_index = 0
            
            # Reset participant turn states
            for participant in self.participants:
                participant.reset_turn_state()
            
            # Fire round start event
            self._fire_event(CombatEvent.ROUND_START, {
                "round": self.round_number,
                "participant_count": len(self.participants)
            })
            
            # Start first turn
            current_actor = self.get_current_actor()
            if current_actor:
                self._start_turn(current_actor)
    
    def _start_turn(self, character_id: str):
        """Start a character's turn."""
        participant = self._require_participant(character_id)
        
        # Process start-of-turn effects
        self._process_condition_effects(participant, "start_turn")
        
        # Fire turn start event
        self._fire_event(CombatEvent.TURN_START, {
            "character_id": character_id,
            "round": self.round_number,
            "turn": self.turn_number,
            "is_pc": participant.is_pc
        })
        
        self.logger.debug(f"Started turn for {character_id}")
    
    def _end_round(self):
        """End the current round and prepare for next."""
        with self._lock:
            # Fire round end event
            self._fire_event(CombatEvent.ROUND_END, {
                "round": self.round_number,
                "turns": self.turn_number
            })
            
            # Check for max rounds
            if self.round_number >= self.max_rounds:
                self.end_combat(CombatResult.DRAW)
                return
            
            # Advance to next round
            self.round_number += 1
            self.current_turn_index = 0
            
            self._start_new_round()
    
    # ==========================================
    # ACTION PROCESSING WITH IDEMPOTENCY
    # ==========================================
    
    def take_action(
        self,
        actor_id: str,
        action_type: ActionType,
        target_ids: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None,
        action_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a combat action with idempotency checks.
        
        Args:
            actor_id: ID of the acting character
            action_type: Type of action being taken
            target_ids: List of target character IDs
            details: Additional action details
            action_id: Optional custom action ID for idempotency
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with action results
            
        Raises:
            InvalidActionError: If action is invalid
            ParticipantNotFoundError: If actor not found
            InvalidCombatStateError: If combat not in progress
        """
        self._require_state(CombatState.IN_PROGRESS)
        participant = self._require_participant(actor_id)
        
        if target_ids is None:
            target_ids = []
        if details is None:
            details = {}
        if action_id is None:
            action_id = str(uuid.uuid4())
        
        with self._lock:
            # Check for duplicate action (idempotency)
            if action_id in self.processed_action_ids:
                self.logger.warning(f"Ignoring duplicate action: {action_id}")
                return {"success": False, "reason": "duplicate_action", "action_id": action_id}
            
            try:
                start_time = time.time()
                
                # Validate action
                self._validate_action(participant, action_type, target_ids, details)
                
                # Create action
                action = CombatAction(
                    action_id=action_id,
                    actor_id=actor_id,
                    action_type=action_type,
                    target_ids=target_ids,
                    details=details,
                    timestamp=datetime.now().isoformat(),
                    round_number=self.round_number,
                    turn_number=self.turn_number,
                    turn_phase=self.current_phase
                )
                
                # Mark as processed for idempotency
                self.processed_action_ids.add(action_id)
                
                # Process action based on type
                result = self._process_action(action, participant)
                action.result = result
                
                # Update participant state
                self._update_participant_action_state(participant, action_type)
                
                # Log action
                self.combat_log.append(action)
                
                # Update statistics
                processing_time = time.time() - start_time
                self._update_performance_stats(processing_time)
                
                # Fire action event
                self._fire_event(CombatEvent.ACTION_TAKEN, {
                    "action": action.to_dict(),
                    "result": result,
                    "processing_time": processing_time
                })
                
                return result
                
            except Exception as e:
                # Remove from processed set on error
                self.processed_action_ids.discard(action_id)
                self.logger.error(f"Error processing action for {actor_id}: {e}")
                raise InvalidActionError(f"Action processing failed: {e}")
    
    def _validate_action(
        self,
        participant: CombatParticipant,
        action_type: ActionType,
        target_ids: List[str],
        details: Dict[str, Any]
    ):
        """Validate an action can be taken."""
        # Check if participant is conscious
        if not participant.is_conscious:
            raise InvalidActionError("Unconscious participants cannot take actions")
        
        # Check if action is allowed by combat profile
        if action_type.value not in self.combat_profile.actions_allowed:
            raise InvalidActionError(f"Action {action_type.value} not allowed in this system")
        
        # Validate targets exist
        for target_id in target_ids:
            if not self.get_participant(target_id):
                raise InvalidActionError(f"Target {target_id} not found")
    
    def _process_action(self, action: CombatAction, participant: CombatParticipant) -> Dict[str, Any]:
        """Process a specific action type."""
        if action.action_type == ActionType.ATTACK:
            return self._process_attack_action(action, participant)
        elif action.action_type == ActionType.SPELL:
            return self._process_spell_action(action, participant)
        elif action.action_type == ActionType.MOVE:
            return self._process_move_action(action, participant)
        elif action.action_type == ActionType.DEFEND:
            return self._process_defend_action(action, participant)
        elif action.action_type == ActionType.HELP:
            return self._process_help_action(action, participant)
        else:
            return self._process_generic_action(action, participant)
    
    def _process_attack_action(self, action: CombatAction, participant: CombatParticipant) -> Dict[str, Any]:
        """Process an attack action."""
        if not action.target_ids:
            raise InvalidActionError("Attack action requires a target")
        
        target_id = action.target_ids[0]
        target = self._require_participant(target_id)
        
        # Use rule system for attack roll
        weapon = action.details.get("weapon", "unarmed")
        modifiers = action.details.get("modifiers", {})
        
        try:
            attack_result = self.rule_system.make_attack_roll(
                participant.character_data,
                target.character_data,
                weapon,
                modifiers=modifiers
            )
            
            result = {
                "success": attack_result.hit,
                "attack_total": attack_result.attack_total,
                "damage": attack_result.damage_total,
                "damage_type": attack_result.damage_type.value if attack_result.damage_type else "physical",
                "is_critical": attack_result.is_critical,
                "target_id": target_id
            }
            
            if attack_result.hit:
                # Apply damage
                current_hp, max_hp = target.get_hp()
                new_hp = max(0, current_hp - attack_result.damage_total)
                target.character_data["current_hp"] = new_hp
                
                # Update statistics
                self.damage_dealt[participant.character_id] = (
                    self.damage_dealt.get(participant.character_id, 0) + attack_result.damage_total
                )
                self.damage_taken[target_id] = (
                    self.damage_taken.get(target_id, 0) + attack_result.damage_total
                )
                
                # Fire damage event
                self._fire_event(CombatEvent.DAMAGE_DEALT, {
                    "attacker_id": participant.character_id,
                    "target_id": target_id,
                    "damage": attack_result.damage_total,
                    "damage_type": result["damage_type"],
                    "is_critical": attack_result.is_critical
                })
                
                # Check if target is down
                if new_hp <= 0:
                    target.is_conscious = False
                    if new_hp <= -max_hp:  # Death threshold
                        target.is_alive = False
                    
                    self._fire_event(CombatEvent.CHARACTER_DOWN, {
                        "character_id": target_id,
                        "is_alive": target.is_alive
                    })
            
            return result
            
        except Exception as e:
            raise InvalidActionError(f"Attack failed: {e}")
    
    def _process_spell_action(self, action: CombatAction, participant: CombatParticipant) -> Dict[str, Any]:
        """Process a spell action."""
        spell_name = action.details.get("spell", "unknown")
        
        # Basic spell processing - can be enhanced per system
        return {
            "success": True,
            "spell": spell_name,
            "message": f"Cast {spell_name}"
        }
    
    def _process_move_action(self, action: CombatAction, participant: CombatParticipant) -> Dict[str, Any]:
        """Process a movement action."""
        distance = action.details.get("distance", 30)
        modifier = self.environment.get_movement_modifier(participant.character_id)
        effective_distance = distance * modifier
        
        return {
            "success": True,
            "distance": effective_distance,
            "movement_modifier": modifier
        }
    
    def _process_defend_action(self, action: CombatAction, participant: CombatParticipant) -> Dict[str, Any]:
        """Process a defend action."""
        participant.temporary_modifiers["defense_bonus"] = 2
        return {
            "success": True,
            "defense_bonus": 2,
            "message": "Taking defensive stance"
        }
    
    def _process_help_action(self, action: CombatAction, participant: CombatParticipant) -> Dict[str, Any]:
        """Process a help action."""
        if not action.target_ids:
            raise InvalidActionError("Help action requires a target")
        
        target_id = action.target_ids[0]
        target = self._require_participant(target_id)
        
        # Add temporary bonus to target's next action
        target.temporary_modifiers["help_bonus"] = target.temporary_modifiers.get("help_bonus", 0) + 2
        
        return {
            "success": True,
            "target_id": target_id,
            "bonus_granted": 2,
            "message": f"Helped {target_id}"
        }
    
    def _process_generic_action(self, action: CombatAction, participant: CombatParticipant) -> Dict[str, Any]:
        """Process a generic/custom action."""
        return {
            "success": True,
            "action_type": action.action_type.value,
            "message": f"Performed {action.action_type.value}"
        }
    
    def _update_participant_action_state(self, participant: CombatParticipant, action_type: ActionType):
        """Update participant's action state."""
        if action_type in [ActionType.ATTACK, ActionType.SPELL, ActionType.DEFEND]:
            participant.has_acted_this_turn = True
        elif action_type == ActionType.MOVE:
            participant.has_moved_this_turn = True
        
        # Update action count
        if participant.character_id not in self.actions_count:
            self.actions_count[participant.character_id] = {}
        
        action_key = action_type.value
        self.actions_count[participant.character_id][action_key] = (
            self.actions_count[participant.character_id].get(action_key, 0) + 1
        )
    
    def _process_condition_effects(self, participant: CombatParticipant, trigger: str):
        """Process condition effects for a participant."""
        try:
            updated_character, messages = self.rule_system.process_condition_effects(
                participant.character_data, trigger
            )
            participant.character_data = updated_character
            
            # Log condition messages
            for message in messages:
                self.logger.info(f"Condition effect: {message}")
                
        except Exception as e:
            self.logger.error(f"Error processing condition effects for {participant.character_id}: {e}")
    
    # ==========================================
    # EVENT SYSTEM WITH ROBUST ERROR HANDLING
    # ==========================================
    
    def subscribe_to_event(self, event: CombatEvent, callback: Callable):
        """
        Subscribe to combat events.
        
        Args:
            event: Combat event type to listen for
            callback: Function to call when event occurs
                     Signature: callback(event: CombatEvent, data: Dict[str, Any])
        """
        with self._lock:
            if event not in self.event_listeners:
                self.event_listeners[event] = []
            self.event_listeners[event].append(callback)
            self.logger.debug(f"Added listener for {event.value}")
    
    def unsubscribe_from_event(self, event: CombatEvent, callback: Callable):
        """
        Unsubscribe from combat events.
        
        Args:
            event: Combat event type to stop listening for
            callback: The callback function to remove
        """
        with self._lock:
            if event in self.event_listeners:
                try:
                    self.event_listeners[event].remove(callback)
                    self.logger.debug(f"Removed listener for {event.value}")
                except ValueError:
                    pass  # Callback wasn't in the list
    
    def _fire_event(self, event: CombatEvent, data: Dict[str, Any]):
        """
        Fire an event to all listeners with error isolation.
        
        Event payloads:
        - COMBAT_START: {"participant_count": int, "environment": str, "combat_profile": dict}
        - COMBAT_END: {"result": str, "duration": str, "rounds": int, "turns": int}
        - ROUND_START: {"round": int, "participant_count": int}
        - ROUND_END: {"round": int, "turns": int}
        - TURN_START: {"character_id": str, "round": int, "turn": int, "is_pc": bool}
        - TURN_END: {"character_id": str, "round": int, "turn": int}
        - ACTION_TAKEN: {"action": dict, "result": dict, "processing_time": float}
        - DAMAGE_DEALT: {"attacker_id": str, "target_id": str, "damage": int, "damage_type": str, "is_critical": bool}
        - CONDITION_APPLIED: {"character_id": str, "condition": str, "duration": int}
        """
        if event in self.event_listeners:
            for callback in self.event_listeners[event]:
                try:
                    callback(event, data)
                except Exception as e:
                    # Isolate listener errors - don't crash the engine
                    self.logger.error(f"Event listener error for {event.value}: {e}")
    
    # ==========================================
    # UTILITY METHODS
    # ==========================================
    
    def get_participant(self, character_id: str) -> Optional[CombatParticipant]:
        """Get a participant by character ID."""
        for participant in self.participants:
            if participant.character_id == character_id:
                return participant
        return None
    
    def get_current_actor(self) -> Optional[str]:
        """Get the ID of the currently active character."""
        if 0 <= self.current_turn_index < len(self.participants):
            return self.participants[self.current_turn_index].character_id
        return None
    
    def get_combat_status(self) -> Dict[str, Any]:
        """Get comprehensive combat status information."""
        current_actor = self.get_current_actor()
        
        return {
            "state": self.state.value,
            "round": self.round_number,
            "turn": self.turn_number,
            "current_actor": current_actor,
            "current_phase": self.current_phase.value,
            "participants": [
                {
                    "character_id": p.character_id,
                    "initiative": p.initiative,
                    "tie_breaker": p.tie_breaker,
                    "is_pc": p.is_pc,
                    "is_alive": p.is_alive,
                    "is_conscious": p.is_conscious,
                    "current_hp": p.get_hp()[0],
                    "max_hp": p.get_hp()[1],
                    "has_acted": p.has_acted_this_turn,
                    "has_moved": p.has_moved_this_turn
                }
                for p in self.participants
            ],
            "environment": {
                "name": self.environment.name,
                "terrain": self.environment.terrain,
                "lighting": self.environment.lighting,
                "weather": self.environment.weather
            },
            "combat_profile": {
                "turn_model": self.combat_profile.turn_model,
                "phases": self.combat_profile.phases,
                "reactions": self.combat_profile.reactions
            }
        }
    
    def export_combat_state(
        self,
        lightweight: bool = False,
        include_statistics: bool = True
    ) -> Dict[str, Any]:
        """
        Export combat state with enhanced options.
        
        Args:
            lightweight: If True, exclude detailed logs and history
            include_statistics: Include combat statistics
            
        Returns:
            Dictionary containing combat state
        """
        with self._lock:
            base_export = {
                "engine_version": self.engine_version,
                "state": self.state.value,
                "round_number": self.round_number,
                "turn_number": self.turn_number,
                "current_turn_index": self.current_turn_index,
                "combat_start_time": self.combat_start_time,
                "combat_end_time": self.combat_end_time,
                "combat_result": self.combat_result.value if self.combat_result else None,
                "participants": [
                    {
                        "character_id": p.character_id,
                        "initiative": p.initiative,
                        "tie_breaker": p.tie_breaker,  # Enhanced: include tie-breakers
                        "is_pc": p.is_pc,
                        "is_alive": p.is_alive,
                        "is_conscious": p.is_conscious,
                        "character_data": p.character_data if not lightweight else {"id": p.character_id}
                    }
                    for p in self.participants
                ],
                "environment": {
                    "name": self.environment.name,
                    "terrain": self.environment.terrain,
                    "weather": self.environment.weather,
                    "lighting": self.environment.lighting,
                    "movement_modifiers": self.environment.movement_modifiers,
                    "action_modifiers": self.environment.action_modifiers
                },
                "combat_profile": {
                    "turn_model": self.combat_profile.turn_model,
                    "phases": self.combat_profile.phases,
                    "actions_allowed": self.combat_profile.actions_allowed,
                    "reactions": self.combat_profile.reactions
                }
            }
            
            # Add statistics if requested
            if include_statistics:
                base_export["statistics"] = {
                    "damage_dealt": self.damage_dealt,
                    "damage_taken": self.damage_taken,
                    "healing_done": self.healing_done,
                    "actions_count": self.actions_count,
                    "performance_stats": self.performance_stats,
                    "total_actions": len(self.combat_log),
                    "duration": self._get_combat_duration()
                }
            
            # Include logs unless lightweight
            if not lightweight:
                base_export["combat_log"] = [action.to_dict() for action in self.combat_log]
                base_export["processed_action_ids"] = list(self.processed_action_ids)
            
            # Generate state hash for integrity
            base_export["state_hash"] = self._generate_state_hash(base_export)
            
            return base_export
    
    def _generate_state_hash(self, export_data: Dict[str, Any]) -> str:
        """Generate a hash of the current state for integrity checking."""
        # Create a stable string representation
        hash_data = {
            "state": export_data["state"],
            "round": export_data["round_number"],
            "turn": export_data["turn_number"],
            "participants": sorted([
                f"{p['character_id']}:{p['initiative']}:{p['tie_breaker']}"
                for p in export_data["participants"]
            ])
        }
        
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.md5(hash_string.encode()).hexdigest()[:8]
    
    def _get_combat_duration(self) -> str:
        """Get combat duration string."""
        if not self.combat_start_time:
            return "0:00:00"
        
        start = datetime.fromisoformat(self.combat_start_time)
        end = datetime.fromisoformat(self.combat_end_time) if self.combat_end_time else datetime.now()
        duration = end - start
        
        total_seconds = int(duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    
    def _log_action(
        self,
        actor_id: str,
        action_type: ActionType,
        target_ids: List[str],
        details: Dict[str, Any],
        result: Dict[str, Any]
    ):
        """Log a combat action."""
        action = CombatAction(
            action_id=str(uuid.uuid4()),
            actor_id=actor_id,
            action_type=action_type,
            target_ids=target_ids,
            details=details,
            timestamp=datetime.now().isoformat(),
            round_number=self.round_number,
            turn_number=self.turn_number,
            turn_phase=self.current_phase,
            result=result
        )
        
        self.combat_log.append(action)
    
    def _update_performance_stats(self, processing_time: float):
        """Update performance tracking statistics."""
        self.performance_stats["actions_processed"] += 1
        self.performance_stats["total_processing_time"] += processing_time
        
        if self.performance_stats["actions_processed"] > 0:
            self.performance_stats["average_action_time"] = (
                self.performance_stats["total_processing_time"] / 
                self.performance_stats["actions_processed"]
            )
    
    def _reset_combat_state(self):
        """Reset all combat state for a new encounter."""
        self.participants.clear()
        self.turn_order.clear()
        self.combat_log.clear()
        self.processed_action_ids.clear()
        self.damage_dealt.clear()
        self.damage_taken.clear()
        self.healing_done.clear()
        self.actions_count.clear()
        self.conditions_applied.clear()
        
        self.current_turn_index = 0
        self.round_number = 0
        self.turn_number = 0
        self.combat_start_time = None
        self.combat_end_time = None
        self.combat_result = None
        
        # Reset performance stats
        self.performance_stats = {
            "actions_processed": 0,
            "average_action_time": 0.0,
            "total_processing_time": 0.0
        }
