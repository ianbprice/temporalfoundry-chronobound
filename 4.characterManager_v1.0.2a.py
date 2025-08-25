"""
ChronoBound CharacterManager v1.0.2a
====================================

Enhanced character management system with atomic operations, advanced validation,
LRU caching, system conversion, and comprehensive filtering capabilities.

Changelog v1.0.2a:
- Validation compatibility: Handle both modern ValidationResult and legacy dict formats
- System conversion validates against target ruleset (with optional system_loader)
- Rule-system filtering uses prefix matching (not substring)
- Cache flush updates metadata before save
- Cache statistics track hit/miss ratios
- Removed unreachable temp file guard in search_characters (glob already excludes .tmp)

Key improvements in v1.0.2:
- ValidationResult with IntEnum severity levels  
- Atomic filesystem saves with temp files and ID sanitization
- LRU cache with targeted invalidation
- ISO date filtering (created_after/before, last_used_after)  
- Robust enum coercion in filters
- System conversion with export/import validation
- Event hooks (on_saved/deleted/restored) 
- Per-character metadata (schema_version, content_hash)
- Optional inter-process file locking
- Daggerheart-specific defaults and stress management

Author: ChronoBound Development Team
License: MIT
"""

import json
import logging
import uuid
import threading
import time
import os
import re
import hashlib
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, Protocol, Mapping
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from datetime import datetime, timedelta
from pathlib import Path
from functools import lru_cache
from collections import OrderedDict
import copy

# Optional inter-process locking
try:
    import portalocker
    HAS_PORTALOCKER = True
except ImportError:
    HAS_PORTALOCKER = False

# Import our interfaces
from irulesystem_interface import IRuleSystem, RuleSystemType, Condition, ConditionType

# ==========================================
# ENUMS AND CONSTANTS
# ==========================================

class ValidationSeverity(IntEnum):
    """Validation severity levels (higher = more severe)."""
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4

class CharacterType(Enum):
    """Types of characters in the system."""
    PLAYER_CHARACTER = "pc"
    NON_PLAYER_CHARACTER = "npc"
    COMPANION = "companion"
    FAMILIAR = "familiar"
    MOUNT = "mount"
    SUMMON = "summon"
    TEMPLATE = "template"

class CharacterStatus(Enum):
    """Character lifecycle status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    RETIRED = "retired"
    DECEASED = "deceased"
    MISSING = "missing"
    TEMPLATE = "template"
    ARCHIVED = "archived"

class ProgressionEvent(Enum):
    """Types of character progression events."""
    LEVEL_UP = "level_up"
    SKILL_INCREASE = "skill_increase"
    TRAIT_CHANGE = "trait_change"
    SPELL_LEARNED = "spell_learned"
    FEAT_GAINED = "feat_gained"
    EQUIPMENT_CHANGE = "equipment_change"

# ==========================================
# DATA CLASSES
# ==========================================

@dataclass
class ValidationResult:
    """Enhanced validation result with severity-based issue tracking."""
    is_valid: bool = True
    severity: ValidationSeverity = ValidationSeverity.INFO
    issues: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)  # Backwards compatibility
    errors: List[str] = field(default_factory=list)    # Backwards compatibility
    
    def add_issue(self, message: str, severity: ValidationSeverity, 
                  field: Optional[str] = None, code: Optional[str] = None):
        """Add validation issue and escalate overall severity."""
        issue = {
            "message": message,
            "severity": severity.name.lower(),
            "field": field,
            "code": code,
            "timestamp": datetime.now().isoformat()
        }
        self.issues.append(issue)
        
        # Escalate overall severity
        if severity > self.severity:
            self.severity = severity
        
        # Update is_valid based on severity
        if severity >= ValidationSeverity.ERROR:
            self.is_valid = False
        
        # Backwards compatibility - populate legacy lists
        if severity == ValidationSeverity.WARNING:
            self.warnings.append(message)
        elif severity >= ValidationSeverity.ERROR:
            self.errors.append(message)

@dataclass 
class CharacterMetadata:
    """Per-character metadata for versioning and integrity."""
    schema_version: str = "1.0.2a"
    rule_system_version: str = "unknown"
    content_hash: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: str = field(default_factory=lambda: datetime.now().isoformat())
    backup_count: int = 0

@dataclass
class SearchFilters:
    """Advanced character search and filtering options."""
    name_pattern: Optional[str] = None
    character_type: Optional[CharacterType] = None
    status: Optional[CharacterStatus] = None
    rule_system: Optional[RuleSystemType] = None
    min_level: Optional[int] = None
    max_level: Optional[int] = None
    created_after: Optional[str] = None  # ISO date string
    created_before: Optional[str] = None  # ISO date string
    last_used_after: Optional[str] = None  # ISO date string
    tags: Optional[List[str]] = None
    voice_style: Optional[str] = None
    limit: Optional[int] = None
    sort_by: str = "name"
    sort_desc: bool = False

# ==========================================
# CHARACTER MANAGER
# ==========================================

class CharacterManager:
    """
    Enhanced character management with atomic operations, caching, and validation.
    
    Features:
    - Atomic filesystem operations with temp files
    - LRU cache with configurable size and targeted invalidation
    - Advanced validation with severity levels
    - System conversion capabilities with target validation
    - ISO date filtering and robust enum coercion
    - Event hooks for lifecycle events
    - Optional inter-process file locking
    - Per-character metadata and content hashing
    """
    
    def __init__(self, 
                 rule_system: IRuleSystem,
                 storage_path: str = "./characters",
                 max_cache_entries: int = 100,
                 cache_ttl: int = 3600,
                 enable_file_locking: bool = False,
                 validate_on_save: bool = True,
                 system_loader: Optional[Callable[[RuleSystemType], IRuleSystem]] = None):
        """Initialize the character manager."""
        self.rule_system = rule_system
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Cache configuration
        self.max_cache_entries = max_cache_entries
        self.cache_ttl = cache_ttl
        self._character_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._cache_timestamps: Dict[str, float] = {}
        
        # Cache statistics tracking
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        
        # Validation settings
        self.validate_on_save = validate_on_save
        
        # System conversion support
        self.system_loader = system_loader
        
        # File locking (optional)
        self.enable_file_locking = enable_file_locking and HAS_PORTALOCKER
        
        # Event hooks
        self._event_hooks: Dict[str, List[Callable]] = {
            "on_saved": [],
            "on_deleted": [], 
            "on_restored": []
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Logging
        self.logger = logging.getLogger(f"{__name__}.CharacterManager")
        
        # Character ID validation pattern
        self._id_pattern = re.compile(r'^[a-z0-9_-]{3,64}$')

    # ==========================================
    # ID VALIDATION AND GENERATION
    # ==========================================
    
    def _validate_character_id(self, character_id: str) -> bool:
        """Validate character ID format (^[a-z0-9_-]{3,64}$)."""
        return bool(self._id_pattern.match(character_id))
    
    def _generate_safe_id(self, name: str) -> str:
        """Generate a safe character ID from name with UUID suffix."""
        # Create slug from name
        slug = re.sub(r'[^a-z0-9]+', '_', name.lower().strip())
        slug = re.sub(r'_+', '_', slug).strip('_')
        
        # Truncate and add short UUID
        short_uuid = str(uuid.uuid4())[:8]
        max_slug_len = 64 - len(short_uuid) - 1  # Account for underscore
        if len(slug) > max_slug_len:
            slug = slug[:max_slug_len]
        
        return f"{slug}_{short_uuid}"
    
    def _sanitize_character_id(self, character_id: str) -> str:
        """Sanitize and validate character ID, generating new one if invalid."""
        if self._validate_character_id(character_id):
            return character_id
        
        # Generate new safe ID
        return self._generate_safe_id(character_id)

    # ==========================================
    # ATOMIC FILE OPERATIONS
    # ==========================================
    
    def _get_character_path(self, character_id: str) -> Path:
        """Get file path for character data."""
        return self.storage_path / f"{character_id}.json"
    
    def _atomic_write_character(self, character_id: str, 
                              data: Dict[str, Any]) -> ValidationResult:
        """Atomically write character data to filesystem."""
        file_path = self._get_character_path(character_id)
        temp_path = file_path.with_suffix('.tmp')
        
        try:
            # Write to temp file
            with open(temp_path, 'w') as f:
                if self.enable_file_locking:
                    portalocker.lock(f, portalocker.LOCK_EX)
                json.dump(data, f, indent=2, sort_keys=True)
            
            # Atomic rename
            os.replace(temp_path, file_path)
            
            result = ValidationResult()
            result.add_issue(f"Character {character_id} saved successfully", 
                           ValidationSeverity.INFO)
            return result
            
        except Exception as e:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
            
            result = ValidationResult()
            result.add_issue(f"Failed to save character {character_id}: {e}", 
                           ValidationSeverity.CRITICAL, code="SAVE_FAILED")
            return result
    
    def _load_character_from_disk(self, character_id: str) -> Optional[Dict[str, Any]]:
        """Load character data from disk with optional locking."""
        file_path = self._get_character_path(character_id)
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r') as f:
                if self.enable_file_locking:
                    portalocker.lock(f, portalocker.LOCK_SH)
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load character {character_id}: {e}")
            return None

    # ==========================================
    # METADATA AND CONTENT HASHING
    # ==========================================
    
    def _calculate_content_hash(self, character: Dict[str, Any]) -> str:
        """Calculate SHA256 hash of canonicalized character data."""
        # Create copy without metadata for hashing
        hashable_data = {k: v for k, v in character.items() if k != "_metadata"}
        canonical_json = json.dumps(hashable_data, sort_keys=True, 
                                   separators=(',', ':'))
        return hashlib.sha256(canonical_json.encode()).hexdigest()
    
    def _update_metadata(self, character: Dict[str, Any]) -> None:
        """Update character metadata with current timestamp and hash."""
        now = datetime.now().isoformat()
        
        if "_metadata" not in character:
            character["_metadata"] = CharacterMetadata().__dict__
        
        metadata = character["_metadata"]
        metadata["last_modified"] = now
        metadata["rule_system_version"] = getattr(self.rule_system, 'version', 'unknown')
        metadata["content_hash"] = self._calculate_content_hash(character)

    # ==========================================
    # LRU CACHE MANAGEMENT
    # ==========================================
    
    def _cache_character(self, character: Dict[str, Any]) -> None:
        """Add character to LRU cache with size management."""
        character_id = character["id"]
        
        with self._lock:
            # Remove existing entry to update position
            if character_id in self._character_cache:
                del self._character_cache[character_id]
            
            # Add to cache
            self._character_cache[character_id] = copy.deepcopy(character)
            self._cache_timestamps[character_id] = time.time()
            
            # Enforce size limit (LRU eviction)
            while len(self._character_cache) > self.max_cache_entries:
                oldest_id = next(iter(self._character_cache))
                del self._character_cache[oldest_id]
                self._cache_timestamps.pop(oldest_id, None)
    
    def _get_cached_character(self, character_id: str) -> Optional[Dict[str, Any]]:
        """Get character from cache if valid (LRU access with hit/miss tracking)."""
        with self._lock:
            if character_id not in self._character_cache:
                self._cache_misses += 1
                return None
            
            # Check TTL
            timestamp = self._cache_timestamps.get(character_id, 0)
            if time.time() - timestamp > self.cache_ttl:
                del self._character_cache[character_id]
                self._cache_timestamps.pop(character_id, None)
                self._cache_misses += 1
                return None
            
            # Cache hit - move to end (most recently used)
            character = self._character_cache.pop(character_id)
            self._character_cache[character_id] = character
            self._cache_hits += 1
            
            return copy.deepcopy(character)
    
    def invalidate_cache(self, character_id: Optional[str] = None) -> None:
        """Invalidate cache entries (targeted or full clear)."""
        with self._lock:
            if character_id:
                self._character_cache.pop(character_id, None)
                self._cache_timestamps.pop(character_id, None)
            else:
                self._character_cache.clear()
                self._cache_timestamps.clear()

    # ==========================================
    # VALIDATION
    # ==========================================
    
    def _normalize_validation_result(self, 
                                   validation_result: Union[ValidationResult, 
                                                          Mapping[str, Any]]) -> ValidationResult:
        """
        Normalize validation results from IRuleSystem into our ValidationResult format.
        Handles both modern ValidationResult objects and legacy dict mappings.
        """
        result = ValidationResult()
        
        # Handle modern ValidationResult objects
        if hasattr(validation_result, 'is_valid'):
            result.is_valid = validation_result.is_valid
            
            # Copy severity if available
            if hasattr(validation_result, 'severity'):
                result.severity = validation_result.severity
            
            # Copy issues if available
            if hasattr(validation_result, 'issues') and validation_result.issues:
                result.issues.extend(validation_result.issues)
            
            # Copy legacy lists if available
            if hasattr(validation_result, 'warnings') and validation_result.warnings:
                for warning in validation_result.warnings:
                    result.add_issue(warning, ValidationSeverity.WARNING, 
                                   code="RULE_SYSTEM")
            
            if hasattr(validation_result, 'errors') and validation_result.errors:
                for error in validation_result.errors:
                    result.add_issue(error, ValidationSeverity.ERROR, 
                                   code="RULE_SYSTEM")
        
        # Handle legacy dict mappings
        elif isinstance(validation_result, Mapping):
            result.is_valid = validation_result.get("is_valid", True)
            
            # Process errors and warnings from legacy format
            for error in validation_result.get("errors", []):
                result.add_issue(f"Rule system error: {error}", ValidationSeverity.ERROR, 
                               code="RULE_SYSTEM")
            
            for warning in validation_result.get("warnings", []):
                result.add_issue(f"Rule system warning: {warning}", 
                               ValidationSeverity.WARNING, code="RULE_SYSTEM")
        
        return result
    
    def validate_character(self, character: Dict[str, Any]) -> ValidationResult:
        """Validate character data with severity-based issues and compatibility."""
        result = ValidationResult()
        
        # Basic structure validation
        if not isinstance(character, dict):
            result.add_issue("Character must be a dictionary", 
                           ValidationSeverity.CRITICAL, code="INVALID_TYPE")
            return result
        
        # Required fields
        required_fields = ["id", "name"]
        for field in required_fields:
            if field not in character:
                result.add_issue(f"Missing required field: {field}", 
                               ValidationSeverity.CRITICAL, field=field)
        
        # ID validation
        if "id" in character:
            if not self._validate_character_id(character["id"]):
                result.add_issue("Invalid character ID format", 
                               ValidationSeverity.ERROR, field="id", code="INVALID_ID")
        
        # Rule system validation with compatibility handling
        try:
            system_validation = self.rule_system.validate_character(character)
            normalized_result = self._normalize_validation_result(system_validation)
            
            # Merge normalized results
            result.issues.extend(normalized_result.issues)
            if normalized_result.severity > result.severity:
                result.severity = normalized_result.severity
            if not normalized_result.is_valid:
                result.is_valid = False
            
            # Merge legacy lists for backwards compatibility
            result.warnings.extend(normalized_result.warnings)
            result.errors.extend(normalized_result.errors)
            
        except Exception as e:
            result.add_issue(f"Rule system validation failed: {e}", 
                           ValidationSeverity.ERROR, code="VALIDATION_EXCEPTION")
        
        return result

    # ==========================================
    # CORE CHARACTER OPERATIONS
    # ==========================================
    
    def create_character(self, 
                        name: str,
                        character_type: CharacterType = CharacterType.PLAYER_CHARACTER,
                        char_class: str = "",
                        race_or_heritage: str = "",
                        background: str = "",
                        voice_style: str = "default",
                        character_id: Optional[str] = None) -> str:
        """Create a new character with Daggerheart defaults."""
        
        # Generate or sanitize ID
        if character_id:
            character_id = self._sanitize_character_id(character_id)
        else:
            character_id = self._generate_safe_id(name)
        
        # Create character with Daggerheart defaults
        character = {
            "id": character_id,
            "name": name,
            "character_type": character_type.value,
            "class": char_class,
            "heritage": race_or_heritage,  # Daggerheart uses "heritage"
            "background": background,
            "voice_style": voice_style,
            "level": 1,
            "status": CharacterStatus.ACTIVE.value,
            
            # Daggerheart-specific defaults
            "traits": {
                "Agility": 0,
                "Strength": 0, 
                "Finesse": 0,
                "Instinct": 0,
                "Presence": 0,
                "Knowledge": 0
            },
            "hp": {"current": 20, "max": 20},
            "stress": {"current": 0, "max": 6},  # Daggerheart stress system
            "evasion": 10,  # Default evasion
            "armor": {"name": "", "threshold": 0, "score": 0},
            
            # Standard fields
            "conditions": [],
            "inventory": [],
            "tags": [],
            "notes": "",
            "experience": 0
        }
        
        # Add metadata
        self._update_metadata(character)
        
        # Validate and save
        validation_result = self.save_character(character)
        
        if (not validation_result.is_valid and 
            validation_result.severity >= ValidationSeverity.ERROR):
            raise ValueError(f"Character creation failed validation: {validation_result.issues}")
        
        return character_id
    
    def get_character(self, character_id: str) -> Optional[Dict[str, Any]]:
        """Get character by ID (cached or from disk)."""
        # Try cache first
        character = self._get_cached_character(character_id)
        if character:
            # Update last_used
            character.setdefault("_metadata", {})["last_used"] = datetime.now().isoformat()
            return character
        
        # Load from disk
        character = self._load_character_from_disk(character_id)
        if character:
            # Update metadata and cache
            self._update_metadata(character)
            character["_metadata"]["last_used"] = datetime.now().isoformat()
            self._cache_character(character)
            
        return character
    
    def save_character(self, character: Dict[str, Any]) -> ValidationResult:
        """Save character with validation and atomic write."""
        character_id = character.get("id")
        if not character_id:
            result = ValidationResult()
            result.add_issue("Character missing ID", ValidationSeverity.CRITICAL)
            return result
        
        # Sanitize ID
        character["id"] = self._sanitize_character_id(character_id)
        
        # Validate if required
        if self.validate_on_save:
            validation_result = self.validate_character(character)
            
            # Block on CRITICAL issues
            if validation_result.severity >= ValidationSeverity.CRITICAL:
                return validation_result
            
            # Optionally block on ERROR issues
            if validation_result.severity >= ValidationSeverity.ERROR:
                # Log but don't block (configurable behavior)
                self.logger.warning(f"Saving character {character_id} with errors: {validation_result.issues}")
        
        # Update metadata
        self._update_metadata(character)
        
        # Atomic write
        save_result = self._atomic_write_character(character["id"], character)
        
        # Cache if successful
        if save_result.is_valid:
            self._cache_character(character)
            
            # Trigger event hooks
            for hook in self._event_hooks["on_saved"]:
                try:
                    hook(character["id"], character)
                except Exception as e:
                    self.logger.error(f"Save hook failed: {e}")
        
        return save_result
    
    def delete_character(self, character_id: str) -> bool:
        """Delete character from disk and cache."""
        file_path = self._get_character_path(character_id)
        
        try:
            if file_path.exists():
                file_path.unlink()
            
            # Remove from cache
            self.invalidate_cache(character_id)
            
            # Trigger event hooks
            for hook in self._event_hooks["on_deleted"]:
                try:
                    hook(character_id, None)
                except Exception as e:
                    self.logger.error(f"Delete hook failed: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete character {character_id}: {e}")
            return False

    # ==========================================
    # ADVANCED SEARCH AND FILTERING
    # ==========================================
    
    def _robust_enum_coercion(self, value: Any, enum_class: type) -> Optional[Any]:
        """Safely coerce value to enum, handling unknown values gracefully."""
        if value is None:
            return None
        
        try:
            if isinstance(value, enum_class):
                return value
            
            # Try by value first
            for enum_member in enum_class:
                if enum_member.value == value:
                    return enum_member
            
            # Try by name
            if isinstance(value, str):
                return enum_class[value.upper()]
                
        except (KeyError, ValueError, AttributeError):
            pass
        
        return None
    
    def _stable_sort_key(self, character: Dict[str, Any], 
                        sort_field: str) -> Tuple[str, Any]:
        """Generate stable sort key that avoids cross-type comparisons."""
        value = character.get(sort_field, "")
        
        # Determine type tag for stable sorting
        if value is None:
            type_tag = "0_none"
            sort_value = ""
        elif isinstance(value, bool):
            type_tag = "1_bool"
            sort_value = value
        elif isinstance(value, (int, float)):
            type_tag = "2_number"
            sort_value = value
        elif isinstance(value, str):
            type_tag = "3_string"
            sort_value = value.lower()
        elif isinstance(value, (list, tuple)):
            type_tag = "4_list"
            sort_value = len(value)
        elif isinstance(value, dict):
            type_tag = "5_dict"
            sort_value = len(value)
        else:
            type_tag = "6_other"
            sort_value = str(value).lower()
        
        return (type_tag, sort_value)
    
    def search_characters(self, filters: SearchFilters) -> List[Dict[str, Any]]:
        """Search characters with advanced filtering and stable sorting."""
        characters = []
        
        # Load all characters
        for char_file in self.storage_path.glob("*.json"):
            try:
                character = self._load_character_from_disk(char_file.stem)
                if character:
                    characters.append(character)
            except Exception as e:
                self.logger.warning(f"Failed to load {char_file}: {e}")
        
        # Apply filters
        filtered_characters = []
        
        for character in characters:
            # Name pattern matching
            if filters.name_pattern:
                if not re.search(filters.name_pattern, character.get("name", ""), 
                                re.IGNORECASE):
                    continue
            
            # Character type filtering with robust enum coercion
            if filters.character_type:
                char_type = self._robust_enum_coercion(character.get("character_type"), 
                                                     CharacterType)
                if char_type != filters.character_type:
                    continue
            
            # Status filtering with robust enum coercion
            if filters.status:
                status = self._robust_enum_coercion(character.get("status"), 
                                                  CharacterStatus)
                if status != filters.status:
                    continue
            
            # Rule system filtering with prefix matching
            if filters.rule_system:
                metadata = character.get("_metadata", {})
                char_rule_system = metadata.get("rule_system_version", "unknown").lower()
                filter_value = filters.rule_system.value.lower()
                if not char_rule_system.startswith(filter_value):
                    continue
            
            # Level range filtering
            char_level = character.get("level", 1)
            if filters.min_level and char_level < filters.min_level:
                continue
            if filters.max_level and char_level > filters.max_level:
                continue
            
            # Date filtering with ISO parsing
            metadata = character.get("_metadata", {})
            
            if filters.created_after:
                try:
                    created_at = datetime.fromisoformat(metadata.get("created_at", ""))
                    filter_date = datetime.fromisoformat(filters.created_after)
                    if created_at < filter_date:
                        continue
                except (ValueError, TypeError):
                    continue
            
            if filters.created_before:
                try:
                    created_at = datetime.fromisoformat(metadata.get("created_at", ""))
                    filter_date = datetime.fromisoformat(filters.created_before)
                    if created_at > filter_date:
                        continue
                except (ValueError, TypeError):
                    continue
            
            if filters.last_used_after:
                try:
                    last_used = datetime.fromisoformat(metadata.get("last_used", ""))
                    filter_date = datetime.fromisoformat(filters.last_used_after)
                    if last_used < filter_date:
                        continue
                except (ValueError, TypeError):
                    continue
            
            # Tag filtering
            if filters.tags:
                char_tags = set(character.get("tags", []))
                filter_tags = set(filters.tags)
                if not filter_tags.issubset(char_tags):
                    continue
            
            # Voice style filtering
            if filters.voice_style:
                if character.get("voice_style") != filters.voice_style:
                    continue
            
            filtered_characters.append(character)
        
        # Stable sorting
        sort_field = filters.sort_by or "name"
        filtered_characters.sort(
            key=lambda char: self._stable_sort_key(char, sort_field),
            reverse=filters.sort_desc
        )
        
        # Limit results
        if filters.limit:
            filtered_characters = filtered_characters[:filters.limit]
        
        return filtered_characters

    # ==========================================
    # SYSTEM CONVERSION
    # ==========================================
    
    def convert_character_system(self, character_id: str, 
                               target: RuleSystemType) -> ValidationResult:
        """
        Convert character between rule systems using export/import.
        Validates against target ruleset if system_loader is available.
        """
        result = ValidationResult()
        
        # Load character
        character = self.get_character(character_id)
        if not character:
            result.add_issue(f"Character {character_id} not found", 
                           ValidationSeverity.CRITICAL, code="NOT_FOUND")
            return result
        
        try:
            # Export character data from current system
            exported_data = self.rule_system.export_character_for_system(character, target)
            
            # Update rule system metadata
            character["_metadata"]["rule_system_version"] = target.value
            
            # Merge exported data
            character.update(exported_data)
            
            # Validate against target system if loader is available
            if self.system_loader:
                try:
                    target_system = self.system_loader(target)
                    target_validation = target_system.validate_character(character)
                    target_result = self._normalize_validation_result(target_validation)
                    
                    if not target_result.is_valid:
                        result.add_issue("Target system validation failed", 
                                       ValidationSeverity.ERROR, code="TARGET_VALIDATION")
                        result.issues.extend(target_result.issues)
                        result.severity = max(result.severity, target_result.severity)
                        result.is_valid = False
                        return result
                        
                except Exception as e:
                    result.add_issue(f"Target system validation error: {e}", 
                                   ValidationSeverity.WARNING, code="TARGET_VALIDATION_ERROR")
            
            # Re-validate with current system after conversion
            validation_result = self.validate_character(character)
            
            if validation_result.is_valid or validation_result.severity < ValidationSeverity.ERROR:
                # Save converted character
                save_result = self.save_character(character)
                result.add_issue(f"Successfully converted character {character_id} to {target.value}", 
                               ValidationSeverity.INFO)
                
                # Invalidate cache to ensure fresh data
                self.invalidate_cache(character_id)
                
                # Merge save issues
                result.issues.extend(save_result.issues)
                if save_result.severity > result.severity:
                    result.severity = save_result.severity
                    result.is_valid = save_result.is_valid
            else:
                result = validation_result
            
        except Exception as e:
            result.add_issue(f"System conversion failed: {e}", 
                           ValidationSeverity.CRITICAL, code="CONVERSION_FAILED")
        
        return result

    # ==========================================
    # EVENT HOOKS
    # ==========================================
    
    def add_event_hook(self, event_type: str, callback: Callable) -> None:
        """Add event hook for character lifecycle events."""
        if event_type in self._event_hooks:
            self._event_hooks[event_type].append(callback)
    
    def remove_event_hook(self, event_type: str, callback: Callable) -> None:
        """Remove event hook."""
        if event_type in self._event_hooks:
            try:
                self._event_hooks[event_type].remove(callback)
            except ValueError:
                pass

    # ==========================================
    # DAGGERHEART SPECIFIC HELPERS
    # ==========================================
    
    def daggerheart_stress_clear_snapshot(self, character_id: str) -> Optional[Dict[str, Any]]:
        """Create snapshot before clearing stress on critical (Daggerheart mechanic)."""
        character = self.get_character(character_id)
        if not character:
            return None
        
        # Create snapshot
        snapshot = {
            "character_id": character_id,
            "timestamp": datetime.now().isoformat(),
            "stress_before": character.get("stress", {}),
            "hp_before": character.get("hp", {}),
            "conditions_before": copy.deepcopy(character.get("conditions", []))
        }
        
        # Clear stress (Daggerheart mechanic)
        if "stress" in character:
            character["stress"]["current"] = 0
        
        # Save character
        self.save_character(character)
        
        return snapshot

    # ==========================================
    # UTILITY METHODS
    # ==========================================
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics with hit/miss tracking."""
        with self._lock:
            total_requests = self._cache_hits + self._cache_misses
            hit_ratio = (self._cache_hits / total_requests 
                        if total_requests > 0 else 0.0)
            
            return {
                "size": len(self._character_cache),
                "max_size": self.max_cache_entries,
                "ttl": self.cache_ttl,
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "hit_ratio": hit_ratio
            }
    
    def backup_character(self, character_id: str) -> bool:
        """Create backup of character file."""
        source_path = self._get_character_path(character_id)
        if not source_path.exists():
            return False
        
        backup_path = source_path.with_suffix(f'.backup.{int(time.time())}.json')
        try:
            backup_path.write_bytes(source_path.read_bytes())
            return True
        except Exception as e:
            self.logger.error(f"Backup failed for {character_id}: {e}")
            return False

    # ==========================================
    # CONTEXT MANAGER
    # ==========================================
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cache flush and metadata updates."""
        # Save cached characters with fresh metadata
        with self._lock:
            for character_id, character in self._character_cache.items():
                try:
                    # Update metadata before saving
                    self._update_metadata(character)
                    self._atomic_write_character(character_id, character)
                except Exception as e:
                    self.logger.error(f"Failed to save cached character {character_id}: {e}")

# ==========================================
# EXPORTS
# ==========================================

__all__ = [
    "CharacterManager", 
    "ValidationResult", 
    "ValidationSeverity",
    "CharacterType", 
    "CharacterStatus", 
    "SearchFilters",
    "CharacterMetadata"
]
