"""
Schema versioning and migration system.

Provides automatic versioning of document schemas with:
- Semantic versioning (semver) for each schema
- Content-based change detection via SHA-256 hashing
- Version history tracking with diffs
- Result migration between schema versions
- Backward-compatible golden dataset management
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from src.config import get_logger
from src.schemas.field_types import FieldDefinition, FieldType


logger = get_logger(__name__)


class ChangeType(str, Enum):
    """Types of changes between schema versions."""

    FIELD_ADDED = "field_added"
    FIELD_REMOVED = "field_removed"
    FIELD_RENAMED = "field_renamed"
    FIELD_TYPE_CHANGED = "field_type_changed"
    FIELD_REQUIRED_CHANGED = "field_required_changed"
    RULE_ADDED = "rule_added"
    RULE_REMOVED = "rule_removed"
    DESCRIPTION_CHANGED = "description_changed"


@dataclass(frozen=True, slots=True)
class FieldChange:
    """A single field-level change between versions."""

    change_type: ChangeType
    field_name: str
    old_value: Any = None
    new_value: Any = None
    description: str = ""


@dataclass(frozen=True, slots=True)
class SchemaDiff:
    """Difference between two schema versions."""

    from_version: str
    to_version: str
    changes: tuple[FieldChange, ...]
    is_breaking: bool
    summary: str

    @property
    def added_fields(self) -> list[str]:
        return [c.field_name for c in self.changes if c.change_type == ChangeType.FIELD_ADDED]

    @property
    def removed_fields(self) -> list[str]:
        return [c.field_name for c in self.changes if c.change_type == ChangeType.FIELD_REMOVED]

    @property
    def has_changes(self) -> bool:
        return len(self.changes) > 0


@dataclass(slots=True)
class SchemaVersion:
    """A versioned snapshot of a schema definition."""

    schema_name: str
    version: str
    schema_hash: str
    fields: list[dict[str, Any]]
    cross_field_rules: list[dict[str, Any]]
    created_at: str
    migration_from: str | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "version": self.version,
            "schema_hash": self.schema_hash,
            "fields": self.fields,
            "cross_field_rules": self.cross_field_rules,
            "created_at": self.created_at,
            "migration_from": self.migration_from,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SchemaVersion":
        return cls(
            schema_name=data["schema_name"],
            version=data["version"],
            schema_hash=data["schema_hash"],
            fields=data.get("fields", []),
            cross_field_rules=data.get("cross_field_rules", []),
            created_at=data.get("created_at", ""),
            migration_from=data.get("migration_from"),
            description=data.get("description", ""),
        )


class SchemaVersionManager:
    """
    Manages schema versions with history, diffs, and migrations.

    Each schema version is stored as a JSON file in the versioning directory.
    Changes are detected automatically via content hashing.
    """

    def __init__(self, storage_dir: Path | str | None = None) -> None:
        self._storage_dir = Path(storage_dir) if storage_dir else Path("data/schema_versions")
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[SchemaVersion]] = {}

    def register_version(self, schema: Any) -> SchemaVersion:
        """
        Register a new version of a schema if it has changed.

        Args:
            schema: A DocumentSchema instance.

        Returns:
            The SchemaVersion (new or existing if unchanged).
        """
        schema_name = schema.name
        current_hash = self._compute_schema_hash(schema)

        latest = self.get_latest(schema_name)
        if latest and latest.schema_hash == current_hash:
            logger.debug("schema_unchanged", schema=schema_name, version=latest.version)
            return latest

        # Determine new version
        if latest:
            new_version = self._bump_version(latest.version, schema)
        else:
            new_version = schema.version if hasattr(schema, "version") else "1.0.0"

        # Serialize field definitions
        fields_data = []
        for f in schema.fields:
            field_dict = {
                "name": f.name,
                "display_name": f.display_name,
                "field_type": f.field_type.value if hasattr(f.field_type, "value") else str(f.field_type),
                "required": f.required,
                "description": getattr(f, "description", ""),
            }
            if hasattr(f, "aliases") and f.aliases:
                field_dict["aliases"] = list(f.aliases) if not isinstance(f.aliases, list) else f.aliases
            fields_data.append(field_dict)

        rules_data = []
        for rule in getattr(schema, "cross_field_rules", []):
            rules_data.append({
                "name": getattr(rule, "name", ""),
                "fields": list(getattr(rule, "fields", [])),
                "rule_type": str(getattr(rule, "rule_type", "")),
            })

        version = SchemaVersion(
            schema_name=schema_name,
            version=new_version,
            schema_hash=current_hash,
            fields=fields_data,
            cross_field_rules=rules_data,
            created_at=datetime.now(UTC).isoformat(),
            migration_from=latest.version if latest else None,
            description=getattr(schema, "description", ""),
        )

        self._save_version(version)
        self._invalidate_cache(schema_name)

        logger.info(
            "schema_version_registered",
            schema=schema_name,
            version=new_version,
            hash=current_hash[:12],
            migrated_from=version.migration_from,
        )

        return version

    def get_latest(self, schema_name: str) -> SchemaVersion | None:
        """Get the latest version of a schema."""
        history = self.get_history(schema_name)
        return history[-1] if history else None

    def get_version(self, schema_name: str, version: str) -> SchemaVersion | None:
        """Get a specific version of a schema."""
        history = self.get_history(schema_name)
        for v in history:
            if v.version == version:
                return v
        return None

    def get_history(self, schema_name: str) -> list[SchemaVersion]:
        """Get full version history for a schema, oldest first."""
        if schema_name in self._cache:
            return self._cache[schema_name]

        schema_dir = self._storage_dir / schema_name
        if not schema_dir.exists():
            return []

        versions: list[SchemaVersion] = []
        for f in sorted(schema_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                versions.append(SchemaVersion.from_dict(data))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("schema_version_load_failed", file=str(f), error=str(e))

        self._cache[schema_name] = versions
        return versions

    def diff(self, schema_name: str, from_version: str, to_version: str) -> SchemaDiff:
        """
        Compute the difference between two schema versions.

        Args:
            schema_name: Name of the schema.
            from_version: Starting version.
            to_version: Ending version.

        Returns:
            SchemaDiff with all changes.
        """
        v_from = self.get_version(schema_name, from_version)
        v_to = self.get_version(schema_name, to_version)

        if not v_from or not v_to:
            return SchemaDiff(
                from_version=from_version,
                to_version=to_version,
                changes=(),
                is_breaking=False,
                summary=f"Version not found: {'from' if not v_from else 'to'}",
            )

        changes = self._compute_changes(v_from, v_to)
        is_breaking = any(
            c.change_type in (ChangeType.FIELD_REMOVED, ChangeType.FIELD_TYPE_CHANGED)
            for c in changes
        )

        summary_parts = []
        added = [c for c in changes if c.change_type == ChangeType.FIELD_ADDED]
        removed = [c for c in changes if c.change_type == ChangeType.FIELD_REMOVED]
        if added:
            summary_parts.append(f"{len(added)} field(s) added")
        if removed:
            summary_parts.append(f"{len(removed)} field(s) removed")
        other = len(changes) - len(added) - len(removed)
        if other:
            summary_parts.append(f"{other} other change(s)")

        return SchemaDiff(
            from_version=from_version,
            to_version=to_version,
            changes=tuple(changes),
            is_breaking=is_breaking,
            summary="; ".join(summary_parts) if summary_parts else "No changes",
        )

    def migrate_result(
        self,
        result: dict[str, Any],
        schema_name: str,
        from_version: str,
        to_version: str,
    ) -> dict[str, Any]:
        """
        Migrate an extraction result from one schema version to another.

        Handles field additions (set null), removals (drop), and renames.

        Args:
            result: Original extraction result dict.
            schema_name: Schema name.
            from_version: Source version.
            to_version: Target version.

        Returns:
            Migrated result dict.
        """
        schema_diff = self.diff(schema_name, from_version, to_version)

        if not schema_diff.has_changes:
            return result

        migrated = dict(result)

        for change in schema_diff.changes:
            if change.change_type == ChangeType.FIELD_ADDED:
                if change.field_name not in migrated:
                    migrated[change.field_name] = None

            elif change.change_type == ChangeType.FIELD_REMOVED:
                migrated.pop(change.field_name, None)

            elif change.change_type == ChangeType.FIELD_RENAMED:
                old_name = change.old_value
                new_name = change.new_value
                if old_name in migrated:
                    migrated[new_name] = migrated.pop(old_name)

        logger.debug(
            "result_migrated",
            schema=schema_name,
            from_version=from_version,
            to_version=to_version,
            changes=len(schema_diff.changes),
        )

        return migrated

    def _compute_schema_hash(self, schema: Any) -> str:
        """Compute a deterministic hash of the schema's content."""
        hasher = hashlib.sha256()

        # Hash field definitions in deterministic order
        for f in sorted(schema.fields, key=lambda x: x.name):
            hasher.update(f.name.encode())
            ft_value = f.field_type.value if hasattr(f.field_type, "value") else str(f.field_type)
            hasher.update(ft_value.encode())
            hasher.update(str(f.required).encode())

        # Hash cross-field rules
        for rule in sorted(getattr(schema, "cross_field_rules", []), key=lambda r: getattr(r, "name", "")):
            hasher.update(getattr(rule, "name", "").encode())

        return hasher.hexdigest()

    def _bump_version(self, current: str, schema: Any) -> str:
        """Compute next semver based on change type."""
        parts = current.split(".")
        if len(parts) != 3:
            return "1.0.1"

        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

        # Detect if changes are breaking (for now, just bump patch)
        latest = self.get_latest(schema.name)
        if latest:
            old_fields = {f["name"] for f in latest.fields}
            new_fields = {f.name for f in schema.fields}

            removed = old_fields - new_fields
            if removed:
                # Breaking change: bump minor
                return f"{major}.{minor + 1}.0"

        return f"{major}.{minor}.{patch + 1}"

    def _compute_changes(
        self, v_from: SchemaVersion, v_to: SchemaVersion,
    ) -> list[FieldChange]:
        """Compute field-level changes between two versions."""
        changes: list[FieldChange] = []

        from_fields = {f["name"]: f for f in v_from.fields}
        to_fields = {f["name"]: f for f in v_to.fields}

        # Added fields
        for name in to_fields:
            if name not in from_fields:
                changes.append(FieldChange(
                    change_type=ChangeType.FIELD_ADDED,
                    field_name=name,
                    new_value=to_fields[name].get("field_type"),
                    description=f"Field '{name}' added",
                ))

        # Removed fields
        for name in from_fields:
            if name not in to_fields:
                changes.append(FieldChange(
                    change_type=ChangeType.FIELD_REMOVED,
                    field_name=name,
                    old_value=from_fields[name].get("field_type"),
                    description=f"Field '{name}' removed",
                ))

        # Changed fields
        for name in from_fields:
            if name in to_fields:
                f_old = from_fields[name]
                f_new = to_fields[name]

                if f_old.get("field_type") != f_new.get("field_type"):
                    changes.append(FieldChange(
                        change_type=ChangeType.FIELD_TYPE_CHANGED,
                        field_name=name,
                        old_value=f_old.get("field_type"),
                        new_value=f_new.get("field_type"),
                        description=f"Field '{name}' type changed",
                    ))

                if f_old.get("required") != f_new.get("required"):
                    changes.append(FieldChange(
                        change_type=ChangeType.FIELD_REQUIRED_CHANGED,
                        field_name=name,
                        old_value=f_old.get("required"),
                        new_value=f_new.get("required"),
                        description=f"Field '{name}' required status changed",
                    ))

        return changes

    def _save_version(self, version: SchemaVersion) -> None:
        """Save a version to disk."""
        schema_dir = self._storage_dir / version.schema_name
        schema_dir.mkdir(parents=True, exist_ok=True)

        filename = f"v{version.version.replace('.', '_')}.json"
        path = schema_dir / filename

        path.write_text(
            json.dumps(version.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _invalidate_cache(self, schema_name: str) -> None:
        """Invalidate cached version history."""
        self._cache.pop(schema_name, None)
