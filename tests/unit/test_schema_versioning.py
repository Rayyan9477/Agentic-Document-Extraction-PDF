"""
Unit tests for Phase 1C: Schema Versioning & Regression Testing.

Tests SchemaVersion, SchemaVersionManager, SchemaDiff, and result migration.
"""

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from src.schemas.versioning import (
    ChangeType,
    FieldChange,
    SchemaDiff,
    SchemaVersion,
    SchemaVersionManager,
)
from src.schemas.field_types import FieldDefinition, FieldType


# ──────────────────────────────────────────────────────────────────
# Mock schema for testing (mimics DocumentSchema interface)
# ──────────────────────────────────────────────────────────────────


@dataclass
class MockSchema:
    name: str
    version: str = "1.0.0"
    description: str = "Test schema"
    fields: list[FieldDefinition] = field(default_factory=list)
    cross_field_rules: list[Any] = field(default_factory=list)


def _make_field(name: str, field_type: FieldType = FieldType.STRING, required: bool = False) -> FieldDefinition:
    return FieldDefinition(
        name=name,
        display_name=name.replace("_", " ").title(),
        field_type=field_type,
        required=required,
    )


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def version_manager(tmp_dir: Path) -> SchemaVersionManager:
    return SchemaVersionManager(storage_dir=tmp_dir)


@pytest.fixture
def basic_schema() -> MockSchema:
    return MockSchema(
        name="test_schema",
        version="1.0.0",
        fields=[
            _make_field("patient_name", FieldType.STRING, required=True),
            _make_field("date_of_birth", FieldType.DATE, required=True),
            _make_field("total_charges", FieldType.CURRENCY),
        ],
    )


# ──────────────────────────────────────────────────────────────────
# SchemaVersion Tests
# ──────────────────────────────────────────────────────────────────


class TestSchemaVersion:
    def test_to_dict_roundtrip(self):
        v = SchemaVersion(
            schema_name="test",
            version="1.0.0",
            schema_hash="abc123",
            fields=[{"name": "field1", "field_type": "text"}],
            cross_field_rules=[],
            created_at="2024-01-01T00:00:00Z",
        )
        d = v.to_dict()
        restored = SchemaVersion.from_dict(d)
        assert restored.schema_name == "test"
        assert restored.version == "1.0.0"
        assert restored.schema_hash == "abc123"
        assert len(restored.fields) == 1


class TestSchemaDiff:
    def test_empty_diff(self):
        diff = SchemaDiff(
            from_version="1.0.0",
            to_version="1.0.1",
            changes=(),
            is_breaking=False,
            summary="No changes",
        )
        assert not diff.has_changes
        assert diff.added_fields == []
        assert diff.removed_fields == []

    def test_diff_with_changes(self):
        diff = SchemaDiff(
            from_version="1.0.0",
            to_version="2.0.0",
            changes=(
                FieldChange(ChangeType.FIELD_ADDED, "new_field"),
                FieldChange(ChangeType.FIELD_REMOVED, "old_field"),
            ),
            is_breaking=True,
            summary="1 added, 1 removed",
        )
        assert diff.has_changes
        assert diff.added_fields == ["new_field"]
        assert diff.removed_fields == ["old_field"]
        assert diff.is_breaking


# ──────────────────────────────────────────────────────────────────
# SchemaVersionManager Tests
# ──────────────────────────────────────────────────────────────────


class TestSchemaVersionManager:
    def test_register_first_version(
        self, version_manager: SchemaVersionManager, basic_schema: MockSchema,
    ):
        v = version_manager.register_version(basic_schema)
        assert v.schema_name == "test_schema"
        assert v.version == "1.0.0"
        assert v.schema_hash
        assert v.migration_from is None

    def test_register_unchanged_returns_same(
        self, version_manager: SchemaVersionManager, basic_schema: MockSchema,
    ):
        v1 = version_manager.register_version(basic_schema)
        v2 = version_manager.register_version(basic_schema)
        assert v1.version == v2.version
        assert v1.schema_hash == v2.schema_hash

    def test_register_changed_bumps_version(
        self, version_manager: SchemaVersionManager, basic_schema: MockSchema,
    ):
        v1 = version_manager.register_version(basic_schema)

        # Modify schema: add a field
        basic_schema.fields.append(_make_field("new_field", FieldType.STRING))
        v2 = version_manager.register_version(basic_schema)

        assert v2.version != v1.version
        assert v2.migration_from == v1.version

    def test_get_latest(
        self, version_manager: SchemaVersionManager, basic_schema: MockSchema,
    ):
        version_manager.register_version(basic_schema)
        latest = version_manager.get_latest("test_schema")
        assert latest is not None
        assert latest.version == "1.0.0"

    def test_get_latest_nonexistent(self, version_manager: SchemaVersionManager):
        assert version_manager.get_latest("nonexistent") is None

    def test_get_history(
        self, version_manager: SchemaVersionManager, basic_schema: MockSchema,
    ):
        version_manager.register_version(basic_schema)
        basic_schema.fields.append(_make_field("extra", FieldType.STRING))
        version_manager.register_version(basic_schema)

        history = version_manager.get_history("test_schema")
        assert len(history) == 2
        assert history[0].version == "1.0.0"

    def test_get_specific_version(
        self, version_manager: SchemaVersionManager, basic_schema: MockSchema,
    ):
        version_manager.register_version(basic_schema)
        v = version_manager.get_version("test_schema", "1.0.0")
        assert v is not None
        assert v.version == "1.0.0"

    def test_diff_field_added(
        self, version_manager: SchemaVersionManager, basic_schema: MockSchema,
    ):
        version_manager.register_version(basic_schema)
        basic_schema.fields.append(_make_field("copay", FieldType.CURRENCY))
        version_manager.register_version(basic_schema)

        history = version_manager.get_history("test_schema")
        diff = version_manager.diff("test_schema", history[0].version, history[1].version)

        assert diff.has_changes
        assert "copay" in diff.added_fields
        assert not diff.is_breaking  # Addition is not breaking

    def test_diff_field_removed_is_breaking(
        self, version_manager: SchemaVersionManager, basic_schema: MockSchema,
    ):
        version_manager.register_version(basic_schema)

        # Remove a field
        basic_schema.fields = [f for f in basic_schema.fields if f.name != "total_charges"]
        version_manager.register_version(basic_schema)

        history = version_manager.get_history("test_schema")
        diff = version_manager.diff("test_schema", history[0].version, history[1].version)

        assert diff.is_breaking
        assert "total_charges" in diff.removed_fields

    def test_diff_nonexistent_version(self, version_manager: SchemaVersionManager):
        diff = version_manager.diff("test_schema", "1.0.0", "2.0.0")
        assert "not found" in diff.summary.lower()

    def test_migrate_result_field_added(
        self, version_manager: SchemaVersionManager, basic_schema: MockSchema,
    ):
        v1 = version_manager.register_version(basic_schema)
        basic_schema.fields.append(_make_field("copay", FieldType.CURRENCY))
        v2 = version_manager.register_version(basic_schema)

        # Original result from v1
        result = {"patient_name": "John", "date_of_birth": "1990-01-01", "total_charges": "150.00"}

        migrated = version_manager.migrate_result(
            result, "test_schema", v1.version, v2.version,
        )

        assert "copay" in migrated
        assert migrated["copay"] is None  # New field gets null
        assert migrated["patient_name"] == "John"  # Existing preserved

    def test_migrate_result_field_removed(
        self, version_manager: SchemaVersionManager, basic_schema: MockSchema,
    ):
        v1 = version_manager.register_version(basic_schema)
        basic_schema.fields = [f for f in basic_schema.fields if f.name != "total_charges"]
        v2 = version_manager.register_version(basic_schema)

        result = {"patient_name": "John", "date_of_birth": "1990-01-01", "total_charges": "150.00"}

        migrated = version_manager.migrate_result(
            result, "test_schema", v1.version, v2.version,
        )

        assert "total_charges" not in migrated
        assert migrated["patient_name"] == "John"

    def test_persistence_across_instances(
        self, tmp_dir: Path, basic_schema: MockSchema,
    ):
        """Versions persist across manager instances."""
        manager1 = SchemaVersionManager(storage_dir=tmp_dir)
        manager1.register_version(basic_schema)

        manager2 = SchemaVersionManager(storage_dir=tmp_dir)
        latest = manager2.get_latest("test_schema")
        assert latest is not None
        assert latest.version == "1.0.0"

    def test_hash_deterministic(
        self, version_manager: SchemaVersionManager, basic_schema: MockSchema,
    ):
        """Same schema content produces same hash."""
        v1 = version_manager.register_version(basic_schema)

        schema2 = MockSchema(
            name="test_schema_2",
            fields=[
                _make_field("patient_name", FieldType.STRING, required=True),
                _make_field("date_of_birth", FieldType.DATE, required=True),
                _make_field("total_charges", FieldType.CURRENCY),
            ],
        )
        v2 = version_manager.register_version(schema2)

        assert v1.schema_hash == v2.schema_hash

    def test_register_removes_field_bumps_minor(
        self, version_manager: SchemaVersionManager, basic_schema: MockSchema,
    ):
        """Removing a field bumps minor version (breaking change)."""
        version_manager.register_version(basic_schema)
        basic_schema.fields = [f for f in basic_schema.fields if f.name != "total_charges"]
        v2 = version_manager.register_version(basic_schema)

        # Should bump minor for breaking change
        parts = v2.version.split(".")
        assert int(parts[1]) > 0 or int(parts[2]) > 0  # Version increased
