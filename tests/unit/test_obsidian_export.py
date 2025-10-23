"""
Unit tests for Obsidian export functionality

Tests markdown generation and file export.
"""

import pytest
import tempfile
from pathlib import Path
from fractal_agent.memory.obsidian_export import ObsidianExporter
from fractal_agent.memory.short_term import ShortTermMemory


class TestObsidianExporter:
    """Test ObsidianExporter functionality"""

    @pytest.fixture
    def temp_vault(self):
        """Create temporary Obsidian vault"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def exporter(self, temp_vault):
        """Create Obsidian exporter instance"""
        return ObsidianExporter(vault_path=temp_vault)

    @pytest.fixture
    def memory_with_session(self, temp_vault):
        """Create memory with a sample session"""
        memory = ShortTermMemory(log_dir=temp_vault / "logs")

        task_id = memory.start_task(
            agent_id="test_agent",
            agent_type="operational",
            task_description="Test task",
            inputs={"test": "data"}
        )

        memory.end_task(
            task_id=task_id,
            outputs={"result": "success"},
            metadata={"tokens": 500}
        )

        return memory

    def test_exporter_initialization(self, exporter):
        """Test exporter initializes correctly"""
        assert exporter is not None
        assert exporter.vault_path.exists()

    def test_export_session_creates_file(self, exporter, memory_with_session):
        """Test exporting session creates markdown file"""
        export_file = exporter.export_session(memory_with_session)

        assert export_file.exists()
        assert export_file.suffix == ".md"

    def test_export_session_contains_yaml_frontmatter(self, exporter, memory_with_session):
        """Test exported file contains YAML frontmatter"""
        export_file = exporter.export_session(memory_with_session)
        content = export_file.read_text()

        assert content.startswith("---")
        assert "session_id:" in content

    def test_export_session_contains_task_info(self, exporter, memory_with_session):
        """Test exported file contains task information"""
        export_file = exporter.export_session(memory_with_session)
        content = export_file.read_text()

        assert "Test task" in content
        assert "operational" in content

    def test_export_session_contains_approval_checkbox(self, exporter, memory_with_session):
        """Test exported file contains approval checkbox"""
        export_file = exporter.export_session(memory_with_session)
        content = export_file.read_text()

        assert "- [ ]" in content  # Unchecked checkbox

    def test_export_multiple_sessions(self, exporter, temp_vault):
        """Test exporting multiple sessions"""
        import time

        # Create two sessions
        memory1 = ShortTermMemory(log_dir=temp_vault / "logs1")
        task1 = memory1.start_task("agent1", "operational", "Task 1", {})
        memory1.end_task(task1, {"result": "1"}, {})

        # Small delay to ensure different timestamps
        time.sleep(1)

        memory2 = ShortTermMemory(log_dir=temp_vault / "logs2")
        task2 = memory2.start_task("agent2", "operational", "Task 2", {})
        memory2.end_task(task2, {"result": "2"}, {})

        # Export both
        file1 = exporter.export_session(memory1)
        file2 = exporter.export_session(memory2)

        assert file1.exists()
        assert file2.exists()
        # Files should be different (different timestamps)
        assert file1 != file2

    def test_export_session_with_multiple_tasks(self, exporter, temp_vault):
        """Test exporting session with multiple tasks"""
        memory = ShortTermMemory(log_dir=temp_vault / "logs")

        # Create multiple tasks
        for i in range(3):
            task_id = memory.start_task(
                f"agent_{i}",
                "operational",
                f"Task {i}",
                {"index": i}
            )
            memory.end_task(task_id, {"result": f"Result {i}"}, {})

        export_file = exporter.export_session(memory)
        content = export_file.read_text()

        # Verify all tasks are included
        assert "Task 0" in content
        assert "Task 1" in content
        assert "Task 2" in content

    def test_export_preserves_task_hierarchy(self, exporter, temp_vault):
        """Test export preserves parent-child task relationships"""
        memory = ShortTermMemory(log_dir=temp_vault / "logs")

        # Create parent task
        parent_id = memory.start_task("control", "control", "Parent task", {})

        # Create child task
        child_id = memory.start_task(
            "operational",
            "operational",
            "Child task",
            {},
            parent_task_id=parent_id
        )

        memory.end_task(child_id, {"result": "child done"}, {})
        memory.end_task(parent_id, {"result": "parent done"}, {})

        export_file = exporter.export_session(memory)
        content = export_file.read_text()

        assert "Parent task" in content
        assert "Child task" in content

    def test_exporter_initialization_with_default_vault(self):
        """Test exporter initializes with default vault path when None"""
        exporter = ObsidianExporter(vault_path=None)

        assert exporter is not None
        assert str(exporter.vault_path) == "obsidian_vault"
        assert exporter.review_folder.exists()

    def test_load_approval_status_nonexistent_file(self, exporter):
        """Test load_approval_status returns None for nonexistent file"""
        fake_file = exporter.review_folder / "nonexistent_file.md"

        status = exporter.load_approval_status(fake_file)

        assert status is None

    def test_load_approval_status_approved(self, exporter, temp_vault):
        """Test load_approval_status detects approved status"""
        test_file = exporter.review_folder / "test_session.md"
        test_file.write_text("""
# Test Session

## Human Review

- [x] Approved
- [ ] Rejected
- [ ] Needs Revision
""")

        status = exporter.load_approval_status(test_file)

        assert status == "approved"

    def test_load_approval_status_approved_uppercase_x(self, exporter, temp_vault):
        """Test load_approval_status detects approved with uppercase X"""
        test_file = exporter.review_folder / "test_session.md"
        test_file.write_text("""
# Test Session

- [X] Approved
- [ ] Rejected
""")

        status = exporter.load_approval_status(test_file)

        assert status == "approved"

    def test_load_approval_status_rejected(self, exporter, temp_vault):
        """Test load_approval_status detects rejected status"""
        test_file = exporter.review_folder / "test_session.md"
        test_file.write_text("""
# Test Session

- [ ] Approved
- [x] Rejected
- [ ] Needs Revision
""")

        status = exporter.load_approval_status(test_file)

        assert status == "rejected"

    def test_load_approval_status_needs_revision(self, exporter, temp_vault):
        """Test load_approval_status detects needs_revision status"""
        test_file = exporter.review_folder / "test_session.md"
        test_file.write_text("""
# Test Session

- [ ] Approved
- [ ] Rejected
- [x] Needs Revision
""")

        status = exporter.load_approval_status(test_file)

        assert status == "needs_revision"

    def test_load_approval_status_no_checkbox_checked(self, exporter, temp_vault):
        """Test load_approval_status returns None when no checkbox is checked"""
        test_file = exporter.review_folder / "test_session.md"
        test_file.write_text("""
# Test Session

- [ ] Approved
- [ ] Rejected
- [ ] Needs Revision
""")

        status = exporter.load_approval_status(test_file)

        assert status is None
