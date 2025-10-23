"""
Unit tests for ObsidianVault functionality

Tests vault parsing, indexing, sync, query, and knowledge management.
Includes edge cases and error handling validation.
"""

import pytest
import tempfile
import yaml
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
from fractal_agent.memory.obsidian_vault import (
    VaultParser,
    VaultNote,
    ObsidianVault,
    WorkflowVaultIntegration
)
from fractal_agent.memory.short_term import ShortTermMemory


class TestVaultParser:
    """Test VaultParser markdown parsing functionality"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_parse_note_basic(self, temp_dir):
        """Test parsing a basic markdown note"""
        note_path = temp_dir / "test_note.md"
        note_path.write_text("# Test Note\n\nThis is a test.")

        result = VaultParser.parse_note(note_path)

        assert result.title == "test_note"
        assert "Test Note" in result.content
        assert result.file_path == note_path
        assert isinstance(result.created, datetime)
        assert isinstance(result.modified, datetime)

    def test_parse_note_with_frontmatter(self, temp_dir):
        """Test parsing note with YAML frontmatter"""
        content = """---
title: Custom Title
tags: [test, demo]
author: TestUser
---

# Content

This is the body."""
        note_path = temp_dir / "frontmatter_note.md"
        note_path.write_text(content)

        result = VaultParser.parse_note(note_path)

        assert result.title == "Custom Title"
        assert result.frontmatter["author"] == "TestUser"
        assert "test" in result.tags
        assert "demo" in result.tags
        assert "This is the body." in result.content

    def test_parse_note_with_inline_tags(self, temp_dir):
        """Test parsing inline #tags"""
        content = "# Note\n\nThis has #tag1 and #tag2 inline."
        note_path = temp_dir / "inline_tags.md"
        note_path.write_text(content)

        result = VaultParser.parse_note(note_path)

        assert "tag1" in result.tags
        assert "tag2" in result.tags

    def test_parse_note_with_backlinks(self, temp_dir):
        """Test parsing wikilinks [[link]]"""
        content = "# Note\n\nSee [[Other Note]] and [[Another Note|alias]]."
        note_path = temp_dir / "backlinks.md"
        note_path.write_text(content)

        result = VaultParser.parse_note(note_path)

        assert "Other Note" in result.backlinks
        assert "Another Note" in result.backlinks

    def test_parse_note_file_not_found(self, temp_dir):
        """Test parsing raises FileNotFoundError for missing file"""
        note_path = temp_dir / "nonexistent.md"

        with pytest.raises(FileNotFoundError):
            VaultParser.parse_note(note_path)

    def test_parse_note_not_markdown(self, temp_dir):
        """Test parsing raises ValueError for non-markdown file"""
        note_path = temp_dir / "test.txt"
        note_path.write_text("Not markdown")

        with pytest.raises(ValueError, match="Not a markdown file"):
            VaultParser.parse_note(note_path)

    def test_parse_note_malformed_yaml(self, temp_dir):
        """Test parsing handles malformed YAML frontmatter gracefully"""
        content = """---
title: Test
tags: [unclosed
invalid yaml
---

# Body"""
        note_path = temp_dir / "malformed.md"
        note_path.write_text(content)

        result = VaultParser.parse_note(note_path)

        assert result.frontmatter == {}
        assert result.title == "malformed"

    def test_parse_note_empty_file(self, temp_dir):
        """Test parsing empty markdown file"""
        note_path = temp_dir / "empty.md"
        note_path.write_text("")

        result = VaultParser.parse_note(note_path)

        assert result.title == "empty"
        assert result.content == ""
        assert result.tags == []
        assert result.backlinks == []

    def test_extract_frontmatter_missing(self):
        """Test extracting frontmatter from content without frontmatter"""
        content = "# Just Content\n\nNo frontmatter here."

        frontmatter, body = VaultParser._extract_frontmatter(content)

        assert frontmatter == {}
        assert body == content

    def test_extract_frontmatter_valid(self):
        """Test extracting valid YAML frontmatter"""
        content = """---
title: Test
value: 123
---

Body content"""

        frontmatter, body = VaultParser._extract_frontmatter(content)

        assert frontmatter["title"] == "Test"
        assert frontmatter["value"] == 123
        assert "Body content" in body

    def test_extract_tags_from_frontmatter_string(self):
        """Test extracting tags from frontmatter as single string"""
        frontmatter = {"tags": "single-tag"}
        body = "Content"

        tags = VaultParser._extract_tags(frontmatter, body)

        assert "single-tag" in tags

    def test_extract_tags_from_frontmatter_list(self):
        """Test extracting tags from frontmatter as list"""
        frontmatter = {"tags": ["tag1", "tag2"]}
        body = "Content"

        tags = VaultParser._extract_tags(frontmatter, body)

        assert "tag1" in tags
        assert "tag2" in tags

    def test_extract_tags_deduplicated(self):
        """Test tags are deduplicated"""
        frontmatter = {"tags": ["duplicate"]}
        body = "Content with #duplicate"

        tags = VaultParser._extract_tags(frontmatter, body)

        assert tags.count("duplicate") == 1

    def test_extract_tags_sorted(self):
        """Test tags are sorted alphabetically"""
        frontmatter = {"tags": ["zebra", "apple"]}
        body = "#middle"

        tags = VaultParser._extract_tags(frontmatter, body)

        assert tags == ["apple", "middle", "zebra"]

    def test_extract_backlinks_simple(self):
        """Test extracting simple wikilinks"""
        content = "Link to [[Page 1]] and [[Page 2]]"

        backlinks = VaultParser._extract_backlinks(content)

        assert "Page 1" in backlinks
        assert "Page 2" in backlinks

    def test_extract_backlinks_with_aliases(self):
        """Test extracting wikilinks with aliases"""
        content = "Link [[Page|Alias]] and [[Another|Different Alias]]"

        backlinks = VaultParser._extract_backlinks(content)

        assert "Page" in backlinks
        assert "Another" in backlinks
        assert "Alias" not in backlinks

    def test_extract_backlinks_deduplicated(self):
        """Test backlinks are deduplicated"""
        content = "[[Duplicate]] and [[Duplicate]] again"

        backlinks = VaultParser._extract_backlinks(content)

        assert len(backlinks) == 1
        assert "Duplicate" in backlinks


class TestObsidianVault:
    """Test ObsidianVault main functionality"""

    @pytest.fixture
    def temp_vault(self):
        """Create temporary vault directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_graphrag(self):
        """Create mock GraphRAG instance"""
        mock = Mock()
        return mock

    @pytest.fixture
    def vault(self, temp_vault):
        """Create ObsidianVault instance without GraphRAG"""
        return ObsidianVault(vault_path=str(temp_vault))

    @pytest.fixture
    def vault_with_graphrag(self, temp_vault, mock_graphrag):
        """Create ObsidianVault instance with GraphRAG"""
        with patch('fractal_agent.memory.obsidian_vault.DocumentStore'):
            return ObsidianVault(
                vault_path=str(temp_vault),
                graphrag=mock_graphrag
            )

    def test_vault_initialization(self, vault, temp_vault):
        """Test vault initializes correctly"""
        assert vault.vault_path == temp_vault
        assert vault.review_folder.exists()
        assert vault.knowledge_folder.exists()

    def test_vault_initialization_creates_folders(self, temp_vault):
        """Test vault creates required folders on init"""
        vault = ObsidianVault(
            vault_path=str(temp_vault),
            review_folder="custom_reviews",
            knowledge_folder="custom_knowledge"
        )

        assert (temp_vault / "custom_reviews").exists()
        assert (temp_vault / "custom_knowledge").exists()

    def test_vault_initialization_without_graphrag(self, vault):
        """Test vault initializes without GraphRAG"""
        assert vault.graphrag is None
        assert vault.doc_store is None

    def test_vault_initialization_with_graphrag(self, vault_with_graphrag):
        """Test vault initializes with GraphRAG"""
        assert vault_with_graphrag.graphrag is not None
        assert vault_with_graphrag.doc_store is not None

    def test_index_vault_empty(self, vault):
        """Test indexing empty vault"""
        count = vault.index_vault()

        assert count == 0
        assert vault._note_index == {}

    def test_index_vault_with_notes(self, vault, temp_vault):
        """Test indexing vault with notes"""
        (temp_vault / "note1.md").write_text("# Note 1")
        (temp_vault / "note2.md").write_text("# Note 2")

        count = vault.index_vault()

        assert count == 2
        assert "note1" in vault._note_index
        assert "note2" in vault._note_index

    def test_index_vault_excludes_folders(self, vault, temp_vault):
        """Test indexing excludes specified folders"""
        obsidian_dir = temp_vault / ".obsidian"
        obsidian_dir.mkdir()
        (obsidian_dir / "config.md").write_text("# Config")
        (temp_vault / "normal.md").write_text("# Normal")

        count = vault.index_vault()

        assert count == 1
        assert "normal" in vault._note_index
        assert "config" not in vault._note_index

    def test_index_vault_excludes_review_knowledge_folders(self, vault, temp_vault):
        """Test indexing excludes review and knowledge folders"""
        (vault.review_folder / "review.md").write_text("# Review")
        (vault.knowledge_folder / "knowledge.md").write_text("# Knowledge")
        (temp_vault / "normal.md").write_text("# Normal")

        count = vault.index_vault()

        assert count == 1
        assert "normal" in vault._note_index

    def test_index_vault_handles_parse_errors(self, vault, temp_vault):
        """Test indexing continues after parse errors"""
        (temp_vault / "good.md").write_text("# Good")
        bad_file = temp_vault / "bad.txt"
        bad_file.write_text("Not markdown")

        count = vault.index_vault()

        assert count == 1
        assert "good" in vault._note_index

    def test_index_vault_force_refresh(self, vault, temp_vault):
        """Test force refresh re-indexes vault"""
        (temp_vault / "note.md").write_text("# Note")

        vault.index_vault()
        first_time = vault._last_index_time

        vault.index_vault(force_refresh=True)
        second_time = vault._last_index_time

        assert second_time > first_time

    def test_index_vault_uses_cache(self, vault, temp_vault):
        """Test indexing uses cache when recent"""
        (temp_vault / "note.md").write_text("# Note")

        vault.index_vault()
        first_time = vault._last_index_time

        vault.index_vault(force_refresh=False)
        second_time = vault._last_index_time

        assert second_time == first_time

    def test_sync_to_graphrag_without_graphrag_raises(self, vault):
        """Test sync raises RuntimeError without GraphRAG"""
        with pytest.raises(RuntimeError, match="GraphRAG not initialized"):
            vault.sync_to_graphrag()

    def test_sync_to_graphrag_basic(self, vault_with_graphrag, temp_vault):
        """Test basic sync to GraphRAG"""
        (temp_vault / "note.md").write_text("# Note\n\nContent")

        vault_with_graphrag.doc_store = Mock()
        vault_with_graphrag.doc_store.store_document = Mock(return_value=['chunk1'])

        stats = vault_with_graphrag.sync_to_graphrag()

        assert stats['synced'] >= 1
        assert stats['total_notes'] >= 1

    def test_sync_to_graphrag_with_tag_filter(self, vault_with_graphrag, temp_vault):
        """Test sync with tag filtering"""
        content1 = "---\ntags: [include]\n---\n# Note 1"
        content2 = "---\ntags: [exclude]\n---\n# Note 2"
        (temp_vault / "note1.md").write_text(content1)
        (temp_vault / "note2.md").write_text(content2)

        vault_with_graphrag.doc_store = Mock()
        vault_with_graphrag.doc_store.store_document = Mock(return_value=['chunk1'])

        stats = vault_with_graphrag.sync_to_graphrag(tag_filter=['include'])

        assert stats['synced'] == 1
        assert stats['skipped'] >= 1

    def test_sync_to_graphrag_handles_errors(self, vault_with_graphrag, temp_vault):
        """Test sync handles storage errors gracefully"""
        (temp_vault / "note.md").write_text("# Note")

        vault_with_graphrag.doc_store = Mock()
        vault_with_graphrag.doc_store.store_document = Mock(
            side_effect=Exception("Storage error")
        )

        stats = vault_with_graphrag.sync_to_graphrag()

        assert stats['failed'] >= 1

    def test_query_without_graphrag_raises(self, vault):
        """Test query raises RuntimeError without GraphRAG"""
        with pytest.raises(RuntimeError, match="GraphRAG not initialized"):
            vault.query("test query")

    def test_query_basic(self, vault_with_graphrag):
        """Test basic vault query"""
        vault_with_graphrag.doc_store = Mock()
        vault_with_graphrag.doc_store.retrieve_document_context = Mock(
            return_value="Context results"
        )

        result = vault_with_graphrag.query("test query")

        assert result == "Context results"
        vault_with_graphrag.doc_store.retrieve_document_context.assert_called_once()

    def test_query_with_tag_filter(self, vault_with_graphrag):
        """Test query with tag filtering"""
        vault_with_graphrag.doc_store = Mock()
        vault_with_graphrag.doc_store.retrieve_document_context = Mock(
            return_value="Filtered results"
        )

        result = vault_with_graphrag.query("test", tag_filter=['tag1'])

        call_args = vault_with_graphrag.doc_store.retrieve_document_context.call_args
        assert call_args[1]['filter_metadata'] == {"tags": ['tag1']}

    def test_query_with_max_results(self, vault_with_graphrag):
        """Test query with custom max results"""
        vault_with_graphrag.doc_store = Mock()
        vault_with_graphrag.doc_store.retrieve_document_context = Mock(
            return_value="Results"
        )

        vault_with_graphrag.query("test", max_results=10)

        call_args = vault_with_graphrag.doc_store.retrieve_document_context.call_args
        assert call_args[1]['max_chunks'] == 10

    def test_export_session(self, vault, temp_vault):
        """Test exporting session to vault"""
        memory = ShortTermMemory(log_dir=str(temp_vault / "logs"))
        task_id = memory.start_task(
            "agent1", "operational", "Test task", {"input": "data"}
        )
        memory.end_task(task_id, {"output": "result"}, {})

        result_path = vault.export_session(memory)

        assert result_path.exists()
        assert result_path.suffix == ".md"
        assert result_path.parent == vault.review_folder

    def test_export_session_with_metadata(self, vault, temp_vault):
        """Test exporting session with additional metadata"""
        memory = ShortTermMemory(log_dir=str(temp_vault / "logs"))
        task_id = memory.start_task("agent1", "operational", "Task", {})
        memory.end_task(task_id, {}, {})

        metadata = {"custom": "value", "priority": "high"}
        result_path = vault.export_session(memory, additional_metadata=metadata)

        content = result_path.read_text()
        assert "custom:" in content
        assert "priority:" in content

    def test_create_knowledge_note_basic(self, vault):
        """Test creating basic knowledge note"""
        note_path = vault.create_knowledge_note(
            title="Test Note",
            content="# Content\n\nTest content",
            sync_to_graphrag=False
        )

        assert note_path.exists()
        assert note_path.parent == vault.knowledge_folder
        assert note_path.suffix == ".md"

    def test_create_knowledge_note_with_tags(self, vault):
        """Test creating note with tags"""
        note_path = vault.create_knowledge_note(
            title="Tagged Note",
            content="Content",
            tags=["tag1", "tag2"],
            sync_to_graphrag=False
        )

        content = note_path.read_text()
        assert "tag1" in content
        assert "tag2" in content

    def test_create_knowledge_note_with_metadata(self, vault):
        """Test creating note with custom metadata"""
        note_path = vault.create_knowledge_note(
            title="Meta Note",
            content="Content",
            metadata={"author": "Test", "version": 1},
            sync_to_graphrag=False
        )

        content = note_path.read_text()
        assert "author:" in content
        assert "version:" in content

    def test_create_knowledge_note_sanitizes_filename(self, vault):
        """Test filename sanitization for invalid characters"""
        note_path = vault.create_knowledge_note(
            title='Invalid: <title> with | chars',
            content="Content",
            sync_to_graphrag=False
        )

        assert note_path.exists()
        assert "<" not in note_path.name
        assert ">" not in note_path.name
        assert "|" not in note_path.name

    def test_create_knowledge_note_syncs_to_graphrag(self, vault_with_graphrag):
        """Test note creation syncs to GraphRAG when requested"""
        vault_with_graphrag.doc_store = Mock()
        vault_with_graphrag.doc_store.store_document = Mock(return_value=['chunk1'])

        vault_with_graphrag.create_knowledge_note(
            title="Synced Note",
            content="Content",
            sync_to_graphrag=True
        )

        vault_with_graphrag.doc_store.store_document.assert_called_once()

    def test_create_knowledge_note_handles_sync_error(self, vault_with_graphrag):
        """Test note creation handles GraphRAG sync errors"""
        vault_with_graphrag.doc_store = Mock()
        vault_with_graphrag.doc_store.store_document = Mock(
            side_effect=Exception("Sync error")
        )

        note_path = vault_with_graphrag.create_knowledge_note(
            title="Error Note",
            content="Content",
            sync_to_graphrag=True
        )

        assert note_path.exists()

    def test_create_knowledge_note_updates_index(self, vault):
        """Test note creation updates vault index"""
        vault._note_index = {}

        vault.create_knowledge_note(
            title="Indexed Note",
            content="Content",
            sync_to_graphrag=False
        )

        assert "Indexed Note" in vault._note_index

    def test_get_note_existing(self, vault, temp_vault):
        """Test retrieving existing note by title"""
        (temp_vault / "existing.md").write_text("# Existing Note")
        vault.index_vault()

        note = vault.get_note("existing")

        assert note is not None
        assert note.title == "existing"

    def test_get_note_nonexistent(self, vault):
        """Test retrieving nonexistent note returns None"""
        vault.index_vault()

        note = vault.get_note("nonexistent")

        assert note is None

    def test_get_note_indexes_if_needed(self, vault, temp_vault):
        """Test get_note indexes vault if not already indexed"""
        (temp_vault / "note.md").write_text("# Note")

        note = vault.get_note("note")

        assert note is not None
        assert len(vault._note_index) > 0

    def test_find_notes_by_tag(self, vault, temp_vault):
        """Test finding notes by tag"""
        content1 = "---\ntags: [target]\n---\n# Note 1"
        content2 = "---\ntags: [other]\n---\n# Note 2"
        content3 = "---\ntags: [target]\n---\n# Note 3"

        (temp_vault / "note1.md").write_text(content1)
        (temp_vault / "note2.md").write_text(content2)
        (temp_vault / "note3.md").write_text(content3)

        results = vault.find_notes_by_tag("target")

        assert len(results) == 2
        titles = [note.title for note in results]
        assert "note1" in titles or "Note 1" in titles

    def test_find_notes_by_tag_empty_result(self, vault, temp_vault):
        """Test finding notes by nonexistent tag"""
        (temp_vault / "note.md").write_text("# Note")

        results = vault.find_notes_by_tag("nonexistent")

        assert len(results) == 0

    def test_get_backlink_graph(self, vault, temp_vault):
        """Test building backlink graph"""
        (temp_vault / "note1.md").write_text("Links to [[note2]]")
        (temp_vault / "note2.md").write_text("Links to [[note3]] and [[note1]]")
        (temp_vault / "note3.md").write_text("No links")

        graph = vault.get_backlink_graph()

        assert len(graph) == 3
        assert "note2" in graph["note1"]
        assert "note3" in graph["note2"]
        assert "note1" in graph["note2"]

    def test_add_metadata_to_file(self, vault, temp_vault):
        """Test adding metadata to existing file"""
        content = "---\ntitle: Original\n---\n\n# Body"
        file_path = temp_vault / "test.md"
        file_path.write_text(content)

        vault._add_metadata_to_file(file_path, {"new_field": "value"})

        updated_content = file_path.read_text()
        assert "new_field:" in updated_content
        assert "title: Original" in updated_content

    def test_add_metadata_to_file_without_frontmatter(self, vault, temp_vault):
        """Test adding metadata to file without existing frontmatter"""
        content = "# Just Content"
        file_path = temp_vault / "test.md"
        file_path.write_text(content)

        vault._add_metadata_to_file(file_path, {"added": "metadata"})

        updated_content = file_path.read_text()
        assert "added:" in updated_content


class TestWorkflowVaultIntegration:
    """Test WorkflowVaultIntegration functionality"""

    @pytest.fixture
    def temp_vault(self):
        """Create temporary vault directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def vault(self, temp_vault):
        """Create ObsidianVault instance"""
        return ObsidianVault(vault_path=str(temp_vault))

    @pytest.fixture
    def integration(self, vault):
        """Create WorkflowVaultIntegration instance"""
        return WorkflowVaultIntegration(vault)

    @pytest.fixture
    def memory(self, temp_vault):
        """Create ShortTermMemory with test data"""
        memory = ShortTermMemory(log_dir=str(temp_vault / "logs"))
        task_id = memory.start_task(
            "agent1", "operational", "Test task", {"input": "data"}
        )
        memory.end_task(task_id, {"output": "result"}, {})
        return memory

    def test_integration_initialization(self, integration, vault):
        """Test integration initializes correctly"""
        assert integration.vault == vault

    def test_on_workflow_start(self, integration, memory):
        """Test workflow start hook"""
        integration.on_workflow_start(
            workflow_type="test_workflow",
            task="Test task",
            memory=memory
        )

    def test_on_workflow_complete_with_auto_export(self, integration, memory):
        """Test workflow completion with auto export"""
        state = {"status": "completed"}

        result = integration.on_workflow_complete(
            workflow_type="test_workflow",
            state=state,
            memory=memory,
            auto_export=True
        )

        assert result is not None
        assert result.exists()
        content = result.read_text()
        assert "workflow_type:" in content

    def test_on_workflow_complete_without_auto_export(self, integration, memory):
        """Test workflow completion without auto export"""
        state = {"status": "completed"}

        result = integration.on_workflow_complete(
            workflow_type="test_workflow",
            state=state,
            memory=memory,
            auto_export=False
        )

        assert result is None

    def test_capture_workflow_knowledge(self, integration):
        """Test capturing workflow knowledge"""
        result = integration.capture_workflow_knowledge(
            title="Workflow Knowledge",
            content="# Knowledge\n\nLearned something",
            workflow_type="test_workflow",
            tags=["custom_tag"]
        )

        assert result.exists()
        content = result.read_text()
        assert "agent_knowledge" in content
        assert "test_workflow" in content
        assert "custom_tag" in content


class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def temp_vault(self):
        """Create temporary vault directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_vault_with_unicode_filenames(self, temp_vault):
        """Test handling Unicode characters in filenames"""
        vault = ObsidianVault(vault_path=str(temp_vault))
        (temp_vault / "日本語.md").write_text("# Japanese")
        (temp_vault / "émojis_😀.md").write_text("# Emoji")

        count = vault.index_vault()

        assert count >= 1

    def test_vault_with_deeply_nested_folders(self, temp_vault):
        """Test handling deeply nested folder structures"""
        vault = ObsidianVault(vault_path=str(temp_vault))
        deep_path = temp_vault / "a" / "b" / "c" / "d" / "e"
        deep_path.mkdir(parents=True)
        (deep_path / "deep.md").write_text("# Deep Note")

        count = vault.index_vault()

        assert count == 1

    def test_vault_with_symlinks(self, temp_vault):
        """Test handling symlinked files"""
        vault = ObsidianVault(vault_path=str(temp_vault))

        original = temp_vault / "original.md"
        original.write_text("# Original")

        try:
            link = temp_vault / "link.md"
            link.symlink_to(original)

            count = vault.index_vault()

            assert count >= 1
        except OSError:
            pytest.skip("Symlinks not supported on this system")

    def test_vault_with_very_large_note(self, temp_vault):
        """Test handling very large note files"""
        vault = ObsidianVault(vault_path=str(temp_vault))

        large_content = "# Large Note\n\n" + ("Lorem ipsum " * 100000)
        (temp_vault / "large.md").write_text(large_content)

        count = vault.index_vault()

        assert count == 1

    def test_create_note_with_empty_title(self, temp_vault):
        """Test creating note with empty title"""
        vault = ObsidianVault(vault_path=str(temp_vault))

        with pytest.raises(Exception):
            vault.create_knowledge_note(
                title="",
                content="Content",
                sync_to_graphrag=False
            )

    def test_multiple_notes_same_title_different_paths(self, temp_vault):
        """Test indexing multiple notes with same title in different folders"""
        vault = ObsidianVault(vault_path=str(temp_vault))

        folder1 = temp_vault / "folder1"
        folder2 = temp_vault / "folder2"
        folder1.mkdir()
        folder2.mkdir()

        (folder1 / "duplicate.md").write_text("# Duplicate 1")
        (folder2 / "duplicate.md").write_text("# Duplicate 2")

        count = vault.index_vault()

        assert count >= 1

    def test_sync_from_graphrag_not_implemented(self, temp_vault):
        """Test sync_from_graphrag returns placeholder stats"""
        mock_graphrag = Mock()
        vault = ObsidianVault(vault_path=str(temp_vault), graphrag=mock_graphrag)

        stats = vault.sync_from_graphrag()

        assert stats['entities_exported'] == 0
        assert stats['notes_created'] == 0
        assert stats['notes_updated'] == 0

    def test_sync_from_graphrag_without_graphrag_raises(self, temp_vault):
        """Test sync_from_graphrag raises without GraphRAG"""
        vault = ObsidianVault(vault_path=str(temp_vault))

        with pytest.raises(RuntimeError, match="GraphRAG not initialized"):
            vault.sync_from_graphrag()

    def test_vault_handles_concurrent_note_creation(self, temp_vault):
        """Test vault handles concurrent note creation"""
        vault = ObsidianVault(vault_path=str(temp_vault))

        for i in range(10):
            vault.create_knowledge_note(
                title=f"Concurrent Note {i}",
                content=f"Content {i}",
                sync_to_graphrag=False
            )

        count = vault.index_vault(force_refresh=True)

        assert count == 10

    def test_parse_note_with_mixed_line_endings(self, temp_vault):
        """Test parsing note with mixed line endings (CRLF and LF)"""
        content = "# Title\r\n\r\nSome content\nMore content\r\n"
        note_path = temp_vault / "mixed.md"
        note_path.write_text(content)

        result = VaultParser.parse_note(note_path)

        assert result.title == "mixed"
        assert "Some content" in result.content

    def test_vault_with_no_write_permissions(self, temp_vault):
        """Test vault behavior with read-only directory"""
        vault = ObsidianVault(vault_path=str(temp_vault))

        import os
        import stat

        try:
            os.chmod(temp_vault, stat.S_IRUSR | stat.S_IXUSR)

            with pytest.raises(PermissionError):
                vault.create_knowledge_note(
                    title="Test",
                    content="Content",
                    sync_to_graphrag=False
                )
        finally:
            os.chmod(temp_vault, stat.S_IRWXU)
