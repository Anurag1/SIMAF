import pytest
import tempfile
import subprocess
import yaml
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from fractal_agent.integrations.obsidian.vault_structure import (
    VaultParser,
    VaultNote,
    VaultStructure,
    TemplateManager,
    GitIntegration,
    ObsidianVault
)


class TestVaultParser:

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_parse_note_basic(self, temp_dir):
        note_path = temp_dir / "test_note.md"
        note_path.write_text("# Test Note\n\nThis is a test.", encoding='utf-8')

        result = VaultParser.parse_note(note_path)

        assert result.title == "test_note"
        assert "Test Note" in result.content
        assert result.file_path == note_path
        assert isinstance(result.created, datetime)
        assert isinstance(result.modified, datetime)

    def test_parse_note_with_frontmatter(self, temp_dir):
        content = """---
title: Custom Title
tags: [test, demo]
author: TestUser
---

# Content

This is the body."""
        note_path = temp_dir / "frontmatter_note.md"
        note_path.write_text(content, encoding='utf-8')

        result = VaultParser.parse_note(note_path)

        assert result.title == "Custom Title"
        assert result.frontmatter["author"] == "TestUser"
        assert "test" in result.tags
        assert "demo" in result.tags

    def test_parse_note_with_inline_tags(self, temp_dir):
        content = "# Note\n\nThis has #tag1 and #tag2 inline."
        note_path = temp_dir / "inline_tags.md"
        note_path.write_text(content, encoding='utf-8')

        result = VaultParser.parse_note(note_path)

        assert "tag1" in result.tags
        assert "tag2" in result.tags

    def test_parse_note_with_backlinks(self, temp_dir):
        content = "# Note\n\nSee [[Other Note]] and [[Another Note|alias]]."
        note_path = temp_dir / "backlinks.md"
        note_path.write_text(content, encoding='utf-8')

        result = VaultParser.parse_note(note_path)

        assert "Other Note" in result.backlinks
        assert "Another Note" in result.backlinks

    def test_parse_note_file_not_found(self, temp_dir):
        note_path = temp_dir / "nonexistent.md"

        with pytest.raises(FileNotFoundError):
            VaultParser.parse_note(note_path)

    def test_parse_note_not_markdown(self, temp_dir):
        note_path = temp_dir / "test.txt"
        note_path.write_text("Not markdown", encoding='utf-8')

        with pytest.raises(ValueError, match="Not a markdown file"):
            VaultParser.parse_note(note_path)

    def test_parse_note_malformed_yaml(self, temp_dir):
        content = """---
title: Test
tags: [unclosed
invalid yaml
---

# Body"""
        note_path = temp_dir / "malformed.md"
        note_path.write_text(content, encoding='utf-8')

        result = VaultParser.parse_note(note_path)

        assert result.frontmatter == {}
        assert result.title == "malformed"

    def test_parse_note_empty_file(self, temp_dir):
        note_path = temp_dir / "empty.md"
        note_path.write_text("", encoding='utf-8')

        result = VaultParser.parse_note(note_path)

        assert result.title == "empty"
        assert result.content == ""
        assert result.tags == []
        assert result.backlinks == []

    def test_extract_frontmatter_missing(self):
        content = "# Just Content\n\nNo frontmatter here."

        frontmatter, body = VaultParser._extract_frontmatter(content)

        assert frontmatter == {}
        assert body == content

    def test_extract_frontmatter_valid(self):
        content = """---
title: Test
value: 123
---

Body content"""

        frontmatter, body = VaultParser._extract_frontmatter(content)

        assert frontmatter["title"] == "Test"
        assert frontmatter["value"] == 123
        assert "Body content" in body

    def test_extract_frontmatter_empty(self):
        content = """---
---

Body"""

        frontmatter, body = VaultParser._extract_frontmatter(content)

        assert frontmatter == {}
        assert "Body" in body

    def test_extract_tags_from_frontmatter_string(self):
        frontmatter = {"tags": "single-tag"}
        body = "Content"

        tags = VaultParser._extract_tags(frontmatter, body)

        assert "single-tag" in tags

    def test_extract_tags_from_frontmatter_list(self):
        frontmatter = {"tags": ["tag1", "tag2"]}
        body = "Content"

        tags = VaultParser._extract_tags(frontmatter, body)

        assert "tag1" in tags
        assert "tag2" in tags

    def test_extract_tags_deduplicated(self):
        frontmatter = {"tags": ["duplicate"]}
        body = "Content with #duplicate"

        tags = VaultParser._extract_tags(frontmatter, body)

        assert tags.count("duplicate") == 1

    def test_extract_tags_sorted(self):
        frontmatter = {"tags": ["zebra", "apple"]}
        body = "#middle"

        tags = VaultParser._extract_tags(frontmatter, body)

        assert tags == ["apple", "middle", "zebra"]

    def test_extract_tags_with_hyphens(self):
        frontmatter = {}
        body = "Tags: #multi-word-tag #another-tag"

        tags = VaultParser._extract_tags(frontmatter, body)

        assert "multi-word-tag" in tags
        assert "another-tag" in tags

    def test_extract_backlinks_simple(self):
        content = "Link to [[Page 1]] and [[Page 2]]"

        backlinks = VaultParser._extract_backlinks(content)

        assert "Page 1" in backlinks
        assert "Page 2" in backlinks

    def test_extract_backlinks_with_aliases(self):
        content = "Link [[Page|Alias]] and [[Another|Different Alias]]"

        backlinks = VaultParser._extract_backlinks(content)

        assert "Page" in backlinks
        assert "Another" in backlinks
        assert "Alias" not in backlinks

    def test_extract_backlinks_deduplicated(self):
        content = "[[Duplicate]] and [[Duplicate]] again"

        backlinks = VaultParser._extract_backlinks(content)

        assert len(backlinks) == 1
        assert "Duplicate" in backlinks

    def test_extract_backlinks_with_paths(self):
        content = "Link to [[folder/page]] and [[another/nested/page]]"

        backlinks = VaultParser._extract_backlinks(content)

        assert "folder/page" in backlinks
        assert "another/nested/page" in backlinks


class TestTemplateManager:

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def template_manager(self, temp_dir):
        return TemplateManager(temp_dir / "templates")

    def test_initialization_creates_directory(self, temp_dir):
        templates_path = temp_dir / "templates"
        manager = TemplateManager(templates_path)

        assert templates_path.exists()
        assert templates_path.is_dir()

    def test_initialization_creates_default_templates(self, template_manager):
        templates = template_manager.list_templates()

        assert "daily" in templates
        assert "project" in templates
        assert "meeting" in templates
        assert "research" in templates
        assert "agent_session" in templates

    def test_list_templates(self, template_manager):
        templates = template_manager.list_templates()

        assert isinstance(templates, list)
        assert len(templates) >= 5
        assert all(isinstance(t, str) for t in templates)

    def test_create_from_template_basic(self, template_manager):
        variables = {"date": "2025-10-19"}
        content = template_manager.create_from_template("daily", variables=variables)

        assert "2025-10-19" in content
        assert "Today's Focus" in content
        assert "Tasks" in content

    def test_create_from_template_with_output_path(self, template_manager, temp_dir):
        output_path = temp_dir / "output" / "test.md"
        variables = {"title": "Test Project", "created": "2025-10-19"}

        content = template_manager.create_from_template(
            "project",
            variables=variables,
            output_path=output_path
        )

        assert output_path.exists()
        assert "Test Project" in output_path.read_text(encoding='utf-8')

    def test_create_from_template_nonexistent(self, template_manager):
        with pytest.raises(FileNotFoundError, match="Template not found"):
            template_manager.create_from_template("nonexistent")

    def test_create_from_template_no_variables(self, template_manager):
        content = template_manager.create_from_template("daily")

        assert "{{date}}" in content

    def test_create_from_template_partial_substitution(self, template_manager):
        variables = {"title": "Partial"}
        content = template_manager.create_from_template("project", variables=variables)

        assert "Partial" in content
        assert "{{created}}" in content

    def test_add_custom_template(self, template_manager):
        custom_content = """---
title: {{title}}
---

# {{title}}

Custom template content."""

        template_manager.add_custom_template("custom", custom_content)

        templates = template_manager.list_templates()
        assert "custom" in templates

        content = template_manager.create_from_template(
            "custom",
            variables={"title": "Test"}
        )
        assert "Test" in content
        assert "Custom template content" in content

    def test_add_custom_template_overwrites_existing(self, template_manager):
        content1 = "First version"
        content2 = "Second version"

        template_manager.add_custom_template("test", content1)
        template_manager.add_custom_template("test", content2)

        result = template_manager.create_from_template("test")
        assert "Second version" in result

    def test_daily_template_structure(self, template_manager):
        content = template_manager.create_from_template("daily")

        assert "type: daily" in content
        assert "Today's Focus" in content
        assert "Notes" in content
        assert "Tasks" in content
        assert "Reflections" in content

    def test_project_template_structure(self, template_manager):
        content = template_manager.create_from_template("project")

        assert "type: project" in content
        assert "status: active" in content
        assert "Overview" in content
        assert "Goals" in content
        assert "Milestones" in content

    def test_meeting_template_structure(self, template_manager):
        content = template_manager.create_from_template("meeting")

        assert "type: meeting" in content
        assert "attendees: []" in content
        assert "Agenda" in content
        assert "Discussion" in content
        assert "Action Items" in content

    def test_research_template_structure(self, template_manager):
        content = template_manager.create_from_template("research")

        assert "type: research" in content
        assert "Summary" in content
        assert "Key Findings" in content
        assert "Questions" in content
        assert "References" in content

    def test_agent_session_template_structure(self, template_manager):
        content = template_manager.create_from_template("agent_session")

        assert "type: agent_session" in content
        assert "status: pending_review" in content
        assert "Summary" in content
        assert "Approval Status" in content


class TestGitIntegration:

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def git_integration(self, temp_dir):
        return GitIntegration(temp_dir)

    def test_initialization(self, git_integration, temp_dir):
        assert git_integration.vault_path == temp_dir
        assert git_integration.auto_commit == False

    def test_initialization_with_auto_commit(self, temp_dir):
        git = GitIntegration(temp_dir, auto_commit=True)
        assert git.auto_commit == True

    def test_is_git_repo_false(self, git_integration):
        assert git_integration.is_git_repo() == False

    def test_init_repo_success(self, git_integration):
        result = git_integration.init_repo(initial_commit=False)

        assert result == True
        assert git_integration.is_git_repo() == True

    def test_init_repo_with_initial_commit(self, git_integration, temp_dir):
        (temp_dir / "test.txt").write_text("test", encoding='utf-8')

        result = git_integration.init_repo(initial_commit=True)

        assert result == True
        status = git_integration.get_status()
        assert status['has_changes'] == False

    def test_init_repo_creates_gitignore(self, git_integration, temp_dir):
        git_integration.init_repo()

        gitignore = temp_dir / ".gitignore"
        assert gitignore.exists()
        content = gitignore.read_text(encoding='utf-8')
        assert ".DS_Store" in content
        assert ".obsidian/workspace" in content

    def test_init_repo_already_initialized(self, git_integration):
        git_integration.init_repo()
        result = git_integration.init_repo()

        assert result == True

    def test_get_status_not_a_repo_raises(self, git_integration):
        with pytest.raises(RuntimeError, match="Not a Git repository"):
            git_integration.get_status()

    def test_get_status_clean_repo(self, git_integration, temp_dir):
        git_integration.init_repo(initial_commit=True)

        status = git_integration.get_status()

        assert status['has_changes'] == False
        assert status['untracked_files'] == []
        assert status['modified_files'] == []
        assert status['staged_files'] == []

    def test_get_status_with_untracked_files(self, git_integration, temp_dir):
        git_integration.init_repo(initial_commit=True)
        (temp_dir / "new.txt").write_text("new", encoding='utf-8')

        status = git_integration.get_status()

        assert status['has_changes'] == True
        assert "new.txt" in status['untracked_files']

    def test_get_status_with_modified_files(self, git_integration, temp_dir):
        (temp_dir / "file.txt").write_text("original", encoding='utf-8')
        git_integration.init_repo(initial_commit=True)
        (temp_dir / "file.txt").write_text("modified", encoding='utf-8')

        status = git_integration.get_status()

        assert status['has_changes'] == True
        assert "file.txt" in status['modified_files']

    def test_get_status_with_staged_files(self, git_integration, temp_dir):
        git_integration.init_repo(initial_commit=True)
        (temp_dir / "staged.txt").write_text("staged", encoding='utf-8')
        git_integration._run_git_command(['add', 'staged.txt'])

        status = git_integration.get_status()

        assert "staged.txt" in status['staged_files']

    def test_has_uncommitted_changes_true(self, git_integration, temp_dir):
        git_integration.init_repo(initial_commit=True)
        (temp_dir / "new.txt").write_text("new", encoding='utf-8')

        assert git_integration.has_uncommitted_changes() == True

    def test_has_uncommitted_changes_false(self, git_integration):
        git_integration.init_repo(initial_commit=True)

        assert git_integration.has_uncommitted_changes() == False

    def test_commit_changes_not_a_repo_raises(self, git_integration):
        with pytest.raises(RuntimeError, match="Not a Git repository"):
            git_integration.commit_changes("test")

    def test_commit_changes_success(self, git_integration, temp_dir):
        git_integration.init_repo(initial_commit=True)
        (temp_dir / "file.txt").write_text("content", encoding='utf-8')

        result = git_integration.commit_changes("Add file")

        assert result == True
        status = git_integration.get_status()
        assert status['has_changes'] == False

    def test_commit_changes_with_author(self, git_integration, temp_dir):
        git_integration.init_repo(initial_commit=True)
        (temp_dir / "file.txt").write_text("content", encoding='utf-8')

        result = git_integration.commit_changes(
            "Add file",
            author="Test User <test@example.com>"
        )

        assert result == True

    def test_commit_changes_no_changes(self, git_integration):
        git_integration.init_repo(initial_commit=True)

        result = git_integration.commit_changes("Empty commit")

        assert result == False

    def test_commit_changes_without_add_all(self, git_integration, temp_dir):
        git_integration.init_repo(initial_commit=True)
        (temp_dir / "file.txt").write_text("content", encoding='utf-8')

        result = git_integration.commit_changes("Commit", add_all=False)

        assert result == False

    def test_get_remote_url_no_repo(self, git_integration):
        result = git_integration.get_remote_url()
        assert result is None

    def test_set_remote_url_not_a_repo_raises(self, git_integration):
        with pytest.raises(RuntimeError, match="Not a Git repository"):
            git_integration.set_remote_url("https://github.com/user/repo.git")

    def test_set_remote_url_add_new(self, git_integration):
        git_integration.init_repo()

        git_integration.set_remote_url("https://github.com/user/repo.git")

        url = git_integration.get_remote_url()
        assert url == "https://github.com/user/repo.git"

    def test_set_remote_url_update_existing(self, git_integration):
        git_integration.init_repo()
        git_integration.set_remote_url("https://github.com/user/old.git")

        git_integration.set_remote_url("https://github.com/user/new.git")

        url = git_integration.get_remote_url()
        assert url == "https://github.com/user/new.git"

    def test_run_git_command_failure(self, git_integration):
        with pytest.raises(subprocess.CalledProcessError):
            git_integration._run_git_command(['invalid-command'])

    def test_run_git_command_with_check_false(self, git_integration):
        result = git_integration._run_git_command(['invalid-command'], check=False)
        assert result.returncode != 0


class TestObsidianVault:

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def vault(self, temp_dir):
        return ObsidianVault(str(temp_dir), init_git=False)

    @pytest.fixture
    def vault_with_git(self, temp_dir):
        return ObsidianVault(str(temp_dir), init_git=True)

    def test_initialization(self, vault, temp_dir):
        assert vault.vault_path == temp_dir
        assert vault.structure.root == temp_dir
        assert isinstance(vault.templates, TemplateManager)
        assert isinstance(vault.git, GitIntegration)

    def test_initialization_creates_vault_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vault_path = Path(tmpdir) / "new_vault"
            vault = ObsidianVault(str(vault_path), init_git=False)

            assert vault_path.exists()

    def test_initialization_with_git(self, vault_with_git):
        assert vault_with_git.git.is_git_repo() == True

    def test_initialization_with_auto_commit(self, temp_dir):
        vault = ObsidianVault(str(temp_dir), init_git=True, auto_commit=True)
        assert vault.git.auto_commit == True

    def test_initialize_structure_creates_folders(self, vault):
        vault.initialize_structure(create_readme=False)

        assert vault.structure.projects.exists()
        assert vault.structure.daily.exists()
        assert vault.structure.templates.exists()
        assert vault.structure.resources.exists()
        assert vault.structure.archive.exists()
        assert vault.structure.agent_reviews.exists()
        assert vault.structure.agent_knowledge.exists()

    def test_initialize_structure_creates_readmes(self, vault):
        vault.initialize_structure(create_readme=True)

        assert (vault.structure.projects / "README.md").exists()
        assert (vault.structure.daily / "README.md").exists()
        assert (vault.vault_path / "README.md").exists()

    def test_initialize_structure_readme_content(self, vault):
        vault.initialize_structure(create_readme=True)

        readme = vault.vault_path / "README.md"
        content = readme.read_text(encoding='utf-8')

        assert "Obsidian Vault" in content
        assert "projects/" in content
        assert "BMAD-METHOD" in content

    def test_initialize_structure_folder_readme_content(self, vault):
        vault.initialize_structure(create_readme=True)

        readme = vault.structure.projects / "README.md"
        content = readme.read_text(encoding='utf-8')

        assert "Projects" in content
        assert "Project notes and documentation" in content

    def test_initialize_structure_doesnt_overwrite_readme(self, vault):
        vault.initialize_structure(create_readme=True)

        readme = vault.vault_path / "README.md"
        original = readme.read_text(encoding='utf-8')
        readme.write_text("Custom content", encoding='utf-8')

        vault.initialize_structure(create_readme=True)
        assert readme.read_text(encoding='utf-8') == "Custom content"

    def test_create_note_basic(self, vault):
        vault.initialize_structure()

        note_path = vault.create_note(
            "Test Note",
            "# Test\n\nContent here"
        )

        assert note_path.exists()
        assert note_path.parent == vault.structure.agent_knowledge
        assert note_path.suffix == ".md"

    def test_create_note_with_tags(self, vault):
        vault.initialize_structure()

        note_path = vault.create_note(
            "Tagged Note",
            "Content",
            tags=["tag1", "tag2"]
        )

        content = note_path.read_text(encoding='utf-8')
        assert "tag1" in content
        assert "tag2" in content

    def test_create_note_with_metadata(self, vault):
        vault.initialize_structure()

        note_path = vault.create_note(
            "Meta Note",
            "Content",
            metadata={"author": "Test", "version": 1}
        )

        content = note_path.read_text(encoding='utf-8')
        assert "author:" in content
        assert "version:" in content

    def test_create_note_in_custom_folder(self, vault):
        vault.initialize_structure()

        note_path = vault.create_note(
            "Custom Note",
            "Content",
            folder=vault.structure.projects
        )

        assert note_path.parent == vault.structure.projects

    def test_create_note_sanitizes_filename(self, vault):
        vault.initialize_structure()

        note_path = vault.create_note(
            'Invalid: <title> with | chars',
            "Content"
        )

        assert note_path.exists()
        assert "<" not in note_path.name
        assert ">" not in note_path.name
        assert "|" not in note_path.name

    def test_create_note_with_template(self, vault):
        vault.initialize_structure()

        note_path = vault.create_note(
            "Template Note",
            "",
            template="project"
        )

        content = note_path.read_text(encoding='utf-8')
        assert "type: project" in content
        assert "Template Note" in content

    def test_create_note_updates_index(self, vault):
        vault.initialize_structure()
        vault._note_index = {}

        vault.create_note("Indexed Note", "Content")

        assert "Indexed Note" in vault._note_index

    def test_read_note_existing(self, vault, temp_dir):
        (temp_dir / "existing.md").write_text("# Existing", encoding='utf-8')
        vault.index_vault()

        note = vault.read_note("existing")

        assert note is not None
        assert note.title == "existing"

    def test_read_note_nonexistent(self, vault):
        vault.index_vault()

        note = vault.read_note("nonexistent")

        assert note is None

    def test_read_note_triggers_indexing(self, vault, temp_dir):
        (temp_dir / "note.md").write_text("# Note", encoding='utf-8')

        note = vault.read_note("note")

        assert note is not None
        assert len(vault._note_index) > 0

    def test_update_note_content(self, vault, temp_dir):
        note_path = temp_dir / "update_test.md"
        note_path.write_text("---\ntitle: Update Test\n---\n\nOriginal", encoding='utf-8')
        vault.index_vault()

        result = vault.update_note("Update Test", content="New content")

        assert result == True
        updated = note_path.read_text(encoding='utf-8')
        assert "New content" in updated

    def test_update_note_metadata(self, vault, temp_dir):
        note_path = temp_dir / "meta_test.md"
        note_path.write_text("---\ntitle: Meta Test\n---\n\nContent", encoding='utf-8')
        vault.index_vault()

        result = vault.update_note("Meta Test", metadata={"status": "updated"})

        assert result == True
        updated = note_path.read_text(encoding='utf-8')
        assert "status:" in updated

    def test_update_note_nonexistent(self, vault):
        vault.index_vault()

        result = vault.update_note("Nonexistent", content="Content")

        assert result == False

    def test_update_note_updates_index(self, vault, temp_dir):
        note_path = temp_dir / "indexed.md"
        note_path.write_text("---\ntitle: Indexed\n---\n\nOriginal", encoding='utf-8')
        vault.index_vault()

        vault.update_note("Indexed", content="Updated")

        assert vault._note_index["Indexed"].content != "Original"

    def test_delete_note_archive(self, vault, temp_dir):
        vault.initialize_structure()
        note_path = temp_dir / "delete_test.md"
        note_path.write_text("---\ntitle: Delete Test\n---\n\nContent", encoding='utf-8')
        vault.index_vault()

        result = vault.delete_note("Delete Test", archive=True)

        assert result == True
        assert not note_path.exists()
        assert (vault.structure.archive / "delete_test.md").exists()

    def test_delete_note_permanent(self, vault, temp_dir):
        vault.initialize_structure()
        note_path = temp_dir / "delete_test.md"
        note_path.write_text("---\ntitle: Delete Test\n---\n\nContent", encoding='utf-8')
        vault.index_vault()

        result = vault.delete_note("Delete Test", archive=False)

        assert result == True
        assert not note_path.exists()
        assert not (vault.structure.archive / "delete_test.md").exists()

    def test_delete_note_removes_from_index(self, vault, temp_dir):
        note_path = temp_dir / "indexed.md"
        note_path.write_text("---\ntitle: Indexed\n---\n\nContent", encoding='utf-8')
        vault.index_vault()

        vault.delete_note("Indexed", archive=False)

        assert "Indexed" not in vault._note_index

    def test_delete_note_nonexistent(self, vault):
        vault.index_vault()

        result = vault.delete_note("Nonexistent")

        assert result == False

    def test_index_vault_empty(self, vault):
        count = vault.index_vault()

        assert count == 0
        assert vault._note_index == {}

    def test_index_vault_with_notes(self, vault, temp_dir):
        (temp_dir / "note1.md").write_text("# Note 1", encoding='utf-8')
        (temp_dir / "note2.md").write_text("# Note 2", encoding='utf-8')

        count = vault.index_vault()

        assert count == 2
        assert "note1" in vault._note_index
        assert "note2" in vault._note_index

    def test_index_vault_excludes_obsidian_folder(self, vault, temp_dir):
        obsidian_dir = temp_dir / ".obsidian"
        obsidian_dir.mkdir()
        (obsidian_dir / "config.md").write_text("# Config", encoding='utf-8')
        (temp_dir / "normal.md").write_text("# Normal", encoding='utf-8')

        count = vault.index_vault()

        assert "normal" in vault._note_index
        assert "config" not in vault._note_index

    def test_index_vault_excludes_templates(self, vault):
        vault.initialize_structure()

        count = vault.index_vault()

        template_titles = [note.title for note in vault._note_index.values()]
        assert "daily" not in template_titles

    def test_index_vault_force_refresh(self, vault, temp_dir):
        (temp_dir / "note.md").write_text("# Note", encoding='utf-8')

        vault.index_vault()
        first_time = vault._last_index_time

        vault.index_vault(force_refresh=True)
        second_time = vault._last_index_time

        assert second_time > first_time

    def test_index_vault_uses_cache(self, vault, temp_dir):
        (temp_dir / "note.md").write_text("# Note", encoding='utf-8')

        vault.index_vault()
        first_time = vault._last_index_time

        vault.index_vault(force_refresh=False)
        second_time = vault._last_index_time

        assert second_time == first_time

    def test_index_vault_handles_parse_errors(self, vault, temp_dir):
        (temp_dir / "good.md").write_text("# Good", encoding='utf-8')
        (temp_dir / "bad.txt").write_text("Not markdown", encoding='utf-8')

        count = vault.index_vault()

        assert "good" in vault._note_index

    def test_find_notes_by_tag(self, vault, temp_dir):
        content1 = "---\ntags: [target]\n---\n# Note 1"
        content2 = "---\ntags: [other]\n---\n# Note 2"
        content3 = "---\ntags: [target]\n---\n# Note 3"

        (temp_dir / "note1.md").write_text(content1, encoding='utf-8')
        (temp_dir / "note2.md").write_text(content2, encoding='utf-8')
        (temp_dir / "note3.md").write_text(content3, encoding='utf-8')

        results = vault.find_notes_by_tag("target")

        assert len(results) == 2

    def test_find_notes_by_tag_empty(self, vault, temp_dir):
        (temp_dir / "note.md").write_text("# Note", encoding='utf-8')

        results = vault.find_notes_by_tag("nonexistent")

        assert len(results) == 0

    def test_get_backlink_graph(self, vault, temp_dir):
        (temp_dir / "note1.md").write_text("Links to [[note2]]", encoding='utf-8')
        (temp_dir / "note2.md").write_text("Links to [[note3]]", encoding='utf-8')
        (temp_dir / "note3.md").write_text("No links", encoding='utf-8')

        graph = vault.get_backlink_graph()

        assert "note2" in graph["note1"]
        assert "note3" in graph["note2"]

    def test_create_daily_note(self, vault):
        vault.initialize_structure()
        date = datetime(2025, 10, 19)

        note_path = vault.create_daily_note(date)

        assert note_path.exists()
        assert note_path.parent == vault.structure.daily
        assert "2025-10-19" in note_path.name

    def test_create_daily_note_today(self, vault):
        vault.initialize_structure()

        note_path = vault.create_daily_note()

        assert note_path.exists()

    def test_create_daily_note_already_exists(self, vault):
        vault.initialize_structure()
        date = datetime(2025, 10, 19)

        path1 = vault.create_daily_note(date)
        path2 = vault.create_daily_note(date)

        assert path1 == path2

    def test_create_project_note(self, vault):
        vault.initialize_structure()

        note_path = vault.create_project_note("Test Project")

        assert note_path.exists()
        assert note_path.parent == vault.structure.projects
        content = note_path.read_text(encoding='utf-8')
        assert "Test Project" in content
        assert "type: project" in content

    def test_create_project_note_with_metadata(self, vault):
        vault.initialize_structure()

        note_path = vault.create_project_note(
            "Project",
            metadata={"status": "active"}
        )

        content = note_path.read_text(encoding='utf-8')
        assert "status:" in content

    def test_get_vault_stats(self, vault):
        vault.initialize_structure()

        stats = vault.get_vault_stats()

        assert "total_notes" in stats
        assert "projects" in stats
        assert "daily_notes" in stats
        assert "agent_reviews" in stats
        assert "agent_knowledge" in stats
        assert "templates" in stats

    def test_get_vault_stats_with_notes(self, vault):
        vault.initialize_structure()
        vault.create_note("Test", "Content", folder=vault.structure.agent_knowledge)
        vault.create_daily_note()
        vault.create_project_note("Project")

        stats = vault.get_vault_stats()

        assert stats['agent_knowledge'] >= 1
        assert stats['daily_notes'] >= 1
        assert stats['projects'] >= 1
        assert stats['total_notes'] >= 3

    def test_sync_vault_with_git(self, vault_with_git):
        with patch.object(vault_with_git.git, 'sync', return_value=True) as mock_sync:
            result = vault_with_git.sync_vault()

            assert result == True
            mock_sync.assert_called_once()


class TestEdgeCases:

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_vault_with_unicode_filenames(self, temp_dir):
        vault = ObsidianVault(str(temp_dir), init_git=False)
        (temp_dir / "日本語.md").write_text("# Japanese", encoding='utf-8')
        (temp_dir / "émojis_😀.md").write_text("# Emoji", encoding='utf-8')

        count = vault.index_vault()

        assert count >= 1

    def test_vault_with_deeply_nested_folders(self, temp_dir):
        vault = ObsidianVault(str(temp_dir), init_git=False)
        deep_path = temp_dir / "a" / "b" / "c" / "d" / "e"
        deep_path.mkdir(parents=True)
        (deep_path / "deep.md").write_text("# Deep", encoding='utf-8')

        count = vault.index_vault()

        assert count == 1

    def test_vault_with_very_large_note(self, temp_dir):
        vault = ObsidianVault(str(temp_dir), init_git=False)

        large_content = "# Large\n\n" + ("Lorem ipsum " * 100000)
        (temp_dir / "large.md").write_text(large_content, encoding='utf-8')

        count = vault.index_vault()

        assert count == 1

    def test_parse_note_with_mixed_line_endings(self, temp_dir):
        content = "# Title\r\n\r\nContent\nMore\r\n"
        note_path = temp_dir / "mixed.md"
        note_path.write_text(content, encoding='utf-8')

        result = VaultParser.parse_note(note_path)

        assert result.title == "mixed"
        assert "Content" in result.content

    def test_multiple_notes_same_title_different_paths(self, temp_dir):
        vault = ObsidianVault(str(temp_dir), init_git=False)

        folder1 = temp_dir / "folder1"
        folder2 = temp_dir / "folder2"
        folder1.mkdir()
        folder2.mkdir()

        (folder1 / "duplicate.md").write_text("# Duplicate", encoding='utf-8')
        (folder2 / "duplicate.md").write_text("# Duplicate", encoding='utf-8')

        count = vault.index_vault()

        assert count >= 1

    def test_create_note_with_empty_content(self, temp_dir):
        vault = ObsidianVault(str(temp_dir), init_git=False)
        vault.initialize_structure()

        note_path = vault.create_note("Empty", "")

        assert note_path.exists()

    def test_template_with_special_characters(self, temp_dir):
        manager = TemplateManager(temp_dir / "templates")

        content = "Special: {{var$1}} and {{var#2}}"
        manager.add_custom_template("special", content)

        result = manager.create_from_template("special", variables={"var$1": "test"})

        assert "test" in result

    def test_git_integration_with_binary_files(self, temp_dir):
        git = GitIntegration(temp_dir)
        git.init_repo(initial_commit=True)

        binary_file = temp_dir / "image.png"
        binary_file.write_bytes(b'\x89PNG\r\n\x1a\n')

        result = git.commit_changes("Add binary")

        assert result == True

    def test_vault_concurrent_operations(self, temp_dir):
        vault = ObsidianVault(str(temp_dir), init_git=False)
        vault.initialize_structure()

        for i in range(10):
            vault.create_note(f"Note {i}", f"Content {i}")

        count = vault.index_vault(force_refresh=True)

        assert count == 10

    def test_backlink_circular_references(self, temp_dir):
        vault = ObsidianVault(str(temp_dir), init_git=False)

        (temp_dir / "note1.md").write_text("Links to [[note2]]", encoding='utf-8')
        (temp_dir / "note2.md").write_text("Links to [[note1]]", encoding='utf-8')

        graph = vault.get_backlink_graph()

        assert "note2" in graph["note1"]
        assert "note1" in graph["note2"]

    def test_frontmatter_with_complex_yaml(self, temp_dir):
        content = """---
title: Complex
nested:
  key1: value1
  key2: value2
list:
  - item1
  - item2
---

# Content"""
        note_path = temp_dir / "complex.md"
        note_path.write_text(content, encoding='utf-8')

        result = VaultParser.parse_note(note_path)

        assert result.title == "Complex"
        assert result.frontmatter["nested"]["key1"] == "value1"
        assert "item1" in result.frontmatter["list"]

    def test_git_status_with_deleted_files(self, temp_dir):
        git = GitIntegration(temp_dir)
        file_path = temp_dir / "file.txt"
        file_path.write_text("content", encoding='utf-8')
        git.init_repo(initial_commit=True)

        file_path.unlink()

        status = git.get_status()

        assert status['has_changes'] == True

    def test_template_manager_with_empty_templates_dir(self, temp_dir):
        templates_path = temp_dir / "empty_templates"
        templates_path.mkdir()

        for template_file in templates_path.glob("*.md"):
            template_file.unlink()

        manager = TemplateManager(templates_path)
        templates = manager.list_templates()

        assert len(templates) >= 5

    def test_vault_index_with_markdown_extension(self, temp_dir):
        vault = ObsidianVault(str(temp_dir), init_git=False)
        (temp_dir / "note.markdown").write_text("# Note", encoding='utf-8')

        count = vault.index_vault()

        assert count == 1
