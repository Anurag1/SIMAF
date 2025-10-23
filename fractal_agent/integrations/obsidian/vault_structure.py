"""
Obsidian Vault Structure and Integration

Comprehensive vault management with initialization, templates, file operations,
and Git synchronization for Obsidian knowledge bases.

Features:
- Vault initialization with standard folder structure
- Template management for various note types
- File operations with frontmatter support
- Git version control integration
- Automated workflows and sync

Author: BMad
Date: 2025-10-19
"""

import re
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import yaml


logger = logging.getLogger(__name__)


@dataclass
class VaultNote:
    """
    Represents a note in the Obsidian vault.

    Attributes:
        file_path: Path to the markdown file
        title: Note title (from frontmatter or filename)
        content: Raw markdown content
        frontmatter: YAML frontmatter metadata
        tags: List of tags
        backlinks: List of notes this note links to
        created: Creation timestamp
        modified: Last modification timestamp
    """
    file_path: Path
    title: str
    content: str
    frontmatter: Dict[str, Any]
    tags: List[str]
    backlinks: List[str]
    created: datetime
    modified: datetime


@dataclass
class VaultStructure:
    """
    Defines standard Obsidian vault folder structure.

    Attributes:
        root: Vault root directory
        projects: Project notes folder
        daily: Daily notes folder
        templates: Template files folder
        resources: Attachments and resources folder
        archive: Archived notes folder
        agent_reviews: Agent session reviews folder
        agent_knowledge: Agent-generated knowledge folder
    """
    root: Path
    projects: Path
    daily: Path
    templates: Path
    resources: Path
    archive: Path
    agent_reviews: Path
    agent_knowledge: Path


class VaultParser:
    """
    Parses Obsidian markdown files with frontmatter, tags, and links.

    Handles:
    - YAML frontmatter extraction
    - Tag parsing (#tag and frontmatter tags)
    - Wikilink extraction ([[link]])
    - Metadata normalization
    """

    @staticmethod
    def parse_note(file_path: Path) -> VaultNote:
        """
        Parse an Obsidian markdown file into a VaultNote.

        Args:
            file_path: Path to markdown file

        Returns:
            VaultNote instance with parsed metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not markdown
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Note not found: {file_path}")

        if file_path.suffix.lower() not in ['.md', '.markdown']:
            raise ValueError(f"Not a markdown file: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        frontmatter, body = VaultParser._extract_frontmatter(content)
        title = frontmatter.get('title', file_path.stem)
        tags = VaultParser._extract_tags(frontmatter, body)
        backlinks = VaultParser._extract_backlinks(body)

        stat = file_path.stat()
        created = datetime.fromtimestamp(stat.st_ctime)
        modified = datetime.fromtimestamp(stat.st_mtime)

        return VaultNote(
            file_path=file_path,
            title=title,
            content=content,
            frontmatter=frontmatter,
            tags=tags,
            backlinks=backlinks,
            created=created,
            modified=modified
        )

    @staticmethod
    def _extract_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
        """
        Extract YAML frontmatter from markdown content.

        Returns:
            Tuple of (frontmatter dict, body content without frontmatter)
        """
        frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n'
        match = re.match(frontmatter_pattern, content, re.DOTALL)

        if match:
            try:
                frontmatter = yaml.safe_load(match.group(1)) or {}
                body = content[match.end():]
                return frontmatter, body
            except yaml.YAMLError as e:
                logger.warning(f"Failed to parse frontmatter: {e}")
                return {}, content

        return {}, content

    @staticmethod
    def _extract_tags(frontmatter: Dict[str, Any], body: str) -> List[str]:
        """
        Extract tags from frontmatter and inline #tags.

        Returns:
            Deduplicated list of tags
        """
        tags = set()

        fm_tags = frontmatter.get('tags', [])
        if isinstance(fm_tags, str):
            fm_tags = [fm_tags]
        tags.update(fm_tags)

        inline_tags = re.findall(r'#([\w-]+)', body)
        tags.update(inline_tags)

        return sorted(list(tags))

    @staticmethod
    def _extract_backlinks(content: str) -> List[str]:
        """
        Extract wikilinks [[link]] from content.

        Returns:
            List of linked note names
        """
        wikilinks = re.findall(r'\[\[([^\]|]+)(?:\|[^\]]*)?\]\]', content)
        return list(set(wikilinks))


class TemplateManager:
    """
    Manages Obsidian note templates.

    Provides templates for:
    - Daily notes
    - Project notes
    - Meeting notes
    - Research notes
    - Agent session reviews

    Usage:
        >>> manager = TemplateManager(templates_path)
        >>> daily_note = manager.create_from_template('daily', title='2025-10-19')
    """

    def __init__(self, templates_path: Path):
        """
        Initialize template manager.

        Args:
            templates_path: Path to templates folder
        """
        self.templates_path = templates_path
        self.templates_path.mkdir(parents=True, exist_ok=True)
        self._initialize_default_templates()
        logger.info(f"Initialized TemplateManager at {templates_path}")

    def _initialize_default_templates(self):
        """Create default templates if they don't exist."""
        templates = {
            'daily.md': self._get_daily_template(),
            'project.md': self._get_project_template(),
            'meeting.md': self._get_meeting_template(),
            'research.md': self._get_research_template(),
            'agent_session.md': self._get_agent_session_template()
        }

        for filename, content in templates.items():
            template_path = self.templates_path / filename
            if not template_path.exists():
                with open(template_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.debug(f"Created template: {filename}")

    @staticmethod
    def _get_daily_template() -> str:
        """Get daily note template."""
        return """---
date: {{date}}
type: daily
tags: [daily]
---

# {{date}}

## Today's Focus

-

## Notes

## Tasks

- [ ]

## Reflections

"""

    @staticmethod
    def _get_project_template() -> str:
        """Get project note template."""
        return """---
title: {{title}}
type: project
status: active
created: {{created}}
tags: [project]
---

# {{title}}

## Overview

## Goals

-

## Milestones

- [ ]

## Resources

-

## Notes

"""

    @staticmethod
    def _get_meeting_template() -> str:
        """Get meeting note template."""
        return """---
title: {{title}}
type: meeting
date: {{date}}
attendees: []
tags: [meeting]
---

# {{title}}

**Date**: {{date}}
**Attendees**:

## Agenda

1.

## Discussion

## Action Items

- [ ]

## Next Steps

"""

    @staticmethod
    def _get_research_template() -> str:
        """Get research note template."""
        return """---
title: {{title}}
type: research
created: {{created}}
tags: [research]
---

# {{title}}

## Summary

## Key Findings

-

## Questions

-

## References

-

## Notes

"""

    @staticmethod
    def _get_agent_session_template() -> str:
        """Get agent session review template."""
        return """---
session_id: {{session_id}}
type: agent_session
created: {{created}}
status: pending_review
tags: [agent_session, needs_review]
---

# Agent Session: {{session_id}}

## Summary

{{summary}}

## Tasks Completed

{{tasks}}

## Review

### Approval Status

- [ ] Approved
- [ ] Rejected
- [ ] Needs Revision

### Reviewer Notes

### Action Items

- [ ]

"""

    def create_from_template(
        self,
        template_name: str,
        variables: Optional[Dict[str, str]] = None,
        output_path: Optional[Path] = None
    ) -> str:
        """
        Create note content from template.

        Args:
            template_name: Template name (without .md extension)
            variables: Variables to substitute in template
            output_path: Optional path to write the note to

        Returns:
            Rendered template content

        Raises:
            FileNotFoundError: If template doesn't exist
        """
        template_path = self.templates_path / f"{template_name}.md"

        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_name}")

        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if variables:
            for key, value in variables.items():
                placeholder = f"{{{{{key}}}}}"
                content = content.replace(placeholder, str(value))

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Created note from template: {output_path}")

        return content

    def list_templates(self) -> List[str]:
        """
        List available templates.

        Returns:
            List of template names (without .md extension)
        """
        templates = [
            f.stem for f in self.templates_path.glob('*.md')
            if f.is_file()
        ]
        return sorted(templates)

    def add_custom_template(self, name: str, content: str):
        """
        Add a custom template.

        Args:
            name: Template name (without .md extension)
            content: Template content with {{variable}} placeholders
        """
        template_path = self.templates_path / f"{name}.md"

        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"Added custom template: {name}")


class GitIntegration:
    """
    Git integration for Obsidian vault version control.

    Provides:
    - Repository initialization
    - Automated commits
    - Push/pull operations
    - Branch management
    - Conflict detection

    Usage:
        >>> git = GitIntegration(vault_path)
        >>> git.init_repo()
        >>> git.commit_changes("Updated daily note")
        >>> git.sync()
    """

    def __init__(self, vault_path: Path, auto_commit: bool = False):
        """
        Initialize Git integration.

        Args:
            vault_path: Path to vault root (Git repository)
            auto_commit: Automatically commit changes on operations
        """
        self.vault_path = vault_path
        self.auto_commit = auto_commit
        logger.info(f"Initialized GitIntegration for {vault_path}")

    def _run_git_command(
        self,
        args: List[str],
        check: bool = True,
        capture_output: bool = True
    ) -> subprocess.CompletedProcess:
        """
        Run a git command in the vault directory.

        Args:
            args: Git command arguments (excluding 'git')
            check: Raise exception on non-zero exit
            capture_output: Capture stdout/stderr

        Returns:
            CompletedProcess instance

        Raises:
            subprocess.CalledProcessError: If command fails and check=True
        """
        cmd = ['git', '-C', str(self.vault_path)] + args

        try:
            result = subprocess.run(
                cmd,
                check=check,
                capture_output=capture_output,
                text=True
            )
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {' '.join(cmd)}")
            logger.error(f"Error: {e.stderr}")
            raise

    def is_git_repo(self) -> bool:
        """
        Check if vault is a Git repository.

        Returns:
            True if Git repository exists
        """
        try:
            self._run_git_command(['rev-parse', '--git-dir'], check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def init_repo(self, initial_commit: bool = True) -> bool:
        """
        Initialize Git repository in vault.

        Args:
            initial_commit: Create initial commit

        Returns:
            True if initialization successful
        """
        if self.is_git_repo():
            logger.info("Git repository already exists")
            return True

        try:
            self._run_git_command(['init'])
            logger.info("Initialized Git repository")

            gitignore_path = self.vault_path / '.gitignore'
            if not gitignore_path.exists():
                gitignore_content = """.DS_Store
.obsidian/workspace*
.obsidian/cache
.trash/
*.tmp
"""
                with open(gitignore_path, 'w', encoding='utf-8') as f:
                    f.write(gitignore_content)
                logger.debug("Created .gitignore")

            if initial_commit:
                self._run_git_command(['add', '.'])
                self._run_git_command([
                    'commit',
                    '-m',
                    'Initial commit: Obsidian vault setup'
                ])
                logger.info("Created initial commit")

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to initialize Git repository: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """
        Get Git repository status.

        Returns:
            Status information dict with keys:
            - has_changes: bool
            - untracked_files: List[str]
            - modified_files: List[str]
            - staged_files: List[str]
        """
        if not self.is_git_repo():
            raise RuntimeError("Not a Git repository")

        result = self._run_git_command(['status', '--porcelain'])
        lines = result.stdout.strip().split('\n') if result.stdout.strip() else []

        status = {
            'has_changes': len(lines) > 0,
            'untracked_files': [],
            'modified_files': [],
            'staged_files': []
        }

        for line in lines:
            if not line:
                continue

            status_code = line[:2]
            filepath = line[3:]

            if status_code[0] in ['A', 'M', 'D']:
                status['staged_files'].append(filepath)
            if status_code[1] == 'M':
                status['modified_files'].append(filepath)
            if status_code == '??':
                status['untracked_files'].append(filepath)

        return status

    def has_uncommitted_changes(self) -> bool:
        """
        Check if there are uncommitted changes.

        Returns:
            True if there are uncommitted changes
        """
        status = self.get_status()
        return status['has_changes']

    def commit_changes(
        self,
        message: str,
        add_all: bool = True,
        author: Optional[str] = None
    ) -> bool:
        """
        Commit changes to repository.

        Args:
            message: Commit message
            add_all: Stage all changes before commit
            author: Optional author string (e.g., "Name <email>")

        Returns:
            True if commit successful
        """
        if not self.is_git_repo():
            raise RuntimeError("Not a Git repository")

        try:
            if add_all:
                self._run_git_command(['add', '.'])

            status = self.get_status()
            if not status['staged_files']:
                logger.info("No changes to commit")
                return False

            commit_args = ['commit', '-m', message]
            if author:
                commit_args.extend(['--author', author])

            self._run_git_command(commit_args)
            logger.info(f"Committed changes: {message}")

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to commit changes: {e}")
            return False

    def push(self, remote: str = 'origin', branch: str = 'main') -> bool:
        """
        Push commits to remote repository.

        Args:
            remote: Remote name
            branch: Branch name

        Returns:
            True if push successful
        """
        if not self.is_git_repo():
            raise RuntimeError("Not a Git repository")

        try:
            self._run_git_command(['push', remote, branch])
            logger.info(f"Pushed to {remote}/{branch}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to push: {e}")
            return False

    def pull(self, remote: str = 'origin', branch: str = 'main') -> bool:
        """
        Pull commits from remote repository.

        Args:
            remote: Remote name
            branch: Branch name

        Returns:
            True if pull successful
        """
        if not self.is_git_repo():
            raise RuntimeError("Not a Git repository")

        try:
            self._run_git_command(['pull', remote, branch])
            logger.info(f"Pulled from {remote}/{branch}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to pull: {e}")
            return False

    def sync(
        self,
        remote: str = 'origin',
        branch: str = 'main',
        commit_message: Optional[str] = None
    ) -> bool:
        """
        Sync vault with remote (commit, pull, push).

        Args:
            remote: Remote name
            branch: Branch name
            commit_message: Optional commit message (default: auto-generated)

        Returns:
            True if sync successful
        """
        if not self.is_git_repo():
            raise RuntimeError("Not a Git repository")

        try:
            if self.has_uncommitted_changes():
                if not commit_message:
                    commit_message = f"Auto-sync: {datetime.now().isoformat()}"

                self.commit_changes(commit_message)

            self.pull(remote, branch)
            self.push(remote, branch)

            logger.info("Vault synced successfully")
            return True

        except Exception as e:
            logger.error(f"Sync failed: {e}")
            return False

    def get_remote_url(self, remote: str = 'origin') -> Optional[str]:
        """
        Get remote repository URL.

        Args:
            remote: Remote name

        Returns:
            Remote URL or None if not configured
        """
        if not self.is_git_repo():
            return None

        try:
            result = self._run_git_command(['remote', 'get-url', remote])
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None

    def set_remote_url(self, url: str, remote: str = 'origin'):
        """
        Set remote repository URL.

        Args:
            url: Remote repository URL
            remote: Remote name
        """
        if not self.is_git_repo():
            raise RuntimeError("Not a Git repository")

        existing_url = self.get_remote_url(remote)

        if existing_url:
            self._run_git_command(['remote', 'set-url', remote, url])
            logger.info(f"Updated remote '{remote}' URL")
        else:
            self._run_git_command(['remote', 'add', remote, url])
            logger.info(f"Added remote '{remote}'")


class ObsidianVault:
    """
    Main Obsidian vault manager with comprehensive functionality.

    Provides:
    - Vault initialization with standard folder structure
    - Template management
    - File operations (create, read, update, delete)
    - Git version control integration
    - Note indexing and search
    - Automated workflows

    Usage:
        >>> vault = ObsidianVault(
        ...     vault_path="./obsidian_vault",
        ...     init_git=True
        ... )
        >>> vault.initialize_structure()
        >>> note = vault.create_note("My Note", content="# Hello World")
        >>> vault.sync_vault()
    """

    def __init__(
        self,
        vault_path: str,
        init_git: bool = True,
        auto_commit: bool = False
    ):
        """
        Initialize Obsidian vault.

        Args:
            vault_path: Path to vault root
            init_git: Initialize Git repository
            auto_commit: Enable automatic commits
        """
        self.vault_path = Path(vault_path)
        self.vault_path.mkdir(parents=True, exist_ok=True)

        self.structure = VaultStructure(
            root=self.vault_path,
            projects=self.vault_path / 'projects',
            daily=self.vault_path / 'daily',
            templates=self.vault_path / 'templates',
            resources=self.vault_path / 'resources',
            archive=self.vault_path / 'archive',
            agent_reviews=self.vault_path / 'agent_reviews',
            agent_knowledge=self.vault_path / 'agent_knowledge'
        )

        self.git = GitIntegration(self.vault_path, auto_commit=auto_commit)
        if init_git:
            self.git.init_repo(initial_commit=False)

        self.templates = TemplateManager(self.structure.templates)

        self._note_index: Dict[str, VaultNote] = {}
        self._last_index_time: Optional[datetime] = None

        logger.info(f"Initialized ObsidianVault at {vault_path}")

    def initialize_structure(self, create_readme: bool = True) -> VaultStructure:
        """
        Create standard vault folder structure.

        Args:
            create_readme: Create README files in folders

        Returns:
            VaultStructure instance
        """
        logger.info("Initializing vault folder structure")

        folders = [
            self.structure.projects,
            self.structure.daily,
            self.structure.templates,
            self.structure.resources,
            self.structure.archive,
            self.structure.agent_reviews,
            self.structure.agent_knowledge
        ]

        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created folder: {folder.name}")

            if create_readme:
                readme_path = folder / 'README.md'
                if not readme_path.exists():
                    readme_content = self._get_folder_readme(folder.name)
                    with open(readme_path, 'w', encoding='utf-8') as f:
                        f.write(readme_content)

        vault_readme = self.vault_path / 'README.md'
        if not vault_readme.exists():
            with open(vault_readme, 'w', encoding='utf-8') as f:
                f.write(self._get_vault_readme())

        logger.info("Vault structure initialized")

        if self.git.is_git_repo():
            self.git.commit_changes("Initialize vault structure")

        return self.structure

    @staticmethod
    def _get_folder_readme(folder_name: str) -> str:
        """Get README content for a folder."""
        descriptions = {
            'projects': 'Project notes and documentation',
            'daily': 'Daily notes and journal entries',
            'templates': 'Note templates',
            'resources': 'Attachments, images, and other resources',
            'archive': 'Archived and historical notes',
            'agent_reviews': 'Agent session reviews for human approval',
            'agent_knowledge': 'Agent-generated knowledge and insights'
        }

        description = descriptions.get(folder_name, 'Notes and documentation')

        return f"""# {folder_name.capitalize()}

{description}

---

*This folder is part of the Obsidian vault structure.*
"""

    @staticmethod
    def _get_vault_readme() -> str:
        """Get vault README content."""
        return """# Obsidian Vault

This is an Obsidian vault with integrated agent memory and knowledge management.

## Structure

- **projects/**: Project notes and documentation
- **daily/**: Daily notes and journal entries
- **templates/**: Note templates
- **resources/**: Attachments and resources
- **archive/**: Archived notes
- **agent_reviews/**: Agent session reviews
- **agent_knowledge/**: Agent-generated knowledge

## Usage

This vault integrates with the Fractal Agent ecosystem for bidirectional knowledge sharing between human curators and AI agents.

---

*Generated by BMAD-METHOD Fractal Agent System*
"""

    def create_note(
        self,
        title: str,
        content: str,
        folder: Optional[Path] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        template: Optional[str] = None
    ) -> Path:
        """
        Create a new note in the vault.

        Args:
            title: Note title
            content: Markdown content
            folder: Target folder (default: agent_knowledge)
            tags: Tags to add
            metadata: Additional frontmatter metadata
            template: Optional template to use

        Returns:
            Path to created note
        """
        if folder is None:
            folder = self.structure.agent_knowledge

        filename = re.sub(r'[<>:"/\\|?*]', '_', title)
        file_path = folder / f"{filename}.md"

        if template:
            variables = {
                'title': title,
                'created': datetime.now().isoformat(),
                **(metadata or {})
            }
            md_content = self.templates.create_from_template(
                template,
                variables=variables
            )
        else:
            frontmatter = {
                'title': title,
                'created': datetime.now().isoformat(),
                'tags': tags or [],
                **(metadata or {})
            }

            md_lines = ['---']
            md_lines.append(yaml.dump(frontmatter, default_flow_style=False))
            md_lines.append('---\n')
            md_lines.append(content)
            md_content = '\n'.join(md_lines)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logger.info(f"Created note: {file_path}")

        if self.git.auto_commit:
            self.git.commit_changes(f"Add note: {title}")

        if self._note_index is not None:
            note = VaultParser.parse_note(file_path)
            self._note_index[title] = note

        return file_path

    def read_note(self, title: str) -> Optional[VaultNote]:
        """
        Read a note from the vault by title.

        Args:
            title: Note title

        Returns:
            VaultNote instance or None if not found
        """
        if not self._note_index:
            self.index_vault()

        return self._note_index.get(title)

    def update_note(
        self,
        title: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing note.

        Args:
            title: Note title
            content: New content (None to keep existing)
            metadata: Metadata to update in frontmatter

        Returns:
            True if update successful
        """
        note = self.read_note(title)
        if not note:
            logger.warning(f"Note not found: {title}")
            return False

        with open(note.file_path, 'r', encoding='utf-8') as f:
            existing_content = f.read()

        frontmatter, body = VaultParser._extract_frontmatter(existing_content)

        if metadata:
            frontmatter.update(metadata)

        if content is not None:
            body = content

        md_lines = ['---']
        md_lines.append(yaml.dump(frontmatter, default_flow_style=False))
        md_lines.append('---\n')
        md_lines.append(body)

        with open(note.file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines))

        logger.info(f"Updated note: {title}")

        if self.git.auto_commit:
            self.git.commit_changes(f"Update note: {title}")

        if title in self._note_index:
            self._note_index[title] = VaultParser.parse_note(note.file_path)

        return True

    def delete_note(self, title: str, archive: bool = True) -> bool:
        """
        Delete a note from the vault.

        Args:
            title: Note title
            archive: Move to archive instead of deleting

        Returns:
            True if deletion successful
        """
        note = self.read_note(title)
        if not note:
            logger.warning(f"Note not found: {title}")
            return False

        if archive:
            archive_path = self.structure.archive / note.file_path.name
            note.file_path.rename(archive_path)
            logger.info(f"Archived note: {title}")
        else:
            note.file_path.unlink()
            logger.info(f"Deleted note: {title}")

        if title in self._note_index:
            del self._note_index[title]

        if self.git.auto_commit:
            action = "Archive" if archive else "Delete"
            self.git.commit_changes(f"{action} note: {title}")

        return True

    def index_vault(
        self,
        force_refresh: bool = False,
        exclude_folders: Optional[List[str]] = None
    ) -> int:
        """
        Index all notes in the vault.

        Args:
            force_refresh: Re-index even if recently indexed
            exclude_folders: Folder names to skip

        Returns:
            Number of notes indexed
        """
        if not force_refresh and self._last_index_time:
            age_minutes = (datetime.now() - self._last_index_time).total_seconds() / 60
            if age_minutes < 5:
                logger.info(f"Using cached index (age: {age_minutes:.1f} min)")
                return len(self._note_index)

        logger.info(f"Indexing vault: {self.vault_path}")

        exclude_folders = exclude_folders or ['.obsidian', '.trash', 'templates']
        exclude_folders.extend([
            self.structure.agent_reviews.name,
            self.structure.resources.name
        ])

        self._note_index = {}

        for md_file in self.vault_path.rglob('*.md'):
            if any(excluded in md_file.parts for excluded in exclude_folders):
                continue

            try:
                note = VaultParser.parse_note(md_file)
                self._note_index[note.title] = note
                logger.debug(f"Indexed: {note.title}")
            except Exception as e:
                logger.warning(f"Failed to parse {md_file}: {e}")

        self._last_index_time = datetime.now()
        logger.info(f"Indexed {len(self._note_index)} notes")

        return len(self._note_index)

    def find_notes_by_tag(self, tag: str) -> List[VaultNote]:
        """
        Find all notes with a specific tag.

        Args:
            tag: Tag to search for

        Returns:
            List of VaultNote instances
        """
        if not self._note_index:
            self.index_vault()

        return [
            note for note in self._note_index.values()
            if tag in note.tags
        ]

    def get_backlink_graph(self) -> Dict[str, List[str]]:
        """
        Build a graph of note backlinks.

        Returns:
            Dict mapping note titles to list of linked note titles
        """
        if not self._note_index:
            self.index_vault()

        graph = {}
        for title, note in self._note_index.items():
            graph[title] = note.backlinks

        return graph

    def create_daily_note(self, date: Optional[datetime] = None) -> Path:
        """
        Create a daily note for the specified date.

        Args:
            date: Date for daily note (default: today)

        Returns:
            Path to created daily note
        """
        if date is None:
            date = datetime.now()

        date_str = date.strftime('%Y-%m-%d')
        note_path = self.structure.daily / f"{date_str}.md"

        if note_path.exists():
            logger.info(f"Daily note already exists: {date_str}")
            return note_path

        variables = {
            'date': date_str,
            'created': datetime.now().isoformat()
        }

        self.templates.create_from_template(
            'daily',
            variables=variables,
            output_path=note_path
        )

        logger.info(f"Created daily note: {date_str}")

        if self.git.auto_commit:
            self.git.commit_changes(f"Add daily note: {date_str}")

        return note_path

    def create_project_note(
        self,
        title: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Create a project note.

        Args:
            title: Project title
            metadata: Additional metadata

        Returns:
            Path to created project note
        """
        filename = re.sub(r'[<>:"/\\|?*]', '_', title)
        note_path = self.structure.projects / f"{filename}.md"

        variables = {
            'title': title,
            'created': datetime.now().isoformat()
        }

        self.templates.create_from_template(
            'project',
            variables=variables,
            output_path=note_path
        )

        logger.info(f"Created project note: {title}")

        if self.git.auto_commit:
            self.git.commit_changes(f"Add project note: {title}")

        return note_path

    def sync_vault(
        self,
        remote: str = 'origin',
        branch: str = 'main',
        commit_message: Optional[str] = None
    ) -> bool:
        """
        Sync vault with Git remote.

        Args:
            remote: Remote name
            branch: Branch name
            commit_message: Optional commit message

        Returns:
            True if sync successful
        """
        return self.git.sync(remote, branch, commit_message)

    def get_vault_stats(self) -> Dict[str, Any]:
        """
        Get vault statistics.

        Returns:
            Statistics dict with folder counts
        """
        stats = {
            'total_notes': 0,
            'projects': 0,
            'daily_notes': 0,
            'agent_reviews': 0,
            'agent_knowledge': 0,
            'templates': 0
        }

        stats['projects'] = len(list(self.structure.projects.glob('*.md')))
        stats['daily_notes'] = len(list(self.structure.daily.glob('*.md')))
        stats['agent_reviews'] = len(list(self.structure.agent_reviews.glob('*.md')))
        stats['agent_knowledge'] = len(list(self.structure.agent_knowledge.glob('*.md')))
        stats['templates'] = len(list(self.structure.templates.glob('*.md')))

        stats['total_notes'] = sum([
            stats['projects'],
            stats['daily_notes'],
            stats['agent_reviews'],
            stats['agent_knowledge']
        ])

        return stats
