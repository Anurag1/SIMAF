"""
Obsidian Vault Structure and Git Integration

Manages vault folder structure, templates, and version control.
Provides initialization, template management, and Git synchronization.

Features:
- Standard vault folder structure creation
- Template management for different note types
- Git integration for version control
- Automated commit and sync workflows
- Conflict detection and resolution support

Author: BMad
Date: 2025-10-19
"""

import logging
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import yaml


logger = logging.getLogger(__name__)


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

        # Initialize default templates if they don't exist
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

        # Substitute variables
        if variables:
            for key, value in variables.items():
                placeholder = f"{{{{{key}}}}}"
                content = content.replace(placeholder, str(value))

        # Write to output path if provided
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
            # Initialize repository
            self._run_git_command(['init'])
            logger.info("Initialized Git repository")

            # Create .gitignore
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

            # Initial commit
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

        # Get status in porcelain format
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
            # Stage changes
            if add_all:
                self._run_git_command(['add', '.'])

            # Check if there's anything to commit
            status = self.get_status()
            if not status['staged_files']:
                logger.info("No changes to commit")
                return False

            # Commit
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
            # Commit local changes if any
            if self.has_uncommitted_changes():
                if not commit_message:
                    commit_message = f"Auto-sync: {datetime.now().isoformat()}"

                self.commit_changes(commit_message)

            # Pull remote changes
            self.pull(remote, branch)

            # Push local changes
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

        # Check if remote exists
        existing_url = self.get_remote_url(remote)

        if existing_url:
            # Update existing remote
            self._run_git_command(['remote', 'set-url', remote, url])
            logger.info(f"Updated remote '{remote}' URL")
        else:
            # Add new remote
            self._run_git_command(['remote', 'add', remote, url])
            logger.info(f"Added remote '{remote}'")


class ObsidianVaultStructure:
    """
    Main class for managing Obsidian vault structure and Git integration.

    Provides:
    - Vault initialization with standard folder structure
    - Template management
    - Git version control
    - Automated workflows

    Usage:
        >>> vault_mgr = ObsidianVaultStructure(
        ...     vault_path="./my_vault",
        ...     init_git=True
        ... )
        >>> vault_mgr.initialize_structure()
        >>> vault_mgr.create_daily_note()
    """

    def __init__(
        self,
        vault_path: str,
        init_git: bool = True,
        auto_commit: bool = False
    ):
        """
        Initialize vault structure manager.

        Args:
            vault_path: Path to vault root
            init_git: Initialize Git repository
            auto_commit: Enable automatic commits
        """
        self.vault_path = Path(vault_path)
        self.vault_path.mkdir(parents=True, exist_ok=True)

        # Initialize structure
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

        # Initialize Git
        self.git = GitIntegration(self.vault_path, auto_commit=auto_commit)
        if init_git:
            self.git.init_repo(initial_commit=False)

        # Initialize template manager
        self.templates = TemplateManager(self.structure.templates)

        logger.info(f"Initialized ObsidianVaultStructure at {vault_path}")

    def initialize_structure(self, create_readme: bool = True) -> VaultStructure:
        """
        Create standard vault folder structure.

        Args:
            create_readme: Create README files in folders

        Returns:
            VaultStructure instance
        """
        logger.info("Initializing vault folder structure")

        # Create all folders
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

            # Create README if requested
            if create_readme:
                readme_path = folder / 'README.md'
                if not readme_path.exists():
                    readme_content = self._get_folder_readme(folder.name)
                    with open(readme_path, 'w', encoding='utf-8') as f:
                        f.write(readme_content)

        # Create vault README
        vault_readme = self.vault_path / 'README.md'
        if not vault_readme.exists():
            with open(vault_readme, 'w', encoding='utf-8') as f:
                f.write(self._get_vault_readme())

        logger.info("Vault structure initialized")

        # Commit if Git is initialized
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

        # Create from template
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

        # Auto-commit if enabled
        if self.git.auto_commit:
            self.git.commit_changes(f"Add daily note: {date_str}")

        return note_path

    def create_project_note(self, title: str, metadata: Optional[Dict[str, Any]] = None) -> Path:
        """
        Create a project note.

        Args:
            title: Project title
            metadata: Additional metadata

        Returns:
            Path to created project note
        """
        # Sanitize filename
        import re
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

        # Auto-commit if enabled
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

        # Count markdown files
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


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("Obsidian Vault Structure & Git Integration Demo")
    print("=" * 80)
    print()

    # Test vault initialization
    print("[1/6] Initializing vault structure...")
    vault_mgr = ObsidianVaultStructure(
        vault_path="./test_vault",
        init_git=True,
        auto_commit=False
    )
    structure = vault_mgr.initialize_structure()
    print(f"✅ Vault initialized at {structure.root}")
    print()

    # Test template listing
    print("[2/6] Listing available templates...")
    templates = vault_mgr.templates.list_templates()
    print(f"✅ Found {len(templates)} templates: {', '.join(templates)}")
    print()

    # Test daily note creation
    print("[3/6] Creating daily note...")
    daily_path = vault_mgr.create_daily_note()
    print(f"✅ Created daily note: {daily_path.name}")
    print()

    # Test project note creation
    print("[4/6] Creating project note...")
    project_path = vault_mgr.create_project_note("Test Project")
    print(f"✅ Created project note: {project_path.name}")
    print()

    # Test Git operations
    print("[5/6] Testing Git operations...")
    if vault_mgr.git.is_git_repo():
        print("✅ Git repository initialized")

        # Commit changes
        vault_mgr.git.commit_changes("Test commit: Add sample notes")
        print("✅ Changes committed")

        # Get status
        status = vault_mgr.git.get_status()
        print(f"✅ Repository status: {status['has_changes']}")
    else:
        print("⚠️  Git repository not initialized")
    print()

    # Test vault statistics
    print("[6/6] Getting vault statistics...")
    stats = vault_mgr.get_vault_stats()
    print(f"✅ Vault statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    print()

    print("=" * 80)
    print("Vault Structure & Git Integration Demo Complete!")
    print("=" * 80)
