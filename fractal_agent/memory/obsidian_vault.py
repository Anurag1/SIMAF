"""
ObsidianVault Integration Layer - Phase 4

Bidirectional integration between Obsidian vault and agent memory systems.
Bridges GraphRAG (long-term memory) with human-curated knowledge in Obsidian.

Features:
- Automatic knowledge extraction from vault notes
- Bidirectional sync between vault and GraphRAG
- Session export with graph-compatible metadata
- Query interface for vault knowledge retrieval
- Support for Obsidian features (tags, backlinks, graph view)

Author: BMad
Date: 2025-10-19
"""

import re
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
import yaml

from .long_term import GraphRAG, DocumentStore
from .short_term import ShortTermMemory
from .obsidian_export import ObsidianExporter
from .embeddings import generate_embedding
from ..utils.file_operations import atomic_read, atomic_write

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

        # Extract frontmatter
        frontmatter, body = VaultParser._extract_frontmatter(content)

        # Extract title (from frontmatter or filename)
        title = frontmatter.get('title', file_path.stem)

        # Extract tags
        tags = VaultParser._extract_tags(frontmatter, body)

        # Extract backlinks (wikilinks)
        backlinks = VaultParser._extract_backlinks(body)

        # Get file timestamps
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

        # Tags from frontmatter
        fm_tags = frontmatter.get('tags', [])
        if isinstance(fm_tags, str):
            fm_tags = [fm_tags]
        tags.update(fm_tags)

        # Inline #tags
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
        # Match [[link]] and [[link|alias]]
        wikilinks = re.findall(r'\[\[([^\]|]+)(?:\|[^\]]*)?\]\]', content)
        return list(set(wikilinks))


class ObsidianVault:
    """
    Main Obsidian vault manager with GraphRAG integration.

    Provides:
    - Vault scanning and indexing
    - Knowledge extraction to GraphRAG
    - Session export with graph metadata
    - Query interface for vault knowledge
    - Bidirectional sync

    Usage:
        >>> vault = ObsidianVault(
        ...     vault_path="./obsidian_vault",
        ...     graphrag=graphrag
        ... )
        >>>
        >>> # Index vault contents
        >>> vault.index_vault()
        >>>
        >>> # Query vault knowledge
        >>> results = vault.query("What is the Viable System Model?")
        >>>
        >>> # Export session
        >>> vault.export_session(memory)
    """

    def __init__(
        self,
        vault_path: str,
        graphrag: Optional[GraphRAG] = None,
        review_folder: str = "agent_reviews",
        knowledge_folder: str = "agent_knowledge"
    ):
        """
        Initialize Obsidian vault integration.

        Args:
            vault_path: Path to Obsidian vault root
            graphrag: GraphRAG instance for knowledge storage (optional)
            review_folder: Folder for session reviews
            knowledge_folder: Folder for agent-generated knowledge
        """
        self.vault_path = Path(vault_path)
        self.review_folder = self.vault_path / review_folder
        self.knowledge_folder = self.vault_path / knowledge_folder

        # Create folders if they don't exist
        self.review_folder.mkdir(parents=True, exist_ok=True)
        self.knowledge_folder.mkdir(parents=True, exist_ok=True)

        # Initialize GraphRAG if provided
        self.graphrag = graphrag
        if self.graphrag:
            self.doc_store = DocumentStore(graphrag)
        else:
            self.doc_store = None
            logger.warning("No GraphRAG instance provided - vault sync disabled")

        # Initialize exporter
        self.exporter = ObsidianExporter(
            vault_path=str(vault_path),
            review_folder=review_folder
        )

        # Vault index cache
        self._note_index: Dict[str, VaultNote] = {}
        self._last_index_time: Optional[datetime] = None

        logger.info(f"Initialized ObsidianVault at {self.vault_path}")

    def index_vault(
        self,
        force_refresh: bool = False,
        exclude_folders: Optional[List[str]] = None
    ) -> int:
        """
        Index all notes in the vault.

        Args:
            force_refresh: Re-index even if recently indexed
            exclude_folders: Folder names to skip (e.g., [".obsidian", "templates"])

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
        exclude_folders.append(self.review_folder.name)
        exclude_folders.append(self.knowledge_folder.name)

        self._note_index = {}

        # Find all markdown files
        for md_file in self.vault_path.rglob('*.md'):
            # Skip excluded folders
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

    def sync_to_graphrag(
        self,
        tag_filter: Optional[List[str]] = None,
        force_resync: bool = False
    ) -> Dict[str, Any]:
        """
        Sync vault notes to GraphRAG for semantic search.

        Args:
            tag_filter: Only sync notes with these tags (None = sync all)
            force_resync: Re-sync even if note already synced

        Returns:
            Sync statistics dict
        """
        if not self.doc_store:
            raise RuntimeError("GraphRAG not initialized - cannot sync")

        if not self._note_index:
            self.index_vault()

        logger.info(f"Syncing {len(self._note_index)} notes to GraphRAG")

        stats = {
            'total_notes': len(self._note_index),
            'synced': 0,
            'skipped': 0,
            'failed': 0
        }

        for title, note in self._note_index.items():
            # Apply tag filter
            if tag_filter and not any(tag in note.tags for tag in tag_filter):
                stats['skipped'] += 1
                continue

            try:
                # Store document in GraphRAG
                metadata = {
                    'title': note.title,
                    'tags': note.tags,
                    'created': note.created.isoformat(),
                    'modified': note.modified.isoformat(),
                    'source': 'obsidian_vault',
                    **note.frontmatter
                }

                chunk_ids = self.doc_store.store_document(
                    file_path=str(note.file_path),
                    content=note.content,
                    metadata=metadata
                )

                stats['synced'] += 1
                logger.debug(f"Synced {title}: {len(chunk_ids)} chunks")

            except Exception as e:
                logger.error(f"Failed to sync {title}: {e}")
                stats['failed'] += 1

        logger.info(
            f"Sync complete: {stats['synced']} synced, "
            f"{stats['skipped']} skipped, {stats['failed']} failed"
        )

        return stats

    def sync_from_graphrag(
        self,
        entity_filter: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Export knowledge from GraphRAG to vault notes.

        Creates notes for knowledge triples that don't exist in vault.

        Args:
            entity_filter: Only export knowledge about these entities

        Returns:
            Export statistics dict
        """
        if not self.graphrag:
            raise RuntimeError("GraphRAG not initialized - cannot sync")

        logger.info("Syncing knowledge from GraphRAG to vault")

        # This is a placeholder - full implementation would:
        # 1. Query GraphRAG for all entities/relationships
        # 2. Check if corresponding notes exist in vault
        # 3. Create new notes for missing entities
        # 4. Update existing notes with new relationships

        stats = {
            'entities_exported': 0,
            'notes_created': 0,
            'notes_updated': 0
        }

        logger.warning("GraphRAG -> Vault sync not fully implemented yet")

        return stats

    def query(
        self,
        query_text: str,
        max_results: int = 5,
        tag_filter: Optional[List[str]] = None
    ) -> str:
        """
        Query vault knowledge using semantic search.

        Args:
            query_text: Natural language query
            max_results: Maximum number of results
            tag_filter: Only search notes with these tags

        Returns:
            Formatted context string from vault
        """
        if not self.doc_store:
            raise RuntimeError("GraphRAG not initialized - cannot query")

        logger.info(f"Querying vault: {query_text}")

        # Build metadata filter
        filter_metadata = None
        if tag_filter:
            filter_metadata = {"tags": tag_filter}

        # Query document store
        context = self.doc_store.retrieve_document_context(
            query=query_text,
            max_chunks=max_results,
            filter_metadata=filter_metadata
        )

        return context

    def search_notes(
        self,
        query: str,
        max_results: int = 5,
        tag_filter: Optional[List[str]] = None,
        search_content: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search notes using text matching (simpler than semantic query).

        Searches note titles and optionally content for keyword matches.
        Useful when GraphRAG isn't available or for simple text searches.

        Args:
            query: Search query (keywords)
            max_results: Maximum number of results
            tag_filter: Only search notes with these tags
            search_content: Whether to search note content (slower but more thorough)

        Returns:
            List of dicts with note information and relevance scores

        Example:
            >>> results = vault.search_notes("VSM coordination")
            >>> for note in results:
            ...     print(f"{note['name']}: {note['score']:.2f}")
        """
        # Ensure vault is indexed
        if not self._note_index:
            self.index_vault()

        logger.info(f"Searching notes for: '{query[:50]}...'")

        query_lower = query.lower()
        query_terms = query_lower.split()

        scored_notes = []

        for title, note in self._note_index.items():
            # Apply tag filter
            if tag_filter and not any(tag in note.tags for tag in tag_filter):
                continue

            # Search title and content
            search_text = title.lower()
            if search_content:
                search_text += " " + note.content.lower()

            # Calculate relevance score
            matches = sum(1 for term in query_terms if term in search_text)
            if matches == 0:
                continue

            # Boost score if title matches
            title_matches = sum(1 for term in query_terms if term in title.lower())
            score = (matches / len(query_terms)) + (title_matches * 0.5)

            # Create result dict
            scored_notes.append({
                "name": note.title,
                "path": str(note.file_path),
                "content": note.content[:500],  # First 500 chars
                "tags": note.tags,
                "score": min(score, 1.0),  # Cap at 1.0
                "modified": note.modified.isoformat()
            })

        # Sort by score (descending)
        scored_notes.sort(key=lambda x: x['score'], reverse=True)

        # Return top results
        results = scored_notes[:max_results]

        logger.info(f"Found {len(scored_notes)} matching notes, returning top {len(results)}")

        return results

    def export_session(
        self,
        memory: ShortTermMemory,
        include_approval: bool = True,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Export agent session to vault for human review.

        Args:
            memory: ShortTermMemory instance with session data
            include_approval: Add human approval template
            additional_metadata: Extra metadata to include

        Returns:
            Path to created review file
        """
        logger.info(f"Exporting session {memory.session_id} to vault")

        # Use existing exporter
        file_path = self.exporter.export_session(
            memory=memory,
            include_approval_template=include_approval
        )

        # Add additional metadata if provided
        if additional_metadata:
            self._add_metadata_to_file(file_path, additional_metadata)

        return file_path

    def create_knowledge_note(
        self,
        title: str,
        content: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        sync_to_graphrag: bool = True
    ) -> Path:
        """
        Create a new knowledge note in the vault.

        Args:
            title: Note title
            content: Markdown content
            tags: Tags to add
            metadata: Additional frontmatter metadata
            sync_to_graphrag: Also store in GraphRAG

        Returns:
            Path to created note
        """
        # Sanitize filename
        filename = re.sub(r'[<>:"/\\|?*]', '_', title)
        file_path = self.knowledge_folder / f"{filename}.md"

        # Build frontmatter
        frontmatter = {
            'title': title,
            'created': datetime.now().isoformat(),
            'source': 'agent_generated',
            'tags': tags or [],
            **(metadata or {})
        }

        # Build markdown content
        md_lines = ['---']
        md_lines.append(yaml.dump(frontmatter, default_flow_style=False))
        md_lines.append('---\n')
        md_lines.append(content)

        markdown = '\n'.join(md_lines)

        # Write file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(markdown)

        logger.info(f"Created knowledge note: {file_path}")

        # Sync to GraphRAG if requested
        if sync_to_graphrag and self.doc_store:
            try:
                self.doc_store.store_document(
                    file_path=str(file_path),
                    content=content,
                    metadata=frontmatter
                )
                logger.debug("Synced note to GraphRAG")
            except Exception as e:
                logger.error(f"Failed to sync to GraphRAG: {e}")

        # Update index
        if self._note_index is not None:
            note = VaultParser.parse_note(file_path)
            self._note_index[title] = note

        return file_path

    def get_note(self, title: str) -> Optional[VaultNote]:
        """
        Get a note from the vault by title.

        Args:
            title: Note title

        Returns:
            VaultNote instance or None if not found
        """
        if not self._note_index:
            self.index_vault()

        return self._note_index.get(title)

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

    def _add_metadata_to_file(
        self,
        file_path: Path,
        metadata: Dict[str, Any]
    ):
        """
        Add metadata to an existing markdown file's frontmatter.

        Args:
            file_path: Path to markdown file
            metadata: Metadata to add/update
        """
        # Use atomic read to safely read file
        content = atomic_read(file_path, format='text')
        if content is None:
            logger.error(f"Failed to read file for metadata update: {file_path}")
            return

        frontmatter, body = VaultParser._extract_frontmatter(content)
        frontmatter.update(metadata)

        # Rebuild file
        md_lines = ['---']
        md_lines.append(yaml.dump(frontmatter, default_flow_style=False))
        md_lines.append('---\n')
        md_lines.append(body)

        # Use atomic write to prevent race conditions
        success = atomic_write(file_path, '\n'.join(md_lines), format='text')
        if not success:
            logger.error(f"Failed to write metadata to file: {file_path}")


class WorkflowVaultIntegration:
    """
    Integration layer between workflows and Obsidian vault.

    Provides workflow hooks for automatic session export and
    knowledge capture.

    Usage:
        >>> integration = WorkflowVaultIntegration(vault)
        >>>
        >>> # In workflow callback
        >>> integration.on_workflow_complete(state, memory)
    """

    def __init__(self, vault: ObsidianVault):
        """
        Initialize workflow integration.

        Args:
            vault: ObsidianVault instance
        """
        self.vault = vault
        logger.info("Initialized WorkflowVaultIntegration")

    def on_workflow_start(
        self,
        workflow_type: str,
        task: str,
        memory: ShortTermMemory
    ):
        """
        Hook called when workflow starts.

        Args:
            workflow_type: Type of workflow (e.g., "multi_agent")
            task: Main task description
            memory: ShortTermMemory instance
        """
        logger.info(f"Workflow started: {workflow_type} - {task}")

        # Could create a placeholder note or log entry
        # For now, just log

    def on_workflow_complete(
        self,
        workflow_type: str,
        state: Dict[str, Any],
        memory: ShortTermMemory,
        auto_export: bool = True
    ) -> Optional[Path]:
        """
        Hook called when workflow completes.

        Args:
            workflow_type: Type of workflow
            state: Final workflow state
            memory: ShortTermMemory instance
            auto_export: Automatically export session to vault

        Returns:
            Path to exported session file if auto_export=True
        """
        logger.info(f"Workflow completed: {workflow_type}")

        if auto_export:
            # Export session to vault
            metadata = {
                'workflow_type': workflow_type,
                'completed': datetime.now().isoformat()
            }

            file_path = self.vault.export_session(
                memory=memory,
                include_approval=True,
                additional_metadata=metadata
            )

            logger.info(f"Auto-exported session to {file_path}")
            return file_path

        return None

    def capture_workflow_knowledge(
        self,
        title: str,
        content: str,
        workflow_type: str,
        tags: Optional[List[str]] = None
    ) -> Path:
        """
        Capture knowledge generated during workflow execution.

        Args:
            title: Knowledge note title
            content: Markdown content
            workflow_type: Type of workflow that generated this
            tags: Additional tags

        Returns:
            Path to created knowledge note
        """
        base_tags = ['agent_knowledge', workflow_type]
        if tags:
            base_tags.extend(tags)

        metadata = {
            'workflow_type': workflow_type,
            'generated_by': 'workflow'
        }

        return self.vault.create_knowledge_note(
            title=title,
            content=content,
            tags=base_tags,
            metadata=metadata,
            sync_to_graphrag=True
        )


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("ObsidianVault Integration Demo")
    print("=" * 80)
    print()

    # Test vault initialization
    print("[1/5] Initializing vault...")
    vault = ObsidianVault(vault_path="./test_vault")
    print(f"✅ Vault initialized at {vault.vault_path}")
    print()

    # Test note creation
    print("[2/5] Creating test knowledge note...")
    note_path = vault.create_knowledge_note(
        title="Test Knowledge Note",
        content="# Test Content\n\nThis is a test note with some knowledge.",
        tags=['test', 'demo'],
        metadata={'source': 'demo'},
        sync_to_graphrag=False
    )
    print(f"✅ Created note: {note_path}")
    print()

    # Test vault indexing
    print("[3/5] Indexing vault...")
    count = vault.index_vault()
    print(f"✅ Indexed {count} notes")
    print()

    # Test note retrieval
    print("[4/5] Retrieving note...")
    note = vault.get_note("Test Knowledge Note")
    if note:
        print(f"✅ Retrieved note: {note.title}")
        print(f"   Tags: {note.tags}")
        print(f"   Created: {note.created}")
    else:
        print("⚠️  Note not found")
    print()

    # Test session export
    print("[5/5] Testing session export...")
    from fractal_agent.memory.short_term import ShortTermMemory

    memory = ShortTermMemory(log_dir="./test_logs")
    task_id = memory.start_task(
        agent_id="test_001",
        agent_type="test",
        task_description="Test task for vault integration",
        inputs={"test": True}
    )
    memory.end_task(
        task_id=task_id,
        outputs={"result": "success"}
    )

    session_path = vault.export_session(memory)
    print(f"✅ Exported session to: {session_path}")
    print()

    print("=" * 80)
    print("ObsidianVault Integration Demo Complete!")
    print("=" * 80)
