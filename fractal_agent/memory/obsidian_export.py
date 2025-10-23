"""
Obsidian Export - Human Review Interface (Phase 1)

Exports session logs to Obsidian-compatible markdown for human review and approval.

Features:
- Task tree visualization
- Agent performance metrics
- Human approval/rejection workflow
- Bidirectional linking between tasks

Author: BMad
Date: 2025-10-18
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from .short_term import ShortTermMemory

logger = logging.getLogger(__name__)


class ObsidianExporter:
    """
    Exports task logs to Obsidian vault for human review.

    Creates markdown files with:
    - YAML frontmatter for metadata
    - Task trees with hierarchical structure
    - Performance metrics
    - Human approval workflow

    Usage:
        >>> memory = ShortTermMemory()
        >>> # ... execute tasks ...
        >>> exporter = ObsidianExporter(vault_path="./obsidian_vault")
        >>> exporter.export_session(memory)
    """

    def __init__(
        self,
        vault_path: Optional[str] = None,
        review_folder: str = "agent_reviews"
    ):
        """
        Initialize Obsidian exporter.

        Args:
            vault_path: Path to Obsidian vault (default: ./obsidian_vault)
            review_folder: Folder within vault for reviews
        """
        if vault_path is None:
            vault_path = "./obsidian_vault"

        self.vault_path = Path(vault_path)
        self.review_folder = self.vault_path / review_folder
        self.review_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized ObsidianExporter: {self.review_folder}")

    def export_session(
        self,
        memory: ShortTermMemory,
        include_approval_template: bool = True
    ) -> Path:
        """
        Export a session to Obsidian markdown.

        Args:
            memory: ShortTermMemory instance with session data
            include_approval_template: Add human approval section

        Returns:
            Path to created markdown file
        """
        session_id = memory.session_id
        file_path = self.review_folder / f"{session_id}.md"

        # Get session summary
        summary = memory.get_session_summary()

        # Generate markdown
        markdown = self._generate_markdown(
            session_id=session_id,
            summary=summary,
            tasks=memory.tasks,
            include_approval=include_approval_template
        )

        # Write file
        with open(file_path, 'w') as f:
            f.write(markdown)

        logger.info(f"Exported session to: {file_path}")

        return file_path

    def _generate_markdown(
        self,
        session_id: str,
        summary: Dict[str, Any],
        tasks: Dict[str, Dict[str, Any]],
        include_approval: bool
    ) -> str:
        """Generate markdown content for session"""

        # YAML frontmatter
        md = ["---"]
        md.append(f"session_id: {session_id}")
        md.append(f"created: {datetime.now().isoformat()}")
        md.append(f"total_tasks: {summary['total_tasks']}")
        md.append(f"completed_tasks: {summary['completed_tasks']}")
        md.append("status: pending_review")
        md.append("tags: [agent_session, needs_review]")
        md.append("---")
        md.append("")

        # Title
        md.append(f"# Agent Session Review: {session_id}")
        md.append("")

        # Summary section
        md.append("## Session Summary")
        md.append("")
        md.append(f"- **Total Tasks**: {summary['total_tasks']}")
        md.append(f"- **Completed**: {summary['completed_tasks']}")
        md.append(f"- **In Progress**: {summary['in_progress_tasks']}")
        md.append(f"- **Average Duration**: {summary['avg_duration_seconds']:.2f}s")
        md.append(f"- **Total Duration**: {summary['total_duration_seconds']:.2f}s")
        md.append("")

        # Task tree section
        md.append("## Task Tree")
        md.append("")

        # Find root tasks (no parent)
        root_tasks = [t for t in tasks.values() if not t.get("parent_task_id")]

        for root in root_tasks:
            md.extend(self._render_task_tree(root, tasks, level=0))

        # Detailed task logs
        md.append("")
        md.append("## Detailed Task Logs")
        md.append("")

        for task_id, task in tasks.items():
            md.extend(self._render_task_detail(task))

        # Human approval section
        if include_approval:
            md.append("")
            md.append("---")
            md.append("")
            md.append("## Human Review")
            md.append("")
            md.append("### Approval Status")
            md.append("")
            md.append("- [ ] Approved")
            md.append("- [ ] Rejected")
            md.append("- [ ] Needs Revision")
            md.append("")
            md.append("### Reviewer Notes")
            md.append("")
            md.append("*Add your notes here...*")
            md.append("")
            md.append("### Action Items")
            md.append("")
            md.append("- [ ] Item 1")
            md.append("- [ ] Item 2")
            md.append("")

        return "\n".join(md)

    def _render_task_tree(
        self,
        task: Dict[str, Any],
        all_tasks: Dict[str, Dict[str, Any]],
        level: int
    ) -> List[str]:
        """Recursively render task tree"""
        md = []
        indent = "  " * level

        # Task status emoji
        status_emoji = {
            "completed": "✅",
            "in_progress": "🔄",
            "failed": "❌"
        }
        emoji = status_emoji.get(task.get("status", ""), "⚪")

        # Task line with link
        task_link = f"[[#{task['task_id']}|{task['task_description']}]]"
        duration = f"{task['duration_seconds']:.2f}s" if task.get("duration_seconds") else "N/A"

        md.append(f"{indent}- {emoji} {task_link} ({duration})")

        # Find and render children
        children = [
            t for t in all_tasks.values()
            if t.get("parent_task_id") == task["task_id"]
        ]

        for child in children:
            md.extend(self._render_task_tree(child, all_tasks, level + 1))

        return md

    def _render_task_detail(self, task: Dict[str, Any]) -> List[str]:
        """Render detailed task information"""
        md = []

        md.append(f"### {task['task_description']} {{#{task['task_id']}}}")
        md.append("")

        # Metadata table
        md.append("| Property | Value |")
        md.append("|----------|-------|")
        md.append(f"| Task ID | `{task['task_id']}` |")
        md.append(f"| Agent ID | `{task['agent_id']}` |")
        md.append(f"| Agent Type | {task['agent_type']} |")
        md.append(f"| Status | {task['status']} |")

        if task.get("parent_task_id"):
            md.append(f"| Parent Task | [[#{task['parent_task_id']}]] |")

        if task.get("duration_seconds"):
            md.append(f"| Duration | {task['duration_seconds']:.2f}s |")

        md.append("")

        # Inputs
        if task.get("inputs"):
            md.append("**Inputs:**")
            md.append("```json")
            md.append(json.dumps(task["inputs"], indent=2))
            md.append("```")
            md.append("")

        # Outputs
        if task.get("outputs"):
            md.append("**Outputs:**")
            md.append("```json")
            md.append(json.dumps(task["outputs"], indent=2))
            md.append("```")
            md.append("")

        # Metadata
        if task.get("metadata"):
            md.append("**Metadata:**")
            md.append("```json")
            md.append(json.dumps(task["metadata"], indent=2))
            md.append("```")
            md.append("")

        md.append("---")
        md.append("")

        return md

    def load_approval_status(self, session_file: Path) -> Optional[str]:
        """
        Parse Obsidian markdown to get human approval status.

        Returns:
            "approved", "rejected", "needs_revision", or None
        """
        if not session_file.exists():
            return None

        with open(session_file, 'r') as f:
            content = f.read()

        # Look for checked approval boxes
        if "- [x] Approved" in content or "- [X] Approved" in content:
            return "approved"
        elif "- [x] Rejected" in content or "- [X] Rejected" in content:
            return "rejected"
        elif "- [x] Needs Revision" in content or "- [X] Needs Revision" in content:
            return "needs_revision"

        return None


# Quick test
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("Obsidian Export Test")
    print("=" * 80)
    print()

    # Create a test session
    from .short_term import ShortTermMemory

    memory = ShortTermMemory(log_dir="./test_logs")

    # Add test tasks
    control_task = memory.start_task(
        agent_id="control_001",
        agent_type="control",
        task_description="Research the Viable System Model",
        inputs={"topic": "VSM"}
    )

    for i in range(3):
        subtask = memory.start_task(
            agent_id=f"operational_{i+1:03d}",
            agent_type="operational",
            task_description=f"Research VSM System {i+1}",
            inputs={"system": i+1},
            parent_task_id=control_task
        )

        memory.end_task(
            task_id=subtask,
            outputs={"findings": f"System {i+1} research complete"},
            metadata={"tokens": 1000}
        )

    memory.end_task(
        task_id=control_task,
        outputs={"report": "Complete VSM analysis"},
        metadata={"total_tokens": 3000}
    )

    # Export to Obsidian
    exporter = ObsidianExporter(vault_path="./test_vault")
    file_path = exporter.export_session(memory)

    print(f"✓ Exported to: {file_path}")
    print()

    # Show preview
    print("Preview:")
    print("-" * 80)
    with open(file_path, 'r') as f:
        lines = f.readlines()[:30]  # First 30 lines
        print("".join(lines))
    print("-" * 80)

    print()
    print("=" * 80)
    print("✓ Obsidian export test complete!")
    print("=" * 80)
