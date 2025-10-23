"""
CLI Review Tool for Obsidian Vault Integration

Production-quality command-line interface for reviewing and managing
agent session reviews in the Obsidian vault using Typer and Rich.

Features:
- List pending/approved/rejected reviews with filtering
- Interactive review workflow with status updates
- Approve/reject operations with reviewer notes
- Batch operations for multiple reviews
- Vault synchronization with Git integration
- Rich terminal UI with tables and formatting

Author: BMad
Date: 2025-10-19
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich import box
import yaml

from .vault_structure import ObsidianVault, VaultParser, VaultNote

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="obsidian-review",
    help="CLI tool for managing agent session reviews in Obsidian vault",
    add_completion=False
)
console = Console()


class ReviewStatus(str, Enum):
    """Review status enumeration."""
    PENDING = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


class ReviewCLI:
    """
    Review CLI manager for Obsidian vault integration.

    Provides methods for review operations and rich terminal output.
    """

    def __init__(self, vault_path: str):
        """
        Initialize review CLI.

        Args:
            vault_path: Path to Obsidian vault root
        """
        self.vault_path = Path(vault_path)

        if not self.vault_path.exists():
            console.print(f"[red]Error:[/red] Vault path does not exist: {vault_path}")
            raise typer.Exit(code=1)

        self.vault = ObsidianVault(
            vault_path=str(vault_path),
            init_git=True,
            auto_commit=False
        )

        self.reviews_path = self.vault.structure.agent_reviews

        if not self.reviews_path.exists():
            console.print(
                f"[yellow]Warning:[/yellow] Reviews folder not found, creating: "
                f"{self.reviews_path}"
            )
            self.reviews_path.mkdir(parents=True, exist_ok=True)

    def get_all_reviews(
        self,
        status_filter: Optional[ReviewStatus] = None
    ) -> List[VaultNote]:
        """
        Get all review notes from the vault.

        Args:
            status_filter: Filter by review status

        Returns:
            List of VaultNote instances
        """
        reviews = []

        for md_file in self.reviews_path.glob("*.md"):
            if md_file.name == "README.md":
                continue

            try:
                note = VaultParser.parse_note(md_file)

                if status_filter:
                    note_status = note.frontmatter.get('status', 'pending_review')
                    if note_status != status_filter.value:
                        continue

                reviews.append(note)
            except Exception as e:
                logger.warning(f"Failed to parse {md_file}: {e}")

        return sorted(reviews, key=lambda n: n.modified, reverse=True)

    def display_reviews_table(
        self,
        reviews: List[VaultNote],
        title: str = "Agent Session Reviews"
    ):
        """
        Display reviews in a rich table.

        Args:
            reviews: List of review notes
            title: Table title
        """
        if not reviews:
            console.print("[yellow]No reviews found.[/yellow]")
            return

        table = Table(title=title, box=box.ROUNDED, show_header=True)
        table.add_column("#", style="cyan", width=3)
        table.add_column("Session ID", style="bright_blue", width=25)
        table.add_column("Status", width=15)
        table.add_column("Created", style="dim", width=19)
        table.add_column("Modified", style="dim", width=19)
        table.add_column("Tags", style="magenta", width=20)

        for idx, review in enumerate(reviews, 1):
            status = review.frontmatter.get('status', 'unknown')

            status_color = {
                'pending_review': 'yellow',
                'approved': 'green',
                'rejected': 'red',
                'needs_revision': 'orange1'
            }.get(status, 'white')

            session_id = review.frontmatter.get('session_id', review.title)
            tags_str = ', '.join(review.tags[:3]) if review.tags else '-'

            table.add_row(
                str(idx),
                session_id,
                f"[{status_color}]{status}[/{status_color}]",
                review.created.strftime('%Y-%m-%d %H:%M:%S'),
                review.modified.strftime('%Y-%m-%d %H:%M:%S'),
                tags_str
            )

        console.print(table)
        console.print(f"\n[dim]Total: {len(reviews)} reviews[/dim]")

    def display_review_detail(self, review: VaultNote):
        """
        Display detailed review information.

        Args:
            review: VaultNote instance
        """
        session_id = review.frontmatter.get('session_id', review.title)
        status = review.frontmatter.get('status', 'unknown')

        status_color = {
            'pending_review': 'yellow',
            'approved': 'green',
            'rejected': 'red',
            'needs_revision': 'orange1'
        }.get(status, 'white')

        console.print(Panel(
            f"[bold]{session_id}[/bold]\n"
            f"Status: [{status_color}]{status}[/{status_color}]\n"
            f"Created: {review.created.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Modified: {review.modified.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Tags: {', '.join(review.tags) if review.tags else 'None'}",
            title="Review Details",
            border_style="blue"
        ))

        frontmatter, body = VaultParser._extract_frontmatter(review.content)

        md = Markdown(body)
        console.print(md)

    def update_review_status(
        self,
        review: VaultNote,
        new_status: ReviewStatus,
        reviewer_notes: Optional[str] = None
    ) -> bool:
        """
        Update review status and metadata.

        Args:
            review: VaultNote to update
            new_status: New review status
            reviewer_notes: Optional reviewer notes to append

        Returns:
            True if update successful
        """
        try:
            with open(review.file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            frontmatter, body = VaultParser._extract_frontmatter(content)

            frontmatter['status'] = new_status.value
            frontmatter['reviewed_at'] = datetime.now().isoformat()
            frontmatter['reviewed_by'] = 'cli_reviewer'

            if 'needs_review' in frontmatter.get('tags', []):
                frontmatter['tags'] = [
                    t for t in frontmatter['tags'] if t != 'needs_review'
                ]
                frontmatter['tags'].append(new_status.value)

            if reviewer_notes:
                notes_section = f"\n\n### Reviewer Notes\n\n{reviewer_notes}\n\n*Reviewed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"

                if "### Reviewer Notes" in body:
                    body = body.split("### Reviewer Notes")[0] + notes_section
                elif "## Review" in body:
                    parts = body.split("## Review")
                    body = parts[0] + "## Review" + notes_section + parts[1] if len(parts) > 1 else body + notes_section
                else:
                    body += notes_section

            md_lines = ['---']
            md_lines.append(yaml.dump(frontmatter, default_flow_style=False))
            md_lines.append('---\n')
            md_lines.append(body)

            with open(review.file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(md_lines))

            logger.info(f"Updated review {review.title} to status {new_status.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to update review: {e}")
            console.print(f"[red]Error updating review:[/red] {e}")
            return False


@app.command()
def list(
    vault_path: str = typer.Option(
        "./obsidian_vault",
        "--vault",
        "-v",
        help="Path to Obsidian vault"
    ),
    status: Optional[str] = typer.Option(
        None,
        "--status",
        "-s",
        help="Filter by status (pending_review, approved, rejected, needs_revision)"
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        "-n",
        help="Limit number of results"
    )
):
    """
    List agent session reviews with optional filtering.

    Examples:
        obsidian-review list
        obsidian-review list --status pending_review
        obsidian-review list --vault ./my_vault --limit 10
    """
    try:
        cli = ReviewCLI(vault_path)

        status_filter = None
        if status:
            try:
                status_filter = ReviewStatus(status)
            except ValueError:
                console.print(
                    f"[red]Error:[/red] Invalid status. "
                    f"Valid options: {', '.join(s.value for s in ReviewStatus)}"
                )
                raise typer.Exit(code=1)

        reviews = cli.get_all_reviews(status_filter=status_filter)

        if limit:
            reviews = reviews[:limit]

        title = "Agent Session Reviews"
        if status_filter:
            title += f" - Status: {status_filter.value}"

        cli.display_reviews_table(reviews, title=title)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command()
def review(
    session_id: str = typer.Argument(..., help="Session ID or review number from list"),
    vault_path: str = typer.Option(
        "./obsidian_vault",
        "--vault",
        "-v",
        help="Path to Obsidian vault"
    ),
    interactive: bool = typer.Option(
        True,
        "--interactive/--no-interactive",
        "-i/-I",
        help="Interactive review mode"
    )
):
    """
    Review a specific agent session interactively.

    Examples:
        obsidian-review review session_abc123
        obsidian-review review 1 --no-interactive
    """
    try:
        cli = ReviewCLI(vault_path)
        reviews = cli.get_all_reviews()

        target_review = None

        if session_id.isdigit():
            idx = int(session_id) - 1
            if 0 <= idx < len(reviews):
                target_review = reviews[idx]
        else:
            for review in reviews:
                if review.frontmatter.get('session_id', '') == session_id:
                    target_review = review
                    break

        if not target_review:
            console.print(f"[red]Error:[/red] Review not found: {session_id}")
            raise typer.Exit(code=1)

        cli.display_review_detail(target_review)

        if interactive:
            console.print("\n" + "=" * 70 + "\n")

            action = Prompt.ask(
                "Action",
                choices=["approve", "reject", "revision", "skip"],
                default="skip"
            )

            if action == "skip":
                console.print("[yellow]Review skipped.[/yellow]")
                return

            add_notes = Confirm.ask("Add reviewer notes?", default=False)
            notes = None

            if add_notes:
                console.print("[dim]Enter notes (press Ctrl+D or Ctrl+Z when done):[/dim]")
                notes_lines = []
                try:
                    while True:
                        line = input()
                        notes_lines.append(line)
                except EOFError:
                    pass
                notes = '\n'.join(notes_lines).strip()

            status_map = {
                'approve': ReviewStatus.APPROVED,
                'reject': ReviewStatus.REJECTED,
                'revision': ReviewStatus.NEEDS_REVISION
            }

            new_status = status_map[action]

            if cli.update_review_status(target_review, new_status, notes):
                console.print(f"[green]✓[/green] Review {action}d successfully!")

                if Confirm.ask("Commit changes to Git?", default=True):
                    if cli.vault.git.is_git_repo():
                        cli.vault.git.commit_changes(
                            f"Review {action}d: {target_review.frontmatter.get('session_id', target_review.title)}"
                        )
                        console.print("[green]✓[/green] Changes committed to Git")
            else:
                console.print("[red]✗[/red] Failed to update review")
                raise typer.Exit(code=1)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command()
def approve(
    session_ids: List[str] = typer.Argument(..., help="Session IDs or review numbers"),
    vault_path: str = typer.Option(
        "./obsidian_vault",
        "--vault",
        "-v",
        help="Path to Obsidian vault"
    ),
    notes: Optional[str] = typer.Option(
        None,
        "--notes",
        "-n",
        help="Reviewer notes"
    ),
    commit: bool = typer.Option(
        True,
        "--commit/--no-commit",
        help="Commit changes to Git"
    )
):
    """
    Approve one or more reviews.

    Examples:
        obsidian-review approve session_abc123
        obsidian-review approve 1 2 3 --notes "All look good"
        obsidian-review approve session_xyz --no-commit
    """
    try:
        cli = ReviewCLI(vault_path)
        reviews = cli.get_all_reviews()

        approved_count = 0

        for session_id in session_ids:
            target_review = None

            if session_id.isdigit():
                idx = int(session_id) - 1
                if 0 <= idx < len(reviews):
                    target_review = reviews[idx]
            else:
                for review in reviews:
                    if review.frontmatter.get('session_id', '') == session_id:
                        target_review = review
                        break

            if not target_review:
                console.print(f"[yellow]Warning:[/yellow] Review not found: {session_id}")
                continue

            if cli.update_review_status(target_review, ReviewStatus.APPROVED, notes):
                console.print(
                    f"[green]✓[/green] Approved: "
                    f"{target_review.frontmatter.get('session_id', target_review.title)}"
                )
                approved_count += 1
            else:
                console.print(
                    f"[red]✗[/red] Failed to approve: "
                    f"{target_review.frontmatter.get('session_id', target_review.title)}"
                )

        if approved_count > 0:
            console.print(f"\n[green]Approved {approved_count} review(s)[/green]")

            if commit and cli.vault.git.is_git_repo():
                cli.vault.git.commit_changes(f"Approved {approved_count} review(s)")
                console.print("[green]✓[/green] Changes committed to Git")
        else:
            console.print("[yellow]No reviews were approved[/yellow]")
            raise typer.Exit(code=1)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command()
def reject(
    session_ids: List[str] = typer.Argument(..., help="Session IDs or review numbers"),
    vault_path: str = typer.Option(
        "./obsidian_vault",
        "--vault",
        "-v",
        help="Path to Obsidian vault"
    ),
    notes: Optional[str] = typer.Option(
        None,
        "--notes",
        "-n",
        help="Rejection reason"
    ),
    commit: bool = typer.Option(
        True,
        "--commit/--no-commit",
        help="Commit changes to Git"
    )
):
    """
    Reject one or more reviews.

    Examples:
        obsidian-review reject session_abc123 --notes "Incorrect approach"
        obsidian-review reject 1 2 --no-commit
    """
    try:
        cli = ReviewCLI(vault_path)
        reviews = cli.get_all_reviews()

        rejected_count = 0

        for session_id in session_ids:
            target_review = None

            if session_id.isdigit():
                idx = int(session_id) - 1
                if 0 <= idx < len(reviews):
                    target_review = reviews[idx]
            else:
                for review in reviews:
                    if review.frontmatter.get('session_id', '') == session_id:
                        target_review = review
                        break

            if not target_review:
                console.print(f"[yellow]Warning:[/yellow] Review not found: {session_id}")
                continue

            if cli.update_review_status(target_review, ReviewStatus.REJECTED, notes):
                console.print(
                    f"[green]✓[/green] Rejected: "
                    f"{target_review.frontmatter.get('session_id', target_review.title)}"
                )
                rejected_count += 1
            else:
                console.print(
                    f"[red]✗[/red] Failed to reject: "
                    f"{target_review.frontmatter.get('session_id', target_review.title)}"
                )

        if rejected_count > 0:
            console.print(f"\n[red]Rejected {rejected_count} review(s)[/red]")

            if commit and cli.vault.git.is_git_repo():
                cli.vault.git.commit_changes(f"Rejected {rejected_count} review(s)")
                console.print("[green]✓[/green] Changes committed to Git")
        else:
            console.print("[yellow]No reviews were rejected[/yellow]")
            raise typer.Exit(code=1)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command()
def batch(
    vault_path: str = typer.Option(
        "./obsidian_vault",
        "--vault",
        "-v",
        help="Path to Obsidian vault"
    ),
    status: str = typer.Option(
        "pending_review",
        "--status",
        "-s",
        help="Filter by status"
    ),
    action: str = typer.Option(
        ...,
        "--action",
        "-a",
        help="Action to perform (approve, reject)"
    ),
    notes: Optional[str] = typer.Option(
        None,
        "--notes",
        "-n",
        help="Reviewer notes"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be done without making changes"
    ),
    commit: bool = typer.Option(
        True,
        "--commit/--no-commit",
        help="Commit changes to Git"
    )
):
    """
    Perform batch operations on multiple reviews.

    Examples:
        obsidian-review batch --action approve --status pending_review
        obsidian-review batch --action reject --status needs_revision --notes "Outdated"
        obsidian-review batch --action approve --dry-run
    """
    try:
        cli = ReviewCLI(vault_path)

        try:
            status_filter = ReviewStatus(status)
        except ValueError:
            console.print(
                f"[red]Error:[/red] Invalid status. "
                f"Valid options: {', '.join(s.value for s in ReviewStatus)}"
            )
            raise typer.Exit(code=1)

        action_map = {
            'approve': ReviewStatus.APPROVED,
            'reject': ReviewStatus.REJECTED
        }

        if action not in action_map:
            console.print(
                f"[red]Error:[/red] Invalid action. "
                f"Valid options: {', '.join(action_map.keys())}"
            )
            raise typer.Exit(code=1)

        new_status = action_map[action]
        reviews = cli.get_all_reviews(status_filter=status_filter)

        if not reviews:
            console.print(f"[yellow]No reviews found with status: {status_filter.value}[/yellow]")
            return

        cli.display_reviews_table(reviews, title=f"Reviews to {action}")
        console.print()

        if dry_run:
            console.print(f"[yellow]DRY RUN:[/yellow] Would {action} {len(reviews)} review(s)")
            return

        if not Confirm.ask(
            f"[bold yellow]Proceed with {action} for {len(reviews)} review(s)?[/bold yellow]",
            default=False
        ):
            console.print("[yellow]Operation cancelled[/yellow]")
            return

        success_count = 0

        for review in reviews:
            if cli.update_review_status(review, new_status, notes):
                console.print(
                    f"[green]✓[/green] {action.capitalize()}d: "
                    f"{review.frontmatter.get('session_id', review.title)}"
                )
                success_count += 1
            else:
                console.print(
                    f"[red]✗[/red] Failed: "
                    f"{review.frontmatter.get('session_id', review.title)}"
                )

        console.print(
            f"\n[green]Successfully {action}d {success_count}/{len(reviews)} review(s)[/green]"
        )

        if commit and cli.vault.git.is_git_repo():
            cli.vault.git.commit_changes(f"Batch {action}: {success_count} review(s)")
            console.print("[green]✓[/green] Changes committed to Git")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command()
def sync(
    vault_path: str = typer.Option(
        "./obsidian_vault",
        "--vault",
        "-v",
        help="Path to Obsidian vault"
    ),
    remote: str = typer.Option(
        "origin",
        "--remote",
        "-r",
        help="Git remote name"
    ),
    branch: str = typer.Option(
        "main",
        "--branch",
        "-b",
        help="Git branch name"
    ),
    commit_message: Optional[str] = typer.Option(
        None,
        "--message",
        "-m",
        help="Custom commit message"
    )
):
    """
    Sync vault with Git remote (commit, pull, push).

    Examples:
        obsidian-review sync
        obsidian-review sync --remote origin --branch main
        obsidian-review sync --message "Manual review sync"
    """
    try:
        cli = ReviewCLI(vault_path)

        if not cli.vault.git.is_git_repo():
            console.print("[red]Error:[/red] Vault is not a Git repository")
            raise typer.Exit(code=1)

        console.print("[blue]Syncing vault with Git remote...[/blue]")

        with console.status("[blue]Checking status...[/blue]"):
            has_changes = cli.vault.git.has_uncommitted_changes()

        if has_changes:
            console.print("[yellow]Uncommitted changes detected[/yellow]")

            if not commit_message:
                commit_message = f"Auto-sync: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            with console.status("[blue]Committing changes...[/blue]"):
                if cli.vault.git.commit_changes(commit_message):
                    console.print("[green]✓[/green] Changes committed")
                else:
                    console.print("[yellow]No changes to commit[/yellow]")

        with console.status(f"[blue]Pulling from {remote}/{branch}...[/blue]"):
            if cli.vault.git.pull(remote, branch):
                console.print(f"[green]✓[/green] Pulled from {remote}/{branch}")
            else:
                console.print(f"[red]✗[/red] Pull failed")
                raise typer.Exit(code=1)

        with console.status(f"[blue]Pushing to {remote}/{branch}...[/blue]"):
            if cli.vault.git.push(remote, branch):
                console.print(f"[green]✓[/green] Pushed to {remote}/{branch}")
            else:
                console.print(f"[red]✗[/red] Push failed")
                raise typer.Exit(code=1)

        console.print("\n[green]✓ Vault synced successfully![/green]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command()
def stats(
    vault_path: str = typer.Option(
        "./obsidian_vault",
        "--vault",
        "-v",
        help="Path to Obsidian vault"
    )
):
    """
    Display vault statistics and review summary.

    Examples:
        obsidian-review stats
        obsidian-review stats --vault ./my_vault
    """
    try:
        cli = ReviewCLI(vault_path)

        all_reviews = cli.get_all_reviews()

        status_counts = {
            'pending_review': 0,
            'approved': 0,
            'rejected': 0,
            'needs_revision': 0
        }

        for review in all_reviews:
            status = review.frontmatter.get('status', 'pending_review')
            if status in status_counts:
                status_counts[status] += 1

        vault_stats = cli.vault.get_vault_stats()

        stats_panel = Panel(
            f"[bold]Total Reviews:[/bold] {len(all_reviews)}\n"
            f"[yellow]Pending:[/yellow] {status_counts['pending_review']}\n"
            f"[green]Approved:[/green] {status_counts['approved']}\n"
            f"[red]Rejected:[/red] {status_counts['rejected']}\n"
            f"[orange1]Needs Revision:[/orange1] {status_counts['needs_revision']}\n\n"
            f"[bold]Vault Statistics:[/bold]\n"
            f"Total Notes: {vault_stats['total_notes']}\n"
            f"Projects: {vault_stats['projects']}\n"
            f"Daily Notes: {vault_stats['daily_notes']}\n"
            f"Agent Knowledge: {vault_stats['agent_knowledge']}\n"
            f"Templates: {vault_stats['templates']}",
            title="Obsidian Vault Statistics",
            border_style="blue"
        )

        console.print(stats_panel)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
