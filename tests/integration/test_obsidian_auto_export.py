"""
Integration Test: Automatic Obsidian Exports (Gap #4)

Tests the automatic Obsidian export functionality:
- Auto-export on session end
- Async background export
- Export status tracking
- Synchronous export mode

Author: BMad
Date: 2025-10-22
"""

import logging
import time
from pathlib import Path
import sys
import tempfile

# Add fractal_agent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fractal_agent.memory.short_term import ShortTermMemory
from fractal_agent.memory.obsidian_vault import ObsidianVault

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_sync_export():
    """
    Test synchronous Obsidian export.
    """
    print("\n" + "=" * 80)
    print("TEST 1: Synchronous Obsidian Export")
    print("=" * 80)

    # Create temp directories
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_path = Path(temp_dir) / "test_vault"
        log_dir = Path(temp_dir) / "test_logs"

        # Initialize vault
        vault = ObsidianVault(vault_path=str(vault_path))
        print(f"✓ Created Obsidian vault at: {vault_path}")

        # Initialize memory with vault
        memory = ShortTermMemory(
            log_dir=str(log_dir),
            obsidian_vault=vault,
            enable_auto_export=True
        )
        print(f"✓ Initialized ShortTermMemory: {memory.session_id}")

        # Add test tasks
        task1 = memory.start_task(
            agent_id="test_001",
            agent_type="research",
            task_description="Test task 1",
            inputs={"test": "input1"}
        )
        memory.end_task(
            task_id=task1,
            outputs={"result": "output1"}
        )

        task2 = memory.start_task(
            agent_id="test_002",
            agent_type="developer",
            task_description="Test task 2",
            inputs={"test": "input2"}
        )
        memory.end_task(
            task_id=task2,
            outputs={"result": "output2"}
        )

        print(f"✓ Logged {len(memory.tasks)} test tasks")

        # End session (synchronous export)
        print("▶ Ending session (sync export)...")
        export_path = memory.end_session(async_export=False)

        # Verify export
        export_status = memory.get_export_status()
        print(f"\n✓ Export Status: {export_status['status']}")
        print(f"✓ Export Path: {export_status['path']}")
        print(f"✓ Export Enabled: {export_status['enabled']}")

        # Check file exists
        if export_path and export_path.exists():
            print(f"✅ Export file created successfully: {export_path}")
            print(f"   File size: {export_path.stat().st_size} bytes")

            # Read first 20 lines
            with open(export_path, 'r') as f:
                lines = f.readlines()[:20]
            print(f"\n📄 Preview (first 20 lines):")
            print("-" * 80)
            print("".join(lines))
            print("-" * 80)

            return True
        else:
            print(f"❌ Export file not found at: {export_path}")
            return False


def test_async_export():
    """
    Test asynchronous Obsidian export (background thread).
    """
    print("\n" + "=" * 80)
    print("TEST 2: Asynchronous Obsidian Export")
    print("=" * 80)

    # Create temp directories
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_path = Path(temp_dir) / "test_vault"
        log_dir = Path(temp_dir) / "test_logs"

        # Initialize vault
        vault = ObsidianVault(vault_path=str(vault_path))
        print(f"✓ Created Obsidian vault at: {vault_path}")

        # Initialize memory with vault
        memory = ShortTermMemory(
            log_dir=str(log_dir),
            obsidian_vault=vault,
            enable_auto_export=True
        )
        print(f"✓ Initialized ShortTermMemory: {memory.session_id}")

        # Add test tasks
        for i in range(5):
            task_id = memory.start_task(
                agent_id=f"test_{i:03d}",
                agent_type="research",
                task_description=f"Test task {i+1}",
                inputs={"index": i}
            )
            memory.end_task(
                task_id=task_id,
                outputs={"result": f"output {i+1}"}
            )

        print(f"✓ Logged {len(memory.tasks)} test tasks")

        # End session (async export)
        print("▶ Ending session (async export)...")
        export_path = memory.end_session(async_export=True)

        # Check initial status (should be "pending")
        export_status = memory.get_export_status()
        print(f"\n✓ Initial Export Status: {export_status['status']}")
        assert export_status['status'] in ['pending', 'exporting', 'completed'], \
            f"Unexpected initial status: {export_status['status']}"

        # Wait for async export to complete (max 5 seconds)
        print("⏳ Waiting for background export to complete...")
        for i in range(50):  # 50 * 0.1s = 5s max
            time.sleep(0.1)
            export_status = memory.get_export_status()
            if export_status['status'] == 'completed':
                print(f"✅ Export completed in {(i+1)*0.1:.1f}s")
                break
            if export_status['status'] == 'failed':
                print(f"❌ Export failed: {export_status}")
                return False

        # Final verification
        export_status = memory.get_export_status()
        print(f"\n✓ Final Export Status: {export_status['status']}")
        print(f"✓ Export Path: {export_status['path']}")

        # Check file exists
        export_path = Path(export_status['path']) if export_status['path'] else None
        if export_path and export_path.exists():
            print(f"✅ Export file created successfully: {export_path}")
            print(f"   File size: {export_path.stat().st_size} bytes")
            return True
        else:
            print(f"❌ Export file not found at: {export_path}")
            return False


def test_disabled_export():
    """
    Test that exports are skipped when disabled.
    """
    print("\n" + "=" * 80)
    print("TEST 3: Disabled Auto-Export")
    print("=" * 80)

    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        log_dir = Path(temp_dir) / "test_logs"

        # Initialize memory WITHOUT vault (auto-export disabled)
        memory = ShortTermMemory(
            log_dir=str(log_dir),
            obsidian_vault=None,
            enable_auto_export=True  # Will be disabled due to no vault
        )
        print(f"✓ Initialized ShortTermMemory: {memory.session_id}")

        # Add test task
        task_id = memory.start_task(
            agent_id="test_001",
            agent_type="research",
            task_description="Test task",
            inputs={"test": "input"}
        )
        memory.end_task(
            task_id=task_id,
            outputs={"result": "output"}
        )

        print(f"✓ Logged {len(memory.tasks)} test tasks")

        # End session (export should be skipped)
        print("▶ Ending session (export disabled)...")
        export_path = memory.end_session(async_export=True)

        # Verify export was skipped
        export_status = memory.get_export_status()
        print(f"\n✓ Export Status: {export_status['status']}")
        print(f"✓ Export Enabled: {export_status['enabled']}")
        print(f"✓ Vault Configured: {export_status['vault_configured']}")

        if not export_status['enabled'] and export_status['status'] is None:
            print("✅ Export correctly skipped (no vault configured)")
            return True
        else:
            print(f"❌ Unexpected behavior: {export_status}")
            return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("AUTOMATIC OBSIDIAN EXPORTS - INTEGRATION TEST (Gap #4)")
    print("=" * 80)

    results = {}

    # Run tests
    try:
        results['sync'] = test_sync_export()
    except Exception as e:
        logger.error(f"Test 1 failed with exception: {e}", exc_info=True)
        results['sync'] = False

    try:
        results['async'] = test_async_export()
    except Exception as e:
        logger.error(f"Test 2 failed with exception: {e}", exc_info=True)
        results['async'] = False

    try:
        results['disabled'] = test_disabled_export()
    except Exception as e:
        logger.error(f"Test 3 failed with exception: {e}", exc_info=True)
        results['disabled'] = False

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Test 1 (Sync Export):      {'✅ PASS' if results['sync'] else '❌ FAIL'}")
    print(f"Test 2 (Async Export):     {'✅ PASS' if results['async'] else '❌ FAIL'}")
    print(f"Test 3 (Disabled Export):  {'✅ PASS' if results['disabled'] else '❌ FAIL'}")
    print("=" * 80)

    # Overall result
    all_passed = all(results.values())
    if all_passed:
        print("\n✅ ALL TESTS PASSED - Gap #4 Implementation Complete!")
        print("📤 Automatic Obsidian exports are now 100% operational")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED - See details above")
        sys.exit(1)
