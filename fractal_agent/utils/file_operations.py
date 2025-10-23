"""
Unified File Operations with Atomic Writes and Observability

Provides safe, observable file operations for the Fractal VSM system.

Features:
- Atomic writes (write-to-temp-then-rename)
- File locking for concurrent safety
- OpenTelemetry tracing
- VSMEvent logging
- Automatic retries on transient failures
- Proper error handling and logging

Usage:
    from fractal_agent.utils.file_operations import atomic_write, atomic_read

    # Safe atomic write
    atomic_write("/path/to/file.json", {"data": "value"})

    # Safe read with retry
    data = atomic_read("/path/to/file.json", format="json")

Author: BMad
Date: 2025-10-20
"""

import os
import json
import tempfile
import fcntl
from pathlib import Path
from typing import Any, Dict, Optional, Union
from contextlib import contextmanager
import logging
import time

# Observability imports
try:
    from ..observability import (
        get_tracer, get_logger, get_event_store, VSMEvent,
        set_span_attributes
    )
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False
    import logging
    logger = logging.getLogger(__name__)

if OBSERVABILITY_AVAILABLE:
    logger = get_logger(__name__)
else:
    logger = logging.getLogger(__name__)


@contextmanager
def file_lock(file_path: Union[str, Path], timeout: float = 5.0):
    """
    Context manager for file locking.

    Args:
        file_path: Path to file to lock
        timeout: Maximum seconds to wait for lock

    Yields:
        File descriptor with lock acquired

    Raises:
        TimeoutError: If lock couldn't be acquired within timeout
    """
    lock_path = Path(str(file_path) + ".lock")
    lock_fd = None

    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)

        start_time = time.time()
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except IOError:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Could not acquire lock for {file_path} within {timeout}s")
                time.sleep(0.1)

        yield lock_fd

    finally:
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
            except:
                pass
        try:
            lock_path.unlink(missing_ok=True)
        except:
            pass


def atomic_write(
    file_path: Union[str, Path],
    content: Union[str, bytes, Dict, Any],
    encoding: str = 'utf-8',
    format: str = 'auto',
    create_dirs: bool = True,
    fsync: bool = False,
    verbose: bool = False
) -> bool:
    """
    Atomically write content to file using write-to-temp-then-rename pattern.

    This prevents corruption from crashes/failures mid-write.

    Args:
        file_path: Destination file path
        content: Content to write (str, bytes, or dict for JSON)
        encoding: Text encoding (default: utf-8)
        format: Output format ('auto', 'text', 'json', 'bytes')
        create_dirs: Create parent directories if missing
        fsync: Force sync to disk (slower but safer)
        verbose: Print status messages

    Returns:
        True if write succeeded, False otherwise
    """
    file_path = Path(file_path)

    # Start observability span
    if OBSERVABILITY_AVAILABLE:
        tracer = get_tracer(__name__)
        event_store = get_event_store()

        with tracer.start_as_current_span("file_write_atomic") as span:
            set_span_attributes({
                "file.path": str(file_path),
                "file.size": len(str(content)) if isinstance(content, str) else len(content) if isinstance(content, bytes) else 0,
                "file.format": format
            })

            event_store.append(VSMEvent(
                tier="Utils",
                event_type="file_write_started",
                data={
                    "path": str(file_path),
                    "format": format,
                    "create_dirs": create_dirs
                }
            ))

    try:
        # Create parent directories if needed
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine format
        if format == 'auto':
            if isinstance(content, dict) or isinstance(content, list):
                format = 'json'
            elif isinstance(content, bytes):
                format = 'bytes'
            else:
                format = 'text'

        # Serialize content
        if format == 'json':
            write_content = json.dumps(content, indent=2, ensure_ascii=False)
            write_mode = 'w'
        elif format == 'bytes':
            write_content = content
            write_mode = 'wb'
        else:
            write_content = str(content)
            write_mode = 'w'

        # Atomic write: write to temp file, then rename
        with file_lock(file_path, timeout=5.0):
            # Create temp file in same directory (ensures same filesystem)
            temp_fd, temp_path = tempfile.mkstemp(
                dir=file_path.parent,
                prefix=f".{file_path.name}.",
                suffix=".tmp"
            )

            try:
                # Write to temp file
                if 'b' in write_mode:
                    os.write(temp_fd, write_content)
                else:
                    os.write(temp_fd, write_content.encode(encoding))

                # Force sync to disk if requested
                if fsync:
                    os.fsync(temp_fd)

                os.close(temp_fd)

                # Atomic rename
                os.replace(temp_path, file_path)

            except Exception as e:
                # Clean up temp file on failure
                try:
                    os.close(temp_fd)
                except:
                    pass
                try:
                    os.unlink(temp_path)
                except:
                    pass
                raise e

        # Verify file exists
        if not file_path.exists():
            raise IOError(f"File not found after atomic write: {file_path}")

        file_size = file_path.stat().st_size

        if verbose:
            print(f"  ✓ Wrote (atomic): {file_path} ({file_size} bytes)")

        logger.info(f"Atomic write successful: {file_path} ({file_size} bytes)")

        if OBSERVABILITY_AVAILABLE:
            event_store.append(VSMEvent(
                tier="Utils",
                event_type="file_write_completed",
                data={
                    "path": str(file_path),
                    "size": file_size,
                    "success": True
                }
            ))

        return True

    except Exception as e:
        logger.error(f"Atomic write failed for {file_path}: {e}")

        if OBSERVABILITY_AVAILABLE:
            event_store.append(VSMEvent(
                tier="Utils",
                event_type="file_write_failed",
                data={
                    "path": str(file_path),
                    "error": str(e)
                }
            ))

        if verbose:
            print(f"  ✗ Write failed: {file_path}: {e}")

        return False


def atomic_read(
    file_path: Union[str, Path],
    encoding: str = 'utf-8',
    format: str = 'auto',
    retries: int = 3,
    retry_delay: float = 0.1
) -> Optional[Union[str, bytes, Dict, Any]]:
    """
    Safely read file with retry logic for transient failures.

    Args:
        file_path: File to read
        encoding: Text encoding (default: utf-8)
        format: Expected format ('auto', 'text', 'json', 'bytes')
        retries: Number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        File content in requested format, or None on failure
    """
    file_path = Path(file_path)

    for attempt in range(retries):
        try:
            with file_lock(file_path, timeout=5.0):
                if not file_path.exists():
                    return None

                # Auto-detect format from extension
                if format == 'auto':
                    if file_path.suffix == '.json':
                        format = 'json'
                    else:
                        format = 'text'

                # Read file
                if format == 'bytes':
                    with open(file_path, 'rb') as f:
                        return f.read()
                else:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()

                    if format == 'json':
                        return json.loads(content)
                    else:
                        return content

        except Exception as e:
            if attempt < retries - 1:
                logger.warning(f"Read attempt {attempt + 1} failed for {file_path}, retrying: {e}")
                time.sleep(retry_delay)
            else:
                logger.error(f"Read failed after {retries} attempts for {file_path}: {e}")
                return None

    return None


def safe_copy(
    src: Union[str, Path],
    dst: Union[str, Path],
    overwrite: bool = False,
    atomic: bool = True
) -> bool:
    """
    Safely copy file from src to dst with optional atomic write.

    Args:
        src: Source file path
        dst: Destination file path
        overwrite: Allow overwriting existing file
        atomic: Use atomic write for destination

    Returns:
        True if copy succeeded, False otherwise
    """
    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        logger.error(f"Source file does not exist: {src}")
        return False

    if dst.exists() and not overwrite:
        logger.error(f"Destination exists and overwrite=False: {dst}")
        return False

    try:
        content = atomic_read(src, format='bytes')
        if content is None:
            return False

        if atomic:
            return atomic_write(dst, content, format='bytes')
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, 'wb') as f:
                f.write(content)
            return True

    except Exception as e:
        logger.error(f"Copy failed from {src} to {dst}: {e}")
        return False
