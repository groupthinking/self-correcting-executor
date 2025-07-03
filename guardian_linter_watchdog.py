#!/usr/bin/env python3
"""
Guardian Agent: Linter Watchdog
===============================

This script continuously monitors the project for Python file changes
and runs a linter to provide immediate feedback on code quality.

This is the first component of the Guardian Agent Protocol.
"""
import asyncio
import subprocess
import logging
from pathlib import Path

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.resolve()
WATCHED_EXTENSIONS = {".py"}
LINT_COMMAND = ["pylint"]
EXCLUDED_DIRS = {"__pycache__", ".git", "venv", "node_modules", ".cursor"}
# ---------------------

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def run_linter(file_path: Path):
    """Run the linter on a specific file."""
    if not any(part in EXCLUDED_DIRS for part in file_path.parts):
        command = LINT_COMMAND + [str(file_path)]
        logger.info(
            f"Guardian: Analyzing {
                file_path.relative_to(PROJECT_ROOT)}...")

        process = await asyncio.create_subprocess_exec(
            *command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.warning(
                f"Guardian: Found issues in {
                    file_path.relative_to(PROJECT_ROOT)}")
            if stdout:
                print("\n--- LINT REPORT ---")
                print(stdout.decode().strip())
                print("--- END REPORT ---\n")
            if stderr:
                logger.error(
                    f"Linter error on {
                        file_path.relative_to(PROJECT_ROOT)}:\n{
                        stderr.decode().strip()}")
        else:
            logger.info(
                f"Guardian: {
                    file_path.relative_to(PROJECT_ROOT)} looks clean!")


async def watch_directory():
    """Watch the project directory for file changes."""
    logger.info("Guardian Agent (Linter Watchdog) is now active.")
    logger.info(f"Watching for changes in: {PROJECT_ROOT}")

    # Simple polling-based watcher
    last_mtimes = {}

    while True:
        for file_path in PROJECT_ROOT.rglob("*"):
            if file_path.is_file() and file_path.suffix in WATCHED_EXTENSIONS:
                try:
                    mtime = file_path.stat().st_mtime
                    if file_path not in last_mtimes:
                        last_mtimes[file_path] = mtime
                        # Optionally lint on first discovery
                        # await run_linter(file_path)
                    elif last_mtimes[file_path] < mtime:
                        last_mtimes[file_path] = mtime
                        await run_linter(file_path)
                except FileNotFoundError:
                    # File might have been deleted
                    if file_path in last_mtimes:
                        del last_mtimes[file_path]

        await asyncio.sleep(2)  # Check for changes every 2 seconds


if __name__ == "__main__":
    try:
        asyncio.run(watch_directory())
    except KeyboardInterrupt:
        logger.info("Guardian Agent deactivated.")
