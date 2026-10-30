#!/usr/bin/env python3
"""Start MCP server with proper error handling and restart capability"""

import asyncio
import subprocess
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def start_mcp_server():
    """Start the MCP server with proper error handling"""

    server_path = Path(__file__).parent / "mcp_server" / "main.py"

    try:
        logger.info("Starting MCP server...")

        # Start server process
        process = subprocess.Popen(
            [sys.executable, str(server_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        logger.info(f"MCP server started with PID: {process.pid}")

        # Monitor for crashes
        while True:
            if process.poll() is not None:
                logger.error("MCP server crashed, restarting...")
                stdout, stderr = process.communicate()
                logger.error(f"STDOUT: {stdout}")
                logger.error(f"STDERR: {stderr}")

                # Restart
                process = subprocess.Popen(
                    [sys.executable, str(server_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                logger.info(f"MCP server restarted with PID: {process.pid}")

            time.sleep(5)  # Check every 5 seconds

    except KeyboardInterrupt:
        logger.info("Stopping MCP server...")
        process.terminate()
        process.wait()

    except Exception as e:
        logger.error(f"Error starting MCP server: {e}")


if __name__ == "__main__":
    start_mcp_server()
