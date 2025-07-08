# Enhanced logging utility
import os
import datetime


def log(message):
    """Log message to both file and console with timestamp"""
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    formatted_message = f"[{timestamp}] {message}"

    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Write to log file
    try:
        with open("logs/mcp.log", "a") as f:
            f.write(formatted_message + "\n")
    except Exception as e:
        print(f"Failed to write to log file: {e}")

    # Also print to console
    print(formatted_message)


def log_json(data, prefix="DATA"):
    """Log JSON data in a structured format"""
    import json

    try:
        json_str = json.dumps(data, indent=2)
        log(f"{prefix}: {json_str}")
    except Exception as e:
        log(f"Failed to log JSON data: {e}")


def get_log_path():
    """Get the path to the current log file"""
    return os.path.abspath("logs/mcp.log")
