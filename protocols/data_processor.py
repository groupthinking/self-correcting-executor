# Real Protocol: Data Processor
import json
import csv
import os
from datetime import datetime


def task(data_path=None, operation="analyze"):
    """Process data files and extract insights"""
    # Use provided data path or try multiple possible data directories
    if data_path and os.path.exists(data_path) and os.path.isdir(data_path):
        data_dir = data_path
    else:
        possible_dirs = [
            os.environ.get("DATA_DIR", "/data"),
            "/data",
            "/app/data",
            "/tmp",
            os.getcwd(),
        ]

        data_dir = None
        for dir_path in possible_dirs:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                data_dir = dir_path
                break

    if not data_dir:
        # Return error when no data directory exists
        return {
            "success": False,
            "error": "No data directory found"
        }

    try:
        processed_count = 0
        total_records = 0
        insights = []

        # Look for JSON and CSV files
        files = os.listdir(data_dir)[:10]  # Limit to 10 files

        if not files:
            # No files found, return success with empty results
            return {
                "success": True,
                "action": "data_processing",
                "message": f"No data files found in {data_dir}",
                "files_processed": 0,
                "total_records": 0,
                "insights": [],
                "timestamp": datetime.utcnow().isoformat(),
            }

        for filename in files:
            file_path = os.path.join(data_dir, filename)

            if filename.endswith(".json"):
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            total_records += len(data)
                        elif isinstance(data, dict):
                            total_records += 1
                        processed_count += 1
                        insights.append(
                            f"{filename}: {type(data).__name__} with {len(data) if isinstance(data, (list, dict)) else 1} items"
                        )
                except:
                    pass

            elif filename.endswith(".csv"):
                try:
                    with open(file_path, "r") as f:
                        reader = csv.reader(f)
                        row_count = sum(1 for row in reader)
                        total_records += row_count
                        processed_count += 1
                        insights.append(f"{filename}: CSV with {row_count} rows")
                except BaseException:
                    pass

        # Always return success if we got this far
        return {
            "success": True,
            "action": "data_processing",
            "directory": data_dir,
            "files_processed": processed_count,
            "total_records": total_records,
            "insights": (
                insights[:5] if insights else ["No data files found to process"]
            ),
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        return {"success": False, "action": "data_processing", "error": str(e)}
