# Real Protocol: User Data Processor
import os
from datetime import datetime


def task():
    """Process and analyze user data from mounted directories"""
    # Check which directories are available
    available_dirs = []
    data_stats = {
        "desktop": {"files": 0, "size": 0},
        "documents": {"files": 0, "size": 0},
        "gptdata": {"files": 0, "size": 0},
    }

    # Map of mount points
    mount_points = {
        "desktop": "/data/desktop",
        "documents": "/data/documents",
        "gptdata": "/data/gptdata",
    }

    insights = []
    processed_files = []

    try:
        # Check each mounted directory
        for name, path in mount_points.items():
            if os.path.exists(path) and os.access(path, os.R_OK):
                available_dirs.append(name)

                # Count files and calculate size
                for root, dirs, files in os.walk(path):
                    # Limit depth to avoid scanning too deep
                    depth = root.replace(path, "").count(os.sep)
                    if depth > 2:  # Only go 2 levels deep
                        dirs[:] = []  # Don't recurse further
                        continue

                    for file in files[:10]:  # Limit files per directory
                        file_path = os.path.join(root, file)
                        try:
                            size = os.path.getsize(file_path)
                            data_stats[name]["files"] += 1
                            data_stats[name]["size"] += size

                            # Process specific file types
                            if file.endswith(".json"):
                                processed_files.append(
                                    {
                                        "path": file_path.replace(path, f"{name}/"),
                                        "type": "json",
                                        "size": size,
                                    }
                                )
                            elif file.endswith(".txt"):
                                processed_files.append(
                                    {
                                        "path": file_path.replace(path, f"{name}/"),
                                        "type": "text",
                                        "size": size,
                                    }
                                )
                        except:
                            pass

        # Generate insights
        total_files = sum(stats["files"] for stats in data_stats.values())
        total_size = sum(stats["size"] for stats in data_stats.values())

        if available_dirs:
            insights.append(
                f"Found {
                    len(available_dirs)} accessible directories"
            )
            insights.append(f"Total files scanned: {total_files}")
            insights.append(f"Total size: {total_size / (1024**2):.2f} MB")

            # Find largest directory
            largest = max(data_stats.items(), key=lambda x: x[1]["size"])
            insights.append(
                f"Largest data source: {largest[0]} ({largest[1]['size'] / (1024**2):.2f} MB)"
            )

        success = len(available_dirs) > 0

        return {
            "success": success,
            "action": "user_data_processing",
            "available_directories": available_dirs,
            "data_statistics": data_stats,
<<<<<<< HEAD
            # Sample of processed files
            "processed_files": processed_files[:10],
=======
        }
    except Exception as e:
        return {
            "success": False,
            "action": "user_data_processing",
            "error": str(e),
            "available_directories": available_dirs,
        }
