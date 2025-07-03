# Real Protocol: System Monitor
import psutil
import platform
import os


def task():
    """Monitor system resources and health"""
    try:
        # Get system info
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Get process count
        process_count = len(psutil.pids())

        # Check if system is healthy
        is_healthy = cpu_percent < 90 and memory.percent < 90 and disk.percent < 95

        system_info = {
            "platform": platform.system(),
            "platform_version": platform.version()[:50],  # Truncate long versions
            "cpu_cores": psutil.cpu_count(),
            "cpu_percent": cpu_percent,
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "memory_used_percent": memory.percent,
            "disk_total_gb": round(disk.total / (1024**3), 2),
            "disk_used_percent": disk.percent,
            "process_count": process_count,
        }

        # Determine if intervention needed
        warnings = []
        if cpu_percent > 80:
            warnings.append(f"High CPU usage: {cpu_percent}%")
        if memory.percent > 80:
            warnings.append(f"High memory usage: {memory.percent}%")
        if disk.percent > 90:
            warnings.append(f"Low disk space: {disk.percent}% used")

        return {
            "success": is_healthy,
            "action": "system_monitoring",
            "healthy": is_healthy,
            "system_info": system_info,
            "warnings": warnings,
        }

    except Exception as e:
        return {"success": False, "action": "system_monitoring", "error": str(e)}
