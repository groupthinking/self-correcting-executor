# Real Protocol: Log Analyzer
import os
import re
from datetime import datetime
from collections import Counter

def task():
    """Analyze log files for errors, warnings, and patterns"""
    log_dir = os.environ.get('LOG_DIR', '/app/logs')
    
    try:
        total_lines = 0
        error_count = 0
        warning_count = 0
        error_patterns = []
        activity_timeline = Counter()
        
        # Analyze all log files in directory
        for filename in os.listdir(log_dir):
            if filename.endswith('.log'):
                file_path = os.path.join(log_dir, filename)
                
                with open(file_path, 'r') as f:
                    for line in f:
                        total_lines += 1
                        
                        # Check for errors
                        if 'error' in line.lower() or 'exception' in line.lower():
                            error_count += 1
                            error_patterns.append(line.strip()[:100])  # First 100 chars
                            
                        # Check for warnings
                        if 'warning' in line.lower() or 'warn' in line.lower():
                            warning_count += 1
                            
                        # Extract timestamps for activity timeline
                        timestamp_match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}):', line)
                        if timestamp_match:
                            hour = timestamp_match.group(1)
                            activity_timeline[hour] += 1
        
        # Calculate health score
        if total_lines > 0:
            error_rate = error_count / total_lines
            success = error_rate < 0.05  # Less than 5% errors
        else:
            success = True  # No logs is not a failure
            
        # Get top activity hours
        top_hours = activity_timeline.most_common(5)
        
        return {
            'success': success,
            'action': 'log_analysis',
            'total_lines_analyzed': total_lines,
            'error_count': error_count,
            'warning_count': warning_count,
            'error_rate': round(error_rate * 100, 2) if total_lines > 0 else 0,
            'recent_errors': error_patterns[-5:],  # Last 5 errors
            'peak_activity_hours': dict(top_hours),
            'log_directory': log_dir
        }
        
    except Exception as e:
        return {
            'success': False,
            'action': 'log_analysis',
            'error': str(e)
        } 