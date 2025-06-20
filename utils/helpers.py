"""
Helper utility functions for the self-correcting MCP runtime
"""

import json
import time
import hashlib
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


def safe_json_parse(json_string: str) -> Optional[Dict[str, Any]]:
    """
    Safely parse JSON string, returning None if parsing fails.
    
    Args:
        json_string: The JSON string to parse
        
    Returns:
        Parsed JSON as dict or None if parsing fails
    """
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return None


def safe_json_dumps(data: Any, indent: int = 2) -> str:
    """
    Safely serialize data to JSON string.
    
    Args:
        data: Data to serialize
        indent: Indentation level for pretty printing
        
    Returns:
        JSON string or empty string if serialization fails
    """
    try:
        return json.dumps(data, indent=indent, default=str)
    except (TypeError, ValueError):
        return ""


def generate_hash(data: Union[str, bytes]) -> str:
    """
    Generate SHA256 hash of input data.
    
    Args:
        data: String or bytes to hash
        
    Returns:
        Hexadecimal hash string
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
    """
    Retry function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        
    Returns:
        Function result or raises last exception
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)


def flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Flatten nested dictionary with dot notation.
    
    Args:
        data: Dictionary to flatten
        prefix: Prefix for keys
        
    Returns:
        Flattened dictionary
    """
    result = {}
    for key, value in data.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_dict(value, new_key))
        else:
            result[new_key] = value
    return result


def ensure_directory_exists(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    import re
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    # Ensure it's not empty or only underscores
    if not sanitized or sanitized.replace('_', '').strip() == '':
        return "unnamed"
    return sanitized


def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def chunk_list(data: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size.
    
    Args:
        data: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"