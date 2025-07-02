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
    Safely parses a JSON string into a Python dictionary.
    
    Attempts to decode the provided JSON string. Returns the resulting dictionary if parsing is successful, or None if the input is invalid or cannot be parsed.
    
    Args:
        json_string (str): The JSON-formatted string to parse.
    
    Returns:
        Optional[Dict[str, Any]]: The parsed dictionary if successful, or None if parsing fails.
    
    Example:
        >>> safe_json_parse('{"foo": 123, "bar": "baz"}')
        {'foo': 123, 'bar': 'baz'}
        >>> safe_json_parse('not a json')
        None
    """
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return None


def safe_json_dumps(data: Any, indent: int = 2) -> str:
    """
    Serializes Python data to a JSON-formatted string with indentation.
    
    If the data cannot be serialized (e.g., due to non-serializable objects), returns an empty string. Non-serializable objects are converted to strings using `str()`.
    
    Args:
        data (Any): The Python object to serialize.
        indent (int, optional): Number of spaces to use for indentation in the output JSON string. Defaults to 2.
    
    Returns:
        str: The JSON-formatted string, or an empty string if serialization fails.
    
    Example:
        >>> safe_json_dumps({'a': 1, 'b': 2})
        '{\\n  "a": 1,\\n  "b": 2\\n}'
    """
    try:
        return json.dumps(data, indent=indent, default=str)
    except (TypeError, ValueError):
        return ""


def generate_hash(data: Union[str, bytes]) -> str:
    """
    Generate a SHA256 hash of the input string or bytes.
    
    Args:
        data (str or bytes): Input data to hash. Strings are encoded as UTF-8 before hashing.
    
    Returns:
        str: Hexadecimal SHA256 hash of the input.
    
    Example:
        >>> generate_hash("hello")
        '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
    """
    Execute a no-argument function with retries and exponential backoff.
    
    Retries the given function up to `max_retries` times, doubling the delay after each failure starting from `base_delay` seconds. If all attempts fail, raises the last encountered exception.
    
    Args:
        func: A callable that takes no arguments and returns any value.
        max_retries: Maximum number of attempts (default is 3).
        base_delay: Initial delay in seconds before retrying (default is 1.0).
    
    Returns:
        The result returned by `func` if successful.
    
    Raises:
        Exception: The last exception raised by `func` if all retries fail.
    
    Example:
        >>> def flaky():
        ...     import random
        ...     if random.random() < 0.7:
        ...         raise ValueError("Try again!")
        ...     return "Success"
        >>> retry_with_backoff(flaky, max_retries=5, base_delay=0.5)
        'Success'
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
    Recursively flattens a nested dictionary into a single-level dictionary with dot-separated keys.
    
    Args:
        data (Dict[str, Any]): The dictionary to flatten.
        prefix (str, optional): Prefix to prepend to each key in the flattened dictionary. Defaults to "".
    
    Returns:
        Dict[str, Any]: A new dictionary with all nested keys flattened and joined by dots.
    
    Example:
        >>> flatten_dict({'a': {'b': 1, 'c': 2}, 'd': 3})
        {'a.b': 1, 'a.c': 2, 'd': 3}
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
    Ensure that a directory exists at the specified path, creating it and any necessary parent directories if needed.
    
    Args:
        path (str or Path): The directory path to check or create.
    
    Returns:
        Path: A Path object representing the ensured directory.
    
    Example:
        >>> ensure_directory_exists("/tmp/mydir")
        PosixPath('/tmp/mydir')
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def sanitize_filename(filename: str) -> str:
    """
    Sanitizes a filename by replacing invalid characters with underscores and trimming leading or trailing spaces and dots.
    
    Parameters:
        filename (str): The filename to sanitize.
    
    Returns:
        str: The sanitized filename. If the result is empty or consists only of underscores, returns "unnamed".
    
    Example:
        >>> sanitize_filename('  my<file>:name?.txt  ')
        'my_file_name.txt'
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
    Recursively merge two dictionaries, combining nested dictionaries and prioritizing values from the second dictionary.
    
    Args:
        dict1 (Dict[str, Any]): The base dictionary to merge into.
        dict2 (Dict[str, Any]): The dictionary whose values take precedence. If both dictionaries have a key with dictionary values, those are merged recursively.
    
    Returns:
        Dict[str, Any]: A new dictionary containing the merged keys and values.
    
    Example:
        >>> merge_dicts({'a': 1, 'b': {'x': 2}}, {'b': {'y': 3}, 'c': 4})
        {'a': 1, 'b': {'x': 2, 'y': 3}, 'c': 4}
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
    Split a list into consecutive sublists (chunks) of a specified maximum size.
    
    Args:
        data (List[Any]): The list to split into chunks.
        chunk_size (int): The maximum number of elements per chunk. Must be greater than zero.
    
    Returns:
        List[List[Any]]: A list of sublists, each containing up to `chunk_size` elements. The final chunk may have fewer elements if the list size is not a multiple of `chunk_size`.
    
    Example:
        >>> chunk_list([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
    """
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def format_duration(seconds: float) -> str:
    """
    Convert a duration in seconds to a human-readable string formatted as seconds, minutes, or hours.
    
    Parameters:
        seconds (float): Duration in seconds.
    
    Returns:
        str: The duration formatted as a string. Uses seconds with two decimals if less than 60, minutes with one decimal if less than 3600, or hours with one decimal otherwise.
    
    Examples:
        >>> format_duration(45)
        '45.00s'
        >>> format_duration(125)
        '2.1m'
        >>> format_duration(5400)
        '1.5h'
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"