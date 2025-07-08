# Python Version Compatibility Fix

## The Issue
You're seeing:
```
ERROR: Ignored the following versions that require a different python version: 
0.0.3 Requires-Python >=3.11; 0.0.4 Requires-Python >=3.11...
```

## Why This Happens
- You have **Python 3.13.3** (which is great! ✅)
- The error is from pip trying older package versions
- Some packages might not have metadata for Python 3.13 yet

## Quick Fix

### Option 1: Force Install Current Versions
```bash
# Install with --force-reinstall to use latest versions
pip install --force-reinstall -r requirements.txt
```

### Option 2: Use Constraints File
Create `constraints.txt`:
```bash
# Ensure we use versions compatible with Python 3.13
mcp>=1.3.3
dwave-ocean-sdk>=6.0
aiohttp>=3.9
pyyaml>=6.0
```

Then install:
```bash
pip install -r requirements.txt -c constraints.txt
```

### Option 3: Update pip First
```bash
# Update pip to latest version
python3 -m pip install --upgrade pip

# Then install requirements
pip install -r requirements.txt
```

## If You Still Get Errors

### For MCP specifically:
```bash
# Install MCP directly from source if needed
pip install git+https://github.com/modelcontextprotocol/python-sdk.git
```

### For D-Wave:
```bash
# D-Wave supports Python 3.13
pip install dwave-ocean-sdk --upgrade
```

## Complete Installation Command
```bash
# This should work for Python 3.13.3:
python3 -m pip install --upgrade pip
pip install --upgrade setuptools wheel
pip install -r requirements.txt --upgrade
```

## Verify Installation
```python
# Test imports
python3 -c "
import mcp
import dwave
import aiohttp
print('✅ All imports successful!')
print(f'MCP version: {mcp.__version__ if hasattr(mcp, \"__version__\") else \"OK\"}')
"
```

## Alternative: Use Python 3.11 (if needed)
If you must use Python 3.11:
```bash
# Using pyenv
pyenv install 3.11.9
pyenv local 3.11.9

# Or using conda
conda create -n myenv python=3.11
conda activate myenv
```

But **Python 3.13.3 should work fine** - the error is just pip being cautious!