"""
Utility helpers used by tests to verify Dependabot config wiring.
The goal is to keep this lightweight and side-effect free.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional


def load_config(path: Optional[str] = None) -> Dict:
    """Load dependabot configuration if it exists."""
    target = Path(path or ".github/dependabot.yml")
    if not target.exists():
        return {}

    try:
        import yaml  # type: ignore
    except Exception:
        return {}

    try:
        return yaml.safe_load(target.read_text()) or {}
    except Exception:
        return {}


def summarize_config(config: Optional[Dict] = None) -> Dict:
    """Provide a minimal summary for assertions."""
    cfg = config or load_config()
    if not cfg:
        return {"alerts": 0, "updates": []}

    updates = cfg.get("updates", [])
    return {
        "alerts": len(updates),
        "updates": [u.get("package-ecosystem") for u in updates],
    }


def main():
    config = load_config()
    summary = summarize_config(config)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
