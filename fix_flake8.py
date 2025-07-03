#!/usr/bin/env python3
"""
Script to fix flake8 line length issues in github_mcp_connector.py
"""

import re


def fix_line_length(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    # Fix specific long lines
    fixes = [
        # Line 115
        (
            r'logger\.error\(\s*f"GitHub API connection failed: {response\.status}"\s*\)',
            'logger.error(\n                f"GitHub API connection failed: {response.status}"\n            )',
        ),
        # Line 146
        (
            r'logger\.error\("Action executed while not connected to GitHub API\."\)',
            'logger.error(\n            "Action executed while not connected to GitHub API."\n        )',
        ),
        # Line 187
        (
            r'f" language:{params\[\'language\'\]}"\s+if params\.get\("language"\)\s+else ""',
            'f" language:{params[\'language\']}"\n                if params.get("language")\n                else ""',
        ),
        # Line 193
        (
            r'"sort": params\.get\("sort", "stars"\),\s+"order": params\.get\("order", "desc"\),',
            '"sort": params.get("sort", "stars"),\n                "order": params.get("order", "desc"),',
        ),
        # Line 200
        (
            r'"per_page": params\.get\("per_page", 10\),',
            '"per_page": params.get("per_page", 10),\n                ',
        ),
        # Line 204
        (
            r"async with self\.session\.get\(url, params=params_dict\) as response:",
            "async with self.session.get(\n                url, params=params_dict\n            ) as response:",
        ),
        # Line 304
        (
            r"url, params={k: v for k, v in params_dict\.items\(\) if v}",
            "url,\n                params={k: v for k, v in params_dict.items() if v}",
        ),
        # Line 328
        (
            r'data\["decoded_content"\] = base64\.b64decode\(\s*data\["content"\]\s*\)\.decode\("utf-8"\)',
            'data["decoded_content"] = base64.b64decode(\n                        data["content"]\n                    ).decode("utf-8")',
        ),
        # Line 465
        (
            r'"error": \(\s*f"Failed to get rate limit: {response\.status}"\s*\),',
            '"error": (\n                            f"Failed to get rate limit: {response.status}"\n                        ),',
        ),
        # Line 479
        (
            r'self\.rate_limit_remaining = rate_limit\["rate_limit"\]\[\s*"remaining"\s*\]',
            'self.rate_limit_remaining = rate_limit["rate_limit"][\n                    "remaining"\n                ]',
        ),
        # Line 491
        (
            r'logger\.warning\(f"Failed to update rate limit: {e}"\)',
            'logger.warning(\n            f"Failed to update rate limit: {e}"\n        )',
        ),
        # Line 557
        (
            r'print\(f"   - Found {search_result\.get\(\'total_count\'\)} repositories"\)',
            "print(\n        f\"   - Found {search_result.get('total_count')} repositories\"\n    )",
        ),
        # Line 580
        (
            r'print\(f"   - Open issues: {repo\.get\(\'open_issues_count\'\)}"\)',
            "print(\n        f\"   - Open issues: {repo.get('open_issues_count')}\"\n    )",
        ),
    ]

    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content)

    with open(file_path, "w") as f:
        f.write(content)

    print(f"Fixed flake8 issues in {file_path}")


if __name__ == "__main__":
    fix_line_length(
        "/workspace/self-correcting-executor/connectors/github_mcp_connector.py"
    )
