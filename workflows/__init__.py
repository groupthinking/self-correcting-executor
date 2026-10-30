"""
Workflow template registry and self-correcting wrappers.

This package consolidates workflow patterns from:
- self-correcting-executor (auto-fix orchestration)
- WRKFLW (construction management flows)
- workflows (TypeScript workflow definitions)
"""

from .registry import (
    get_template,
    list_templates,
    match_template_for_intent,
    materialize_workflow,
)

__all__ = [
    "get_template",
    "list_templates",
    "match_template_for_intent",
    "materialize_workflow",
]
