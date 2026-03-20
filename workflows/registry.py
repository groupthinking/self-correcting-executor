from typing import Any, Dict, List, Optional

from .templates import WORKFLOW_TEMPLATES, WorkflowTemplate, get_template


def list_templates() -> List[WorkflowTemplate]:
    """Return all known workflow templates."""
    return list(WORKFLOW_TEMPLATES.values())


def match_template_for_intent(intent: str) -> Optional[WorkflowTemplate]:
    """Simple tag-based matcher that maps a natural language intent to a template."""
    intent_lower = intent.lower()

    for template in WORKFLOW_TEMPLATES.values():
        for tag in template.tags:
            if tag.lower() in intent_lower:
                return template
    return None


def materialize_workflow(
    intent: str, preferred_template: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Render the best matching workflow with self-correcting metadata."""
    template = None

    if preferred_template:
        template = get_template(preferred_template)

    if not template:
        template = match_template_for_intent(intent)

    if not template:
        return None

    return template.materialize(intent)
