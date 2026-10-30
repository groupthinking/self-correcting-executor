from typing import Any, Dict


def positive_attempts(value: Any, default: int = 1) -> int:
    """Invalid retry counts must never silently skip a workflow step."""
    return value if type(value) is int and value > 0 else default


def apply_self_correction(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """Apply retry defaults only when the workflow explicitly enables them."""
    if not workflow:
        return workflow
    correction = workflow.get("self_correction") or {}
    if correction.get("enabled") is not True:
        return workflow

    policy = correction.get("policy") or {}
    max_attempts = positive_attempts(policy.get("max_attempts", 2))
    for step in workflow.get("steps", []):
        retry = dict(step.get("retry_policy") or {})
        retry["max_attempts"] = positive_attempts(
            retry.get("max_attempts", max_attempts)
        )
        retry.setdefault("strategy", "self_heal")
        step["retry_policy"] = retry
    return workflow
