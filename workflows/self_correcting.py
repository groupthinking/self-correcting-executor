from typing import Any, Dict


def apply_self_correction(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enrich a workflow dictionary with retry policies so the orchestration
    engine can self-heal on transient failures.
    """
    if not workflow:
        return workflow

    policy = workflow.get("self_correction", {}).get("policy", {})
    max_attempts = policy.get("max_attempts", 2)

    for step in workflow.get("steps", []):
        retry = step.get("retry_policy") or {}
        retry.setdefault("max_attempts", max_attempts)
        retry.setdefault("strategy", "self_heal")
        step["retry_policy"] = retry

    return workflow
