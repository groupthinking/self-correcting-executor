from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy

from .self_correcting import apply_self_correction
from typing import Any, Dict, List, Optional
from pathlib import Path
import json

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml is optional but expected
    yaml = None


@dataclass
class WorkflowTemplate:
    """Declarative workflow template with optional resilience metadata."""

    name: str
    source_repo: str
    summary: str
    tags: List[str]
    steps: List[Dict[str, Any]]
    resilience: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    required_inputs: List[str] = field(default_factory=list)

    def materialize(self, intent: str) -> Dict[str, Any]:
        """Render a workflow dictionary ready for execution."""
        workflow = {
            "template": self.name,
            "source_repo": self.source_repo,
            "intent": intent,
            "steps": [],
            "self_correction": {
                "enabled": self.resilience.get("enabled", True) is True,
                "policy": {
                    "max_attempts": self.resilience.get("max_attempts", 2),
                    "capture_telemetry": True,
                    "fallback_protocol": self.resilience.get(
                        "fallback_protocol", "log_analyzer"
                    ),
                },
            },
            "metadata": deepcopy(self.metadata),
            "required_inputs": list(self.required_inputs),
        }

        workflow["steps"] = deepcopy(self.steps)
        for step in workflow["steps"]:
            step.setdefault("inputs", {})
        return apply_self_correction(workflow)


def _wrkflw_templates() -> WorkflowTemplate:
    """Workflows imported from the WRKFLW repository."""
    return WorkflowTemplate(
        name="construction_daily_log",
        required_inputs=["field_capture"],
        source_repo="WRKFLW",
        summary=(
            "ConstructionOnline daily log with RFI routing and AI summary; "
            "ported from WRKFLW backend + PRD flows."
        ),
        tags=[
            "construction",
            "daily log",
            "rfi",
            "wrkflw",
            "field capture",
        ],
        steps=[
            {
                "type": "protocol",
                "name": "data_processor",
                "description": "Normalize field updates and attachments",
                "inputs": {"source": "field_capture"},
                "outputs": ["normalized_data"],
            },
            {
                "type": "analyzer",
                "name": "pattern_detector",
                "description": "Detect risks and schedule slippage from logs",
                "inputs": {"data": "normalized_data"},
                "outputs": ["patterns", "insights"],
            },
            {
                "type": "protocol",
                "name": "log_analyzer",
                "description": "Summarize daily activity for PM review",
                "inputs": {"source": "insights"},
                "outputs": ["draft_summary"],
            },
            {
                "type": "agent",
                "name": "code_generator",
                "description": "Generate API artifacts for RFI + task endpoints",
                "inputs": {
                    "intent": "Generate RFI + task endpoints",
                    "context": {
                        "endpoint_name": "rfi",
                        "description": "Create and approve RFIs with audit trail",
                        "parameters": {"project_id": "int", "status": "str"},
                    },
                },
                "outputs": ["generated_code", "instructions"],
            },
        ],
        resilience={
            "max_attempts": 2,
            "fallback_protocol": "log_analyzer",
            "origin": "https://github.com/groupthinking/WRKFLW",
        },
        metadata={
            "imported_from": "WRKFLW (archived)",
            "patterns": [
                "Project creation + membership",
                "Daily log capture",
                "RFI approval",
                "AI daily summary",
            ],
        },
    )


def _typescript_workflows_template() -> WorkflowTemplate:
    """Templates based on the workflows TypeScript definitions repository."""
    return WorkflowTemplate(
        name="typescript_service_pipeline",
        required_inputs=["service_health_url", "cache_config"],
        source_repo="workflows",
        summary=(
            "TypeScript workflow definitions: service scaffold, CI verification, "
            "and deployment hooks, converted to orchestration steps."
        ),
        tags=["typescript", "api", "ci", "deployment", "workflows"],
        steps=[
            {
                "type": "agent",
                "name": "code_generator",
                "description": "Scaffold TypeScript service endpoints",
                "inputs": {
                    "intent": "Generate TypeScript API",
                    "context": {
                        "endpoint_name": "service",
                        "description": "Create service scaffold with health + CRUD",
                        "parameters": {"id": "string", "payload": "object"},
                    },
                },
                "outputs": ["generated_code"],
            },
            {
                "type": "protocol",
                "name": "file_validator",
                "description": "Validate generated files before deployment",
                "inputs": {"target": "generated_code"},
                "outputs": ["validation_result"],
            },
            {
                "type": "protocol",
                "name": "api_health_checker",
                "description": "Run health checks mirroring CI workflow definitions",
                "inputs": {"service": "generated_code", "url": "service_health_url"},
                "outputs": ["health_report"],
            },
            {
                "type": "protocol",
                "name": "redis_cache_manager",
                "description": "Warm cache with schema + health report metadata",
                "inputs": {
                    "service": "generated_code",
                    "health_report": "health_report",
                    "config": "cache_config",
                },
                "outputs": ["deployment_artifacts"],
            },
        ],
        resilience={
            "max_attempts": 3,
            "fallback_protocol": "file_validator",
            "origin": "https://github.com/groupthinking/workflows",
        },
        metadata={
            "imported_from": "workflows (archived)",
            "patterns": ["TypeScript API definition", "CI validation", "deployment"],
        },
    )


def _self_correcting_core_template() -> WorkflowTemplate:
    """Native self-correcting executor workflow, preserved as the canonical core."""
    return WorkflowTemplate(
        name="self_correcting_core",
        required_inputs=["user_data"],
        source_repo="self-correcting-executor",
        summary="Default intent-driven pipeline with self-healing and knowledge capture.",
        tags=["self-heal", "executor", "core"],
        steps=[
            {
                "type": "protocol",
                "name": "data_processor",
                "description": "Normalize incoming data and intent context",
                "inputs": {"source": "user_data"},
                "outputs": ["processed_data"],
            },
            {
                "type": "analyzer",
                "name": "pattern_detector",
                "description": "Detect patterns to adjust next actions",
                "inputs": {"data": "processed_data"},
                "outputs": ["patterns", "insights"],
            },
            {
                "type": "protocol",
                "name": "system_monitor",
                "description": "Verify system health to guard subsequent steps",
                "inputs": {"scope": "orchestration"},
                "outputs": ["system_status"],
            },
        ],
        resilience={
            "max_attempts": 2,
            "fallback_protocol": "system_monitor",
            "origin": "self-correcting-executor",
        },
        metadata={
            "imported_from": "self-correcting-executor",
            "patterns": ["Auto-fix", "A2A negotiation", "Knowledge graph"],
        },
    )


WORKFLOW_TEMPLATES: Dict[str, WorkflowTemplate] = {
    "construction_daily_log": _wrkflw_templates(),
    "typescript_service_pipeline": _typescript_workflows_template(),
    "self_correcting_core": _self_correcting_core_template(),
}


def _load_external_templates_from_data_dir(
    data_dir: Optional[Path] = None,
) -> Dict[str, WorkflowTemplate]:
    """Load templates from workflows/data (yaml or json)."""
    data_dir = data_dir if data_dir is not None else Path(__file__).parent / "data"
    templates: Dict[str, WorkflowTemplate] = {}

    if not data_dir.exists():
        return templates

    for file in sorted(data_dir.glob("*.*")):
        try:
            content: Dict[str, Any] = {}
            if file.suffix.lower() in [".yaml", ".yml"] and yaml:
                content = yaml.safe_load(file.read_text()) or {}
            elif file.suffix.lower() == ".json":
                content = json.loads(file.read_text())
            else:
                continue

            if not isinstance(content, dict) or not isinstance(
                content.get("name"), str
            ):
                continue
            if not content["name"].strip():
                continue
            steps = content.get("steps")
            if (
                not isinstance(steps, list)
                or not steps
                or not all(
                    isinstance(step, dict)
                    and step.get("type") in {"protocol", "analyzer", "agent"}
                    and isinstance(step.get("name"), str)
                    and bool(step["name"])
                    and isinstance(step.get("inputs", {}), dict)
                    and isinstance(step.get("retry_policy", {}), dict)
                    for step in steps
                )
            ):
                continue
            if not isinstance(content.get("resilience", {}), dict):
                continue
            if not isinstance(content.get("metadata", {}), dict):
                continue
            if not isinstance(content.get("tags", []), list) or not all(
                isinstance(tag, str) for tag in content.get("tags", [])
            ):
                continue

            required_inputs = content.get("required_inputs", [])
            if not isinstance(required_inputs, list) or not all(
                isinstance(name, str) and name for name in required_inputs
            ):
                continue

            templates[content["name"]] = WorkflowTemplate(
                name=content["name"],
                source_repo=content.get("source_repo", "external"),
                summary=content.get("summary", ""),
                tags=content.get("tags", []),
                steps=content.get("steps", []),
                resilience=content.get("resilience", {}),
                metadata=content.get("metadata", {}),
                required_inputs=content.get("required_inputs", []),
            )
        except Exception:
            # Skip malformed entries; keep core templates intact
            continue

    return templates


# Merge external templates (if present)
WORKFLOW_TEMPLATES.update(_load_external_templates_from_data_dir())


def list_template_names() -> List[str]:
    return list(WORKFLOW_TEMPLATES.keys())


def get_template(name: str) -> Optional[WorkflowTemplate]:
    return WORKFLOW_TEMPLATES.get(name)
