"""Execute one explicit WRKFLW action through trusted application bindings.

Run: python -m examples.construction_lifecycle --help

The --bindings module must expose configure(store), returning the keyword
arguments for ConstructionLifecycle: authenticate, authorize, adapters, and
optional repairs/clock. No cloud providers or identities are supplied here.
Credentials and invitation tokens come from named environment variables.

Example (input.json contains {"name":"Site A","location":"Chicago"}):
  python -m examples.construction_lifecycle --database ./project.sqlite \
      --bindings my_application.construction_bindings --credentials-env APP_AUTH \
      --action create_project --input input.json

Subsequent --input payloads use the method's named arguments, for example:
  list_timezones: {}
  lookup_time: {"timezones":["UTC","America/Chicago"]}
  invite: {"project_id":"...","email":"...","role":"foreman"}
  queue_daily_log: {"project_id":"...","notes":"Steel arrived",
                   "attachments":[{"path":"site.png","content_type":"image/png"}]}
  sync_daily_log: {"queue_id":"..."}
  create_task: {"project_id":"...","title":"Install steel","assignee_id":"..."}
  update_task: {"task_id":"...","status":"in_progress","comment":"Started"}
  create_rfi: {"project_id":"...","description":"Confirm beam drawing"}
  rfi_transition: {"rfi_id":"...","action":"submit","approver_id":"..."}
  draft_summary: {"project_id":"...","query":"Daily progress","prompt_version":"v1"}
  review_summary / publish_summary: {"summary_id":"..."}

Capture queues actual bytes offline; sync invokes validation, signed upload and
notification adapters. Summary draft invokes actual retrieval/embed/index/search
and generation bindings. Review and PDF publication remain separate PM actions.
This is a local orchestration example, not the archived SaaS/mobile UI.
"""

import argparse
import asyncio
import importlib
import json
import os
from pathlib import Path

from workflows.build_plan import WorkflowStore
from workflows.construction import ConstructionLifecycle
from workflows.correction import invoke

ACTIONS = (
    "list_timezones",
    "lookup_time",
    "create_project",
    "invite",
    "accept_invitation",
    "queue_daily_log",
    "sync_daily_log",
    "attachment_link",
    "create_task",
    "update_task",
    "create_rfi",
    "revise_rfi",
    "rfi_transition",
    "draft_summary",
    "review_summary",
    "publish_summary",
    "pending_notifications",
    "retry_notification",
)


def parser():
    command = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    command.add_argument(
        "--database", required=True, help="Explicit durable SQLite file"
    )
    command.add_argument(
        "--bindings",
        required=True,
        help="Trusted application module exposing configure(store)",
    )
    command.add_argument("--credentials-env", required=True)
    command.add_argument("--action", required=True, choices=ACTIONS)
    command.add_argument(
        "--input",
        required=True,
        type=Path,
        help="JSON object of named action arguments",
    )
    command.add_argument(
        "--token-env",
        help="Required for accept_invitation; token is not read from the JSON file",
    )
    return command


def environment(name):
    value = os.environ.get(name) if name else None
    if not value:
        raise ValueError(f"Required environment variable is unavailable: {name}")
    return value


async def execute(arguments):
    credentials = environment(arguments.credentials_env)
    payload = json.loads(arguments.input.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Action input must be a JSON object")
    if arguments.action == "accept_invitation":
        payload["token"] = environment(arguments.token_env)
    if arguments.action == "queue_daily_log":
        for item in payload.get("attachments", []):
            path = Path(item.pop("path"))
            if not path.is_absolute():
                path = arguments.input.parent / path
            item["content"] = path.read_bytes()
            item.setdefault("filename", path.name)
    module = importlib.import_module(arguments.bindings)
    configure = getattr(module, "configure", None)
    if not callable(configure):
        raise ValueError("Bindings module must expose configure(store)")
    store = WorkflowStore(arguments.database)
    try:
        bindings = await invoke(configure, store)
        if not isinstance(bindings, dict):
            raise ValueError("configure(store) must return explicit lifecycle bindings")
        lifecycle = ConstructionLifecycle(store, **bindings)
        result = await getattr(lifecycle, arguments.action)(credentials, **payload)
        if arguments.action == "queue_daily_log":
            return {key: result[key] for key in ["id", "project_id", "state"]}
        return result
    finally:
        store.close()


def main():
    arguments = parser().parse_args()
    try:
        print(json.dumps(asyncio.run(execute(arguments)), indent=2))
    except (ValueError, PermissionError, RuntimeError, KeyError, ImportError) as error:
        raise SystemExit(f"Construction action failed: {error}") from error


if __name__ == "__main__":
    main()
