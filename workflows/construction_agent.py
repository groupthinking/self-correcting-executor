"""Allowlisted native-engine adapter for authenticated WRKFLW operations."""

from copy import deepcopy
from .correction import invoke

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


class ConstructionWorkflowAgent:
    def __init__(self, lifecycle, credentials, *, confirm=None):
        # The authenticated host supplies this identity; source documents cannot.
        self.lifecycle, self.credentials = lifecycle, credentials
        self.confirm = confirm

    async def execute(self, arguments):
        if set(arguments) != {"operation", "parameters"}:
            raise ValueError("Construction agent expects operation and parameters")
        operation, parameters = arguments["operation"], deepcopy(
            arguments["parameters"]
        )
        if operation not in ACTIONS or not isinstance(parameters, dict):
            raise ValueError("Unsupported construction operation")
        if "credentials" in parameters:
            raise PermissionError("Source inputs cannot replace host identity")
        if operation not in {
            "list_timezones",
            "lookup_time",
            "attachment_link",
            "pending_notifications",
        }:
            if (
                not callable(self.confirm)
                or await invoke(self.confirm, operation, deepcopy(parameters))
                is not True
            ):
                raise PermissionError(
                    "Host confirmation is required before a construction mutation"
                )
        return await getattr(self.lifecycle, operation)(self.credentials, **parameters)
