"""Executable WRKFLW PRD lifecycle patterns, with host-owned external adapters.

This is a local orchestration adapter, not the archived SaaS/mobile UI. Host
callables authenticate/authorize every action and implement storage, delivery,
embeddings, vector search, generation and PDF rendering. No provider defaults,
credentials, recipients, risk rules or successful delivery are fabricated.
"""

from base64 import b64encode
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import math
import secrets
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .correction import invoke, run_correcting
from .field_capture import normalize_field_capture

ROLES = {"admin", "project_manager", "foreman", "subcontractor"}
MANAGERS = {"admin", "project_manager"}
SECTIONS = (
    "overview",
    "issues",
    "photos_of_note",
    "tasks_due",
    "risk_flags",
    "recommendations",
)
TASK_NEXT = {"not_started": "in_progress", "in_progress": "complete"}


def _digest(value):
    return sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _required_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value


def _audit(value):
    if isinstance(value, bytes):
        return {"byte_count": len(value), "sha256": sha256(value).hexdigest()}
    if isinstance(value, dict):
        return {
            key: (
                "[redacted]"
                if key in {"token", "signed_url", "url", "content", "data"}
                else _audit(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_audit(item) for item in value]
    return value


class ConstructionLifecycle:
    """Adapters receive one payload mapping and return validated receipts/data.

    Required authentication: authenticate(credentials) -> {id,email,role};
    authorize(principal, action, project_or_none) -> True. Project membership
    roles come from accepted invitations, never a caller-supplied role.
    External adapters must honor payload idempotency_key to reconcile retries.
    A failed provider call stays failed/pending until an observed retry succeeds.
    """

    def __init__(
        self,
        store,
        *,
        authenticate=None,
        authorize=None,
        adapters=None,
        repairs=None,
        attempts=2,
        clock=None,
    ):
        self.store, self.authenticate, self.authorize = store, authenticate, authorize
        self.adapters, self.repairs = adapters or {}, repairs or {}
        self.attempts = attempts
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def _now(self):
        value = self.clock()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError("clock must return an aware datetime")
        return value.astimezone(timezone.utc)

    def _get(self, kind, key):
        return self.store.get(f"construction_{kind}", key)

    def _put(self, kind, value):
        return self.store.put(f"construction_{kind}", value["id"], value)

    def _event(self, project, actor, action, resource, **metadata):
        self.store.event(
            project,
            {
                "type": "construction",
                "actor_id": actor,
                "action": action,
                "resource_id": resource,
                "timestamp": self._now().isoformat(),
                **_audit(metadata),
            },
        )

    def _provider(self, name):
        provider = self.adapters.get(name)
        if not callable(provider):
            raise ValueError(f"Configure construction adapter: {name}")
        return provider

    async def _identity(self, credentials):
        if not callable(self.authenticate) or not callable(self.authorize):
            raise PermissionError(
                "Authentication and authorization adapters are required"
            )
        person = await invoke(self.authenticate, credentials)
        if not isinstance(person, dict) or not person.get("id"):
            raise PermissionError("Authenticated identity required")
        return deepcopy(person)

    async def _access(self, credentials, action, project_id=None, roles=None):
        person = await self._identity(credentials)
        project = self._get("projects", project_id) if project_id else None
        role = project["members"].get(person["id"]) if project else person.get("role")
        if role not in ROLES or (roles and role not in roles):
            raise PermissionError("Project role does not permit this action")
        if (
            await invoke(self.authorize, deepcopy(person), action, deepcopy(project))
            is not True
        ):
            raise PermissionError("Host authorization denied")
        return person, project

    async def _call(self, name, payload):
        provider = self._provider(name)
        original_project = payload.get("project_id")
        # These fields came from the authenticated action and its saved source.
        # Protect the whole payload, including notification bodies and object
        # keys, instead of an incomplete field-name list. Repairs can supply new
        # technical settings (for example temperature), not replace the action.
        protected = deepcopy(payload)

        async def operation(inputs):
            if any(
                key not in inputs or inputs[key] != value
                for key, value in protected.items()
            ):
                raise PermissionError(
                    "Repair cannot change project scope or source evidence"
                )
            return await invoke(provider, inputs)

        def observe(item):
            entry = _audit(item)
            if name == "invitation_link" and "output" in entry:
                entry["output"] = "[redacted invitation link]"
            self.store.event(
                original_project or "construction",
                {"type": "construction_adapter", "adapter": name, **entry},
            )

        result = await run_correcting(
            operation,
            payload,
            attempts=self.attempts if name in self.repairs else 1,
            repair=self.repairs.get(name),
            observe=observe,
        )
        for attempt in result["attempts"]:
            if "correction" in attempt:
                self.store.event(
                    original_project or "construction",
                    {
                        "type": "construction_repair",
                        "adapter": name,
                        "attempt": attempt["attempt"],
                        **_audit(attempt["correction"]),
                    },
                )
        if result["status"] != "completed":
            raise RuntimeError(result["error"])
        return result["output"]

    async def _notify(self, project_id, event_id, recipients, kind, data):
        if not recipients:
            raise ValueError("Notification requires explicit recipients")
        try:
            receipt = self._get("notifications", event_id)
            if receipt.get("delivered") is True:
                return receipt
        except KeyError:
            pass
        payload = {
            "project_id": project_id,
            "idempotency_key": event_id,
            "recipients": recipients,
            "kind": kind,
            "data": deepcopy(data),
        }
        # Keep the exact authorized recipients/content for explicit recovery. This
        # private local outbox can contain invitation links; never export its body
        # in telemetry or pending-notification listings.
        pending = {
            "id": event_id,
            "project_id": project_id,
            "delivered": False,
            "payload": payload,
            "at": self._now().isoformat(),
        }
        self._put("notifications", pending)
        try:
            receipt = await self._call("notify", payload)
        except Exception as error:
            pending["error"] = str(error)
            self._put("notifications", pending)
            raise RuntimeError(
                f"Notification pending: {event_id}; retry_notification can resume it ({error})"
            ) from error
        if (
            not isinstance(receipt, dict)
            or receipt.get("delivered") is not True
            or not receipt.get("receipt")
        ):
            raise RuntimeError("Notification adapter did not confirm delivery")
        return self._put(
            "notifications",
            {
                "id": event_id,
                "project_id": project_id,
                "delivered": True,
                "receipt": receipt["receipt"],
                "kind": kind,
                "recipients": recipients,
                "at": self._now().isoformat(),
            },
        )

    async def pending_notifications(self, credentials, project_id):
        await self._access(credentials, "pending_notifications", project_id, MANAGERS)
        return [
            {"id": item["id"], "project_id": project_id, "delivered": False}
            for item in self.store.all("construction_notifications")
            if item["project_id"] == project_id and item.get("delivered") is not True
        ]

    async def retry_notification(self, credentials, notification_id):
        item = self._get("notifications", notification_id)
        await self._access(
            credentials, "retry_notification", item["project_id"], MANAGERS
        )
        if item.get("delivered") is True:
            return item
        payload = item["payload"]
        return await self._notify(
            payload["project_id"],
            notification_id,
            payload["recipients"],
            payload["kind"],
            payload["data"],
        )

    @staticmethod
    def _managers(project):
        return [key for key, role in project["members"].items() if role in MANAGERS]

    async def list_timezones(self, credentials):
        """Retain the archived endpoint's seven advertised zones."""
        await self._access(credentials, "list_timezones")
        zones = [
            "UTC",
            "America/New_York",
            "America/Los_Angeles",
            "Europe/London",
            "Europe/Berlin",
            "Asia/Tokyo",
            "Australia/Sydney",
        ]
        for name in zones:
            ZoneInfo(name)
        return {"timezones": zones}

    async def lookup_time(self, credentials, *, timezones):
        """Convert one actual UTC instant using the host IANA timezone database."""
        await self._access(credentials, "lookup_time")
        if (
            not isinstance(timezones, list)
            or not timezones
            or any(not isinstance(name, str) or not name for name in timezones)
        ):
            raise ValueError("Provide explicit nonempty IANA timezones")
        try:
            selected = [(name, ZoneInfo(name)) for name in timezones]
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise ValueError("Invalid timezone requested") from error
        instant = self._now()
        return {
            "times": [
                {
                    "timezone": name,
                    "iso": instant.astimezone(zone).isoformat(),
                    "formatted": instant.astimezone(zone).strftime(
                        "%Y-%m-%d %H:%M:%S %Z"
                    ),
                }
                for name, zone in selected
            ]
        }

    async def create_project(
        self,
        credentials,
        *,
        name,
        location="",
        start_date=None,
        end_date=None,
        baseline_budget=None,
    ):
        actor, _ = await self._access(credentials, "create_project", roles=MANAGERS)
        project = self._put(
            "projects",
            {
                "id": str(uuid4()),
                "name": _required_text(name, "name"),
                "location": location,
                "start_date": start_date,
                "end_date": end_date,
                "baseline_budget": baseline_budget,
                "members": {actor["id"]: actor["role"]},
                "created_at": self._now().isoformat(),
            },
        )
        self._event(project["id"], actor["id"], "project_created", project["id"])
        return project

    async def invite(self, credentials, project_id, *, email, role):
        actor, _ = await self._access(credentials, "invite", project_id, MANAGERS)
        if role not in ROLES:
            raise ValueError("Unknown project role")
        _required_text(email, "email")
        self._provider("notify")
        self._provider("invitation_link")
        token = secrets.token_urlsafe(32)
        token_id = sha256(token.encode()).hexdigest()
        invitation = self._put(
            "invitations",
            {
                "id": token_id,
                "project_id": project_id,
                "email": email.casefold(),
                "role": role,
                "state": "pending",
                "expires_at": (self._now() + timedelta(hours=48)).isoformat(),
                "created_by": actor["id"],
            },
        )
        url = await self._call(
            "invitation_link",
            {
                "project_id": project_id,
                "token": token,
                "expires_at": invitation["expires_at"],
            },
        )
        if not isinstance(url, str) or not url.startswith("https://"):
            raise ValueError("Invitation link must use HTTPS")
        await self._notify(
            project_id,
            f"invite/{token_id}",
            [email],
            "project_invitation",
            {"url": url, "role": role, "expires_at": invitation["expires_at"]},
        )
        self._event(project_id, actor["id"], "invitation_sent", token_id)
        return invitation

    async def accept_invitation(self, credentials, token):
        actor = await self._identity(credentials)
        invitation = self._get("invitations", sha256(token.encode()).hexdigest())
        project = self._get("projects", invitation["project_id"])
        if (
            invitation["state"] != "pending"
            or self._now() >= datetime.fromisoformat(invitation["expires_at"])
            or actor.get("email", "").casefold() != invitation["email"]
        ):
            raise PermissionError(
                "Invitation invalid, expired, used, or belongs to another email"
            )
        if (
            await invoke(self.authorize, actor, "accept_invitation", project)
            is not True
        ):
            raise PermissionError("Host authorization denied")
        project["members"][actor["id"]] = invitation["role"]
        self._put("projects", project)
        invitation.update(
            state="accepted",
            accepted_by=actor["id"],
            accepted_at=self._now().isoformat(),
        )
        self._put("invitations", invitation)
        self._event(project["id"], actor["id"], "invitation_accepted", invitation["id"])
        return project

    async def queue_daily_log(
        self,
        credentials,
        project_id,
        *,
        notes,
        attachments=(),
        weather=None,
        completed_tasks=(),
        time_entries=(),
    ):
        actor, _ = await self._access(
            credentials, "capture_daily_log", project_id, MANAGERS | {"foreman"}
        )
        photos = []
        for item in attachments:
            data = item.get("content")
            if not isinstance(data, bytes):
                raise ValueError("Offline attachments require actual bytes")
            photos.append(
                {
                    "id": str(uuid4()),
                    "filename": _required_text(item.get("filename"), "filename"),
                    "content_type": _required_text(
                        item.get("content_type"), "content_type"
                    ),
                    "size": len(data),
                    "checksum": sha256(data).hexdigest(),
                    "content": b64encode(data).decode(),
                    "caption": item.get("caption", ""),
                }
            )
        queued = self._put(
            "queue",
            {
                "id": str(uuid4()),
                "project_id": project_id,
                "author_id": actor["id"],
                "notes": notes,
                "weather": weather,
                "completed_tasks": list(completed_tasks),
                "time_entries": list(time_entries),
                "attachments": photos,
                "uploaded": {},
                "state": "queued",
                "created_at": self._now().isoformat(),
            },
        )
        self._event(project_id, actor["id"], "daily_log_queued", queued["id"])
        return queued

    async def sync_daily_log(self, credentials, queue_id):
        queued = self._get("queue", queue_id)
        actor, project = await self._access(
            credentials, "sync_daily_log", queued["project_id"], MANAGERS | {"foreman"}
        )
        if actor["id"] != queued["author_id"]:
            raise PermissionError("Sync requires the authenticated capture author")
        if queued["state"] == "synced":
            return self._get("daily_logs", queue_id)
        self._provider("notify")
        for attachment in queued["attachments"]:
            if attachment["id"] in queued["uploaded"]:
                continue
            payload = {
                **attachment,
                "project_id": queued["project_id"],
                "idempotency_key": attachment["id"],
                "author_id": actor["id"],
            }
            if await self._call("validate_upload", payload) is not True:
                raise ValueError(
                    "Attachment rejected by configured type/size/security validation"
                )
            signature = await self._call(
                "sign_upload", {k: v for k, v in payload.items() if k != "content"}
            )
            if (
                not isinstance(signature, dict)
                or not signature.get("object_key")
                or not str(signature.get("signed_url", "")).startswith("https://")
            ):
                raise ValueError("A signed HTTPS upload and object key are required")
            expiry = datetime.fromisoformat(signature.get("expires_at", ""))
            if expiry.tzinfo is None or self._now() >= expiry:
                raise ValueError(
                    "Signed upload URL is expired or lacks an aware expiry"
                )
            receipt = await self._call("upload", {**payload, "signature": signature})
            if (
                not isinstance(receipt, dict)
                or receipt.get("object_key") != signature["object_key"]
                or receipt.get("checksum") != attachment["checksum"]
                or receipt.get("size") != attachment["size"]
            ):
                raise RuntimeError("Upload receipt does not match captured bytes")
            saved = self._put(
                "attachments",
                {
                    "id": attachment["id"],
                    "project_id": queued["project_id"],
                    "object_key": receipt["object_key"],
                    "checksum": receipt["checksum"],
                    "size": receipt["size"],
                    "content_type": attachment["content_type"],
                    "caption": attachment["caption"],
                    "uploaded_by": actor["id"],
                },
            )
            queued["uploaded"][attachment["id"]] = saved
            self._put("queue", queued)
        try:
            log = self._get(
                "daily_logs", queue_id
            )  # Server wins on replay; never overwrite published log.
        except KeyError:
            log = self._put(
                "daily_logs",
                {
                    "id": queue_id,
                    "project_id": queued["project_id"],
                    "author_id": actor["id"],
                    "date": queued["created_at"][:10],
                    "notes": queued["notes"],
                    "weather": queued["weather"],
                    "completed_tasks": queued["completed_tasks"],
                    "time_entries": queued["time_entries"],
                    "attachments": list(queued["uploaded"].values()),
                    "created_at": queued["created_at"],
                    "synced_at": self._now().isoformat(),
                },
            )
            self._event(project["id"], actor["id"], "daily_log_created", queue_id)
        queued["state"] = "awaiting_notification"
        self._put("queue", queued)
        await self._notify(
            project["id"],
            f"daily-log/{queue_id}",
            self._managers(project),
            "daily_log_created",
            {"daily_log_id": queue_id},
        )
        queued["state"] = "synced"
        self._put("queue", queued)
        return log

    async def attachment_link(self, credentials, attachment_id):
        attachment = self._get("attachments", attachment_id)
        await self._access(credentials, "read_attachment", attachment["project_id"])
        return await self._call("read_attachment", deepcopy(attachment))

    async def create_task(
        self,
        credentials,
        project_id,
        *,
        title,
        assignee_id,
        description="",
        due_date=None,
        priority=None,
    ):
        actor, project = await self._access(
            credentials, "create_task", project_id, MANAGERS | {"foreman"}
        )
        if assignee_id not in project["members"]:
            raise ValueError("Assignee must belong to the project")
        self._provider("notify")
        task = self._put(
            "tasks",
            {
                "id": str(uuid4()),
                "project_id": project_id,
                "title": _required_text(title, "title"),
                "description": description,
                "assignee_id": assignee_id,
                "due_date": due_date,
                "priority": priority,
                "status": "not_started",
                "history": [],
                "comments": [],
                "attachments": [],
                "created_at": self._now().isoformat(),
            },
        )
        await self._notify(
            project_id,
            f"task/{task['id']}",
            [assignee_id],
            "task_assigned",
            {"task_id": task["id"]},
        )
        self._event(project_id, actor["id"], "task_created", task["id"])
        return task

    async def update_task(
        self,
        credentials,
        task_id,
        *,
        status,
        comment=None,
        attachments=(),
        notify_pm=False,
    ):
        task = self._get("tasks", task_id)
        actor, project = await self._access(
            credentials, "update_task", task["project_id"]
        )
        if (
            actor["id"] != task["assignee_id"]
            and project["members"][actor["id"]] not in MANAGERS
        ):
            raise PermissionError(
                "Only the assignee or project manager can update the task"
            )
        previous = task["status"]
        if status != previous and TASK_NEXT.get(previous) != status:
            raise ValueError(
                "Task transitions are Not Started -> In Progress -> Complete"
            )
        for key in attachments:
            if self._get("attachments", key)["project_id"] != project["id"]:
                raise PermissionError("Attachment belongs to another project")
        at = self._now().isoformat()
        task["history"].append(
            {"from": previous, "to": status, "actor_id": actor["id"], "at": at}
        )
        if comment is not None:
            task["comments"].append(
                {"text": comment, "actor_id": actor["id"], "at": at}
            )
        task["attachments"].extend(
            key for key in attachments if key not in task["attachments"]
        )
        task.update(status=status, updated_at=at)
        self._put("tasks", task)
        self._event(
            project["id"],
            actor["id"],
            "task_completed" if status == "complete" else "task_updated",
            task_id,
        )
        if status == "complete" and notify_pm:
            await self._notify(
                project["id"],
                f"task-complete/{task_id}",
                self._managers(project),
                "task_completed",
                {"task_id": task_id},
            )
        return task

    async def create_rfi(
        self, credentials, project_id, *, description, due_date=None, attachments=()
    ):
        actor, _ = await self._access(
            credentials, "create_rfi", project_id, MANAGERS | {"foreman"}
        )
        for key in attachments:
            if self._get("attachments", key)["project_id"] != project_id:
                raise PermissionError("Attachment belongs to another project")
        rfi = self._put(
            "rfis",
            {
                "id": str(uuid4()),
                "project_id": project_id,
                "creator_id": actor["id"],
                "description": _required_text(description, "description"),
                "due_date": due_date,
                "attachments": list(attachments),
                "status": "draft",
                "approver_id": None,
                "history": [],
                "created_at": self._now().isoformat(),
            },
        )
        self._event(project_id, actor["id"], "rfi_created", rfi["id"])
        return rfi

    async def revise_rfi(self, credentials, rfi_id, *, description):
        rfi = self._get("rfis", rfi_id)
        actor, project = await self._access(
            credentials, "revise_rfi", rfi["project_id"], MANAGERS | {"foreman"}
        )
        if rfi["status"] != "draft":
            raise ValueError("Only a draft RFI can be revised")
        if (
            actor["id"] != rfi["creator_id"]
            and project["members"][actor["id"]] not in MANAGERS
        ):
            raise PermissionError(
                "Only the creator or project manager can revise the draft"
            )
        before = rfi["description"]
        rfi["description"] = _required_text(description, "description")
        rfi["history"].append(
            {
                "from": "draft",
                "to": "draft",
                "actor_id": actor["id"],
                "at": self._now().isoformat(),
                "before": before,
                "after": description,
            }
        )
        self._event(project["id"], actor["id"], "rfi_revised", rfi_id)
        return self._put("rfis", rfi)

    async def rfi_transition(
        self, credentials, rfi_id, *, action, approver_id=None, comment=None
    ):
        rfi = self._get("rfis", rfi_id)
        actor, project = await self._access(
            credentials, "rfi_review", rfi["project_id"], MANAGERS
        )
        self._provider("notify")
        old = rfi["status"]
        if action == "submit" and old == "draft":
            if (
                approver_id in {rfi["creator_id"], actor["id"]}
                or project["members"].get(approver_id) not in MANAGERS
            ):
                raise ValueError("Approver must be another PM or Admin")
            rfi.update(approver_id=approver_id, status="pending_approval")
        elif action in {"approve", "request_changes"} and old == "pending_approval":
            if actor["id"] != rfi["approver_id"]:
                raise PermissionError("Only the selected approver can review this RFI")
            rfi["status"] = "open" if action == "approve" else "draft"
        elif action == "close" and old == "open":
            rfi["status"] = "closed"
        else:
            raise ValueError("Invalid RFI lifecycle transition")
        rfi["history"].append(
            {
                "from": old,
                "to": rfi["status"],
                "actor_id": actor["id"],
                "at": self._now().isoformat(),
                "comment": comment,
            }
        )
        self._put("rfis", rfi)
        recipients = [rfi["approver_id"]] if action == "submit" else [rfi["creator_id"]]
        await self._notify(
            project["id"],
            f"rfi/{rfi_id}/{len(rfi['history'])}",
            recipients,
            f"rfi_{action}",
            {"rfi_id": rfi_id, "status": rfi["status"]},
        )
        self._event(project["id"], actor["id"], f"rfi_{action}", rfi_id)
        return rfi

    @staticmethod
    def _vectors(result, ids):
        if (
            not isinstance(result, dict)
            or not result.get("model")
            or not isinstance(result.get("vectors"), dict)
        ):
            raise ValueError(
                "Embedding adapter requires model and vectors keyed by source ID"
            )
        if set(result["vectors"]) != set(ids):
            raise ValueError("Embedding IDs do not match source records")
        dimensions = set()
        for vector in result["vectors"].values():
            if (
                not isinstance(vector, list)
                or not vector
                or any(
                    type(v) not in (int, float) or not math.isfinite(v) for v in vector
                )
            ):
                raise ValueError("Embedding vectors must contain finite numbers")
            dimensions.add(len(vector))
        if len(dimensions) != 1:
            raise ValueError("Embedding dimensions must agree")
        return result

    async def draft_summary(
        self, credentials, project_id, *, query, prompt_version, lookback_hours=24
    ):
        actor, project = await self._access(
            credentials, "draft_summary", project_id, MANAGERS
        )
        for name in ("retrieve", "embed", "index", "search", "generate", "notify"):
            self._provider(name)
        if type(lookback_hours) is not int or not 24 <= lookback_hours <= 72:
            raise ValueError(
                "Summary retrieval requires a 24-72 hour window from the source PRD"
            )
        until = self._now()
        since = until - timedelta(hours=lookback_hours)
        window = {
            "project_id": project_id,
            "since": since.isoformat(),
            "until": until.isoformat(),
        }
        records = await self._call("retrieve", window)
        capture = normalize_field_capture(
            {
                **records,
                "project_id": project_id,
                "report_date": until.date().isoformat(),
            }
        )
        for item in capture["sources"]:
            record = item["record"]
            if record.get("project_id", record.get("projectId")) != project_id:
                raise PermissionError(
                    "Retrieved records require explicit matching project scope"
                )
            stamp = record.get("updated_at") or record.get("created_at")
            try:
                observed = datetime.fromisoformat(stamp)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Retrieved records require an aware source timestamp"
                ) from error
            if observed.tzinfo is None or not since <= observed <= until:
                raise ValueError(
                    "Retrieved record falls outside the requested time window"
                )
        sources = {item["source_id"]: item for item in capture["sources"]}
        documents = [
            {
                "id": key,
                "text": json.dumps(item["record"], sort_keys=True),
                "project_id": project_id,
            }
            for key, item in sources.items()
        ]
        embedded = self._vectors(
            await self._call("embed", {**window, "documents": documents}), sources
        )
        indexed = await self._call(
            "index", {**window, "documents": documents, **embedded}
        )
        if (
            not isinstance(indexed, dict)
            or set(indexed.get("ids", {})) != set(sources)
            or not all(isinstance(key, str) and key for key in indexed["ids"].values())
            or len(set(indexed["ids"].values())) != len(sources)
        ):
            raise ValueError("Vector index must return stored IDs for each source")
        query_embedded = self._vectors(
            await self._call(
                "embed",
                {
                    **window,
                    "documents": [
                        {
                            "id": "query",
                            "text": _required_text(query, "query"),
                            "project_id": project_id,
                        }
                    ],
                },
            ),
            ["query"],
        )
        if query_embedded["model"] != embedded["model"] or len(
            query_embedded["vectors"]["query"]
        ) != len(next(iter(embedded["vectors"].values()))):
            raise ValueError("Query and document embedding models/dimensions differ")
        matches = await self._call(
            "search",
            {
                **window,
                "vector": query_embedded["vectors"]["query"],
                "model": embedded["model"],
            },
        )
        if not isinstance(matches, list) or not matches:
            raise ValueError("Configured vector search returned no sources")
        for match in matches:
            if (
                match.get("project_id") != project_id
                or match.get("source_id") not in sources
                or match.get("id") != indexed["ids"][match["source_id"]]
            ):
                raise PermissionError(
                    "Vector retrieval provenance is missing or outside the project"
                )
        selected = {
            match["source_id"]: sources[match["source_id"]] for match in matches
        }
        generated = await self._call(
            "generate",
            {
                **window,
                "sources": list(selected.values()),
                "retrieval": matches,
                "sections": list(SECTIONS),
                "prompt_version": _required_text(prompt_version, "prompt_version"),
            },
        )
        if (
            not isinstance(generated, dict)
            or not generated.get("model")
            or set(generated.get("content", {})) != set(SECTIONS)
            or any(not isinstance(v, str) for v in generated["content"].values())
            or not isinstance(generated.get("citations"), list)
            or not generated["citations"]
            or not set(generated["citations"]) <= set(selected)
        ):
            raise ValueError(
                "Generated summary needs all source-defined sections and retrieved citations"
            )
        summary = self._put(
            "summaries",
            {
                "id": str(uuid4()),
                **window,
                "status": "draft",
                "content": generated["content"],
                "model": generated["model"],
                "prompt_version": prompt_version,
                "sources": list(selected.values()),
                "retrieval": matches,
                "embedding_model": embedded["model"],
                "citations": generated["citations"],
                "generated_at": self._now().isoformat(),
                "created_by": actor["id"],
            },
        )
        await self._notify(
            project_id,
            f"summary/{summary['id']}",
            self._managers(project),
            "summary_draft",
            {"summary_id": summary["id"]},
        )
        return summary

    async def review_summary(self, credentials, summary_id, *, content=None):
        summary = self._get("summaries", summary_id)
        actor, _ = await self._access(
            credentials, "review_summary", summary["project_id"], {"project_manager"}
        )
        if summary["status"] not in {"draft", "reviewed"}:
            raise ValueError("Published summaries cannot be edited in place")
        if content is not None:
            if set(content) != set(SECTIONS) or any(
                not isinstance(v, str) for v in content.values()
            ):
                raise ValueError("Preserve all required summary sections")
            summary["content"] = deepcopy(content)
        summary.update(
            status="reviewed",
            review={
                "actor_id": actor["id"],
                "at": self._now().isoformat(),
                "content_sha256": _digest(summary["content"]),
            },
        )
        self._event(summary["project_id"], actor["id"], "summary_reviewed", summary_id)
        return self._put("summaries", summary)

    async def publish_summary(self, credentials, summary_id):
        summary = self._get("summaries", summary_id)
        actor, _ = await self._access(
            credentials, "publish_summary", summary["project_id"], {"project_manager"}
        )
        if summary["status"] == "published":
            return summary
        if summary["status"] != "reviewed" or summary["review"][
            "content_sha256"
        ] != _digest(summary["content"]):
            raise PermissionError(
                "Current summary content requires PM review before publication"
            )
        self._provider("store_pdf")
        pdf = await self._call("render_pdf", deepcopy(summary))
        if not isinstance(pdf, bytes) or not pdf.startswith(b"%PDF-"):
            raise ValueError("Renderer must produce PDF bytes")
        receipt = await self._call(
            "store_pdf",
            {
                "project_id": summary["project_id"],
                "idempotency_key": summary_id,
                "content": b64encode(pdf).decode(),
                "checksum": sha256(pdf).hexdigest(),
                "content_type": "application/pdf",
            },
        )
        if (
            not isinstance(receipt, dict)
            or receipt.get("checksum") != sha256(pdf).hexdigest()
            or not receipt.get("object_key")
        ):
            raise RuntimeError("PDF storage receipt missing or mismatched")
        summary.update(
            status="published",
            pdf=receipt,
            published_by=actor["id"],
            published_at=self._now().isoformat(),
        )
        self._event(summary["project_id"], actor["id"], "summary_published", summary_id)
        return self._put("summaries", summary)
