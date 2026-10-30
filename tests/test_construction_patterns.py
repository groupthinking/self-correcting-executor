"""Controlled adapters execute real local persistence/signing/retrieval contracts."""

import asyncio
from base64 import b64decode
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import hmac
import json
import math
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from workflows.build_plan import WorkflowStore
from workflows.construction import ConstructionLifecycle, SECTIONS
from workflows.correction import Repair


def run(awaitable):
    return asyncio.run(awaitable)


def pdf_document(text):
    """A real one-page PDF with byte offsets, used by the controlled renderer."""
    text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 10 Tf 20 80 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 100] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(stream)).encode()
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
    ]
    output = b"%PDF-1.4\n"
    offsets = [0]
    for index, body in enumerate(objects, 1):
        offsets.append(len(output))
        output += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"
    start = len(output)
    output += b"xref\n0 6\n0000000000 65535 f \n"
    output += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    return (
        output
        + f"trailer\n<< /Root 1 0 R /Size 6 >>\nstartxref\n{start}\n%%EOF\n".encode()
    )


class ControlledHost:
    def __init__(self, root, store):
        self.root, self.store = root, store
        self.now = datetime(2026, 10, 29, 18, tzinfo=timezone.utc)
        self.people = {
            key: {"id": key, "email": f"{key}@example.test", "role": role}
            for key, role in [
                ("pm", "project_manager"),
                ("pm2", "project_manager"),
                ("foreman", "foreman"),
                ("sub", "subcontractor"),
                ("outsider", "project_manager"),
            ]
        }
        self.notifications, self.uploads, self.retrieval_calls = [], [], []
        self.notification_failure = False
        self.objects = root / "objects"
        self.objects.mkdir()
        self.key = b"controlled-test-signing-key"

    def authenticate(self, credentials):
        return self.people.get(credentials)

    def authorize(self, actor, action, project):
        return True  # Controlled host grant; lifecycle still enforces source roles/membership.

    def notify(self, payload):
        if self.notification_failure:
            return {"status": "failed", "error": "controlled delivery outage"}
        destination = self.root / "received-notifications.jsonl"
        with destination.open("a") as handle:
            handle.write(json.dumps(payload) + "\n")
        self.notifications.append(payload)
        return {
            "delivered": True,
            "receipt": f"{destination.name}:{len(self.notifications)}",
        }

    def sign_upload(self, payload):
        key = payload["idempotency_key"]
        signature = hmac.new(
            self.key, f'{key}:{payload["checksum"]}'.encode(), "sha256"
        ).hexdigest()
        return {
            "object_key": key,
            "signed_url": f"https://controlled-files.test/{key}?signature={signature}",
            "expires_at": (self.now + timedelta(minutes=5)).isoformat(),
        }

    def upload(self, payload):
        signature = payload["signature"]
        expected = hmac.new(
            self.key,
            f'{signature["object_key"]}:{payload["checksum"]}'.encode(),
            "sha256",
        ).hexdigest()
        actual = parse_qs(urlparse(signature["signed_url"]).query)["signature"][0]
        if not hmac.compare_digest(expected, actual):
            raise PermissionError("Invalid controlled signature")
        data = b64decode(payload["content"])
        checksum = sha256(data).hexdigest()
        assert checksum == payload["checksum"]
        destination = self.objects / signature["object_key"]
        destination.write_bytes(data)
        assert destination.read_bytes() == data
        self.uploads.append(signature["object_key"])
        return {
            "object_key": signature["object_key"],
            "checksum": checksum,
            "size": len(data),
        }

    def retrieve(self, payload):
        self.retrieval_calls.append(payload)
        return {
            kind: [
                row
                for row in self.store.all(f"construction_{kind}")
                if row["project_id"] == payload["project_id"]
            ]
            for kind in ["daily_logs", "tasks", "rfis"]
        }

    def embed(self, payload):
        # Deterministic local embedding implementation; no hosted model is claimed.
        return {
            "model": "controlled-term-vector",
            "vectors": {
                doc["id"]: [
                    float(doc["text"].lower().count("steel") + 1),
                    float(len(doc["text"].split()) + 1),
                ]
                for doc in payload["documents"]
            },
        }

    def index(self, payload):
        ids = {}
        for doc in payload["documents"]:
            key = f'{payload["project_id"]}/{doc["id"]}'
            self.store.put(
                "controlled_vectors",
                key,
                {
                    "id": key,
                    "source_id": doc["id"],
                    "project_id": payload["project_id"],
                    "vector": payload["vectors"][doc["id"]],
                },
            )
            ids[doc["id"]] = key
        return {"ids": ids}

    def search(self, payload):
        query = payload["vector"]
        matches = []
        for row in self.store.all("controlled_vectors"):
            if row["project_id"] != payload["project_id"]:
                continue
            score = sum(a * b for a, b in zip(query, row["vector"])) / math.sqrt(
                sum(a * a for a in query) * sum(a * a for a in row["vector"])
            )
            matches.append(
                {k: row[k] for k in ["id", "source_id", "project_id"]}
                | {"score": score}
            )
        return sorted(matches, key=lambda item: item["score"], reverse=True)

    def generate(self, payload):
        return {
            "model": "controlled-test-generator",
            "content": {
                key: f'{key}: {len(payload["sources"])} retrieved source records'
                for key in SECTIONS
            },
            "citations": [row["source_id"] for row in payload["sources"]],
        }

    def store_pdf(self, payload):
        data = b64decode(payload["content"])
        target = self.objects / f'{payload["idempotency_key"]}.pdf'
        target.write_bytes(data)
        return {
            "object_key": target.name,
            "checksum": sha256(target.read_bytes()).hexdigest(),
        }

    def lifecycle(self, **kwargs):
        adapters = {
            "notify": self.notify,
            "invitation_link": lambda p: f'https://app.example.test/join?token={p["token"]}',
            "validate_upload": lambda p: p["content_type"] == "image/png"
            and p["size"] < 10000,
            "sign_upload": self.sign_upload,
            "upload": self.upload,
            "read_attachment": lambda p: f'https://controlled-files.test/authorized/{p["object_key"]}',
            "retrieve": self.retrieve,
            "embed": self.embed,
            "index": self.index,
            "search": self.search,
            "generate": self.generate,
            "render_pdf": lambda p: pdf_document(p["content"]["overview"]),
            "store_pdf": self.store_pdf,
        }
        adapters.update(kwargs.pop("adapters", {}))
        return ConstructionLifecycle(
            self.store,
            authenticate=self.authenticate,
            authorize=self.authorize,
            adapters=adapters,
            clock=lambda: self.now,
            **kwargs,
        )

    async def project(self, lifecycle):
        project = await lifecycle.create_project(
            "pm", name="Actual controlled project", location="Site A"
        )
        for person in ["pm2", "foreman", "sub"]:
            await lifecycle.invite(
                "pm",
                project["id"],
                email=self.people[person]["email"],
                role=self.people[person]["role"],
            )
            token = parse_qs(urlparse(self.notifications[-1]["data"]["url"]).query)[
                "token"
            ][0]
            await lifecycle.accept_invitation(person, token)
        return project["id"]


@pytest.fixture
def host(tmp_path):
    store = WorkflowStore(tmp_path / "construction.sqlite")
    instance = ControlledHost(tmp_path, store)
    yield instance
    instance.store.close()


def test_authentication_membership_and_invitation_expiry(host):
    async def scenario():
        lifecycle = host.lifecycle()
        with pytest.raises(PermissionError):
            await ConstructionLifecycle(host.store).create_project("pm", name="blocked")
        project = await lifecycle.create_project("pm", name="project")
        invitation = await lifecycle.invite(
            "pm", project["id"], email="sub@example.test", role="subcontractor"
        )
        token = parse_qs(urlparse(host.notifications[-1]["data"]["url"]).query)[
            "token"
        ][0]
        assert datetime.fromisoformat(invitation["expires_at"]) - host.now == timedelta(
            hours=48
        )
        with pytest.raises(PermissionError):
            await lifecycle.accept_invitation("outsider", token)
        host.now += timedelta(hours=48)
        with pytest.raises(PermissionError):
            await lifecycle.accept_invitation("sub", token)
        assert "sub" not in lifecycle._get("projects", project["id"])["members"]
        project_id = await host.project(lifecycle)
        with pytest.raises(PermissionError):
            await lifecycle.invite(
                "sub", project_id, email="outsider@example.test", role="admin"
            )
        with pytest.raises(PermissionError):
            await lifecycle.create_task(
                "outsider", project_id, title="blocked", assignee_id="sub"
            )

    run(scenario())


def test_offline_bytes_survive_restart_and_failed_notification_resumes_without_reupload(
    host,
):
    async def scenario():
        lifecycle = host.lifecycle()
        project = await host.project(lifecycle)
        queued = await lifecycle.queue_daily_log(
            "foreman",
            project,
            notes="Steel delivered",
            weather="clear",
            attachments=[
                {
                    "filename": "site.png",
                    "content_type": "image/png",
                    "content": b"\x89PNG\r\nactual controlled bytes",
                    "caption": "Steel delivery",
                }
            ],
        )
        host.store.close()
        host.store = WorkflowStore(host.root / "construction.sqlite")
        lifecycle = host.lifecycle()
        host.notification_failure = True
        with pytest.raises(RuntimeError, match="delivery outage"):
            await lifecycle.sync_daily_log("foreman", queued["id"])
        assert lifecycle._get("queue", queued["id"])["state"] == "awaiting_notification"
        assert len(host.uploads) == 1
        host.notification_failure = False
        log = await lifecycle.sync_daily_log("foreman", queued["id"])
        assert (
            log["notes"] == "Steel delivered"
            and log["attachments"][0]["caption"] == "Steel delivery"
        )
        assert len(host.uploads) == 1
        assert (
            host.objects / host.uploads[0]
        ).read_bytes() == b"\x89PNG\r\nactual controlled bytes"
        assert lifecycle._get("queue", queued["id"])["state"] == "synced"
        assert (
            json.loads(
                (host.root / "received-notifications.jsonl")
                .read_text()
                .splitlines()[-1]
            )["kind"]
            == "daily_log_created"
        )
        with pytest.raises(PermissionError):
            await lifecycle.attachment_link("outsider", log["attachments"][0]["id"])
        assert "/authorized/" in await lifecycle.attachment_link(
            "sub", log["attachments"][0]["id"]
        )
        assert await lifecycle.sync_daily_log("foreman", queued["id"]) == log

    run(scenario())


def test_upload_validation_failure_keeps_queue_and_never_claims_upload(host):
    async def scenario():
        lifecycle = host.lifecycle()
        project = await host.project(lifecycle)
        queued = await lifecycle.queue_daily_log(
            "foreman",
            project,
            notes="Document",
            attachments=[
                {
                    "filename": "bad.exe",
                    "content_type": "application/executable",
                    "content": b"not a photo",
                }
            ],
        )
        with pytest.raises(ValueError, match="rejected"):
            await lifecycle.sync_daily_log("foreman", queued["id"])
        assert (
            host.uploads == []
            and lifecycle._get("queue", queued["id"])["state"] == "queued"
        )

    run(scenario())


def test_task_transition_actor_time_and_rfi_approval(host):
    async def scenario():
        lifecycle = host.lifecycle()
        project = await host.project(lifecycle)
        task = await lifecycle.create_task(
            "pm", project, title="Install steel", assignee_id="sub"
        )
        with pytest.raises(ValueError):
            await lifecycle.update_task("sub", task["id"], status="complete")
        await lifecycle.update_task(
            "sub", task["id"], status="in_progress", comment="Started"
        )
        task = await lifecycle.update_task(
            "sub", task["id"], status="complete", notify_pm=True
        )
        assert task["history"][-1] == {
            "from": "in_progress",
            "to": "complete",
            "actor_id": "sub",
            "at": host.now.isoformat(),
        }
        assert any(
            event.get("action") == "task_completed"
            for event in host.store.events(project)
        )
        rfi = await lifecycle.create_rfi(
            "foreman", project, description="Confirm beam drawing"
        )
        with pytest.raises(ValueError):
            await lifecycle.rfi_transition("pm", rfi["id"], action="approve")
        await lifecycle.rfi_transition(
            "pm", rfi["id"], action="submit", approver_id="pm2"
        )
        with pytest.raises(PermissionError):
            await lifecycle.rfi_transition("pm", rfi["id"], action="approve")
        revised = await lifecycle.rfi_transition(
            "pm2", rfi["id"], action="request_changes", comment="Need drawing number"
        )
        assert revised["status"] == "draft"
        revised = await lifecycle.revise_rfi(
            "foreman", rfi["id"], description="Confirm beam drawing A-42"
        )
        assert revised["description"].endswith("A-42")
        with pytest.raises(ValueError):
            await lifecycle.rfi_transition(
                "pm", rfi["id"], action="submit", approver_id="pm"
            )
        await lifecycle.rfi_transition(
            "pm", rfi["id"], action="submit", approver_id="pm2"
        )
        assert (await lifecycle.rfi_transition("pm2", rfi["id"], action="approve"))[
            "status"
        ] == "open"
        assert (await lifecycle.rfi_transition("pm", rfi["id"], action="close"))[
            "status"
        ] == "closed"
        assert host.notifications[-1]["kind"] == "rfi_close"

    run(scenario())


def test_summary_executes_real_controlled_index_search_and_pm_pdf_publication(host):
    async def scenario():
        lifecycle = host.lifecycle()
        project = await host.project(lifecycle)
        queued = await lifecycle.queue_daily_log(
            "foreman", project, notes="Steel installed"
        )
        await lifecycle.sync_daily_log("foreman", queued["id"])
        summary = await lifecycle.draft_summary(
            "pm", project, query="steel progress", prompt_version="daily-v1"
        )
        assert summary["status"] == "draft" and set(summary["content"]) == set(SECTIONS)
        assert (
            host.retrieval_calls[-1]["since"]
            == (host.now - timedelta(hours=24)).isoformat()
        )
        assert host.store.all("controlled_vectors")
        assert summary["retrieval"][0]["id"] == f'{project}/daily_logs/{queued["id"]}'
        with pytest.raises(PermissionError):
            await lifecycle.publish_summary("pm", summary["id"])
        with pytest.raises(PermissionError):
            await lifecycle.review_summary("sub", summary["id"])
        modified = {**summary["content"], "overview": "PM verified steel installation"}
        reviewed = await lifecycle.review_summary("pm", summary["id"], content=modified)
        assert reviewed["review"]["actor_id"] == "pm"
        published = await lifecycle.publish_summary("pm", summary["id"])
        pdf = (host.objects / published["pdf"]["object_key"]).read_bytes()
        assert (
            pdf.startswith(b"%PDF-1.4")
            and pdf.endswith(b"%%EOF\n")
            and b"PM verified steel installation" in pdf
        )
        assert published["pdf"]["checksum"] == sha256(pdf).hexdigest()
        with pytest.raises(ValueError):
            await lifecycle.review_summary("pm", summary["id"], content=modified)

    run(scenario())


def test_summary_fail_closed_on_missing_provider_and_cross_project_retrieval(host):
    async def scenario():
        lifecycle = host.lifecycle()
        project = await host.project(lifecycle)
        queued = await lifecycle.queue_daily_log("foreman", project, notes="Steel")
        await lifecycle.sync_daily_log("foreman", queued["id"])
        missing = host.lifecycle(adapters={"embed": None})
        with pytest.raises(ValueError, match="embed"):
            await missing.draft_summary(
                "pm", project, query="steel", prompt_version="v1"
            )
        bad = host.lifecycle(
            adapters={
                "search": lambda p: [
                    {
                        "id": "other/doc",
                        "source_id": f'daily_logs/{queued["id"]}',
                        "project_id": "other",
                    }
                ]
            }
        )
        with pytest.raises(PermissionError, match="provenance"):
            await bad.draft_summary("pm", project, query="steel", prompt_version="v1")
        assert host.store.all("construction_summaries") == []

    run(scenario())


def test_provider_repair_changes_actual_inputs_and_records_successful_rerun(host):
    async def scenario():
        calls = []

        def generate(payload):
            calls.append(payload.get("temperature", 1))
            if payload.get("temperature") != 0:
                raise RuntimeError(
                    "controlled generator requires deterministic temperature"
                )
            return host.generate(payload)

        def repair(error, inputs, attempt):
            return Repair(
                {**inputs, "temperature": 0},
                before={"temperature": 1},
                after={"temperature": 0},
                reason="Use supported deterministic setting",
            )

        lifecycle = host.lifecycle(
            adapters={"generate": generate}, repairs={"generate": repair}
        )
        project = await host.project(lifecycle)
        queued = await lifecycle.queue_daily_log("foreman", project, notes="Steel")
        await lifecycle.sync_daily_log("foreman", queued["id"])
        summary = await lifecycle.draft_summary(
            "pm", project, query="steel", prompt_version="v1"
        )
        assert summary["status"] == "draft" and calls == [1, 0]
        events = host.store.events(project)
        assert any(
            event.get("type") == "construction_repair"
            and event["before"] != event["after"]
            for event in events
        )
        assert any(
            event.get("adapter") == "generate" and event.get("status") == "completed"
            for event in events
        )

    run(scenario())


@pytest.mark.parametrize("field", ["data", "object_key"])
def test_repair_cannot_replace_authorized_notification_or_attachment(host, field):
    async def scenario():
        calls = []
        selected = {"body": "approved"} if field == "data" else "selected-object"
        replacement = {"body": "replacement"} if field == "data" else "other-object"
        original = {"project_id": "selected-project", field: selected}

        def operation(payload):
            calls.append(payload)
            raise RuntimeError("Controlled transient failure")

        def repair(error, inputs, attempt):
            return Repair(
                {**inputs, field: replacement},
                before=inputs[field],
                after=replacement,
                reason="Invalid attempt to replace authorized content",
            )

        adapter = "notify" if field == "data" else "read_attachment"
        lifecycle = host.lifecycle(
            adapters={adapter: operation}, repairs={adapter: repair}
        )
        with pytest.raises(RuntimeError, match="Repair cannot change"):
            await lifecycle._call(adapter, original)
        assert calls == [original]
        attempts = [
            event
            for event in host.store.events("selected-project")
            if event.get("type") == "construction_adapter"
        ]
        assert [item["status"] for item in attempts] == ["failed", "failed"]

    run(scenario())


def test_expired_upload_or_mismatched_receipt_cannot_publish_log(host):
    async def scenario():
        lifecycle = host.lifecycle()
        project = await host.project(lifecycle)
        queued = await lifecycle.queue_daily_log(
            "foreman",
            project,
            notes="Photo",
            attachments=[
                {
                    "filename": "site.png",
                    "content_type": "image/png",
                    "content": b"actual bytes",
                }
            ],
        )

        def expired(payload):
            return {**host.sign_upload(payload), "expires_at": host.now.isoformat()}

        with pytest.raises(ValueError, match="expired"):
            await host.lifecycle(adapters={"sign_upload": expired}).sync_daily_log(
                "foreman", queued["id"]
            )
        with pytest.raises(RuntimeError, match="receipt"):
            await host.lifecycle(
                adapters={
                    "upload": lambda p: {
                        "object_key": p["signature"]["object_key"],
                        "checksum": "wrong",
                        "size": p["size"],
                    }
                }
            ).sync_daily_log("foreman", queued["id"])
        assert host.store.all("construction_daily_logs") == []
        assert host.store.all("construction_attachments") == []
        assert lifecycle._get("queue", queued["id"])["state"] == "queued"

    run(scenario())


def test_summary_review_is_bound_to_exact_content_and_durable_record(host):
    async def scenario():
        lifecycle = host.lifecycle()
        project = await host.project(lifecycle)
        queued = await lifecycle.queue_daily_log("foreman", project, notes="Steel")
        await lifecycle.sync_daily_log("foreman", queued["id"])
        summary = await lifecycle.draft_summary(
            "pm", project, query="steel", prompt_version="v1"
        )
        reviewed = await lifecycle.review_summary("pm", summary["id"])
        tampered = {
            **reviewed,
            "content": {**reviewed["content"], "overview": "Changed after review"},
        }
        host.store.put("construction_summaries", summary["id"], tampered)
        with pytest.raises(PermissionError, match="PM review"):
            await lifecycle.publish_summary("pm", summary["id"])
        await lifecycle.review_summary("pm", summary["id"])
        published = await lifecycle.publish_summary("pm", summary["id"])
        host.store.close()
        host.store = WorkflowStore(host.root / "construction.sqlite")
        assert host.lifecycle()._get("summaries", summary["id"]) == published

    run(scenario())


def test_cli_requires_credentials_before_loading_host_bindings(tmp_path, monkeypatch):
    from examples.construction_lifecycle import parser, execute

    source = tmp_path / "input.json"
    source.write_text('{"name":"No default identity"}')
    args = parser().parse_args(
        [
            "--database",
            str(tmp_path / "no-write.sqlite"),
            "--bindings",
            "missing_host_module",
            "--credentials-env",
            "CONSTRUCTION_TEST_MISSING_AUTH",
            "--action",
            "create_project",
            "--input",
            str(source),
        ]
    )
    monkeypatch.delenv("CONSTRUCTION_TEST_MISSING_AUTH", raising=False)
    with pytest.raises(ValueError, match="environment variable"):
        run(execute(args))
    assert not (tmp_path / "no-write.sqlite").exists()


def test_timezone_lookup_uses_actual_zoneinfo_offsets_and_rejects_unknown_zone(host):
    async def scenario():
        lifecycle = host.lifecycle()
        zones = await lifecycle.list_timezones("pm")
        assert zones["timezones"] == [
            "UTC",
            "America/New_York",
            "America/Los_Angeles",
            "Europe/London",
            "Europe/Berlin",
            "Asia/Tokyo",
            "Australia/Sydney",
        ]
        result = await lifecycle.lookup_time(
            "foreman", timezones=["UTC", "America/New_York", "Asia/Tokyo"]
        )
        times = [datetime.fromisoformat(item["iso"]) for item in result["times"]]
        assert times[0] == times[1] == times[2] == host.now
        assert times[1].utcoffset() == timedelta(hours=-4)
        assert times[2].utcoffset() == timedelta(hours=9)
        assert "EDT" in result["times"][1]["formatted"]
        with pytest.raises(ValueError, match="Invalid timezone"):
            await lifecycle.lookup_time("pm", timezones=["UTC", "Definitely/Invalid"])
        with pytest.raises(ValueError, match="explicit"):
            await lifecycle.lookup_time("pm", timezones=[])

    run(scenario())


def test_failed_rfi_notification_resumes_without_repeating_transition(host):
    async def scenario():
        lifecycle = host.lifecycle()
        project = await host.project(lifecycle)
        rfi = await lifecycle.create_rfi("pm", project, description="Confirm steel")
        host.notification_failure = True
        with pytest.raises(RuntimeError, match="Notification pending"):
            await lifecycle.rfi_transition(
                "pm", rfi["id"], action="submit", approver_id="pm2"
            )
        saved = lifecycle._get("rfis", rfi["id"])
        assert saved["status"] == "pending_approval" and len(saved["history"]) == 1
        pending = await lifecycle.pending_notifications("pm", project)
        assert len(pending) == 1 and "payload" not in pending[0]
        host.notification_failure = False
        receipt = await lifecycle.retry_notification("pm", pending[0]["id"])
        assert receipt["delivered"] is True
        assert lifecycle._get("rfis", rfi["id"]) == saved
        assert await lifecycle.pending_notifications("pm", project) == []
        before = len(host.notifications)
        await lifecycle.retry_notification("pm", pending[0]["id"])
        assert len(host.notifications) == before

    run(scenario())


def test_native_catalog_runs_project_then_invite_with_nested_output_id(host):
    from orchestrator import OrchestrationEngine
    from workflows.construction_agent import ConstructionWorkflowAgent

    async def scenario():
        engine = OrchestrationEngine()
        engine.agents["construction_lifecycle"] = ConstructionWorkflowAgent(
            host.lifecycle(), "pm", confirm=lambda operation, parameters: True
        )
        result = await engine.execute_intent(
            "project onboarding",
            options={
                "workflow_template": "construction_project_onboarding",
                "inputs": {
                    "project_parameters": {"name": "Actual project"},
                    "invite_email": "foreman@example.test",
                    "invite_role": "foreman",
                },
            },
        )
        assert result["status"] == "completed", result
        project = result["outputs"]["created_project"]
        assert result["outputs"]["invitation"]["project_id"] == project["id"]
        assert host.notifications[-1]["recipients"] == ["foreman@example.test"]
        with pytest.raises(PermissionError, match="host identity"):
            await engine.agents["construction_lifecycle"].execute(
                {
                    "operation": "create_project",
                    "parameters": {"name": "No", "credentials": "outsider"},
                }
            )

    run(scenario())


def test_agent_mutation_requires_host_confirmation_and_retrieval_enforces_window(host):
    from workflows.construction_agent import ConstructionWorkflowAgent

    async def scenario():
        lifecycle = host.lifecycle()
        agent = ConstructionWorkflowAgent(lifecycle, "pm")
        with pytest.raises(PermissionError, match="confirmation"):
            await agent.execute(
                {"operation": "create_project", "parameters": {"name": "Unconfirmed"}}
            )
        assert host.store.all("construction_projects") == []
        project = await host.project(lifecycle)
        queued = await lifecycle.queue_daily_log(
            "foreman", project, notes="Older record"
        )
        await lifecycle.sync_daily_log("foreman", queued["id"])
        host.now += timedelta(hours=25)
        with pytest.raises(ValueError, match="time window"):
            await lifecycle.draft_summary(
                "pm", project, query="records", prompt_version="v1"
            )
        summary = await lifecycle.draft_summary(
            "pm", project, query="records", prompt_version="v1", lookback_hours=48
        )
        assert summary["status"] == "draft"

    run(scenario())
