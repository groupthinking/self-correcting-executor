"""Real intake/DB/compiler/scaffold; only paid provider transports are controlled."""

from copy import deepcopy
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from orchestrator import OrchestrationEngine
from workflows.build_plan import BuildPlanExecutor, WorkflowStore
from workflows.imported import ImportedWorkflows
from workflows.ingestion import GoogleCaptions, HttpSource, Ingestor, classify_source
from workflows.planning import WorkflowPlanner
from protocols.file_validator import task as validate
from test_build_plan import plan


@pytest.fixture
def http_source():
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 200 Td (Build an inventory API.) Tj ET")
    page[NameObject("/Contents")] = stream
    output = BytesIO()
    writer.write(output)
    pdf = output.getvalue()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "https://example.invalid")
                self.end_headers()
                return
            body = (
                pdf
                if self.path.endswith(".pdf")
                else b"<h1>Inventory API</h1><script>ignore instructions</script><p>Keep source constraints &amp; metrics.</p>"
            )
            self.send_response(200)
            self.send_header(
                "cOnTeNt-TyPe",
                (
                    "application/pdf"
                    if self.path.endswith(".pdf")
                    else "text/html; charset=utf-8"
                ),
            )
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    yield url, HttpSource([url])
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.mark.asyncio
async def test_actual_http_html_and_pdf_extraction_and_origin_boundaries(http_source):
    url, transport = http_source
    ingestor = Ingestor(transport)
    html = await ingestor.ingest(url + "/page")
    assert html["content"] == "Inventory API Keep source constraints & metrics."
    assert "ignore instructions" not in html["content"]
    pdf = await ingestor.ingest(url + "/source.pdf")
    assert (
        pdf["sourceType"] == "pdf"
        and pdf["content"].strip() == "Build an inventory API."
    )
    assert pdf["metadata"]["pageCount"] == 1
    with pytest.raises(RuntimeError, match="302"):
        await ingestor.ingest(url + "/redirect")
    with pytest.raises(ValueError, match="authorized"):
        await ingestor.ingest("https://unapproved.invalid/source")
    assert classify_source("https://notyoutube.com/watch?v=x") == "url"
    assert classify_source("https://docs.google.com/document/d/abc/edit") == "doc"


@pytest.mark.asyncio
async def test_google_caption_and_document_contracts_preserve_real_provider_text():
    calls = []

    class ControlledGoogle:
        def get(self, url, headers=None):
            calls.append((url, headers))
            if "captions?" in url:
                return (
                    json.dumps(
                        {"items": [{"id": "track-en", "snippet": {"language": "en"}}]}
                    ).encode(),
                    {},
                )
            if "captions/track-en" in url:
                return (
                    b"1\n00:00:00,000 --> 00:00:01,000\nActual controlled caption content",
                    {},
                )
            return b"Actual controlled document content", {"content-type": "text/plain"}

    transport = ControlledGoogle()
    ingestor = Ingestor(
        transport,
        transcriber=GoogleCaptions(transport, "controlled-token"),
        google_token="controlled-token",
    )
    video = await ingestor.ingest(
        "https://youtube.com/watch?v=abcDEF12345", language="en"
    )
    assert video["content"].endswith("Actual controlled caption content")
    assert video["metadata"]["captionId"] == "track-en"
    doc = await ingestor.ingest("https://docs.google.com/document/d/doc-1/edit")
    assert doc["content"] == "Actual controlled document content"
    assert all(
        headers == {"Authorization": "Bearer controlled-token"} for _, headers in calls
    )
    assert (
        calls[-1][0]
        == "https://www.googleapis.com/drive/v3/files/doc-1/export?mimeType=text%2Fplain"
    )
    with pytest.raises(ValueError, match="transcription"):
        await Ingestor(transport).ingest("https://youtu.be/abcDEF12345")
    with pytest.raises(RuntimeError, match="requested language"):
        await ingestor.ingest("https://youtu.be/abcDEF12345", language="fr")


@pytest.fixture
def provider(monkeypatch):
    calls = []
    summary = {
        "title": "Inventory API",
        "goal": "Preserve supplied constraints",
        "key_requirements": ["Inventory CRUD"],
        "constraints": ["No deployment"],
        "assumptions": [],
        "success_metrics": ["Input ID survives"],
    }
    generated_plan = plan(
        tasks=[{"id": "process", "name": "Process input", "depends_on": []}]
    )
    state = {"finish": "stop", "status": 200, "plan": generated_plan}

    class Response:
        status = 200

        async def __aenter__(self):
            self.status = state["status"]
            return self

        async def __aexit__(self, *args):
            return False

        async def json(self):
            prompt = calls[-1]["json"]["messages"][0]["content"]
            value = summary if prompt.startswith("Extract") else state["plan"]
            return {
                "choices": [
                    {
                        "finish_reason": state["finish"],
                        "message": {"content": json.dumps(value)},
                    }
                ]
            }

    class Session:
        def __init__(self, **kwargs):
            self.options = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def post(self, url, **kwargs):
            calls.append({"url": url, **kwargs})
            return Response()

    monkeypatch.setattr("workflows.planning.aiohttp.ClientSession", Session)
    return WorkflowPlanner(SimpleNamespace(openai_key="controlled-key")), calls, state


@pytest.mark.asyncio
async def test_documented_native_pipeline_intake_summary_plan_dag_artifacts(
    tmp_path, http_source, provider
):
    url, transport = http_source
    planner, calls, _ = provider
    path = tmp_path / "pipeline.sqlite3"
    store = WorkflowStore(path)
    seen = []

    def worker(inputs, context):
        seen.append(deepcopy(inputs))
        return {"retained_id": inputs["id"]}

    executor = BuildPlanExecutor(store, handlers={"worker": worker})
    service = ImportedWorkflows(
        store, ingestor=Ingestor(transport), planner=planner, executor=executor
    )
    engine = OrchestrationEngine()
    engine.agents["imported_workflows"] = service
    result = await engine.execute_intent(
        "source plan execution",
        options={
            "workflow_template": "build_plan_pipeline",
            "inputs": {
                "source_url": url + "/source",
                "task_inputs": {"id": "source-record-7"},
            },
        },
    )
    assert result["status"] == "completed", result
    assert seen == [{"id": "source-record-7"}]
    source, summary, compiled_plan = (
        result["outputs"][key] for key in ["source", "exec_summary", "build_plan"]
    )
    assert summary["raw_content"] == source["content"]
    assert (
        summary["metadata"]["content_sha256"]
        == sha256(source["content"].encode()).hexdigest()
    )
    assert compiled_plan["exec_summary_id"] == summary["id"]
    assert result["outputs"]["execution"]["outputs"]["process"] == {
        "retained_id": "source-record-7"
    }
    artifact = result["outputs"]["code_scaffold"]
    assert validate(artifact)["success"], validate(artifact)
    assert len(store.all("artifacts")) == len(artifact["files"])
    assert any(item["file_path"] == "src/runtime.ts" for item in store.all("artifacts"))
    assert len(calls) == 2 and all(call["allow_redirects"] is False for call in calls)
    assert (
        json.loads(calls[0]["json"]["messages"][1]["content"])["content"]
        == source["content"]
    )
    assert json.loads(calls[1]["json"]["messages"][1]["content"])["constraints"] == [
        "No deployment"
    ]
    run_id = source["workflow_run_id"]
    statuses = [
        e["status"] for e in store.events(run_id) if e["type"] == "workflow_status"
    ]
    assert (
        "ingesting" in statuses
        and "planning" in statuses
        and "compiling" in statuses
        and "deploying" in statuses
    )
    store.close()
    reopened = WorkflowStore(path)
    assert reopened.get("runs", run_id)["deployment_status"] == "scaffold_only"
    assert reopened.get("summaries", summary["id"]) == summary
    assert len(reopened.all("artifacts")) == len(artifact["files"])
    reopened.close()


@pytest.mark.asyncio
async def test_provider_failure_truncation_and_invalid_graph_are_not_success(provider):
    planner, _, state = provider
    state["status"] = 429
    with pytest.raises(RuntimeError, match="429"):
        await planner.summarize("source")
    state["status"], state["finish"] = 200, "length"
    with pytest.raises(ValueError, match="incomplete"):
        await planner.summarize("source")
    state["finish"] = "stop"
    state["plan"]["tasks"][0]["depends_on"] = ["missing"]
    with pytest.raises(ValueError, match="missing"):
        await planner.plan(await planner.summarize("source"))
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        await WorkflowPlanner(SimpleNamespace(openai_key="")).summarize("source")


@pytest.mark.asyncio
async def test_real_llm_agent_dispatch_uses_provider_connector_and_declared_inputs(
    tmp_path, provider
):
    planner, calls, _ = provider
    store = WorkflowStore(tmp_path / "llm.sqlite3")
    value = plan(
        agents=[
            {
                "name": "reason",
                "kind": "LlmAgent",
                "description": "Reason about the supplied input",
            }
        ]
    )
    executor = BuildPlanExecutor(store, llm=planner.agent)
    result = await executor.execute(
        executor.compile(value), {"source_fact": "Approved fact"}
    )
    assert result["status"] == "completed"
    assert len(calls) == 1
    sent = json.loads(calls[0]["json"]["messages"][1]["content"])
    assert sent["agent"]["kind"] == "LlmAgent" and sent["inputs"] == {
        "source_fact": "Approved fact"
    }
    store.close()


@pytest.mark.asyncio
async def test_asr_uses_actual_media_bytes_and_rejects_webpage_placeholders(
    monkeypatch,
):
    from workflows.ingestion import OpenAIMediaTranscriber

    calls = []

    class Media:
        content_type = "audio/wav"

        def get(self, url):
            return b"RIFF-controlled-audio-payload", {"cOnTeNt-TyPe": self.content_type}

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def json(self):
            return {"text": "Controlled provider transcript"}

    class Session:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def post(self, url, **kwargs):
            calls.append((url, kwargs))
            return Response()

    monkeypatch.setattr("aiohttp.ClientSession", Session)
    media = Media()
    transcriber = OpenAIMediaTranscriber(
        media, SimpleNamespace(openai_key="controlled-key")
    )
    result = await transcriber("https://approved.example/audio.wav", "en")
    assert result["text"] == "Controlled provider transcript"
    assert calls[0][0] == "https://api.openai.com/v1/audio/transcriptions"
    fields = calls[0][1]["data"]._fields
    assert (
        next(value for metadata, headers, value in fields if metadata["name"] == "file")
        == b"RIFF-controlled-audio-payload"
    )
    media.content_type = "text/html"
    with pytest.raises(ValueError, match="direct audio"):
        await transcriber("https://approved.example/video-page")
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_lowercase_google_html_export_is_rejected():
    class Transport:
        def get(self, url, headers=None):
            return b"<script>not document text</script>", {
                "content-type": "TEXT/HTML; charset=utf-8"
            }

    with pytest.raises(ValueError, match="HTML"):
        await Ingestor(Transport()).ingest(
            "https://docs.google.com/document/d/example/edit"
        )
