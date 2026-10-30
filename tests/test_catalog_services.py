"""Execute actual selected catalog components with local service transports."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import socketserver
import subprocess
import threading
from types import SimpleNamespace

import pytest
import requests

from agents.specialized.code_generator import CodeGeneratorAgent
from orchestrator import OrchestrationEngine
from protocols import (
    api_health_checker,
    file_validator,
    redis_cache_manager,
    system_monitor,
)
from workflows.service_scaffold import generate_typescript_service, artifact_digest


@pytest.fixture(scope="module")
def artifact():
    return generate_typescript_service({})


@pytest.fixture(scope="module")
def service_url(tmp_path_factory, artifact):
    """Compile the real generated files and expose them through a test-only Node adapter."""
    directory = tmp_path_factory.mktemp("typescript-service")
    for item in artifact["files"]:
        destination = directory / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(item["content"])
    compiler = (
        Path(__file__).resolve().parents[1]
        / "tools/typescript-validation/node_modules/typescript/bin/tsc"
    )
    assert compiler.is_file(), "Run npm ci --prefix tools/typescript-validation"
    node = shutil.which("node")
    assert node, "Node 22+ is required for catalog tests"
    subprocess.run(
        [node, str(compiler), "--project", "tsconfig.json"],
        cwd=directory,
        check=True,
        capture_output=True,
        text=True,
    )
    adapter = directory / "test-server.mjs"
    adapter.write_text("""import { createServer } from 'node:http';
import { createService } from './dist/service.js';
// Controlled persistence belongs to this test, not the generated production scaffold.
const items = new Map();
const handle = createService({
  async list() { return [...items.values()]; },
  async get(id) { if (id === "storage-failure") throw new Error("private storage details"); return items.get(id) ?? null; },
  async create(item) { if (items.has(item.id)) return false; items.set(item.id, item); return true; },
  async update(item) { if (!items.has(item.id)) return false; items.set(item.id, item); return true; },
  async delete(id) { return items.delete(id); }
});
const server = createServer(async (req, res) => {
  if (req.url === '/unhealthy') { res.writeHead(503); res.end(); return; }
  if (req.url === '/redirect') { res.writeHead(302, {location: '/health'}); res.end(); return; }
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  const body = Buffer.concat(chunks);
  const request = new Request('http://localhost' + req.url, {
    method: req.method, headers: req.headers,
    ...(body.length ? { body } : {})
  });
  const response = await handle(request);
  res.writeHead(response.status, Object.fromEntries(response.headers));
  res.end(Buffer.from(await response.arrayBuffer()));
});
server.listen(0, '127.0.0.1', () => console.log(server.address().port));
""")
    process = subprocess.Popen(
        [node, str(adapter)],
        cwd=directory,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        line = process.stdout.readline().strip()
        assert line.isdigit(), (
            process.stderr.read() if process.poll() is not None else line
        )
        yield f"http://127.0.0.1:{line}"
    finally:
        process.terminate()
        process.communicate(timeout=10)


@pytest.fixture
def redis_transport():
    """Small RESP fixture: production redis.Redis performs actual TCP commands."""
    state = {"values": {}, "commands": [], "corrupt_readback": False}

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            while True:
                line = self.rfile.readline()
                if not line:
                    return
                assert line.startswith(b"*")
                args = []
                for _ in range(int(line[1:])):
                    size = self.rfile.readline()
                    assert size.startswith(b"$")
                    args.append(self.rfile.read(int(size[1:])))
                    assert self.rfile.read(2) == b"\r\n"
                command = args[0].upper()
                state["commands"].append(args)
                if command == b"CLIENT":
                    self.wfile.write(b"+OK\r\n")
                elif command == b"PING":
                    self.wfile.write(b"+PONG\r\n")
                elif command == b"SET":
                    assert args[3].upper() == b"EX" and int(args[4]) > 0
                    state["values"][args[1]] = args[2]
                    self.wfile.write(b"+OK\r\n")
                elif command == b"GET":
                    value = (
                        b"corrupt"
                        if state["corrupt_readback"]
                        else state["values"].get(args[1])
                    )
                    self.wfile.write(
                        b"$-1\r\n"
                        if value is None
                        else b"$" + str(len(value)).encode() + b"\r\n" + value + b"\r\n"
                    )
                else:
                    self.wfile.write(b"-ERR unsupported test command\r\n")
                self.wfile.flush()

    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = True
        allow_reuse_address = True

    server = Server(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"redis://127.0.0.1:{server.server_address[1]}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.asyncio
async def test_entire_typescript_catalog_real_compiler_http_and_redis(
    service_url, redis_transport
):
    engine = OrchestrationEngine()
    result = await engine.execute_intent(
        "TypeScript API pipeline",
        options={
            "workflow_template": "typescript_service_pipeline",
            "inputs": {
                "service_health_url": service_url + "/health",
                "cache_config": {
                    "redis_url": redis_transport["url"],
                    "key": "test:service-metadata",
                    "ttl_seconds": 90,
                },
            },
        },
    )
    assert result["status"] == "completed", result
    assert [step["step"] for step in result["steps_completed"]] == [
        "code_generator",
        "file_validator",
        "api_health_checker",
        "redis_cache_manager",
    ]
    outputs = result["outputs"]
    digest = artifact_digest(outputs["generated_code"])
    assert outputs["validation_result"]["artifact_sha256"] == digest
    assert outputs["health_report"]["artifact_sha256"] == digest
    stored = json.loads(redis_transport["values"][b"test:service-metadata"])
    assert stored["schema"] == {"id": "string", "payload": "object"}
    assert stored["health_report"] == outputs["health_report"]
    assert outputs["deployment_artifacts"]["cache_verified"] is True
    assert (
        outputs["deployment_artifacts"]["deployment_status"]
        == "not_deployed_by_workflow"
    )
    assert any(
        command[0] == b"SET" and command[-1] == b"90"
        for command in redis_transport["commands"]
    )


def test_generated_crud_executes_actual_storage_contract(service_url):
    base = service_url + "/api/v1/service"
    record = {"id": "entry", "payload": {"progress": "observed"}}
    assert requests.get(service_url + "/health", timeout=5).json() == {
        "status": "ok",
        "service": "service",
    }
    assert requests.post(base, json=record, timeout=5).status_code == 201
    assert requests.post(base, json=record, timeout=5).status_code == 409
    assert requests.get(base + "/entry", timeout=5).json() == record
    assert requests.get(base, timeout=5).json() == [record]
    assert (
        requests.put(
            base + "/entry", json={"payload": {"progress": "updated"}}, timeout=5
        ).status_code
        == 200
    )
    assert requests.get(base + "/entry", timeout=5).json()["payload"] == {
        "progress": "updated"
    }
    assert (
        requests.put(
            base + "/entry", json={"id": "other", "payload": {}}, timeout=5
        ).status_code
        == 400
    )
    assert (
        requests.post(base, json={"id": "bad", "payload": []}, timeout=5).status_code
        == 400
    )
    assert requests.delete(base + "/entry", timeout=5).status_code == 204
    assert requests.get(base + "/entry", timeout=5).status_code == 404
    assert requests.delete(base + "/entry", timeout=5).status_code == 404
    failure = requests.get(base + "/storage-failure", timeout=5)
    assert failure.status_code == 503
    assert failure.json() == {"error": "Storage unavailable"}


def test_validator_deterministically_rejects_real_type_error(artifact):
    broken = deepcopy(artifact)
    broken["files"][0]["content"] += '\nconst wrong: number = "not a number";\n'
    first = file_validator.task(target=broken)
    second = file_validator.task(target=broken)
    assert first == second
    assert first["success"] is False
    assert "TS2322" in first["diagnostics"]


@pytest.mark.parametrize(
    "path",
    [
        "../escape.ts",
        "/escape.ts",
        "src/../escape.ts",
        "src\\escape.ts",
        "C:/escape.ts",
        "@flags.ts",
    ],
)
def test_validator_rejects_unsafe_paths(artifact, path):
    invalid = deepcopy(artifact)
    invalid["files"][0]["path"] = path
    assert file_validator.task(invalid)["success"] is False


def test_validator_missing_toolchain_fails_not_skips(artifact, monkeypatch):
    monkeypatch.setenv("TYPESCRIPT_COMPILER", "/nonexistent/compiler")
    assert "requires Node" in file_validator.task(artifact)["error"]
    assert file_validator.task()["success"] is False


@pytest.mark.parametrize("path,status", [("/unhealthy", 503), ("/redirect", 302)])
def test_health_does_not_hide_bad_status_or_follow_redirects(
    artifact, service_url, path, status
):
    result = api_health_checker.task(artifact, service_url + path)
    assert result["success"] is False
    assert result["status"] == status
    assert result["deployment_verified"] is False


def test_health_requires_explicit_url(artifact):
    assert api_health_checker.task(artifact)["success"] is False
    assert (
        api_health_checker.task(artifact, "https://user:secret@example.com/health")[
            "success"
        ]
        is False
    )


def test_cache_readback_failure_and_exact_artifact_binding(
    artifact, service_url, redis_transport
):
    health = api_health_checker.task(artifact, service_url + "/health")
    config = {
        "redis_url": redis_transport["url"],
        "key": "test:failed",
        "ttl_seconds": 15,
    }
    redis_transport["corrupt_readback"] = True
    assert "readback" in redis_cache_manager.task(artifact, health, config)["error"]
    bad = deepcopy(health)
    bad["artifact_sha256"] = "different"
    count = len(redis_transport["commands"])
    assert "exact artifact" in redis_cache_manager.task(artifact, bad, config)["error"]
    assert len(redis_transport["commands"]) == count


def test_cache_requires_explicit_connection_key_and_ttl(artifact, monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_HOST", raising=False)
    health = {"success": True, "artifact_sha256": artifact_digest(artifact)}
    assert "cache_config" in redis_cache_manager.task(artifact, health)["error"]
    for config in [
        {"key": "", "ttl_seconds": 10},
        {"key": "test", "ttl_seconds": 0},
        {"key": "test", "ttl_seconds": True},
    ]:
        assert redis_cache_manager.task(artifact, health, config)["success"] is False
    assert (
        "explicitly"
        in redis_cache_manager.task(
            artifact, health, {"key": "test", "ttl_seconds": 10}
        )["error"]
    )


@pytest.mark.asyncio
async def test_typescript_missing_configuration_keeps_actual_upstream_outputs():
    result = await OrchestrationEngine().execute_intent(
        "TypeScript API",
        options={
            "workflow_template": "typescript_service_pipeline",
        },
    )
    assert result["status"] == "failed"
    assert (
        "generated_code" in result["outputs"]
        and "validation_result" in result["outputs"]
    )
    assert "service_health_url" in result["steps_completed"][-1]["error"]
    assert "deployment_artifacts" not in result["outputs"]


@pytest.mark.asyncio
async def test_core_catalog_runs_real_analyzer_and_host_metrics(tmp_path, monkeypatch):
    (tmp_path / "records.json").write_text('[{"value":1},{"value":2}]')
    # Controlled measurements exercise the actual host-health policy, not a replaced task.
    monkeypatch.setattr(system_monitor.psutil, "cpu_percent", lambda **kwargs: 12)
    monkeypatch.setattr(
        system_monitor.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=25, total=8 * 1024**3),
    )
    monkeypatch.setattr(
        system_monitor.psutil,
        "disk_usage",
        lambda _: SimpleNamespace(percent=40, total=80 * 1024**3),
    )
    result = await OrchestrationEngine().execute_intent(
        "core",
        options={
            "workflow_template": "self_correcting_core",
            "inputs": {"user_data": str(tmp_path)},
        },
    )
    assert result["status"] == "completed", result
    assert result["outputs"]["processed_data"]["total_records"] == 2
    assert result["outputs"]["system_status"]["scope"] == "orchestration"
    assert result["outputs"]["system_status"]["system_info"]["cpu_percent"] == 12
    monkeypatch.setattr(system_monitor.psutil, "cpu_percent", lambda **kwargs: 95)
    assert system_monitor.task(scope="orchestration")["success"] is False
    assert system_monitor.task(scope="unknown")["success"] is False


@pytest.mark.asyncio
async def test_generator_preserves_requested_language_and_rejects_unsupported_schema():
    result = await CodeGeneratorAgent().execute(
        {"intent": "Generate TypeScript API", "context": {"endpoint_name": "events"}}
    )
    assert result["generation_type"] == "typescript_service"
    assert "src/service.ts" in result["files_created"]
    assert result["generated_code"]["language"] == "typescript"
    assert "/api/v1/events" in result["generated_code"]["files"][0]["content"]
    with pytest.raises(ValueError, match="requires string id"):
        generate_typescript_service({"parameters": {"project": "int"}})
