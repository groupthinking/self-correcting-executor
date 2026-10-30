"""Typed service scaffolds in the archived workflows files/content bundle format."""

from copy import deepcopy
import hashlib
import json
import re

SERVICE_SOURCE = """// Generated scaffold: supply a real Store and HTTP adapter before deployment.
export type Payload = Record<string, unknown>;
export interface Item { id: string; payload: Payload }
export interface Store {
  list(): Promise<Item[]>;
  get(id: string): Promise<Item | null>;
  create(item: Item): Promise<boolean>;
  update(item: Item): Promise<boolean>;
  delete(id: string): Promise<boolean>;
}
const base = __BASE__;
const reply = (status: number, body: unknown): Response =>
  Response.json(body, { status });

export function createService(store: Store): (request: Request) => Promise<Response> {
  return async (request: Request): Promise<Response> => {
    const path = new URL(request.url).pathname;
    if (path === "/health" && request.method === "GET") {
      return reply(200, { status: "ok", service: __NAME__ });
    }
    const itemPath = path.startsWith(base + "/");
    if (path !== base && !itemPath) return reply(404, { error: "Not found" });
    let id: string | undefined;
    try {
      id = itemPath ? decodeURIComponent(path.slice(base.length + 1)) : undefined;
    } catch { return reply(400, { error: "Invalid item id" }); }
    if (itemPath && (!id || id.includes("/"))) {
      return reply(400, { error: "Invalid item id" });
    }
    try {
      if (request.method === "GET") {
        if (!id) return reply(200, await store.list());
        const item = await store.get(id);
        return item ? reply(200, item) : reply(404, { error: "Not found" });
      }
      if ((request.method === "POST" && !id) || (request.method === "PUT" && id)) {
        let body: unknown;
        try { body = await request.json(); }
        catch { return reply(400, { error: "Invalid JSON" }); }
        if (typeof body !== "object" || body === null || Array.isArray(body)) {
          return reply(400, { error: "Expected id and payload" });
        }
        const values = body as Record<string, unknown>;
        const itemId = id ?? values.id;
        if (typeof itemId !== "string" || !itemId.trim() || itemId.includes("/") ||
            typeof values.payload !== "object" || values.payload === null ||
            Array.isArray(values.payload)) {
          return reply(400, { error: "Expected string id and object payload" });
        }
        if (id && values.id !== undefined && values.id !== id) {
          return reply(400, { error: "Body id does not match path" });
        }
        const item = { id: itemId, payload: values.payload as Payload };
        if (request.method === "POST") {
          return await store.create(item) ? reply(201, item) : reply(409, { error: "Already exists" });
        }
        return await store.update(item) ? reply(200, item) : reply(404, { error: "Not found" });
      }
      if (request.method === "DELETE" && id) {
        return await store.delete(id) ? new Response(null, { status: 204 }) : reply(404, { error: "Not found" });
      }
      return reply(405, { error: "Method not allowed" });
    } catch {
      return reply(503, { error: "Storage unavailable" });
    }
  };
}
"""


def generate_typescript_service(context):
    """Produce callable CRUD/health handlers; never choose a production store."""
    name = context.get("endpoint_name", "service")
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,62}", name):
        raise ValueError("endpoint_name must be a lowercase URL path segment")
    parameters = context.get("parameters", {"id": "string", "payload": "object"})
    if parameters != {"id": "string", "payload": "object"}:
        raise ValueError(
            "TypeScript service scaffold requires string id and object payload"
        )
    source = SERVICE_SOURCE.replace("__BASE__", json.dumps(f"/api/v1/{name}"))
    source = source.replace("__NAME__", json.dumps(name))
    config = {
        "compilerOptions": {
            "target": "ES2022",
            "module": "ES2022",
            "lib": ["ES2022", "DOM"],
            "strict": True,
            "outDir": "dist",
            "rootDir": "src",
            "declaration": True,
        },
        "include": ["src/**/*.ts"],
    }
    package = {
        "name": name,
        "version": "0.1.0",
        "private": True,
        "type": "module",
        "scripts": {"build": "tsc --project tsconfig.json"},
        "devDependencies": {"typescript": "5.9.3"},
    }
    readme = (
        "# TypeScript service scaffold\n\n"
        "Run npm install and npm run build. Import createService from dist/service.js "
        "and provide a Store implementation plus an HTTP adapter supporting standard "
        "Request/Response (Node 22+). The returned handler implements GET /health and "
        f"GET/POST /api/v1/{name}, GET/PUT/DELETE /api/v1/{name}/:id. "
        "Store.create returns false for duplicate IDs; update/delete return false for "
        "missing IDs. Storage errors return 503. No persistence, authentication, "
        "authorization, domain policy or deployment is selected for you. The health "
        "route confirms handler availability; it does not probe the supplied store.\n"
    )
    files = [
        {"path": "src/service.ts", "content": source, "language": "typescript"},
        {
            "path": "tsconfig.json",
            "content": json.dumps(config, indent=2),
            "language": "json",
        },
        {
            "path": "package.json",
            "content": json.dumps(package, indent=2),
            "language": "json",
        },
        {"path": "README.md", "content": readme, "language": "markdown"},
    ]
    return {
        "language": "typescript",
        "name": name,
        "files": files,
        "schema": {"id": "string", "payload": "object"},
        "health_path": "/health",
        "deployment_status": "scaffold_only",
    }


def artifact_digest(service):
    """Bind validation/cache reports to complete, immutable caller artifacts."""
    if not isinstance(service, dict) or service.get("language") != "typescript":
        raise ValueError("Expected a TypeScript service artifact bundle")
    files = service.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Service artifact files are required")
    return hashlib.sha256(
        json.dumps(service, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def artifact_metadata(service):
    return {
        "artifact_sha256": artifact_digest(service),
        "name": service.get("name"),
        "schema": deepcopy(service.get("schema")),
        "health_path": service.get("health_path"),
    }
