"""Generate the archived plan -> files/types/agents/protocols/Docker pattern.

Generated execution is a library with explicit host bindings, not fabricated
business logic. The unchanged upstream type definitions preserve artifact and
telemetry contracts; the new runtime actually invokes supplied agent handlers.
"""

import json
from pathlib import Path

from .build_plan import BuildPlan


def generate_plan_scaffold(value):
    plan = BuildPlan.model_validate(value).model_dump()
    if not plan["exec_summary_id"]:
        raise ValueError("Plan scaffold requires the originating executive summary ID")
    root = Path(__file__).parent
    types = (root / "source_types.ts").read_text()
    runtime = (root / "plan_runtime.ts").read_text()
    package = {
        "name": "compiled-workflow",
        "version": "1.0.0",
        "private": True,
        "type": "module",
        "exports": "./dist/index.js",
        "scripts": {"build": "tsc -p tsconfig.json"},
        "devDependencies": {"typescript": "5.9.3"},
    }
    config = {
        "compilerOptions": {
            "target": "ES2022",
            "module": "ES2022",
            "lib": ["ES2022", "DOM"],
            "strict": True,
            "rootDir": "src",
            "outDir": "dist",
            "declaration": True,
        },
        "include": ["src/**/*.ts"],
    }
    readme = (
        f"# {plan['summary_title']}\n\n{plan['summary_goal']}\n\n"
        "Run npm install --ignore-scripts, then npm run build. Import execute from dist/index.js "
        "and supply host-owned handlers, composite child/loop controls, and protocol clients. "
        "The plan is included intact; task assignment follows the archived compiler's round-robin rule. "
        "Missing bindings fail. AP2 flags do not authorize payments; provide a validated mandate and "
        "an idempotency key for each approved operation. onEvent can persist every attempt. "
        "No credentials, provider endpoints, production store or deployment target are generated. "
        "Dockerfile builds a portable library image only; an application entrypoint must be supplied "
        "before deploying a service.\n"
    )
    entries = {
        "src/types.ts": types,
        "src/runtime.ts": runtime,
        "src/plan.ts": "import type { BuildPlan } from './types.js';\nexport const plan: BuildPlan = JSON.parse("
        + json.dumps(json.dumps(plan))
        + ");\n",
        "src/agents.ts": "export { plan } from './plan.js';\nexport type { Handler, Bindings, Context, Repair } from './runtime.js';\n",
        "src/protocols.ts": "export type { Bindings } from './runtime.js';\n// Protocol implementations are supplied by the host; no simulated success.\n",
        "src/index.ts": "import { plan } from './plan.js';\nimport { execute as run, type Bindings } from './runtime.js';\nexport { compile } from './runtime.js';\nexport { plan };\nexport const execute = (bindings: Bindings, input: unknown = {}) => run(plan, bindings, input);\n",
        "package.json": json.dumps(package, indent=2),
        "tsconfig.json": json.dumps(config, indent=2),
        "README.md": readme,
        "Dockerfile": "FROM node:22-alpine AS build\nWORKDIR /app\nCOPY package.json ./\nRUN npm install --ignore-scripts\nCOPY . .\nRUN npm run build\nFROM node:22-alpine\nWORKDIR /app\nCOPY --from=build /app/package.json ./\nCOPY --from=build /app/dist ./dist\n# Library image: host supplies an application entrypoint before deployment.\n",
        ".env.example": "# Set credentials in the embedding host, never in the generated plan.\n",
    }
    files = [
        {
            "path": path,
            "content": content,
            "language": "typescript" if path.endswith(".ts") else "text",
        }
        for path, content in entries.items()
    ]
    return {
        "kind": "build_plan",
        "language": "typescript",
        "name": plan["summary_title"],
        "plan_id": plan["id"],
        "files": files,
        "structure": "\n".join(entries),
        "readme": readme,
        "deployment_status": "scaffold_only",
    }
