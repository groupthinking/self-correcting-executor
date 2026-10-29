"""Deterministic generated-artifact validation; never execute artifact scripts."""

import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile

from workflows.service_scaffold import artifact_digest


def task(target=None):
    """Validate a TypeScript files/content bundle with the actual TS compiler."""
    try:
        digest = artifact_digest(target)
        compiler = os.environ.get("TYPESCRIPT_COMPILER") or str(
            Path(__file__).resolve().parents[1]
            / "tools/typescript-validation/node_modules/typescript/bin/tsc"
        )
        node = shutil.which("node")
        if not node or not Path(compiler).is_file():
            raise ValueError(
                "TypeScript validation requires Node and npm ci --prefix tools/typescript-validation"
            )
        compiler = str(Path(compiler).resolve())
        names = set()
        files = target["files"]
        with tempfile.TemporaryDirectory(prefix="workflow-validate-") as directory:
            root = Path(directory)
            source_paths = []
            for item in files:
                if not isinstance(item, dict):
                    raise ValueError("Artifact files must be objects")
                name, content = item.get("path"), item.get("content")
                if not isinstance(name, str) or not isinstance(content, str):
                    raise ValueError("Artifact path and content must be strings")
                path = PurePosixPath(name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or "\\" in name
                    or ":" in name
                    or str(path) != name
                    or name in names
                    or name.startswith(("-", "@"))
                    or path.suffix not in {".ts", ".json", ".md"}
                ):
                    raise ValueError("Unsafe, duplicate or unsupported artifact path")
                names.add(name)
                destination = root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(content)
                if path.suffix == ".json":
                    json.loads(content)
                if path.suffix == ".ts":
                    source_paths.append(str(destination))
            if not {"src/service.ts", "package.json", "tsconfig.json"}.issubset(names):
                raise ValueError(
                    "Service source, package.json and tsconfig.json are required"
                )
            # Fixed flags prevent a supplied config/plugin/script from running or
            # changing the acceptance contract. No npm install or artifact execution.
            command = [
                node,
                compiler,
                "--noEmit",
                "--strict",
                "--target",
                "ES2022",
                "--module",
                "ES2022",
                "--lib",
                "ES2022,DOM",
                "--typeRoots",
                str(root / ".empty-types"),
                *sorted(source_paths),
            ]
            # With no node_modules in the isolated directory, ambient @types cannot
            # change results. An empty typeRoots also excludes ancestor packages.
            empty_types = root / ".empty-types"
            empty_types.mkdir()
            result = subprocess.run(
                command,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            diagnostics = (result.stdout + result.stderr).replace(
                str(root), "<artifact>"
            )
            return {
                "success": result.returncode == 0,
                "action": "file_validation",
                "language": "typescript",
                "artifact_sha256": digest,
                "files_checked": sorted(names),
                "diagnostics": diagnostics,
                "error": "TypeScript compilation failed" if result.returncode else None,
            }
    except (ValueError, OSError, TypeError, subprocess.TimeoutExpired) as exc:
        return {"success": False, "action": "file_validation", "error": str(exc)}
