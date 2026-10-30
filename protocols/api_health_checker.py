"""Probe the caller-selected service, without substituting unrelated public APIs."""

import time
from urllib.parse import urlsplit
import requests

from workflows.service_scaffold import artifact_digest


def task(service=None, url=None):
    """Check one explicit full health URL and bind the observation to the artifact."""
    try:
        digest = artifact_digest(service)
        if not isinstance(url, str):
            raise ValueError("service_health_url is required")
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.fragment
        ):
            raise ValueError(
                "service_health_url must be an HTTP(S) URL without credentials or fragment"
            )
        started = time.monotonic()
        with requests.get(
            url, timeout=5, allow_redirects=False, stream=True
        ) as response:
            status = response.status_code
        healthy = status == 200
        return {
            "success": healthy,
            "action": "api_health_check",
            "healthy": healthy,
            "artifact_sha256": digest,
            "url": url,
            "status": status,
            "response_time_ms": round((time.monotonic() - started) * 1000, 2),
            "deployment_verified": False,
            "error": (
                None
                if healthy
                else "Configured health endpoint did not return HTTP 200"
            ),
        }
    except (ValueError, TypeError) as exc:
        return {"success": False, "action": "api_health_check", "error": str(exc)}
    except requests.RequestException:
        return {
            "success": False,
            "action": "api_health_check",
            "error": "Configured health endpoint request failed",
        }
