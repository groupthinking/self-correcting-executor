"""Cache the actual schema and health report under caller-selected Redis settings."""

import json
import os
from urllib.parse import urlsplit
import redis

from workflows.service_scaffold import artifact_metadata


def task(service=None, health_report=None, config=None):
    """Warm metadata and verify readback; never invent a deployment target or key."""
    client = None
    try:
        metadata = artifact_metadata(service)
        if (
            not isinstance(health_report, dict)
            or health_report.get("success") is not True
            or health_report.get("artifact_sha256") != metadata["artifact_sha256"]
        ):
            raise ValueError(
                "A successful health_report for this exact artifact is required"
            )
        if not isinstance(config, dict):
            raise ValueError("cache_config with key and ttl_seconds is required")
        key, ttl = config.get("key"), config.get("ttl_seconds")
        if not isinstance(key, str) or not key.strip():
            raise ValueError("cache_config.key must be nonempty")
        if type(ttl) is not int or ttl <= 0:
            raise ValueError("cache_config.ttl_seconds must be a positive integer")
        url = config.get("redis_url") or os.environ.get("REDIS_URL")
        if url:
            if not isinstance(url, str) or urlsplit(url).scheme not in {
                "redis",
                "rediss",
            }:
                raise ValueError("Redis URL must use redis:// or rediss://")
            client = redis.Redis.from_url(
                url, decode_responses=True, socket_connect_timeout=5, socket_timeout=5
            )
        else:
            host = os.environ.get("REDIS_HOST")
            if not host:
                raise ValueError(
                    "Set REDIS_URL/REDIS_HOST or cache_config.redis_url explicitly"
                )
            client = redis.Redis(
                host=host,
                port=int(os.environ.get("REDIS_PORT", "6379")),
                password=os.environ.get("REDIS_PASSWORD"),
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
        client.ping()
        payload = {**metadata, "health_report": health_report}
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if not client.set(key, serialized, ex=ttl):
            raise RuntimeError("Redis did not acknowledge metadata write")
        if client.get(key) != serialized:
            raise RuntimeError("Redis metadata readback did not match")
        return {
            "success": True,
            "action": "redis_cache_management",
            "deployment_artifacts": {
                **payload,
                "cache_key": key,
                "ttl_seconds": ttl,
                "cache_verified": True,
                "deployment_status": "not_deployed_by_workflow",
            },
        }
    except (ValueError, TypeError, RuntimeError) as exc:
        return {"success": False, "action": "redis_cache_management", "error": str(exc)}
    except redis.RedisError:
        # Client errors can contain credential-bearing URLs; keep them out of telemetry.
        return {
            "success": False,
            "action": "redis_cache_management",
            "error": "Redis metadata write/read failed",
        }
    finally:
        if client is not None:
            client.close()
