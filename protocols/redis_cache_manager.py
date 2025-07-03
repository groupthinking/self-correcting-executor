# Redis-Powered Protocol: Cache Manager
import redis
import json
import time
import os
from datetime import datetime


def task():
    """Manage Redis cache and demonstrate caching benefits"""
    try:
        # Connect to Redis
        r = redis.Redis(
            host=os.environ.get("REDIS_HOST", "mcp_redis"),
            port=int(os.environ.get("REDIS_PORT", "6379")),
            decode_responses=True,
        )

        # Test Redis connectivity
        r.ping()

        # Demonstrate cache usage
        cache_key = "expensive_computation"
        cache_stats_key = "cache_stats"

        # Check if we have cached result
        cached_result = r.get(cache_key)

        if cached_result:
            # Cache hit
            computation_time = 0
            result = json.loads(cached_result)
            cache_status = "hit"

            # Update cache statistics
            r.hincrby(cache_stats_key, "hits", 1)
        else:
            # Cache miss - simulate expensive computation
            start_time = time.time()

            # Simulate expensive operation
            import hashlib

            result = {
                "computed_value": hashlib.sha256(str(time.time()).encode()).hexdigest(),
                "computation_timestamp": datetime.utcnow().isoformat(),
            }
            time.sleep(0.5)  # Simulate processing time

            computation_time = time.time() - start_time
            cache_status = "miss"

            # Store in cache with 5 minute expiration
            r.setex(cache_key, 300, json.dumps(result))

            # Update cache statistics
            r.hincrby(cache_stats_key, "misses", 1)

        # Get cache statistics
        stats = r.hgetall(cache_stats_key) or {"hits": "0", "misses": "0"}
        hits = int(stats.get("hits", 0))
        misses = int(stats.get("misses", 0))
        total_requests = hits + misses
        hit_rate = (hits / total_requests * 100) if total_requests > 0 else 0

        # Store protocol execution metrics in Redis
        execution_key = f"protocol:execution:{int(time.time())}"
        r.hset(
            execution_key,
            mapping={
                "protocol": "redis_cache_manager",
                "status": cache_status,
                "computation_time": computation_time,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
        r.expire(execution_key, 3600)  # Keep for 1 hour

        # Get Redis info
        info = r.info("memory")
        memory_used = info.get("used_memory_human", "unknown")

        # List all protocol execution keys
        recent_executions = []
        for key in r.scan_iter(match="protocol:execution:*", count=10):
            exec_data = r.hgetall(key)
            if exec_data:
                recent_executions.append(exec_data)

        return {
            "success": True,
            "action": "redis_cache_management",
            "cache_status": cache_status,
            "computation_time_seconds": round(computation_time, 3),
            "cached_result": result,
            "cache_statistics": {
                "hits": hits,
                "misses": misses,
                "total_requests": total_requests,
                "hit_rate_percent": round(hit_rate, 2),
            },
            "redis_info": {"memory_used": memory_used, "connected": True},
            "recent_executions": recent_executions[-5:],  # Last 5
            "benefits": {
                "time_saved": f"{0.5 * hits:.1f} seconds saved from cache hits",
                "efficiency": (
                    "High" if hit_rate > 70 else "Medium" if hit_rate > 30 else "Low"
                ),
            },
        }

    except Exception as e:
        return {
            "success": False,
            "action": "redis_cache_management",
            "error": str(e),
            "note": "Redis might not be accessible from container",
        }
