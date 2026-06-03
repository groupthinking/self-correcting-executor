#!/usr/bin/env python3
"""
Pattern Detector for analyzing execution patterns and generating insights.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

from utils.db_tracker import ensure_tables_exist, get_db_connection

logger = logging.getLogger(__name__)

# The executions table is ensured once per process (the writer,
# utils.db_tracker.track_outcome, also creates it on demand), so the read path
# does not issue CREATE TABLE statements on every call.
_tables_ready = False


class PatternDetector:
    """
    Analyzes execution patterns and generates insights and recommendations.

    Execution history is read from the shared PostgreSQL ``protocol_executions``
    table that ``utils.db_tracker`` writes to, so the detector sees the same
    data the rest of the system records.
    """

    async def detect_patterns(self) -> Dict[str, Any]:
        """
        Detect patterns in execution history from the PostgreSQL
        ``protocol_executions`` table.

        The queries use the synchronous psycopg2 driver, so the blocking work
        runs in a worker thread to avoid stalling the event loop.

        Returns:
            Dict[str, Any]: Detected patterns
        """
        return await asyncio.to_thread(self._detect_patterns_sync)

    def _detect_patterns_sync(self) -> Dict[str, Any]:
        """Run the blocking database queries for :meth:`detect_patterns`."""
        global _tables_ready
        conn = None
        try:
            # Ensure the executions table exists once per process rather than on
            # every call (the write path also creates it on demand).
            if not _tables_ready:
                ensure_tables_exist()
                _tables_ready = True

            conn = get_db_connection()
            cursor = conn.cursor()

            # Look at execution history for the last 7 days.
            seven_days_ago = datetime.utcnow() - timedelta(days=7)

            # Detect repeated failures.
            cursor.execute(
                """
                SELECT protocol_name, COUNT(*) AS failure_count
                FROM protocol_executions
                WHERE success = FALSE AND execution_time > %s
                GROUP BY protocol_name
                HAVING COUNT(*) > 1
                ORDER BY failure_count DESC
                """,
                (seven_days_ago,),
            )

            repeated_failures = []
            for protocol, count in cursor.fetchall():
                severity = "high" if count > 3 else "medium"
                repeated_failures.append(
                    {
                        "protocol": protocol,
                        "failure_count": count,
                        "severity": severity,
                    }
                )

            # Detect slow protocols. Duration is stored inside the JSONB
            # ``details`` payload, so average only the numeric values present.
            cursor.execute(
                r"""
                SELECT protocol_name,
                       AVG((details ->> 'duration')::float) AS avg_duration,
                       COUNT(*) AS exec_count
                FROM protocol_executions
                WHERE execution_time > %s
                  AND details ? 'duration'
                  AND (details ->> 'duration') ~ '^[0-9]+(\.[0-9]+)?$'
                GROUP BY protocol_name
                HAVING COUNT(*) > 2
                ORDER BY avg_duration DESC
                LIMIT 5
                """,
                (seven_days_ago,),
            )

            slow_protocols = []
            for protocol, avg_duration, count in cursor.fetchall():
                if avg_duration is not None and avg_duration > 1.0:
                    slow_protocols.append(
                        {
                            "protocol": protocol,
                            "avg_duration": float(avg_duration),
                            "execution_count": count,
                        }
                    )

            # Detect most-used protocols.
            cursor.execute(
                """
                SELECT protocol_name, COUNT(*) AS usage_count
                FROM protocol_executions
                WHERE execution_time > %s
                GROUP BY protocol_name
                ORDER BY usage_count DESC
                LIMIT 5
                """,
                (seven_days_ago,),
            )

            top_protocols = [
                {"protocol": protocol, "usage_count": count}
                for protocol, count in cursor.fetchall()
            ]

            cursor.close()

            return {
                "failure_patterns": {
                    "repeated_failures": repeated_failures,
                },
                "performance_patterns": {
                    "slow_protocols": slow_protocols,
                },
                "usage_patterns": {
                    "top_protocols": top_protocols,
                },
            }

        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return {
                "failure_patterns": {"repeated_failures": []},
                "performance_patterns": {"slow_protocols": []},
                "usage_patterns": {"top_protocols": []},
                "error": str(e),
            }
        finally:
            if conn is not None:
                conn.close()

    async def generate_insights(
        self,
        failure_patterns: Dict[str, Any],
        performance_patterns: Dict[str, Any],
        usage_patterns: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Generate insights from detected patterns.

        Args:
            failure_patterns: Detected failure patterns
            performance_patterns: Detected performance patterns
            usage_patterns: Detected usage patterns

        Returns:
            List[Dict[str, Any]]: List of insights
        """
        insights = []

        # Failure insights
        for repeated in failure_patterns["repeated_failures"]:
            insights.append(
                {
                    "type": "repeated_failure",
                    "severity": repeated["severity"],
                    "message": f"Protocol {repeated['protocol']} has failed {repeated['failure_count']} times",
                }
            )

        # Performance insights
        for slow in performance_patterns["slow_protocols"]:
            insights.append(
                {
                    "type": "performance_issue",
                    "severity": "medium",
                    "message": f"Protocol {slow['protocol']} averages {slow['avg_duration']:.2f}s execution time",
                    "recommendation": "Optimize algorithm or add caching",
                    "data": slow,
                }
            )

        # Usage insights
        if "top_protocols" in usage_patterns and usage_patterns["top_protocols"]:
            top_protocol = usage_patterns["top_protocols"][0]
            insights.append(
                {
                    "type": "high_usage",
                    "message": f"Protocol {top_protocol['protocol']} is most used ({top_protocol['usage_count']} times)",
                }
            )

        return insights

    async def generate_mutation_recommendations(
        self, insights: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate mutation recommendations based on insights.

        Args:
            insights: List of insights

        Returns:
            List[Dict[str, Any]]: List of mutation recommendations
        """
        recommendations = []

        for insight in insights:
            if insight["type"] == "repeated_failure":
                protocol = insight["message"].split()[1]  # Extract protocol name
                recommendations.append(
                    {
                        "protocol": protocol,
                        "mutation_type": "error_handling",
                        "priority": "high",
                        "suggested_changes": [
                            "Add retry logic with exponential backoff",
                            "Implement better error handling",
                            "Add input validation",
                            "Consider circuit breaker pattern",
                        ],
                        "reason": insight["message"],
                    }
                )
            elif insight["type"] == "performance_issue":
                protocol = insight["data"]["protocol"]
                recommendations.append(
                    {
                        "protocol": protocol,
                        "mutation_type": "performance_optimization",
                        "priority": "medium",
                        "suggested_changes": [
                            "Add caching layer",
                            "Optimize database queries",
                            "Implement pagination",
                        ],
                        "reason": insight["message"],
                    }
                )

        return recommendations

    async def analyze(self) -> Dict[str, Any]:
        """
        Analyze execution history and generate insights and recommendations.

        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            patterns = await self.detect_patterns()

            insights = await self.generate_insights(
                patterns["failure_patterns"],
                patterns["performance_patterns"],
                patterns["usage_patterns"],
            )

            recommendations = await self.generate_mutation_recommendations(insights)

            return {
                "success": True,
                "patterns": patterns,
                "insights": insights,
                "recommendations": recommendations,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error in pattern analysis: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
