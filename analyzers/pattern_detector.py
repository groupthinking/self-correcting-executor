#!/usr/bin/env python3
"""
Pattern Detector for analyzing execution patterns and generating insights.
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class PatternDetector:
    """
    Analyzes execution patterns and generates insights and recommendations.
    """

    def __init__(self, db_path: str = "data/executor.db"):
        self.db_path = db_path

    async def detect_patterns(self) -> Dict[str, Any]:
        """
        Detect patterns in execution history.

        Returns:
            Dict[str, Any]: Detected patterns
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get execution history for the last 7 days
            seven_days_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()

            # Detect failure patterns
            cursor.execute(
                """
                SELECT protocol_name, COUNT(*) as failure_count
                FROM execution_history
                WHERE success = 0 AND timestamp > ?
                GROUP BY protocol_name
                HAVING COUNT(*) > 1
                ORDER BY failure_count DESC
                """,
                (seven_days_ago,),
            )

            repeated_failures = []
            for row in cursor.fetchall():
                protocol, count = row
                severity = "high" if count > 3 else "medium"
                repeated_failures.append(
                    {
                        "protocol": protocol,
                        "failure_count": count,
                        "severity": severity,
                    }
                )

            # Detect performance patterns
            cursor.execute(
                """
                SELECT protocol_name, AVG(duration) as avg_duration, COUNT(*) as exec_count
                FROM execution_history
                WHERE timestamp > ?
                GROUP BY protocol_name
                HAVING COUNT(*) > 2
                ORDER BY avg_duration DESC
                LIMIT 5
                """,
                (seven_days_ago,),
            )

            slow_protocols = []
            for row in cursor.fetchall():
                protocol, avg_duration, count = row
                if avg_duration > 1.0:  # More than 1 second
                    slow_protocols.append(
                        {
                            "protocol": protocol,
                            "avg_duration": avg_duration,
                            "execution_count": count,
                        }
                    )

            # Detect usage patterns
            cursor.execute(
                """
                SELECT protocol_name, COUNT(*) as usage_count
                FROM execution_history
                WHERE timestamp > ?
                GROUP BY protocol_name
                ORDER BY usage_count DESC
                LIMIT 5
                """,
                (seven_days_ago,),
            )

            top_protocols = []
            for row in cursor.fetchall():
                protocol, count = row
                top_protocols.append(
                    {
                        "protocol": protocol,
                        "usage_count": count,
                    }
                )

            conn.close()

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
