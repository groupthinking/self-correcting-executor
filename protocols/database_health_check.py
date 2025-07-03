# Real Protocol: Database Health Check
import psycopg2
import os
from datetime import datetime


def task():
    """Check PostgreSQL database connectivity and health"""
    try:
        # Get database credentials from environment
        db_config = {
            "host": os.environ.get("POSTGRES_HOST", "mcp_db"),
            "port": os.environ.get("POSTGRES_PORT", "5432"),
            "user": os.environ.get("POSTGRES_USER", "mcp"),
            "password": os.environ.get("POSTGRES_PASSWORD", "mcp"),
            "database": os.environ.get("POSTGRES_DB", "mcp"),
        }

        # Connect to database
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        # Check database version
        cursor.execute("SELECT version();")
        db_version = cursor.fetchone()[0]

        # Check database size
        cursor.execute("SELECT pg_database_size(current_database());")
        db_size = cursor.fetchone()[0]

        # Check connection count
        cursor.execute("SELECT count(*) FROM pg_stat_activity;")
        connection_count = cursor.fetchone()[0]

        # Check table count
        cursor.execute(
            """
            SELECT count(*)
            FROM information_schema.tables
            WHERE table_schema = 'public';
        """
        )
        table_count = cursor.fetchone()[0]

        # Create a test table if it doesn't exist
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS protocol_executions (
                id SERIAL PRIMARY KEY,
                protocol_name VARCHAR(100),
                execution_time TIMESTAMP,
                success BOOLEAN,
                details JSONB
            );
        """
        )

        # Insert a test record
        cursor.execute(
            """
            INSERT INTO protocol_executions (protocol_name, execution_time, success, details)
            VALUES (%s, %s, %s, %s);
        """,
            (
                "database_health_check",
                datetime.utcnow(),
                True,
                '{"action": "health_check"}',
            ),
        )

        conn.commit()
        cursor.close()
        conn.close()

        return {
            "success": True,
            "action": "database_health_check",
            "database_info": {
                # Truncate long version string
                "version": db_version.split(",")[0],
                "size_bytes": db_size,
                "size_mb": round(db_size / (1024 * 1024), 2),
                "connection_count": connection_count,
                "table_count": table_count,
            },
            "test_write": "successful",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        return {
            "success": False,
            "action": "database_health_check",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }
