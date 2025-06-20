# Database-backed outcome tracking
import psycopg2
import json
import os
from datetime import datetime
from utils.logger import log

def get_db_connection():
    """Get database connection using environment variables"""
    return psycopg2.connect(
        host=os.environ.get('POSTGRES_HOST', 'mcp_db'),
        port=os.environ.get('POSTGRES_PORT', '5432'),
        user=os.environ.get('POSTGRES_USER', 'mcp'),
        password=os.environ.get('POSTGRES_PASSWORD', 'mcp'),
        database=os.environ.get('POSTGRES_DB', 'mcp')
    )

def ensure_tables_exist():
    """Create necessary tables if they don't exist"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Create protocol_executions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS protocol_executions (
                id SERIAL PRIMARY KEY,
                protocol_name VARCHAR(100),
                execution_time TIMESTAMP,
                success BOOLEAN,
                details JSONB
            );
        """)
        
        # Create protocol_mutations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS protocol_mutations (
                id SERIAL PRIMARY KEY,
                protocol_name VARCHAR(100) NOT NULL,
                mutation_time TIMESTAMP NOT NULL,
                previous_failure_rate FLOAT,
                new_code TEXT,
                backup_code TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create indexes for better performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_executions_protocol 
            ON protocol_executions(protocol_name);
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_executions_time 
            ON protocol_executions(execution_time);
        """)
        
        conn.commit()
        log("Database tables initialized successfully")
        
    except Exception as e:
        log(f"Error creating tables: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def track_outcome(protocol_name, outcome):
    """Track protocol outcome in database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Ensure tables exist
        ensure_tables_exist()
        
        # Insert execution record
        cursor.execute("""
            INSERT INTO protocol_executions 
            (protocol_name, execution_time, success, details)
            VALUES (%s, %s, %s, %s)
        """, (
            protocol_name,
            datetime.utcnow(),
            outcome.get('success', False),
            json.dumps(outcome)
        ))
        
        conn.commit()
        log(f"Outcome tracked in database for {protocol_name}: {outcome.get('success', 'unknown')}")
        
    except Exception as e:
        log(f"Failed to track outcome in database: {e}")
        conn.rollback()
        # Fall back to file-based tracking
        _track_to_file(protocol_name, outcome)
    finally:
        cursor.close()
        conn.close()

def _track_to_file(protocol_name, outcome):
    """Fallback file-based tracking"""
    os.makedirs('memory', exist_ok=True)
    enhanced_outcome = {
        **outcome,
        'protocol': protocol_name,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    memory_file = f'memory/{protocol_name}.json'
    try:
        with open(memory_file, 'a') as f:
            f.write(json.dumps(enhanced_outcome) + '\n')
        log(f"Outcome tracked to file for {protocol_name} (database unavailable)")
    except Exception as e:
        log(f"Failed to track outcome to file: {e}")

def get_protocol_stats(protocol_name):
    """Get statistics for a specific protocol from database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes,
                SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as failures,
                MAX(execution_time) as last_execution
            FROM protocol_executions
            WHERE protocol_name = %s
        """, (protocol_name,))
        
        result = cursor.fetchone()
        if result and result[0] > 0:
            total, successes, failures, last_execution = result
            return {
                'protocol': protocol_name,
                'total_executions': total,
                'successes': successes,
                'failures': failures,
                'success_rate': successes / total if total > 0 else 0,
                'failure_rate': failures / total if total > 0 else 0,
                'last_execution': last_execution.isoformat() if last_execution else None
            }
        else:
            return None
            
    except Exception as e:
        log(f"Error reading stats from database: {e}")
        # Fall back to file-based stats
        return _get_file_stats(protocol_name)
    finally:
        cursor.close()
        conn.close()

def _get_file_stats(protocol_name):
    """Fallback file-based statistics"""
    memory_file = f'memory/{protocol_name}.json'
    if not os.path.exists(memory_file):
        return None
    
    total = 0
    successes = 0
    failures = 0
    
    try:
        with open(memory_file, 'r') as f:
            for line in f:
                if line.strip():
                    outcome = json.loads(line)
                    total += 1
                    if outcome.get('success', False):
                        successes += 1
                    else:
                        failures += 1
    except Exception as e:
        log(f"Error reading file stats: {e}")
        return None
    
    return {
        'protocol': protocol_name,
        'total_executions': total,
        'successes': successes,
        'failures': failures,
        'success_rate': successes / total if total > 0 else 0,
        'failure_rate': failures / total if total > 0 else 0
    }

def get_all_stats():
    """Get statistics for all protocols from database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT 
                protocol_name,
                COUNT(*) as total,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes,
                SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as failures
            FROM protocol_executions
            GROUP BY protocol_name
            ORDER BY protocol_name
        """)
        
        stats = []
        for row in cursor.fetchall():
            protocol_name, total, successes, failures = row
            stats.append({
                'protocol': protocol_name,
                'total_executions': total,
                'successes': successes,
                'failures': failures,
                'success_rate': successes / total if total > 0 else 0,
                'failure_rate': failures / total if total > 0 else 0
            })
        
        return stats
        
    except Exception as e:
        log(f"Error reading all stats from database: {e}")
        # Fall back to file-based stats
        return _get_all_file_stats()
    finally:
        cursor.close()
        conn.close()

def _get_all_file_stats():
    """Fallback file-based statistics for all protocols"""
    stats = []
    memory_dir = 'memory'
    if not os.path.exists(memory_dir):
        return stats
    
    for filename in os.listdir(memory_dir):
        if filename.endswith('.json'):
            protocol_name = filename[:-5]
            protocol_stats = _get_file_stats(protocol_name)
            if protocol_stats:
                stats.append(protocol_stats)
    
    return stats

def track_mutation(protocol_name, failure_rate, new_code, backup_code):
    """Track protocol mutation in database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO protocol_mutations 
            (protocol_name, mutation_time, previous_failure_rate, new_code, backup_code)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            protocol_name,
            datetime.utcnow(),
            failure_rate,
            new_code,
            backup_code
        ))
        
        conn.commit()
        log(f"Mutation tracked in database for {protocol_name}")
        
    except Exception as e:
        log(f"Failed to track mutation in database: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def get_mutation_history(protocol_name):
    """Get mutation history for a protocol"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT 
                mutation_time,
                previous_failure_rate,
                new_code
            FROM protocol_mutations
            WHERE protocol_name = %s
            ORDER BY mutation_time DESC
            LIMIT 10
        """, (protocol_name,))
        
        history = []
        for row in cursor.fetchall():
            mutation_time, failure_rate, new_code = row
            history.append({
                'mutation_time': mutation_time.isoformat(),
                'previous_failure_rate': failure_rate,
                'code_preview': new_code[:200] + '...' if len(new_code) > 200 else new_code
            })
        
        return history
        
    except Exception as e:
        log(f"Error reading mutation history: {e}")
        return []
    finally:
        cursor.close()
        conn.close() 