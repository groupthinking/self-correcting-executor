# Database-Powered Protocol: Execution Analytics
import psycopg2
import os
from datetime import datetime, timedelta
import json

def task():
    """Analyze execution patterns and provide insights from database"""
    try:
        # Connect to database
        conn = psycopg2.connect(
            host=os.environ.get('POSTGRES_HOST', 'mcp_db'),
            port=os.environ.get('POSTGRES_PORT', '5432'),
            user=os.environ.get('POSTGRES_USER', 'mcp'),
            password=os.environ.get('POSTGRES_PASSWORD', 'mcp'),
            database=os.environ.get('POSTGRES_DB', 'mcp')
        )
        cursor = conn.cursor()
        
        # Get overall statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as total_executions,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_runs,
                COUNT(DISTINCT protocol_name) as unique_protocols,
                MIN(execution_time) as first_execution,
                MAX(execution_time) as last_execution
            FROM protocol_executions
        """)
        
        overall_stats = cursor.fetchone()
        total, successes, unique_protocols, first_exec, last_exec = overall_stats
        
        # Get per-protocol performance
        cursor.execute("""
            SELECT 
                protocol_name,
                COUNT(*) as runs,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes,
                AVG(CASE WHEN success THEN 1 ELSE 0 END) * 100 as success_rate
            FROM protocol_executions
            GROUP BY protocol_name
            ORDER BY success_rate DESC
        """)
        
        protocol_performance = []
        for row in cursor.fetchall():
            protocol_name, runs, successes, success_rate = row
            protocol_performance.append({
                'protocol': protocol_name,
                'runs': runs,
                'successes': successes,
                'success_rate': round(success_rate, 2)
            })
        
        # Get recent failure patterns
        cursor.execute("""
            SELECT 
                protocol_name,
                (details->>'error')::text as error_message,
                COUNT(*) as occurrences
            FROM protocol_executions
            WHERE success = false 
            AND execution_time > NOW() - INTERVAL '1 hour'
            AND details->>'error' IS NOT NULL
            GROUP BY protocol_name, error_message
            ORDER BY occurrences DESC
            LIMIT 5
        """)
        
        failure_patterns = []
        for row in cursor.fetchall():
            protocol, error, count = row
            failure_patterns.append({
                'protocol': protocol,
                'error': error,
                'occurrences': count
            })
        
        # Get mutation effectiveness
        cursor.execute("""
            SELECT 
                pm.protocol_name,
                pm.previous_failure_rate * 100 as before_mutation,
                COALESCE(current_stats.success_rate, 0) as after_mutation
            FROM protocol_mutations pm
            LEFT JOIN (
                SELECT 
                    protocol_name,
                    AVG(CASE WHEN success THEN 1 ELSE 0 END) * 100 as success_rate
                FROM protocol_executions
                WHERE execution_time > (
                    SELECT MAX(mutation_time) 
                    FROM protocol_mutations 
                    WHERE protocol_name = protocol_executions.protocol_name
                )
                GROUP BY protocol_name
            ) current_stats ON pm.protocol_name = current_stats.protocol_name
            ORDER BY pm.mutation_time DESC
            LIMIT 5
        """)
        
        mutation_effectiveness = []
        for row in cursor.fetchall():
            protocol, before, after = row
            mutation_effectiveness.append({
                'protocol': protocol,
                'failure_rate_before': round(before, 2),
                'success_rate_after': round(after or 0, 2),
                'improvement': round((after or 0) - (100 - before), 2)
            })
        
        cursor.close()
        conn.close()
        
        # Generate insights
        insights = []
        if total > 0:
            overall_success_rate = (successes / total) * 100
            insights.append(f"Overall success rate: {overall_success_rate:.1f}%")
            
            if overall_success_rate < 50:
                insights.append("⚠️ System performance below 50% - review failing protocols")
            elif overall_success_rate > 80:
                insights.append("✅ System performing well with >80% success rate")
                
            if len(failure_patterns) > 0:
                insights.append(f"Most common error: '{failure_patterns[0]['error']}' ({failure_patterns[0]['occurrences']} times)")
        
        return {
            'success': True,
            'action': 'execution_analytics',
            'overall_stats': {
                'total_executions': total,
                'successful_runs': successes,
                'unique_protocols': unique_protocols,
                'time_range': {
                    'first': first_exec.isoformat() if first_exec else None,
                    'last': last_exec.isoformat() if last_exec else None
                }
            },
            'protocol_performance': protocol_performance,
            'recent_failures': failure_patterns,
            'mutation_effectiveness': mutation_effectiveness,
            'insights': insights,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            'success': False,
            'action': 'execution_analytics',
            'error': str(e)
        }
