# Enhanced outcome tracking and analysis
import json
import os
from utils.logger import log

def track_outcome(protocol_name, outcome):
    """Track protocol outcome to memory for later analysis"""
    # Ensure memory directory exists
    os.makedirs('memory', exist_ok=True)
    
    # Add metadata to outcome
    enhanced_outcome = {
        **outcome,
        'protocol': protocol_name,
        'timestamp': __import__('datetime').datetime.utcnow().isoformat()
    }
    
    memory_file = f'memory/{protocol_name}.json'
    try:
        with open(memory_file, 'a') as f:
            f.write(json.dumps(enhanced_outcome) + '\n')
        log(f"Outcome tracked for {protocol_name}: {outcome.get('success', 'unknown')}")
    except Exception as e:
        log(f"Failed to track outcome for {protocol_name}: {e}")

def get_protocol_stats(protocol_name):
    """Get statistics for a specific protocol"""
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
        log(f"Error reading stats for {protocol_name}: {e}")
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
    """Get statistics for all protocols"""
    stats = []
    memory_dir = 'memory'
    if not os.path.exists(memory_dir):
        return stats
    
    for filename in os.listdir(memory_dir):
        if filename.endswith('.json'):
            protocol_name = filename[:-5]  # Remove .json extension
            protocol_stats = get_protocol_stats(protocol_name)
            if protocol_stats:
                stats.append(protocol_stats)
    
    return stats

def clear_memory(protocol_name=None):
    """Clear memory for a specific protocol or all protocols"""
    if protocol_name:
        memory_file = f'memory/{protocol_name}.json'
        if os.path.exists(memory_file):
            os.remove(memory_file)
            log(f"Memory cleared for protocol: {protocol_name}")
    else:
        memory_dir = 'memory'
        if os.path.exists(memory_dir):
            for filename in os.listdir(memory_dir):
                if filename.endswith('.json'):
                    os.remove(os.path.join(memory_dir, filename))
            log("All protocol memory cleared")