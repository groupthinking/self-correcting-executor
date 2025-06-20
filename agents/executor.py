# Protocol Executor: Loads and executes protocols with adaptive behavior
from protocols.loader import load_protocol
from utils.logger import log

def execute_task(protocol_name='default_protocol'):
    """Execute a specific protocol and return the outcome"""
    # Try database tracker first, fall back to file tracker
    try:
        from utils.db_tracker import track_outcome
    except:
        from utils.tracker import track_outcome
    
    log(f"Executing protocol: {protocol_name}")
    
    # Load the protocol
    protocol = load_protocol(protocol_name)
    if not protocol:
        log(f"Failed to load protocol: {protocol_name}")
        return {'success': False, 'error': 'Protocol not found'}
    
    # Execute the protocol's task function
    try:
        outcome = protocol['task']()
        
        # Track the outcome
        track_outcome(protocol_name, outcome)
        
        log(f"Protocol {protocol_name} completed with outcome: {outcome}")
        return outcome
        
    except Exception as e:
        log(f"Protocol {protocol_name} failed with error: {e}")
        error_outcome = {'success': False, 'error': str(e)}
        track_outcome(protocol_name, error_outcome)
        return error_outcome