# Core execution agent
import json
from protocols import loader
from utils.logger import log
from utils.tracker import track_outcome

def execute_task(protocol_name):
    """Execute a protocol and track its outcome"""
    try:
        protocol = loader.load_protocol(protocol_name)
        log(f"Executing protocol: {protocol_name}")
        outcome = protocol['task']()
        track_outcome(protocol_name, outcome)
        log(f"Protocol {protocol_name} completed with outcome: {outcome}")
        return outcome
    except Exception as e:
        error_outcome = {'success': False, 'error': str(e)}
        track_outcome(protocol_name, error_outcome)
        log(f"Protocol {protocol_name} failed with error: {e}")
        return error_outcome