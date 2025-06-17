# Core execution agent
import json
from protocols import loader
from utils.logger import log
from utils.tracker import track_outcome

def execute_task(protocol_name):
    protocol = loader.load_protocol(protocol_name)
    log(f"Executing protocol: {protocol_name}")
    outcome = protocol['task']()
    track_outcome(protocol_name, outcome)