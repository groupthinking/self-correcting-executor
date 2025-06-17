# Track outcomes and self-correct
import json

def track_outcome(protocol_name, outcome):
    # Simple outcome store
    with open(f'memory/{protocol_name}.json', 'a') as f:
        f.write(json.dumps(outcome) + '\n')