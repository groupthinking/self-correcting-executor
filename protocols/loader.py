# Load protocols dynamically
import importlib

def load_protocol(name):
    module = importlib.import_module(f"protocols.{name}")
    return {
        'name': name,
        'task': module.task
    }