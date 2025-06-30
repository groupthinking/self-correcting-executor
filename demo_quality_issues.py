#!/usr/bin/env python3
"""
Demo file with intentional code quality issues for Guardian Agent V2.0
"""

import os

# TODO: This function needs proper implementation
def demo_function():
    # FIXME: Remove this print statement
    print("Hello World")
    
    # Intentional pylint issues
    unused_variable = 42
    x=1+2+3  # Poor formatting
    
    if True:
        pass  # TODO: Add actual logic here
    
    # HACK: This is a temporary solution
    return "demo"

# More issues
class DemoClass:
    def __init__(self):
        # NotImplementedError placeholder
        raise NotImplementedError("Class not implemented yet")
    
    def method_with_issues(self):
        """Method with various quality issues"""
        # TODO: Implement this method properly
        import sys  # Import not at top
        print(sys.version)  # Using print instead of logging