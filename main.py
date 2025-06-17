# Entry point for the autonomous self-correcting executor
# Integrates with UMDR MCP stack for full runtime capabilities

from agents.executor import execute_task
from agents.mutator import mutate_protocol, analyze_all_protocols
from utils.logger import log
from utils.tracker import get_protocol_stats
import sys

def run_self_correcting_executor(protocol='default_protocol', iterations=1):
    """Run the self-correcting executor with automatic mutation"""
    log(f"🚀 Starting Self-Correcting MCP Executor")
    log(f"Protocol: {protocol}, Iterations: {iterations}")
    
    for i in range(iterations):
        log(f"\n--- Iteration {i+1}/{iterations} ---")
        
        # Execute the protocol
        outcome = execute_task(protocol)
        
        # Immediate mutation check after each execution
        mutated = mutate_protocol(protocol)
        
        # Log iteration summary
        stats = get_protocol_stats(protocol)
        if stats:
            log(f"Current stats - Success rate: {stats['success_rate']:.2%}, "
                f"Total executions: {stats['total_executions']}")
        
        if mutated:
            log(f"🔄 Protocol {protocol} was mutated due to poor performance")
        
    log(f"✅ Self-correcting executor completed {iterations} iterations")
    
    # Final analysis
    final_stats = get_protocol_stats(protocol)
    if final_stats:
        log(f"Final performance - Success rate: {final_stats['success_rate']:.2%}")
    
    return final_stats

if __name__ == '__main__':
    # Command line arguments
    protocol = sys.argv[1] if len(sys.argv) > 1 else 'default_protocol'
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    # Run the self-correcting executor
    run_self_correcting_executor(protocol, iterations)