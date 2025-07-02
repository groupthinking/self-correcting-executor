#!/usr/bin/env python3
"""Simple Quantum Computing Example - No MCP Required"""

import asyncio
from connectors.dwave_quantum_connector import DWaveQuantumConnector

async def solve_optimization_problem():
    """Direct quantum optimization without MCP overhead"""
    
    # 1. Create connector
    quantum = DWaveQuantumConnector()
    
    # 2. Connect (you'll need a D-Wave token)
    connected = await quantum.connect({
        'api_token': 'your-dwave-token'  # Get free at: https://cloud.dwavesys.com/leap/
    })
    
    if not connected:
        print("Failed to connect. Do you have a D-Wave account?")
        return
    
    # 3. Solve a simple QUBO problem
    # Example: Minimize x0 + x1 - 2*x0*x1
    result = await quantum.solve_qubo({
        'qubo': {
            (0, 0): 1,    # x0 coefficient
            (1, 1): 1,    # x1 coefficient  
            (0, 1): -2    # x0*x1 coefficient
        },
        'num_reads': 100
    })
    
    print(f"Best solution: {result['best_solution']}")
    print(f"Best energy: {result['best_energy']}")
    
    # 4. Disconnect
    await quantum.disconnect()

# Run it
if __name__ == "__main__":
    asyncio.run(solve_optimization_problem())