"""
Quantum Simulator Protocol
Uses dimod for local quantum algorithm testing without D-Wave API access.
"""

import dimod


def execute(protocol_context):
    """
    Demonstrates local quantum optimization using dimod's ExactSolver.
    This works without any API keys or external connections.
    """
    try:
        # Create a simple Binary Quadratic Model (BQM)
        # This represents a basic optimization problem
        linear = {0: -1, 1: 1, 2: -1}  # Linear coefficients
        quadratic = {(0, 1): 2, (1, 2): -1}  # Quadratic coefficients
        
        # Create the BQM
        bqm = dimod.BinaryQuadraticModel(
            linear, quadratic, 0.0, dimod.BINARY
        )
        
        # Solve using local exact solver (no API required)
        sampler = dimod.ExactSolver()
        sampleset = sampler.sample(bqm)
        
        # Get the best solution
        best_sample = sampleset.first
        
        return {
            "status": "completed",
            "quantum_solution": {
                "variables": dict(best_sample.sample),
                "energy": best_sample.energy,
                "num_occurrences": best_sample.num_occurrences,
                "solver_used": "dimod.ExactSolver (local)",
                "problem_size": len(linear),
                "message": "Quantum optimization completed locally"
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Quantum simulation failed: {str(e)}"
        }


if __name__ == "__main__":
    # Test the quantum simulator
    result = execute({})
    print("Quantum Simulation Result:")
    print(result)