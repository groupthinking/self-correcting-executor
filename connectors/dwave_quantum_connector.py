#!/usr/bin/env python3
"""
Real D-Wave Quantum MCP Connector

LEGITIMATE quantum computing integration using D-Wave Ocean SDK and Leap cloud service.
Based on official D-Wave examples and documentation.

Requirements:
- D-Wave Ocean SDK
- Valid D-Wave Leap account and API token
- Internet connection to D-Wave cloud

References:
- https://github.com/dwave-examples
- https://cloud.dwavesys.com/leap/
- https://github.com/dwave-examples/advantage2.git
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np

# Real D-Wave imports (requires: pip install dwave-ocean-sdk)
try:
    from dwave.system import (
        DWaveSampler,
        EmbeddingComposite,
        FixedEmbeddingComposite,
    )
    from dwave.system.composites import LazyFixedEmbeddingComposite
    from dwave.cloud import Client
    # SimulatedAnnealingSampler removed - real QPU only
    import dimod
    from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel
    import dwave.inspector

    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False

from connectors.mcp_base import MCPConnector

logger = logging.getLogger(__name__)


@dataclass
class QuantumResult:
    """Real quantum annealing result from D-Wave"""

    samples: List[Dict[str, int]]
    energies: List[float]
    num_occurrences: List[int]
    timing: Dict[str, float]
    info: Dict[str, Any]
    chain_break_fraction: float
    success: bool
    error_message: Optional[str] = None


class DWaveQuantumConnector(MCPConnector):
    """
    Real D-Wave Quantum MCP Connector

    Provides authentic quantum annealing capabilities through D-Wave Leap cloud service.
    Uses actual D-Wave Ocean SDK - no simulations or fake results.
    """

    def __init__(
        self,
        api_token: Optional[str] = None,
        solver_name: Optional[str] = None,
    ):
        super().__init__("dwave_quantum", "quantum_computing")
        self.api_token = api_token
        self.solver_name = (
            solver_name  # e.g., "Advantage_system6.4" or "Advantage2_prototype"
        )
        self.sampler = None
        self.client = None
        self.solver_info = {}

        if not DWAVE_AVAILABLE:
            logger.error(
                "D-Wave Ocean SDK not installed. Run: pip install dwave-ocean-sdk"
            )

    async def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to D-Wave Leap cloud service"""
        if not DWAVE_AVAILABLE:
            logger.error("D-Wave Ocean SDK not available")
            return False

        try:
            # Get configuration
            self.api_token = config.get("api_token", self.api_token)
            self.solver_name = config.get("solver_name", self.solver_name)

            # Initialize D-Wave client
            if self.api_token:
                self.client = Client.from_config(token=self.api_token)
            else:
                # Try to use default configuration
                self.client = Client.from_config()

            # Get available solvers
            solvers = self.client.get_solvers()
            qpu_solvers = [s for s in solvers if hasattr(s, "qubits")]

            if not qpu_solvers:
                # NO SIMULATION - Real QPU required
                logger.error("No QPU solvers available. Real quantum hardware required.")
                raise RuntimeError(
                    "No D-Wave QPU available. Please ensure:\n"
                    "1. Valid DWAVE_API_TOKEN is set\n"
                    "2. Active D-Wave Leap account with QPU access\n"
                    "3. Internet connection to D-Wave cloud\n"
                    "Visit: https://cloud.dwavesys.com/leap/"
                )
            else:
                # Use specified solver or first available QPU
                if self.solver_name:
                    solver = next(
                        (s for s in qpu_solvers if self.solver_name in s.id),
                        qpu_solvers[0],
                    )
                else:
                    solver = qpu_solvers[0]

                self.sampler = EmbeddingComposite(DWaveSampler(solver=solver.id))
                self.solver_info = {
                    "name": solver.id,
                    "type": "QPU",
                    "num_qubits": len(solver.nodes),
                    "num_couplers": len(solver.edges),
                    "topology": getattr(solver, "topology", "Unknown"),
                    "programming_thermalization": solver.properties.get(
                        "programming_thermalization_range"
                    ),
                    "annealing_time_range": solver.properties.get(
                        "annealing_time_range"
                    ),
                }

            self.connected = True
            logger.info(f"Connected to D-Wave solver: {self.solver_info['name']}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to D-Wave: {e}")
            self.connected = False
            return False

    async def disconnect(self) -> bool:
        """Disconnect from D-Wave service"""
        if self.client:
            self.client.close()
        self.connected = False
        return True

    async def get_context(self):
        """Get quantum system context"""
        return self.context

    async def send_context(self, context) -> bool:
        """Send context to quantum system"""
        self.context = context
        return True
    
    async def ensure_real_qpu(self) -> bool:
        """Ensure we have real QPU access - no simulations allowed"""
        if not self.connected:
            raise RuntimeError("Not connected to D-Wave service")
        
        if self.solver_info.get('type') != 'QPU':
            raise RuntimeError(
                f"Real QPU required but got {self.solver_info.get('type')}. "
                "No simulations allowed in production."
            )
        
        return True
    
    async def list_solvers(self) -> List[Dict[str, Any]]:
        """List all available D-Wave solvers"""
        if not self.client:
            raise RuntimeError("Not connected to D-Wave service")
        
        solvers = self.client.get_solvers()
        solver_list = []
        
        for solver in solvers:
            solver_info = {
                "id": solver.id,
                "name": solver.name,
                "is_qpu": hasattr(solver, 'qubits'),
                "status": solver.status
            }
            
            if hasattr(solver, 'qubits'):
                solver_info.update({
                    "num_qubits": len(solver.nodes),
                    "num_couplers": len(solver.edges),
                    "topology": getattr(solver, 'topology', 'Unknown')
                })
            
            solver_list.append(solver_info)
        
        return solver_list
    
    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute quantum action"""
        if not self.connected:
            return {"error": "Not connected to D-Wave service"}

        actions = {
            "solve_qubo": self.solve_qubo,
            "solve_ising": self.solve_ising,
            "traveling_salesman": self.solve_tsp,
            "max_cut": self.solve_max_cut,
            "knapsack": self.solve_knapsack,
            "get_solver_info": self.get_solver_info,
        }

        handler = actions.get(action)
        if handler:
            try:
                result = await handler(params)
                return result
            except Exception as e:
                return {"error": str(e), "action": action}

        return {"error": f"Unknown action: {action}"}

    async def solve_qubo(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve Quadratic Unconstrained Binary Optimization (QUBO) problem

        Based on D-Wave examples: https://github.com/dwave-examples
        """
        try:
            qubo_dict = params.get("qubo", {})
            num_reads = params.get("num_reads", 100)
            annealing_time = params.get("annealing_time", 20)  # microseconds

            if not qubo_dict:
                return {"error": "QUBO dictionary required"}

            # Convert string keys to tuples if needed
            if isinstance(list(qubo_dict.keys())[0], str):
                # Handle string representation like "x0*x1": coeff
                processed_qubo = {}
                for key, value in qubo_dict.items():
                    if "*" in key:
                        vars = key.split("*")
                        i, j = int(vars[0][1:]), int(vars[1][1:])
                        processed_qubo[(i, j)] = value
                    else:
                        i = int(key[1:])
                        processed_qubo[(i, i)] = value
                qubo_dict = processed_qubo

            # Create BQM
            bqm = dimod.BinaryQuadraticModel.from_qubo(qubo_dict)

            # Sample using real D-Wave hardware or simulator
            sampleset = self.sampler.sample(
                bqm,
                num_reads=num_reads,
                annealing_time=(
                    annealing_time if hasattr(self.sampler, "annealing_time") else None
                ),
                return_embedding=True,
            )

            # Process results
            best_sample = sampleset.first.sample
            best_energy = sampleset.first.energy

            # Calculate chain break information if available
            chain_break_fraction = 0.0
            if (
                hasattr(sampleset, "data_vectors")
                and "chain_break_fraction" in sampleset.data_vectors
            ):
                chain_break_fraction = np.mean(
                    sampleset.data_vectors["chain_break_fraction"]
                )

            return {
                "success": True,
                "best_solution": best_sample,
                "best_energy": best_energy,
                "num_solutions": len(sampleset),
                "chain_break_fraction": chain_break_fraction,
                "timing": sampleset.info.get("timing", {}),
                "solver_info": self.solver_info,
                "all_samples": [
                    dict(sample) for sample in sampleset.samples()[:10]
                ],  # First 10
                "energies": list(sampleset.data_vectors["energy"][:10]),
            }

        except Exception as e:
            logger.error(f"QUBO solving failed: {e}")
            return {"success": False, "error": str(e)}

    async def solve_ising(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve Ising model problem

        Based on real D-Wave Ising formulations
        """
        try:
            h = params.get("h", {})  # Linear terms
            J = params.get("J", {})  # Quadratic terms
            num_reads = params.get("num_reads", 100)

            # Create BQM from Ising model
            bqm = dimod.BinaryQuadraticModel.from_ising(h, J)

            # Sample using D-Wave
            sampleset = self.sampler.sample(bqm, num_reads=num_reads)

            best_sample = sampleset.first.sample
            best_energy = sampleset.first.energy

            return {
                "success": True,
                "best_solution": best_sample,
                "best_energy": best_energy,
                "solver_info": self.solver_info,
                "timing": sampleset.info.get("timing", {}),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def solve_tsp(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve Traveling Salesman Problem using quantum annealing

        Based on: https://github.com/dwave-examples/tsp
        """
        try:
            cities = params.get("cities", [])
            distances = params.get("distances", {})

            if len(cities) < 3:
                return {"error": "Need at least 3 cities for TSP"}

            n = len(cities)

            # Create QUBO formulation for TSP
            # Variables: x_i_t = 1 if city i is visited at time t
            Q = {}

            # Constraint: Each city visited exactly once
            for i in range(n):
                for t1 in range(n):
                    for t2 in range(t1 + 1, n):
                        Q[(i * n + t1, i * n + t2)] = (
                            2  # Penalty for visiting city i at multiple times
                        )

            # Constraint: Each time slot has exactly one city
            for t in range(n):
                for i1 in range(n):
                    for i2 in range(i1 + 1, n):
                        Q[(i1 * n + t, i2 * n + t)] = (
                            2  # Penalty for multiple cities at time t
                        )

            # Objective: Minimize total distance
            for i in range(n):
                for j in range(n):
                    if i != j:
                        dist = distances.get(
                            f"{cities[i]}-{cities[j]}",
                            distances.get((i, j), 1),
                        )
                        for t in range(n):
                            t_next = (t + 1) % n
                            Q[(i * n + t, j * n + t_next)] = dist

            # Solve QUBO
            result = await self.solve_qubo(
                {"qubo": Q, "num_reads": params.get("num_reads", 100)}
            )

            if result.get("success"):
                # Convert solution back to route
                solution = result["best_solution"]
                route = [""] * n
                for var, val in solution.items():
                    if val == 1:
                        city_idx = var // n
                        time_idx = var % n
                        route[time_idx] = cities[city_idx]

                # Calculate total distance
                total_distance = 0
                for i in range(n):
                    current_city = route[i]
                    next_city = route[(i + 1) % n]
                    current_idx = cities.index(current_city)
                    next_idx = cities.index(next_city)
                    total_distance += distances.get(
                        f"{current_city}-{next_city}",
                        distances.get((current_idx, next_idx), 1),
                    )

                result["route"] = route
                result["total_distance"] = total_distance
                result["problem_type"] = "TSP"

            return result

        except Exception as e:
            return {"success": False, "error": str(e), "problem_type": "TSP"}

    async def solve_max_cut(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve Maximum Cut problem

        Based on D-Wave graph partitioning examples
        """
        try:
            graph_edges = params.get("edges", [])
            weights = params.get("weights", {})

            # Create QUBO for Max-Cut
            Q = {}

            for i, (u, v) in enumerate(graph_edges):
                weight = weights.get((u, v), weights.get((v, u), 1))
                # Max-Cut: maximize sum of weights for cut edges
                # QUBO formulation: -weight * (x_u + x_v - 2*x_u*x_v)
                Q[(u, u)] = Q.get((u, u), 0) - weight
                Q[(v, v)] = Q.get((v, v), 0) - weight
                Q[(u, v)] = Q.get((u, v), 0) + 2 * weight

            result = await self.solve_qubo(
                {"qubo": Q, "num_reads": params.get("num_reads", 100)}
            )

            if result.get("success"):
                solution = result["best_solution"]
                set_a = [node for node, val in solution.items() if val == 0]
                set_b = [node for node, val in solution.items() if val == 1]

                # Calculate cut value
                cut_value = 0
                for u, v in graph_edges:
                    if (u in set_a and v in set_b) or (u in set_b and v in set_a):
                        cut_value += weights.get((u, v), weights.get((v, u), 1))

                result["partition_a"] = set_a
                result["partition_b"] = set_b
                result["cut_value"] = cut_value
                result["problem_type"] = "Max-Cut"

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "problem_type": "Max-Cut",
            }

    async def solve_knapsack(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve 0-1 Knapsack problem using quantum annealing
        """
        try:
            items = params.get("items", [])  # List of {'weight': w, 'value': v}
            capacity = params.get("capacity", 10)

            Q = {}
            penalty = max(item["value"] for item in items) * 2  # Large penalty

            # Objective: maximize value (minimize negative value)
            for i, item in enumerate(items):
                Q[(i, i)] = -item["value"]

            # Constraint: weight <= capacity
            # (sum(w_i * x_i) - capacity)^2 penalty term
            n = len(items)  # Define n as the number of items
            for i in range(n):
                for j in range(i, n):
                    weight_product = items[i]["weight"] * items[j]["weight"]
                    if i == j:
                        Q[(i, i)] += penalty * (
                            weight_product - 2 * capacity * items[i]["weight"]
                        )
                    else:
                        Q[(i, j)] = Q.get((i, j), 0) + penalty * weight_product

            # Add capacity^2 term (constant, doesn't affect optimization)

            result = await self.solve_qubo(
                {"qubo": Q, "num_reads": params.get("num_reads", 100)}
            )

            if result.get("success"):
                solution = result["best_solution"]
                selected_items = [i for i, val in solution.items() if val == 1]
                total_weight = sum(items[i]["weight"] for i in selected_items)
                total_value = sum(items[i]["value"] for i in selected_items)

                result["selected_items"] = selected_items
                result["total_weight"] = total_weight
                result["total_value"] = total_value
                result["capacity_used"] = total_weight / capacity
                result["feasible"] = total_weight <= capacity
                result["problem_type"] = "Knapsack"

            return result

        except Exception as e:
            return {"success": False, "error": str(e), "problem_type": "Knapsack"}
    
    async def get_solver_info(self, params: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Get information about the connected D-Wave solver"""
        if not self.connected:
            return {"error": "Not connected to D-Wave service"}

        return {
            "success": True,
            "solver_info": self.solver_info,
            "available": DWAVE_AVAILABLE,
            "connection_status": ("connected" if self.connected else "disconnected"),
        }

    async def ensure_real_qpu(self) -> bool:
        """Ensure we're using a real QPU, not a simulator"""
        if not self.solver_info or self.solver_info.get("type") != "QPU":
            raise RuntimeError("No D-Wave QPU available - real QPU required")
        return True


# Example usage and testing


async def example_usage():
    """Example of using the real D-Wave quantum connector"""

    # NOTE: Requires valid D-Wave Leap account and API token
    connector = DWaveQuantumConnector()

    # Connect (will use default config or environment variables)
    success = await connector.connect({})

    if success:
        print("✅ Connected to D-Wave quantum system")

        # Get solver information
        solver_info = await connector.execute_action("get_solver_info", {})
        print(f"Solver: {solver_info}")

        # Solve a simple QUBO problem
        # Example: x0 + x1 - 2*x0*x1 (prefer x0=1, x1=0 or x0=0, x1=1)
        qubo_result = await connector.execute_action(
            "solve_qubo",
            {"qubo": {(0, 0): 1, (1, 1): 1, (0, 1): -2}, "num_reads": 100},
        )
        print(f"QUBO Result: {qubo_result}")

        # Solve TSP
        tsp_result = await connector.execute_action(
            "traveling_salesman",
            {
                "cities": ["A", "B", "C"],
                "distances": {("A", "B"): 2, ("B", "C"): 3, ("C", "A"): 1},
                "num_reads": 50,
            },
        )
        print(f"TSP Result: {tsp_result}")

        await connector.disconnect()
    else:
        print("❌ Failed to connect to D-Wave")
        print("Required: D-Wave Ocean SDK and valid Leap account")
        print("Install: pip install dwave-ocean-sdk")
        print("Setup: https://cloud.dwavesys.com/leap/")


if __name__ == "__main__":
    if DWAVE_AVAILABLE:
        asyncio.run(example_usage())
    else:
        print("D-Wave Ocean SDK not installed")
        print("Install with: pip install dwave-ocean-sdk")
        print("Sign up for D-Wave Leap: https://cloud.dwavesys.com/leap/")
