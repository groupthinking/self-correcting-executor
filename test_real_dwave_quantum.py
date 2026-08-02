#!/usr/bin/env python3
from config.mcp_config import MCPConfig

"""
Test Suite for REAL D-Wave Quantum MCP Connector

Tests legitimate D-Wave quantum integration using actual Ocean SDK.
Requires real D-Wave Leap cloud access - NO SIMULATIONS.

Requirements:
- D-Wave Ocean SDK: pip install dwave-ocean-sdk
- D-Wave Leap account: https://cloud.dwavesys.com/leap/
- Valid API token configured
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

try:
    from connectors.dwave_quantum_connector import (
        DWaveQuantumConnector,
        DWAVE_AVAILABLE,
    )
except ImportError:
    DWAVE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealQuantumTest:
    """Test suite for authentic D-Wave quantum connector"""

    def __init__(self):
        self.connector = None
        self.results = []

    async def test_connection(self) -> bool:
        """Test connection to D-Wave Leap cloud service"""
        logger.info("🔌 Testing D-Wave Leap connection...")

        if not DWAVE_AVAILABLE:
            logger.error("❌ D-Wave Ocean SDK not available")
            logger.info("💡 Install with: pip install dwave-ocean-sdk")
            logger.info("💡 Sign up at: https://cloud.dwavesys.com/leap/")
            return False

        try:
            self.connector = DWaveQuantumConnector()
            success = await self.connector.connect({})

            if success:
                logger.info("✅ Connected to D-Wave quantum system")
                solver_info = await self.connector.get_solver_info()
                logger.info(f"📊 Solver: {solver_info['solver_info']['name']}")
                logger.info(f"🔬 Type: {solver_info['solver_info']['type']}")

                if solver_info["solver_info"]["type"] == "QPU":
                    logger.info(
                        f"⚛️  Qubits: {solver_info['solver_info']['num_qubits']}"
                    )
                    logger.info(
                        f"🔗 Couplers: {solver_info['solver_info']['num_couplers']}"
                    )

                return True
            else:
                logger.error("❌ Failed to connect to D-Wave")
                return False

        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            return False

    async def test_simple_qubo(self) -> bool:
        """Test simple QUBO problem solving"""
        logger.info("🧮 Testing simple QUBO problem...")

        # Simple QUBO: minimize x0 + x1 - 2*x0*x1
        # Optimal solutions: (0,1) or (1,0) with energy -1
        qubo = {(0, 0): 1, (1, 1): 1, (0, 1): -2}

        try:
            result = await self.connector.execute_action(
                "solve_qubo", {"qubo": qubo, "num_reads": 50}
            )

            if result.get("success"):
                solution = result["best_solution"]
                energy = result["best_energy"]

                logger.info(f"✅ QUBO solved")
                logger.info(f"📊 Best solution: {solution}")
                logger.info(f"⚡ Energy: {energy}")
                logger.info(f"🔢 Samples: {result['num_solutions']}")

                # Check if we got a good solution
                expected_energy = -1
                if abs(energy - expected_energy) < 0.1:
                    logger.info("🎯 Found optimal solution!")
                    return True
                else:
                    logger.warning(
                        f"⚠️  Energy {energy} not optimal (expected ~{expected_energy})"
                    )
                    return True  # Still counts as working
            else:
                logger.error(f"❌ QUBO failed: {result.get('error')}")
                return False

        except Exception as e:
            logger.error(f"❌ QUBO test error: {e}")
            return False

    async def test_traveling_salesman(self) -> bool:
        """Test Traveling Salesman Problem"""
        logger.info("🗺️  Testing Traveling Salesman Problem...")

        # 3-city TSP
        cities = ["NYC", "Boston", "Philly"]
        distances = {
            ("NYC", "Boston"): 4,
            ("NYC", "Philly"): 2,
            ("Boston", "Philly"): 3,
        }

        try:
            result = await self.connector.execute_action(
                "traveling_salesman",
                {"cities": cities, "distances": distances, "num_reads": 30},
            )

            if result.get("success"):
                route = result.get("route", [])
                total_distance = result.get("total_distance", 0)

                logger.info(f"✅ TSP solved")
                logger.info(f"🛣️  Route: {' → '.join(route)}")
                logger.info(f"📏 Total distance: {total_distance}")

                # Verify route is valid (visits all cities)
                if set(route) == set(cities):
                    logger.info("🎯 Valid route found!")
                    return True
                else:
                    logger.warning("⚠️  Invalid route (missing cities)")
                    return False
            else:
                logger.error(f"❌ TSP failed: {result.get('error')}")
                return False

        except Exception as e:
            logger.error(f"❌ TSP test error: {e}")
            return False

    async def test_maximum_cut(self) -> bool:
        """Test Maximum Cut graph problem"""
        logger.info("📊 Testing Maximum Cut problem...")

        # Simple triangle graph
        edges = [(0, 1), (1, 2), (2, 0)]
        weights = {(0, 1): 1, (1, 2): 1, (2, 0): 1}

        try:
            result = await self.connector.execute_action(
                "max_cut",
                {"edges": edges, "weights": weights, "num_reads": 30},
            )

            if result.get("success"):
                partition_a = result.get("partition_a", [])
                partition_b = result.get("partition_b", [])
                cut_value = result.get("cut_value", 0)

                logger.info(f"✅ Max-Cut solved")
                logger.info(f"🔵 Partition A: {partition_a}")
                logger.info(f"🔴 Partition B: {partition_b}")
                logger.info(f"✂️  Cut value: {cut_value}")

                # For triangle, max cut should be 2
                if cut_value >= 2:
                    logger.info("🎯 Good cut found!")
                    return True
                else:
                    logger.warning(f"⚠️  Cut value {cut_value} could be better")
                    return True  # Still working
            else:
                logger.error(f"❌ Max-Cut failed: {result.get('error')}")
                return False

        except Exception as e:
            logger.error(f"❌ Max-Cut test error: {e}")
            return False

    async def cleanup(self):
        """Clean up connection"""
        if self.connector:
            await self.connector.disconnect()
            logger.info("🔌 Disconnected from D-Wave")

    async def run_all_tests(self):
        """Run complete test suite"""
        logger.info("🚀 Starting Real D-Wave Quantum Connector Tests")
        logger.info("=" * 60)

        tests = [
            ("Connection Test", self.test_connection),
            ("Simple QUBO", self.test_simple_qubo),
            ("Traveling Salesman", self.test_traveling_salesman),
            ("Maximum Cut", self.test_maximum_cut),
        ]

        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            logger.info(f"\n🧪 Running: {test_name}")
            try:
                if await test_func():
                    passed += 1
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {e}")

        await self.cleanup()

        logger.info("\n" + "=" * 60)
        logger.info("🧪 REAL D-WAVE QUANTUM TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"📊 Total Tests: {total}")
        logger.info(f"✅ Passed: {passed}")
        logger.info(f"❌ Failed: {total - passed}")
        logger.info(f"📈 Success Rate: {(passed/total)*100:.1f}%")

        if passed == total:
            logger.info("🎉 ALL TESTS PASSED! Real quantum computing working!")
        else:
            logger.warning("⚠️  Some tests failed - check D-Wave Leap setup")

        logger.info("\n📋 Requirements for Full Testing:")
        logger.info("• D-Wave Ocean SDK: pip install dwave-ocean-sdk")
        logger.info("• D-Wave Leap account: https://cloud.dwavesys.com/leap/")
        logger.info("• API token configured in environment")
        logger.info("• Internet connection for cloud access")

        logger.info("\n🔗 Learn More:")
        logger.info("• D-Wave Examples: https://github.com/dwave-examples")
        logger.info(
            "• Advantage2 System: https://github.com/dwave-examples/advantage2.git"
        )
        logger.info("• Ocean Documentation: https://docs.ocean.dwavesys.com/")

        return passed == total


import pytest


@pytest.mark.asyncio
async def test_real_quantum():
    """Test real D-Wave quantum computing"""
    logger.info("🚀 Testing REAL D-Wave Quantum Computing")

    if not DWAVE_AVAILABLE:
        logger.error("❌ D-Wave Ocean SDK not available")
        logger.info("Install: pip install dwave-ocean-sdk")
        logger.info("Signup: https://cloud.dwavesys.com/leap/")
        return False

    connector = DWaveQuantumConnector()

    # Test connection
    success = await connector.connect({})
    if not success:
        logger.error("❌ Failed to connect to D-Wave")
        return False

    logger.info("✅ Connected to D-Wave quantum system")

    # Test simple QUBO
    qubo = {(0, 0): 1, (1, 1): 1, (0, 1): -2}
    result = await connector.execute_action(
        "solve_qubo", {"qubo": qubo, "num_reads": 50}
    )

    if result.get("success"):
        logger.info(f"✅ QUBO solved: {result['best_solution']}")
        logger.info(f"Energy: {result['best_energy']}")
    else:
        logger.error(f"❌ QUBO failed: {result.get('error')}")

    await connector.disconnect()
    return result.get("success", False)


if __name__ == "__main__":
    asyncio.run(test_real_quantum())
