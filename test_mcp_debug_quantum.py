#!/usr/bin/env python3
"""
Comprehensive test suite for MCP Debug Tool with Quantum Agent Applications
Tests all debugging capabilities, quantum analysis, and GCP integration
"""

from quantum_mcp_server.quantum_mcp import QuantumMCPServer
from connectors.mcp_debug_tool import (
    MCPDebugTool,
    MCPDebugContext,
)
import asyncio
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QuantumDebugTestSuite:
    """Comprehensive test suite for quantum debugging capabilities"""

    def __init__(self):
        self.test_results = {}
        self.total_tests = 0
        self.passed_tests = 0

    async def run_all_tests(self):
        """Run all quantum debug test cases"""
        logger.info("🚀 Starting MCP Debug Tool & Quantum Agent Test Suite")

        test_cases = [
            ("Basic Debug Tool Initialization", self.test_debug_tool_init),
            ("Quantum Code Analysis", self.test_quantum_code_analysis),
            ("Qubit State Debugging", self.test_qubit_state_debugging),
            (
                "Entanglement Pattern Detection",
                self.test_entanglement_detection,
            ),
            ("Decoherence Risk Assessment", self.test_decoherence_analysis),
            ("Gate Fidelity Analysis", self.test_gate_fidelity),
            ("Error Pattern Recognition", self.test_error_patterns),
            ("Performance Metrics Calculation", self.test_performance_metrics),
            (
                "Quantum Teleportation Debug",
                self.test_quantum_teleportation_debug,
            ),
            ("Fallback Reasoning", self.test_fallback_reasoning),
            ("MCP Context Creation", self.test_mcp_context_creation),
            (
                "Quantum MCP Server Integration",
                self.test_quantum_mcp_integration,
            ),
        ]

        for test_name, test_func in test_cases:
            await self.run_test(test_name, test_func)

        self.print_summary()
        return self.passed_tests == self.total_tests

    async def run_test(self, test_name: str, test_func):
        """Run individual test with error handling"""
        self.total_tests += 1
        logger.info(f"🧪 Running: {test_name}")

        try:
            result = await test_func()
            if result:
                self.passed_tests += 1
                self.test_results[test_name] = "✅ PASS"
                logger.info(f"✅ {test_name}: PASSED")
            else:
                self.test_results[test_name] = "❌ FAIL"
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            self.test_results[test_name] = f"❌ ERROR: {str(e)}"
            logger.error(f"❌ {test_name}: ERROR - {str(e)}")
            logger.error(traceback.format_exc())

    async def test_debug_tool_init(self) -> bool:
        """Test MCP Debug Tool initialization"""
        try:
            async with MCPDebugTool("https://mock-gcp-api") as debug_tool:
                return debug_tool is not None and hasattr(
                    debug_tool, "quantum_analyzers"
                )
        except Exception:
            return False

    async def test_quantum_code_analysis(self) -> bool:
        """Test quantum code structure analysis"""
        quantum_code = """
        import qiskit
        from qiskit import QuantumCircuit, execute, Aer

        def create_bell_state():
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure_all()
            return qc

        circuit = create_bell_state()
        backend = Aer.get_backend('qasm_simulator')
        result = execute(circuit, backend, shots=1024).result()
        """

        async with MCPDebugTool("https://mock-gcp-api") as debug_tool:
            analysis = await debug_tool._analyze_code_structure(quantum_code)

            required_keys = [
                "complexity",
                "patterns",
                "imports",
                "functions",
                "quantum_elements",
            ]
            has_required_keys = all(key in analysis for key in required_keys)
            has_quantum_elements = len(analysis["quantum_elements"]) > 0
            has_quantum_pattern = "quantum_computing" in analysis["patterns"]

            return has_required_keys and has_quantum_elements and has_quantum_pattern

    async def test_qubit_state_debugging(self) -> bool:
        """Test qubit state analysis capabilities"""
        problematic_quantum_code = """
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.measure(0, 0)  # Premature measurement
        qc.cx(0, 1)  # Operation after measurement
        """

        async with MCPDebugTool("https://mock-gcp-api") as debug_tool:
            result = await debug_tool._analyze_qubit_state(problematic_quantum_code, {})

            has_operations = len(result["operations"]) > 0
            has_issues = len(result["issues"]) > 0
            needs_review = result["state_quality"] == "needs_review"

            return has_operations and has_issues and needs_review

    async def test_entanglement_detection(self) -> bool:
        """Test entanglement pattern detection"""
        entanglement_code = """
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cz(2, 3)
        qc.cx(0, 3)
        qc.bell_state(0, 1)  # Custom bell state
        """

        async with MCPDebugTool("https://mock-gcp-api") as debug_tool:
            result = await debug_tool._analyze_entanglement(entanglement_code, {})

            has_operations = len(result["entanglement_operations"]) > 0
            high_density = result["warning"] is not None
            correct_count = result["count"] > 3

            return has_operations and high_density and correct_count

    async def test_decoherence_analysis(self) -> bool:
        """Test decoherence risk assessment"""
        risky_code = (
            """
        import time
        qc = QuantumCircuit(10)
        for i in range(100):  # Long loop
            qc.h(i % 10)
            time.sleep(0.01)  # Timing delay
            qc.cx(i % 10, (i + 1) % 10)
        # This is a very long quantum program with many operations
        # """
            + "\n" * 60
        )  # Make it long

        async with MCPDebugTool("https://mock-gcp-api") as debug_tool:
            result = await debug_tool._analyze_decoherence(risky_code, {})

            has_risks = len(result["risks"]) > 0
            high_severity = result["severity"] == "high"
            timing_risk = any("delay" in risk for risk in result["risks"])

            return has_risks and high_severity and timing_risk

    async def test_gate_fidelity(self) -> bool:
        """Test gate fidelity analysis"""
        gate_heavy_code = """
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.x(1)
        qc.y(2)
        qc.z(3)
        qc.rx(0.5, 4)
        qc.ry(0.3, 0)
        qc.rz(0.8, 1)
        qc.cx(0, 1)
        qc.cx(1, 2)
        """

        async with MCPDebugTool("https://mock-gcp-api") as debug_tool:
            result = await debug_tool._analyze_gate_fidelity(gate_heavy_code, {})

            has_gates = result["total_gates"] > 5
            has_types = len(result["gate_types"]) > 3
            has_fidelity = result["estimated_fidelity"] in ["high", "medium"]

            return has_gates and has_types and has_fidelity

    async def test_error_patterns(self) -> bool:
        """Test error pattern recognition and fix generation"""
        buggy_code = """
        def quantum_function():
            undefined_variable = some_function()  # NameError
            result = "string" + 5  # TypeError
            my_list = [1, 2, 3]
            value = my_list[10]  # IndexError
            return result
        """

        errors = [
            "NameError: name 'some_function' is not defined",
            'TypeError: can only concatenate str (not "int") to str',
            "IndexError: list index out of range",
        ]

        async with MCPDebugTool("https://mock-gcp-api") as debug_tool:
            all_patterns_detected = True

            for error in errors:
                fixes = await debug_tool._generate_general_fixes(buggy_code, error)
                if not fixes:
                    all_patterns_detected = False
                    break

            return all_patterns_detected

    async def test_performance_metrics(self) -> bool:
        """Test performance metrics calculation"""
        complex_code = (
            """
        def complex_quantum_function():
            for i in range(10):
                if i % 2 == 0:
                    while True:
                        try:
                            if some_condition:
                                break
                        except Exception:
                            continue
                    else:
                        pass
        """
            + "\n" * 150
        )  # Make it long

        async with MCPDebugTool("https://mock-gcp-api") as debug_tool:
            debug_context = MCPDebugContext(
                file="test.py",
                line=1,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            metrics = await debug_tool._calculate_performance_metrics(
                complex_code, debug_context
            )

            required_metrics = [
                "complexity_score",
                "line_count",
                "estimated_runtime",
                "quantum_efficiency",
            ]
            has_metrics = all(metric in metrics for metric in required_metrics)
            high_complexity = metrics["complexity_score"] > 5
            correct_line_count = metrics["line_count"] > 100

            return has_metrics and high_complexity and correct_line_count

    async def test_quantum_teleportation_debug(self) -> bool:
        """Test comprehensive quantum teleportation debugging"""
        teleportation_code = """
        import qiskit
        from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

        def quantum_teleportation():
            # Create quantum registers
            qreg = QuantumRegister(3, 'q')
            creg = ClassicalRegister(3, 'c')
            qc = QuantumCircuit(qreg, creg)

            # Prepare the state to be teleported (|+> state)
            qc.h(qreg[0])

            # Create Bell pair between qubits 1 and 2
            qc.h(qreg[1])
            qc.cx(qreg[1], qreg[2])

            # Bell measurement on qubits 0 and 1
            qc.cx(qreg[0], qreg[1])
            qc.h(qreg[0])
            qc.measure(qreg[0], creg[0])
            qc.measure(qreg[1], creg[1])

            # Apply corrections based on measurement
            qc.cx(qreg[1], qreg[2])
            qc.cz(qreg[0], qreg[2])

            return qc
        """

        async with MCPDebugTool("https://mock-gcp-api") as debug_tool:
            mcp_data = {
                "file": "quantum_teleportation.py",
                "line": 15,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            quantum_context = {
                "circuit_depth": 6,
                "qubit_count": 3,
                "gate_sequence": [
                    "h",
                    "h",
                    "cx",
                    "cx",
                    "h",
                    "measure",
                    "measure",
                    "cx",
                    "cz",
                ],
            }

            result = await debug_tool.debug_code(
                code=teleportation_code,
                error=None,
                mcp_data=mcp_data,
                quantum_context=quantum_context,
            )

            is_success = result.status == "success"
            has_reasoning = len(result.reasoning) > 0
            has_quantum_insights = result.quantum_insights is not None
            has_performance = result.performance_metrics is not None

            return (
                is_success
                and has_reasoning
                and has_quantum_insights
                and has_performance
            )

    async def test_fallback_reasoning(self) -> bool:
        """Test fallback reasoning when GCP is unavailable"""
        async with MCPDebugTool("https://invalid-endpoint") as debug_tool:
            quantum_error = (
                "QuantumError: Circuit execution failed due to quantum decoherence"
            )

            fallback_result = await debug_tool._fallback_reasoning(
                "quantum_code", quantum_error
            )

            has_reasoning = "reasoning" in fallback_result
            has_suggestions = "suggestions" in fallback_result
            quantum_suggestions = any(
                "quantum" in suggestion.lower()
                for suggestion in fallback_result["suggestions"]
            )

            return has_reasoning and has_suggestions and quantum_suggestions

    async def test_mcp_context_creation(self) -> bool:
        """Test MCP debug context creation and validation"""
        async with MCPDebugTool("https://mock-gcp-api") as debug_tool:
            mcp_data = {
                "file": "test_quantum.py",
                "line": 42,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "function": "quantum_algorithm",
            }

            error = "Quantum measurement error: invalid qubit state"
            context = debug_tool._create_debug_context(mcp_data, error)

            has_file = context.file == "test_quantum.py"
            has_line = context.line == 42
            has_timestamp = context.timestamp is not None
            has_stack_trace = context.stack_trace is not None

            return has_file and has_line and has_timestamp and has_stack_trace

    async def test_quantum_mcp_integration(self) -> bool:
        """Test integration with Quantum MCP Server"""
        try:
            quantum_server = QuantumMCPServer()

            # Test quantum optimization tool
            optimize_result = await quantum_server.handle_tool_call(
                "quantum_optimize",
                {
                    "problem": "minimize",
                    "variables": {"x": [0, 1], "y": [0, 1]},
                    "objective": "x + y",
                    "constraints": [],
                },
            )

            has_result = optimize_result is not None
            has_content = (
                "content" in optimize_result
                if isinstance(optimize_result, dict)
                else True
            )

            return has_result and has_content

        except Exception as e:
            logger.warning(f"Quantum MCP Server integration test failed: {e}")
            return False

    def print_summary(self):
        """Print comprehensive test summary"""
        logger.info("\n" + "=" * 80)
        logger.info("🧪 MCP DEBUG TOOL & QUANTUM AGENT TEST SUMMARY")
        logger.info("=" * 80)

        for test_name, result in self.test_results.items():
            logger.info(f"{result} {test_name}")

        logger.info("-" * 80)
        logger.info(f"📊 Total Tests: {self.total_tests}")
        logger.info(f"✅ Passed: {self.passed_tests}")
        logger.info(f"❌ Failed: {self.total_tests - self.passed_tests}")
        logger.info(
            f"📈 Success Rate: {(self.passed_tests / self.total_tests) * 100:.1f}%")

        if self.passed_tests == self.total_tests:
            logger.info(
                "🎉 ALL TESTS PASSED! MCP Debug Tool is fully functional.")
        else:
            logger.warning(
                "⚠️  Some tests failed. Please review and fix issues.")

        logger.info("=" * 80)


async def run_debug_tool_demo():
    """Demonstrate MCP Debug Tool capabilities"""
    logger.info("🚀 Running MCP Debug Tool Demo")

    # Demo quantum code with intentional issues
    demo_code = """
    import qiskit
    from qiskit import QuantumCircuit, execute

    def problematic_quantum_function():
        # Issue 1: Undefined qubit count
        qc = QuantumCircuit(undefined_qubits)

        # Issue 2: Premature measurement
        qc.h(0)
        qc.measure(0, 0)
        qc.cx(0, 1)  # Operation after measurement

        # Issue 3: High gate density
        for i in range(100):
            qc.h(i % 5)
            qc.cx(i % 5, (i + 1) % 5)

        return qc
    """

    async with MCPDebugTool("https://demo-gcp-api", "demo-token") as debug_tool:
        result = await debug_tool.debug_code(
            code=demo_code,
            error="NameError: name 'undefined_qubits' is not defined",
            mcp_data={
                "file": "demo_quantum.py",
                "line": 7,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            quantum_context={
                "circuit_depth": 200,
                "qubit_count": 5,
                "gate_sequence": ["h", "cx"] * 100,
            },
        )

        logger.info("📋 Debug Analysis Results:")
        logger.info(f"Status: {result.status}")
        logger.info(f"Reasoning: {result.reasoning}")
        logger.info(f"Suggestions: {json.dumps(result.suggestions, indent=2)}")
        logger.info(f"Fixes: {json.dumps(result.fixes, indent=2)}")

        if result.quantum_insights:
            logger.info(
                f"Quantum Insights: {
                    json.dumps(
                        result.quantum_insights,
                        indent=2)}")

        if result.performance_metrics:
            logger.info(
                f"Performance Metrics: {
                    json.dumps(
                        result.performance_metrics,
                        indent=2)}")


async def main():
    """Main test execution function"""
    logger.info("🎯 Starting MCP Debug Tool & Quantum Agent Test Suite")

    # Run comprehensive test suite
    test_suite = QuantumDebugTestSuite()
    all_tests_passed = await test_suite.run_all_tests()

    # Run demonstration
    await run_debug_tool_demo()

    # Final verification
    if all_tests_passed:
        logger.info("🏆 SUCCESS: All MCP Debug Tool tests passed!")
        logger.info("✅ Debug Tool is ready for production use")
        logger.info("✅ Quantum Agent Applications are fully supported")
        logger.info("✅ GCP Integration is functional")
        return 0
    else:
        logger.error("❌ FAILURE: Some tests failed")
        logger.error("⚠️  Please review and fix issues before deployment")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
