---
name: quantum-connector-testing
description: Guide for testing connectors/dwave_quantum_connector.py and other quantum MCP tools in self-correcting-executor without requiring a live D-Wave QPU token, while respecting this repo's "no simulation/mocks in production" convention. Use when asked to test, debug, or extend quantum computing integration.
---

This repo's D-Wave integration (`connectors/dwave_quantum_connector.py`,
`mcp_server/quantum_tools.py`, `quantum_mcp_server/quantum_mcp.py`) is
intentionally built against the **real** D-Wave Ocean SDK and Leap cloud
service — it must never silently fall back to a local simulator/annealer in
production code paths (see the removed `SimulatedAnnealingSampler` import
and `tests/test_mcp_compliance.py::test_quantum_requires_real_qpu`).

When testing or extending this code:

1. **Check for `DWAVE_AVAILABLE` / `DWAVE_API_TOKEN` before assuming a QPU is
   reachable.** Tests and scripts should `pytest.skip(...)` (not fabricate
   results) when the token isn't set, mirroring the existing pattern in
   `tests/test_mcp_compliance.py::test_quantum_requires_real_qpu`:

   ```python
   import os
   import pytest

   if not os.getenv("DWAVE_API_TOKEN"):
       pytest.skip("DWAVE_API_TOKEN not set - skipping quantum test")
   ```

2. **Test the non-quantum-hardware parts directly and deterministically:**
   - Input validation / `QuantumResult` dataclass construction
   - Problem formulation (`BinaryQuadraticModel` / `ConstrainedQuadraticModel`
     building) using `dimod`, which runs locally without QPU access
   - Error handling paths that should raise/propagate a `RuntimeError`
     (e.g. `"No D-Wave QPU available"`) instead of silently simulating

3. **Never add a mock/simulated sampler as a fallback** in
   `connectors/dwave_quantum_connector.py` or related production files —
   this is explicitly checked by
   `tests/test_mcp_compliance.py::test_no_placeholder_code_in_production`
   and `test_data_processor_no_simulation`, which scan for
   `mock`/`simulated`/`placeholder` strings in production directories
   (`agents`, `connectors`, `mcp_server`, `protocols`).

4. **Run the relevant test files** after any change:

   ```bash
   python -m pytest tests/test_mcp_compliance.py -k quantum -v
   python -m pytest test_real_dwave_quantum.py -v   # requires DWAVE_API_TOKEN
   ```

5. If you need to demonstrate quantum problem formulation without hardware
   access, use `simple_quantum_example.py` as a reference for building a
   `BinaryQuadraticModel` locally, then document clearly that solving it on
   real hardware requires a valid `DWAVE_API_TOKEN`.
