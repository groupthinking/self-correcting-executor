# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Self-Correcting Executor is a sophisticated multi-agent system that combines MCP (Model Context Protocol) integration, quantum computing capabilities, and intelligent orchestration. The system has evolved from a simple protocol runner to include autonomous agents, data-driven mutations, and distributed workflows.

## Architecture

The system is organized into 6 distinct component types:
- **Protocols**: Executable tasks (e.g., `file_validator`, `api_health_checker`)
- **Agents**: Autonomous decision-making entities with A2A communication
- **Connectors**: MCP-compliant interfaces to external systems
- **Analyzers**: Data processing and insight generation
- **Services**: Background infrastructure services
- **Workflows**: Multi-step orchestrated processes

Key directories:
- `agents/` - Autonomous agents with A2A framework
- `protocols/` - Executable protocol implementations
- `connectors/` - MCP connectors and integrations
- `analyzers/` - Pattern detection and analysis
- `frontend/` - React/TypeScript UI with quantum visualizations
- `docs/architecture/` - Comprehensive architecture documentation

## Development Commands

### Standard Development
```bash
make up                # Start development stack
make down              # Stop development stack
make logs              # Follow container logs
make health            # Check API health (localhost:8080/health)
make test              # Run pytest tests
make build             # Build Docker containers
```

### Quantum Development Stack
```bash
make quantum           # Start quantum development environment
make quantum-down      # Stop quantum stack
make quantum-logs      # Follow quantum container logs
make quantum-test      # Run quantum-specific tests
make setup-dwave       # Configure D-Wave Leap authentication
make verify-quantum    # Test quantum hardware connection
```

### Frontend Development
```bash
cd frontend/
npm run dev            # Start Vite dev server (localhost:3000)
npm run build          # Build production bundle
npm run lint           # Run ESLint
```

### Testing
```bash
python test_mcp_debug_simple.py       # Simple MCP debugging
python test_real_dwave_quantum.py     # Real quantum hardware tests
python test_mcp_ecosystem_expansion.py # MCP ecosystem tests
make test-debug                       # Debug test runner
```

## Code Standards

The project follows comprehensive coding standards defined in `.cursorrules`:

### Python
- Use black formatting (88 character line length)
- Type hints required for all functions
- Comprehensive docstrings (Google style)
- pytest for testing with fixtures
- Async/await patterns preferred
- Proper error handling and logging

### TypeScript/React
- ESLint with strict rules + Prettier formatting
- Functional components with hooks
- Proper TypeScript types and interfaces
- React performance optimization (memo, proper state management)
- Accessibility compliance

### MCP Integration Standards
- Structured logging for MCP operations
- Timeout and retry logic with exponential backoff
- Proper connection pooling and health checks
- Comprehensive type definitions for MCP schemas

## Key Concepts

### A2A (Agent-to-Agent) Communication
Agents communicate autonomously using the A2A framework for resource negotiation and task coordination.

### MCP Integration
Universal context sharing through Model Context Protocol enables seamless integration with external systems (GitHub, Claude, etc.).

### Pattern-Driven Mutations
The system analyzes execution patterns and applies intelligent mutations to improve performance and reliability.

### Quantum Computing
Real quantum hardware integration via D-Wave Ocean SDK for optimization problems and quantum algorithms.

## API Endpoints

### V2 Architecture (Primary)
- `POST /api/v2/intent` - Execute natural language intents
- `POST /api/v2/a2a/send` - Agent communication
- `POST /api/v2/mcp/connect` - Connect external MCP services
- `GET /api/v2/patterns` - Analyze execution patterns

### Legacy V1
- `POST /api/v1/execute` - Run individual protocols
- `GET /api/v1/protocols` - List available protocols

## Database

Uses PostgreSQL with key tables:
- `protocol_executions` - Execution history and metrics
- `protocol_mutations` - Applied mutations and triggers
- `execution_insights` - Generated insights for decision making

## Environment Setup

The project uses Docker with devcontainer support. Two main environments:
1. **Standard**: `docker-compose.yml` - Core MCP and agent services
2. **Quantum**: `docker-compose.quantum.yml` - Includes quantum computing stack

## Security

- Token-based API authentication
- Role-based component access control
- Protocol execution sandboxing
- Comprehensive audit logging
- No hardcoded secrets (use environment variables)

## Important Files

- `main.py` - Primary application entry point
- `orchestrator.py` - Multi-agent workflow coordination
- `agents/executor.py` - Core execution agent
- `connectors/mcp_base.py` - MCP protocol implementation
- `analyzers/pattern_detector.py` - Execution pattern analysis
- `docs/architecture/ARCHITECTURE.md` - Detailed system architecture