# Self-Correcting Executor Architecture v2.0

## Overview

The Self-Correcting Executor has evolved from a simple protocol runner to a sophisticated multi-component system with intelligent orchestration, MCP integration, and data-driven mutations.

## Core Principles

1. **Not Everything is a Protocol** - Different component types serve different purposes
2. **MCP-First Design** - Universal context sharing through Model Context Protocol
3. **Data-Driven Evolution** - Mutations guided by execution patterns and insights
4. **Agent Autonomy** - Agents communicate and negotiate via A2A framework

## Architecture Layers

```
┌─ FRONTEND LAYER ────────────────────────────────┐
│ Command+control interface for knowledge workflow │
├─ API GATEWAY LAYER ──────────────────────────────┤
│ Standardized access protocols for AI integration │
├─ ORCHESTRATION LAYER ───────────────────────────┤
│ Multi-agent workflow optimization engine         │
├─ AGENT RUNTIME LAYER ────────────────────────────┤
│ Specialized AI execution environment             │
├─ PROTOCOL IMPLEMENTATION LAYER ─────────────────┤
│ MCP Core with context+security management        │
├─ PERSISTENCE LAYER ─────────────────────────────┤
│ Knowledge graph with relationship acceleration   │
└──────────────────────────────────────────────────┘
```

## Component Types

### 1. Protocols
- **Purpose**: Executable tasks with defined inputs/outputs
- **Examples**: file_validator, api_health_checker
- **Key Features**: Can be mutated, returns success/failure

### 2. Agents
- **Purpose**: Autonomous entities that make decisions
- **Examples**: executor, mutator, negotiator
- **Key Features**: A2A communication, maintains state, reasoning capability

### 3. Connectors (MCP)
- **Purpose**: Universal interfaces to external systems
- **Examples**: github_mcp, claude_mcp, sap_mcp
- **Key Features**: Context bridging, authentication, MCP protocol compliance

### 4. Analyzers
- **Purpose**: Data processing and insight generation
- **Examples**: pattern_detector, anomaly_finder
- **Key Features**: Processes large datasets, feeds mutation engine

### 5. Services
- **Purpose**: Background services and infrastructure
- **Examples**: cache_manager, queue_processor
- **Key Features**: Long-running processes, resource management

### 6. Workflows
- **Purpose**: Multi-step orchestrated processes
- **Examples**: rag_pipeline, a2a_negotiation
- **Key Features**: Coordinates components, decision trees

## Key Subsystems

### A2A (Agent-to-Agent) Communication
```python
# Agents communicate autonomously
await agent.send_message(
    recipient="negotiator",
    message_type="resource_request",
    content={"need": "gpu", "duration": "2h"}
)
```

### MCP Integration
```python
# Universal context sharing
context = MCPContext()
context.task = {"intent": "analyze_code"}
await connector.send_context(context)
result = await connector.execute_action("get_code", {"file": "main.py"})
```

### Pattern-Driven Mutations
```python
# Analyze patterns → Generate insights → Apply mutations
analysis = await pattern_detector.analyze_execution_patterns()
mutations = await insight_mutator.apply_recommendations(analysis)
```

## Data Flow

1. **Intent Processing**
   - User provides intent (natural language)
   - Orchestrator analyzes and decomposes

2. **Component Discovery**
   - Required components identified
   - Capabilities matched to needs

3. **Workflow Generation**
   - Optimal execution path determined
   - Parallel operations identified

4. **Execution**
   - Components execute in coordination
   - Context shared via MCP
   - Agents negotiate resources

5. **Learning**
   - Execution patterns analyzed
   - Insights generated
   - System evolves through mutations

## API Endpoints

### V1 (Legacy)
- `POST /api/v1/execute` - Run protocols
- `GET /api/v1/protocols` - List protocols
- `POST /api/v1/mutate` - Force mutation

### V2 (New Architecture)
- `POST /api/v2/intent` - Execute intent through orchestration
- `GET /api/v2/patterns` - Analyze execution patterns
- `POST /api/v2/mutate-intelligent` - Data-driven mutations
- `POST /api/v2/a2a/send` - Agent communication
- `POST /api/v2/mcp/connect` - Connect external services
- `POST /api/v2/mcp/execute` - Execute MCP actions

## Database Schema

### protocol_executions
- Tracks all component executions
- Stores success/failure, duration, errors

### protocol_mutations
- Records all mutations applied
- Links to execution patterns that triggered them

### execution_insights
- Stores generated insights
- Feeds future decision making

## Security Considerations

- **Authentication**: Token-based for API access
- **Authorization**: Role-based component access
- **Sandboxing**: Protocol execution isolation
- **Audit Trail**: All actions logged

## Future Roadmap

1. **Phase 1**: Web UI Development
2. **Phase 2**: Distributed Workers
3. **Phase 3**: Quantum Optimization
4. **Phase 4**: Enterprise Features

## Development Guidelines

1. **Component Creation**
   - Determine correct type (not everything is a protocol)
   - Implement MCP interface if external-facing
   - Add to appropriate registry

2. **Testing**
   - Unit tests for components
   - Integration tests for workflows
   - Pattern analysis validation

3. **Monitoring**
   - Execution metrics to database
   - Pattern detection runs hourly
   - Alert on repeated failures 