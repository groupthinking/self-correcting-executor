# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Self-Correcting Executor** - a sophisticated multi-agent AI system that autonomously executes tasks, learns from patterns, and evolves through data-driven mutations. The system features MCP (Model Context Protocol) integration, quantum computing capabilities, Agent-to-Agent (A2A) communication, and a comprehensive web interface.

## Architecture

### **Multi-Layer Architecture**
```
┌─ FRONTEND LAYER ────────────────────────────────┐
│ React/TypeScript UI + Dashboard                │
├─ API GATEWAY LAYER ──────────────────────────────┤
│ FastAPI + MCP Protocol Compliance               │
├─ ORCHESTRATION LAYER ───────────────────────────┤
│ Multi-agent workflow optimization engine         │
├─ AGENT RUNTIME LAYER ────────────────────────────┤
│ Specialized AI agents with A2A communication    │
├─ PROTOCOL IMPLEMENTATION LAYER ─────────────────┤
│ Self-executing protocols with mutation logic    │
├─ PERSISTENCE LAYER ─────────────────────────────┤
│ PostgreSQL + Redis + Analytics tracking         │
└──────────────────────────────────────────────────┘
```

### **Core Component Types**
1. **Protocols** (`protocols/`) - Executable tasks with mutation capability
2. **Agents** (`agents/`) - Autonomous decision-making entities  
3. **Connectors** (`connectors/`) - MCP-compliant external integrations
4. **Analyzers** (`analyzers/`) - Pattern detection and insight generation
5. **Orchestrator** - Workflow coordination and optimization
6. **Frontend** (`frontend/`) - React-based management interface

## Common Commands

### **Development Environment**
```bash
# Docker-based development (recommended)
make up                    # Start all services
make down                  # Stop all services
make logs                  # Follow service logs
make health                # Check service health

# Alternative docker-compose
docker-compose up -d       # Start in detached mode
docker-compose logs -f     # Follow logs
```

### **Direct Python Execution**
```bash
# Install dependencies
pip install -r requirements.txt

# Run main executor
python main.py [protocol_name] [iterations]

# Examples
python main.py default_protocol 10
python main.py file_validator 5
```

### **MCP Server Operations**
```bash
# Start MCP server (for Claude Desktop integration)
python mcp_server/main.py

# Test MCP server locally
python test_mcp_debug_simple.py
python test_mcp_debug_quantum.py
```

### **Testing & Validation**
```bash
# Run specific test suites
python test_real_dwave_quantum.py       # Quantum integration tests
python test_mcp_ecosystem_expansion.py   # MCP expansion tests

# Protocol validation
python -m protocols.file_validator       # Test file validation
```

### **Frontend Development**
```bash
cd frontend
npm install                # Install dependencies
npm run dev               # Development server
npm run build             # Production build
```

## Key Configuration Files

### **Core Configuration**
- **`requirements.txt`** - Python dependencies (117 packages including quantum, ML, MCP)
- **`docker-compose.yml`** - Multi-service orchestration
- **`docker-compose.quantum.yml`** - Quantum-enhanced configuration
- **`Dockerfile`** + **`Dockerfile.quantum`** - Container definitions
- **`devcontainer.json`** - VS Code/Cursor development container

### **Security & Standards**
- **`security_config.yaml`** - Security policies and authentication
- **`PROJECT_STANDARDS.md`** - Development standards and guidelines
- **`config/component_types.yaml`** - Component type definitions

### **Environment Variables Required**
```bash
# Core API Keys
export ANTHROPIC_API_KEY="your_anthropic_key"
export OPENAI_API_KEY="your_openai_key"
export DWAVE_API_TOKEN="your_quantum_token"

# Database Configuration
export POSTGRES_HOST="localhost"
export POSTGRES_USER="executor_user"
export POSTGRES_PASSWORD="secure_password"
export POSTGRES_DB="self_correcting_executor"

# Redis Configuration  
export REDIS_HOST="localhost"
export REDIS_PORT="6379"

# GitHub Integration
export GITHUB_TOKEN="your_github_token"
```

## Core Workflows

### **1. Self-Correcting Execution**
```python
# Entry point: main.py
run_self_correcting_executor(protocol='default_protocol', iterations=5)

# Flow:
# 1. Execute protocol → 2. Analyze outcome → 3. Apply mutations → 4. Repeat
```

### **2. MCP Integration Workflow**
```python
# MCP Server: mcp_server/main.py
# Provides tools: code_analyzer, protocol_validator, self_corrector
# Integrates with Claude Desktop via JSON-RPC over stdin/stdout
```

### **3. Agent-to-Agent Communication**
```python
# A2A Framework: agents/a2a_framework.py
await agent.send_message(
    recipient="negotiator",
    message_type="resource_request", 
    content={"need": "quantum_processor", "duration": "30min"}
)
```

### **4. Quantum Integration**
```python
# D-Wave Quantum: connectors/dwave_quantum_connector.py
# Provides quantum annealing for optimization problems
# Requires D-Wave Ocean SDK and valid API token
```

## Development Patterns

### **Adding New Protocols**
1. Create file in `protocols/` directory
2. Implement `execute()` function returning success/failure
3. Add to protocol registry in `utils/protocol_registry.py`
4. Include unit tests and mutation logic

### **Creating New Agents**
1. Inherit from base agent class in `agents/`
2. Implement A2A communication interface
3. Define decision-making logic and state management
4. Register with orchestrator for workflow participation

### **MCP Connector Development**
1. Implement base connector interface from `connectors/mcp_base.py`
2. Add authentication and context sharing
3. Ensure JSON-RPC compliance for external systems
4. Include error handling and retry logic

### **Frontend Component Development**
```typescript
// React/TypeScript patterns in frontend/src/
// Key components: Dashboard, IntentExecutor, PatternVisualizer
// Use existing component architecture and styling
```

## Database Schema

### **Core Tables**
- **`protocol_executions`** - Execution history with success/failure tracking
- **`protocol_mutations`** - Applied mutations and their triggers  
- **`execution_insights`** - Generated patterns and recommendations
- **`agent_communications`** - A2A message history and negotiations

### **Analytics & Monitoring**
- Real-time execution metrics
- Pattern detection algorithms (hourly analysis)
- Mutation effectiveness tracking
- Agent performance optimization

## API Endpoints

### **V1 (Legacy Protocol API)**
```bash
POST /api/v1/execute        # Execute protocols directly
GET  /api/v1/protocols      # List available protocols  
POST /api/v1/mutate         # Force protocol mutation
```

### **V2 (Advanced Orchestration)**
```bash
POST /api/v2/intent              # Natural language intent processing
GET  /api/v2/patterns           # Execution pattern analysis
POST /api/v2/mutate-intelligent # Data-driven mutations
POST /api/v2/a2a/send          # Agent communication
POST /api/v2/mcp/connect       # MCP connector management
POST /api/v2/mcp/execute       # Execute MCP actions
```

## Specialized Components

### **Quantum Computing Integration**
- **D-Wave Ocean SDK** integration for quantum annealing
- **Quantum optimization** for complex scheduling problems
- **Hybrid classical-quantum** algorithms for protocol optimization
- Test with: `python test_real_dwave_quantum.py`

### **Machine Learning Stack**
- **LangChain** for LLM orchestration and chaining
- **Transformers/PyTorch** for custom model training
- **Continuous learning** system for pattern adaptation
- **Multi-modal analysis** for complex data processing

### **Enterprise Features**
- **Multi-tenant architecture** with role-based access
- **Audit trail** for all executions and mutations
- **Distributed workers** for horizontal scaling
- **Enterprise authentication** and security policies

## Troubleshooting

### **Common Issues**

#### **MCP Server Not Starting**
```bash
# Check MCP dependencies
python -c "import mcp.server.stdio; print('MCP available')"

# Verify JSON-RPC communication
python mcp_server/main.py < test_request.json

# Check Claude Desktop configuration
cat ~/.config/Claude/claude_desktop_config.json
```

#### **Quantum Integration Failures**
```bash
# Verify D-Wave token
echo $DWAVE_API_TOKEN

# Test quantum connectivity
python -c "
from dwave.system import DWaveSampler
sampler = DWaveSampler()
print(f'Connected to: {sampler.properties[\"chip_id\"]}')
"
```

#### **Database Connection Issues**
```bash
# Check PostgreSQL connectivity
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB

# Verify Redis connectivity  
redis-cli -h $REDIS_HOST -p $REDIS_PORT ping

# Reset database schema
python -c "from utils.db_tracker import reset_schema; reset_schema()"
```

#### **Protocol Execution Failures**
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python main.py your_protocol 1

# Check protocol registry
python -c "from utils.protocol_registry import list_protocols; print(list_protocols())"

# Validate protocol syntax
python -m protocols.your_protocol_name
```

### **Performance Optimization**

#### **Memory Management**
- Monitor agent memory usage in long-running processes
- Configure Redis cache expiration for large datasets
- Use protocol execution pooling for concurrent operations

#### **Quantum Resource Optimization**
- Batch quantum operations to minimize API calls
- Implement quantum result caching for repeated problems
- Use hybrid algorithms when pure quantum is not optimal

#### **Database Performance**
- Index execution patterns for faster pattern detection
- Archive old execution data to separate tables
- Use connection pooling for high-throughput scenarios

## Security Considerations

### **Authentication & Authorization**
- **API Keys**: Stored in environment variables, never committed
- **Database Credentials**: Encrypted connection strings
- **Role-Based Access**: Different permissions for different component types
- **Audit Logging**: All mutations and executions tracked

### **Sandboxing & Isolation**
- **Protocol Execution**: Isolated containers for untrusted code
- **Agent Communication**: Authenticated message passing
- **External Connectors**: Rate limiting and input validation
- **Quantum Operations**: Secure token management for D-Wave

## Advanced Features

### **Pattern-Driven Evolution**
The system continuously analyzes execution patterns and automatically evolves:
- **Success Rate Analysis** → Protocol parameter optimization
- **Failure Pattern Detection** → Automatic mutation triggers  
- **Resource Usage Patterns** → Agent allocation optimization
- **Communication Patterns** → A2A protocol refinement

### **Multi-Modal Intelligence**
- **Code Analysis**: AST parsing and complexity metrics
- **Natural Language Processing**: Intent understanding and generation
- **Time Series Analysis**: Execution pattern recognition
- **Graph Analysis**: Agent communication networks

### **Enterprise Integration**
- **GitHub Connector**: Repository analysis and code generation
- **SAP Connector**: Enterprise system integration
- **Chrome Extension**: Browser-based automation
- **API Gateway**: Standardized external access

## Deployment

### **Development (Local)**
```bash
make up                    # Full stack with hot reload
make dev                   # Development mode with debugging
```

### **Production (Docker)**
```bash
docker-compose -f docker-compose.yml up -d
docker-compose -f docker-compose.quantum.yml up -d  # With quantum
```

### **Enterprise (Kubernetes)**
```bash
# Use provided Kubernetes manifests (when available)
kubectl apply -f k8s/
```

### **Cloud (AWS/GCP/Azure)**
- **Terraform configurations** for infrastructure as code
- **Kubernetes Helm charts** for application deployment  
- **Monitoring stack** with Prometheus/Grafana integration

## Monitoring & Observability

### **Metrics Collection**
- **Execution Metrics**: Success rates, duration, resource usage
- **Agent Metrics**: Communication frequency, decision accuracy
- **System Metrics**: Memory, CPU, database performance
- **Business Metrics**: Task completion rates, user satisfaction

### **Alerting**
- **Failed Execution Threshold**: > 50% failure rate triggers alert
- **Agent Unresponsiveness**: Communication timeout detection
- **Resource Exhaustion**: Memory/CPU/storage threshold alerts
- **Security Events**: Authentication failures, suspicious activity

This system represents a cutting-edge implementation of self-evolving AI with quantum computing integration, comprehensive MCP protocol support, and enterprise-grade architecture. The self-correction capabilities ensure continuous improvement through data-driven mutations and pattern analysis.