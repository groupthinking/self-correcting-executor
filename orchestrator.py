# Intelligent Orchestration Engine
# Coordinates agents, protocols, and services based on intent

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from utils.logger import log

# Import specialized agents
try:
    from agents.specialized.code_generator import code_generator_agent
    from agents.specialized.filesystem_agent import file_system_agent

    specialized_agents_available = True
except ImportError as e:
    print(f"Warning: Could not import specialized agents: {e}")
    specialized_agents_available = False
    code_generator_agent = None
    file_system_agent = None

# Import LLM connector
try:
    from connectors.llm_connector import llm_connector

    llm_available = True
except ImportError:
    llm_available = False
    llm_connector = None


class OrchestrationEngine:
    """
    Multi-layer orchestration matching the architecture:
    - Frontend Layer: Command interface
    - API Gateway: Standardized access
    - Orchestration: This layer - workflow optimization
    - Agent Runtime: Specialized execution
    - Protocol Implementation: MCP core
    - Persistence: Knowledge graph
    """

    def __init__(self):
        self.agents = {}
        self.connectors = {}
        self.active_workflows = {}
        self.message_bus = MessageBus()
        self.knowledge_graph = KnowledgeGraph()

        # Register specialized agents
        if specialized_agents_available:
            if code_generator_agent:
                self.agents["code_generator"] = code_generator_agent
            if file_system_agent:
                self.agents["file_system_agent"] = file_system_agent

    async def execute_intent(
        self,
        intent: str,
        sources: Optional[List[str]] = None,
        options: Optional[Dict] = None,
    ):
        """
        Main entry point - processes user intent through the full stack
        """
        log(f"🎯 Processing intent: {intent}")

        # 1. Intent Analysis
        analyzed_intent = await self.analyze_intent(intent)

        # 2. Component Discovery
        required_components = await self.discover_components(
            analyzed_intent, sources or []
        )

        # 3. Workflow Generation
        workflow = await self.generate_workflow(analyzed_intent, required_components)

        # 4. Execute Workflow
        result = await self.execute_workflow(workflow)

        # 5. Learn from Execution
        await self.learn_from_execution(workflow, result)

        return result

    async def analyze_intent(self, intent: str) -> Dict:
        """Analyze user intent to determine required actions"""
        intent_lower = intent.lower()

        # Determine action type based on keywords
        if any(
            keyword in intent_lower
            for keyword in ["generate", "create", "code", "api", "endpoint"]
        ):
            action = "generate_code"
            target = "api"
        elif any(
            keyword in intent_lower
            for keyword in ["list", "show files", "directory", "ls"]
        ):
            action = "list_directory"
            target = "filesystem"
        elif any(
            keyword in intent_lower for keyword in ["read", "open", "cat", "show file"]
        ):
            action = "read_file"
            target = "filesystem"
        elif any(
            keyword in intent_lower
            for keyword in ["multi-modal", "llm", "ideate", "learn"]
        ):
            action = "multimodal_analysis"
            target = "llm"
        elif any(
            keyword in intent_lower for keyword in ["analyze", "pattern", "insight"]
        ):
            action = "analyze"
            target = "data"
        elif any(keyword in intent_lower for keyword in ["check", "health", "status"]):
            action = "monitor"
            target = "system"
        else:
            action = "process"
            target = "general"

        return {
            "original_intent": intent,
            "parsed_intent": {
                "action": action,
                "target": target,
                "details": self._extract_details(intent, action),
                "constraints": [],
                "requirements": [],
            },
            "confidence": 0.95,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _extract_details(self, intent: str, action: str) -> Dict:
        """Extract specific details like paths from the intent string."""
        if action in ["list_directory", "read_file"]:
            # Simple path extraction: assumes path is the last word
            parts = intent.split()
            path = parts[-1] if len(parts) > 1 and "/" in parts[-1] else "."
            # A more robust NLP/regex solution would be better here in a real
            # system
            return {"path": path}
        return {}

    async def discover_components(
            self,
            intent: Dict,
            sources: List[str]) -> Dict:
        """Discover which components are needed"""
        components = {
            "agents": [],
            "protocols": [],
            "connectors": [],
            "analyzers": [],
            "services": [],
        }

        # Based on intent, determine needed components
        action = intent["parsed_intent"]["action"]

        if action == "generate_code":
            components["agents"].append("code_generator")
        elif action in ["list_directory", "read_file"]:
            components["agents"].append("file_system_agent")
        elif action == "analyze":
            components["analyzers"].append("pattern_detector")
            components["protocols"].append("data_processor")
        elif action == "monitor":
            components["protocols"].append("system_monitor")
        else:
            # Default workflow
            components["protocols"].append("data_processor")
            components["analyzers"].append("pattern_detector")

        if sources:
            for source in sources:
                if source == "github":
                    components["connectors"].append("github_mcp")
                elif source == "claude":
                    components["connectors"].append("claude_mcp")

        return components

    async def generate_workflow(self, intent: Dict, components: Dict) -> Dict:
        """Generate optimized workflow from components"""
        workflow = {
            "id": f"wf_{datetime.utcnow().timestamp()}",
            "intent": intent,
            "steps": [],
            "parallel_groups": [],
            "decision_points": [],
        }

        # Build workflow steps based on action
        action = intent["parsed_intent"]["action"]

        if action == "generate_code" and "code_generator" in components["agents"]:
            # Code generation workflow
            workflow["steps"].append(
                {
                    "type": "agent",
                    "name": "code_generator",
                    "inputs": {
                        "intent": intent["original_intent"],
                        "context": {
                            "endpoint_name": "generated-endpoint",
                            "description": "Auto-generated API endpoint",
                        },
                    },
                    "outputs": ["generated_code", "instructions"],
                }
            )
        elif (
            action in ["list_directory", "read_file"]
            and "file_system_agent" in components["agents"]
        ):
            # Filesystem workflow
            workflow["steps"].append(
                {
                    "type": "agent",
                    "name": "file_system_agent",
                    "inputs": {
                        "action": action,
                        "path": intent["parsed_intent"]
                        .get("details", {})
                        .get("path", "."),
                    },
                    "outputs": ["file_data"],
                }
            )
        else:
            # Default data processing workflow
            if "data_processor" in components["protocols"]:
                workflow["steps"].append(
                    {
                        "type": "protocol",
                        "name": "data_processor",
                        "inputs": {"source": "user_data"},
                        "outputs": ["processed_data"],
                    }
                )

            if "pattern_detector" in components["analyzers"]:
                workflow["steps"].append(
                    {
                        "type": "analyzer",
                        "name": "pattern_detector",
                        "inputs": {"data": "processed_data"},
                        "outputs": ["patterns", "insights"],
                    }
                )

        return workflow

    async def execute_workflow(self, workflow: Dict) -> Dict:
        """Execute the generated workflow"""
        results = {
            "workflow_id": workflow["id"],
            "status": "running",
            "steps_completed": [],
            "outputs": {},
        }

        # Execute each step
        for step in workflow["steps"]:
            try:
                if step["type"] == "protocol":
                    result = await self.execute_protocol(step["name"], step["inputs"])
                elif step["type"] == "analyzer":
                    result = await self.execute_analyzer(step["name"], step["inputs"])
                elif step["type"] == "agent":
                    result = await self.execute_agent(step["name"], step["inputs"])

                results["steps_completed"].append(
                    {
                        "step": step["name"],
                        "status": "success",
                        "output": result,
                    }
                )

                # Store outputs for next steps
                for output_key in step.get("outputs", []):
                    results["outputs"][output_key] = result

            except Exception as e:
                results["steps_completed"].append(
                    {"step": step["name"], "status": "failed", "error": str(e)}
                )
                results["status"] = "failed"
                break

        if results["status"] == "running":
            results["status"] = "completed"

        return results

    async def execute_protocol(self, name: str, inputs: Dict) -> Any:
        """Execute a protocol"""
        from protocols.loader import load_protocol

        protocol = load_protocol(name)
        if protocol:
            return protocol["task"]()
        raise Exception(f"Protocol {name} not found")

    async def execute_analyzer(self, name: str, inputs: Dict) -> Any:
        """Execute an analyzer"""
        # Load and execute analyzer
        # This would be implemented based on analyzer type
        return {"analysis": "complete", "insights": []}

    async def execute_agent(self, name: str, inputs: Dict) -> Any:
        """Execute an agent task"""
        if name in self.agents:
            return await self.agents[name].execute(inputs)
        raise Exception(f"Agent {name} not found")

    async def learn_from_execution(self, workflow: Dict, result: Dict):
        """Learn from execution to improve future workflows"""
        # Track execution metrics
        execution_data = {
            "workflow_id": workflow["id"],
            "intent": workflow["intent"]["original_intent"],
            "success": result["status"] == "completed",
            "duration": datetime.utcnow().timestamp()
            - float(workflow["id"].split("_")[1]),
            "steps_count": len(result["steps_completed"]),
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Store in knowledge graph for future optimization
        await self.knowledge_graph.store_execution(execution_data)

        # Trigger mutation if needed
        if not execution_data["success"]:
            await self.trigger_improvement(workflow, result)

    async def trigger_improvement(self, workflow: Dict, result: Dict):
        """Trigger system improvement based on failure"""
        log(f"🔧 Triggering improvement for workflow {workflow['id']}")
        # This would analyze the failure and potentially:
        # - Mutate protocols
        # - Adjust workflow generation
        # - Update component selection logic


class MessageBus:
    """Handles A2A (Agent to Agent) communication"""

    def __init__(self):
        self.subscribers = {}
        self.message_queue = asyncio.Queue()

    async def publish(self, topic: str, message: Dict):
        """Publish message to topic"""
        await self.message_queue.put(
            {
                "topic": topic,
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    async def subscribe(self, topic: str, callback):
        """Subscribe to topic"""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)

    async def process_messages(self):
        """Process message queue"""
        while True:
            msg = await self.message_queue.get()
            topic = msg["topic"]
            if topic in self.subscribers:
                for callback in self.subscribers[topic]:
                    await callback(msg["message"])


class KnowledgeGraph:
    """Manages relationships and accelerates learning"""

    def __init__(self):
        self.graph = {}  # Simple in-memory for now

    async def store_execution(self, data: Dict):
        """Store execution data in graph"""
        node_id = data["workflow_id"]
        self.graph[node_id] = {
            "data": data,
            "relationships": [],
            "insights": [],
        }

    async def find_similar_executions(self, intent: str) -> List[Dict]:
        """Find similar past executions"""
        similar = []
        for node_id, node in self.graph.items():
            if intent.lower() in node["data"]["intent"].lower():
                similar.append(node["data"])
        return similar

    async def get_optimization_hints(self, workflow: Dict) -> List[str]:
        """Get hints for optimizing workflow"""
        hints = []
        # Analyze past executions for patterns
        # This would use ML/pattern recognition
        return hints


# CLI Interface matching your example
async def run_mcp(
        intent: str,
        sources: List[str],
        quantum: bool = False) -> Dict:
    """Run MCP orchestration from CLI"""
    engine = OrchestrationEngine()

    options = {"quantum_optimization": quantum, "sources": sources}

    result = await engine.execute_intent(intent, sources, options)
    return result
