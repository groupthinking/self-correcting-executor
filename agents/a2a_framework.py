# A2A (Agent-to-Agent) Communication Framework
# Enables autonomous agents to negotiate, collaborate, and share context

import asyncio
import json
from typing import Dict, List, Any, Callable
from datetime import datetime
from abc import ABC, abstractmethod
import uuid


class A2AMessage:
    """Standard message format for agent communication"""

    def __init__(
        self,
        sender: str,
        recipient: str,
        message_type: str,
        content: Dict,
        conversation_id: str = None,
    ):
        self.id = str(uuid.uuid4())
        self.sender = sender
        self.recipient = recipient
        self.message_type = message_type
        self.content = content
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "sender": self.sender,
            "recipient": self.recipient,
            "message_type": self.message_type,
            "content": self.content,
            "conversation_id": self.conversation_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "A2AMessage":
        msg = cls(
            sender=data["sender"],
            recipient=data["recipient"],
            message_type=data["message_type"],
            content=data["content"],
            conversation_id=data.get("conversation_id"),
        )
        msg.id = data["id"]
        msg.timestamp = data["timestamp"]
        return msg


class BaseAgent(ABC):
    """Base class for all agents with A2A capabilities"""

    def __init__(self, agent_id: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.conversations = {}
        self.message_handlers = {}
        self.state = {}

    @abstractmethod
    async def process_intent(self, intent: Dict) -> Dict:
        """Process an intent and return result"""
        pass

    async def send_message(
        self, recipient: str, message_type: str, content: Dict
    ) -> A2AMessage:
        """Send message to another agent"""
        msg = A2AMessage(
            sender=self.agent_id,
            recipient=recipient,
            message_type=message_type,
            content=content,
        )

        # Send through message bus
        await message_bus.send(msg)
        return msg

    async def receive_message(self, message: A2AMessage):
        """Receive and process message from another agent"""
        # Store in conversation history
        if message.conversation_id not in self.conversations:
            self.conversations[message.conversation_id] = []
        self.conversations[message.conversation_id].append(message)

        # Process based on message type
        handler = self.message_handlers.get(message.message_type)
        if handler:
            response = await handler(message)
            if response:
                await self.send_message(
                    recipient=message.sender,
                    message_type=f"{message.message_type}_response",
                    content=response,
                )

    def register_handler(self, message_type: str, handler: Callable):
        """Register handler for specific message type"""
        self.message_handlers[message_type] = handler


class NegotiationAgent(BaseAgent):
    """Agent specialized in negotiating between other agents"""

    def __init__(self):
        super().__init__("negotiator", ["negotiate", "mediate", "coordinate"])
        self.register_handler("negotiate_request", self.handle_negotiation)

    async def process_intent(self, intent: Dict) -> Dict:
        """Process negotiation intent"""
        if intent.get("action") == "negotiate":
            return await self.negotiate_between_agents(
                intent["agents"], intent["topic"], intent.get("constraints", {})
            )
        return {"error": "Unknown intent"}

    async def negotiate_between_agents(
        self, agents: List[str], topic: str, constraints: Dict
    ) -> Dict:
        """Coordinate negotiation between multiple agents"""
        negotiation_id = str(uuid.uuid4())

        # Start negotiation with each agent
        proposals = {}
        for agent in agents:
            response = await self.send_message(
                recipient=agent,
                message_type="proposal_request",
                content={
                    "negotiation_id": negotiation_id,
                    "topic": topic,
                    "constraints": constraints,
                },
            )
            # Wait for proposals (simplified - real implementation would be async)
            proposals[agent] = None

        # Analyze proposals and find optimal solution
        solution = await self.find_optimal_solution(proposals, constraints)

        # Communicate decision
        for agent in agents:
            await self.send_message(
                recipient=agent,
                message_type="negotiation_result",
                content={"negotiation_id": negotiation_id, "solution": solution},
            )

        return {
            "negotiation_id": negotiation_id,
            "participants": agents,
            "solution": solution,
            "status": "completed",
        }

    async def find_optimal_solution(self, proposals: Dict, constraints: Dict) -> Dict:
        """Find optimal solution from proposals"""
        # This would use optimization algorithms
        # For now, return a simple solution
        return {"agreed_terms": {}, "consensus_level": 0.85}

    async def handle_negotiation(self, message: A2AMessage) -> Dict:
        """Handle incoming negotiation request"""
        # Process negotiation request
        return {"status": "accepted", "terms": {}}


class DataAnalysisAgent(BaseAgent):
    """Agent specialized in data analysis"""

    def __init__(self):
        super().__init__("data_analyst", ["analyze", "process", "insights"])
        self.register_handler("analysis_request", self.handle_analysis_request)

    async def process_intent(self, intent: Dict) -> Dict:
        """Process data analysis intent"""
        if intent.get("action") == "analyze":
            return await self.analyze_data(
                intent["data_source"], intent.get("analysis_type", "general")
            )
        return {"error": "Unknown intent"}

    async def analyze_data(self, data_source: str, analysis_type: str) -> Dict:
        """Perform data analysis"""
        # This would connect to actual data sources
        # For now, return mock analysis
        return {
            "data_source": data_source,
            "analysis_type": analysis_type,
            "insights": [
                "Pattern detected in time series",
                "Anomaly found at timestamp X",
                "Correlation between A and B",
            ],
            "confidence": 0.92,
        }

    async def handle_analysis_request(self, message: A2AMessage) -> Dict:
        """Handle incoming analysis request"""
        content = message.content
        result = await self.analyze_data(
            content.get("data_source"), content.get("analysis_type", "general")
        )
        return result


class A2AMessageBus:
    """Central message bus for agent communication"""

    def __init__(self):
        self.agents = {}
        self.message_queue = asyncio.Queue()
        self.running = False

    def register_agent(self, agent: BaseAgent):
        """Register agent with message bus"""
        self.agents[agent.agent_id] = agent

    async def send(self, message: A2AMessage):
        """Send message through bus"""
        await self.message_queue.put(message)

    async def start(self):
        """Start message processing"""
        self.running = True
        while self.running:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)

                # Deliver to recipient
                recipient = self.agents.get(message.recipient)
                if recipient:
                    await recipient.receive_message(message)
                else:
                    # Log undeliverable message
                    print(f"Agent {message.recipient} not found")

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error processing message: {e}")

    def stop(self):
        """Stop message processing"""
        self.running = False


# Global message bus instance
message_bus = A2AMessageBus()


# Example usage
async def demonstrate_a2a():
    """Demonstrate A2A communication"""

    # Create agents
    negotiator = NegotiationAgent()
    analyst = DataAnalysisAgent()

    # Register with message bus
    message_bus.register_agent(negotiator)
    message_bus.register_agent(analyst)

    # Start message bus
    bus_task = asyncio.create_task(message_bus.start())

    # Example: Analyst requests negotiation
    await analyst.send_message(
        recipient="negotiator",
        message_type="negotiate_request",
        content={
            "topic": "resource_allocation",
            "requirements": {"cpu": "4 cores", "memory": "16GB"},
        },
    )

    # Let messages process
    await asyncio.sleep(1)

    # Stop message bus
    message_bus.stop()
    await bus_task
