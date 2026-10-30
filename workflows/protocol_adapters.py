"""Adapters to the repository's real local MCP/A2A implementations.

Remote AP2 providers implement ProtocolBindings' explicit mandate contract; the
archived client supplied no endpoint/wire specification, so none is invented.
"""

import json


class LocalMCP:
    def __init__(self, server):
        self.server = server

    async def list_tools(self):
        return await self.server.list_tools()

    async def call_tool(self, name, arguments):
        content = await self.server.call_tool(name, arguments)
        return [item.model_dump(mode="json") for item in content]


class LocalA2A:
    def __init__(self, registry, capabilities):
        self.registry = registry
        self.capabilities = capabilities

    async def discover_agents(self, capability):
        return [
            {"name": name, "capabilities": self.capabilities[name]}
            for name in self.registry.list_agents()
            if capability in self.capabilities.get(name, [])
        ]

    async def dispatch_task(self, agent_id, task):
        agent = self.registry.get_agent(agent_id)
        if agent is None:
            raise ValueError("A2A agent is absent from the configured registry")
        result = await agent.process_intent(task["parameters"])
        if isinstance(result, dict) and (
            result.get("success") is False
            or result.get("status") in {"failed", "error", "pending"}
            or bool(result.get("error"))
        ):
            raise RuntimeError(result.get("error", "A2A agent failed"))
        # Reject nonserializable results before claiming a durable trajectory.
        json.dumps(result, allow_nan=False)
        return {"taskId": task["taskId"], "status": "completed", "result": result}
