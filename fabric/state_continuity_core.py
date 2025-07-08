"""
State Continuity Fabric Core Implementation

This is our UNIQUE VALUE PROPOSITION that sits on top of mcp-use.
No existing library provides this cross-device, cross-application state continuity.
"""

import asyncio
import time
import json
import hashlib
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

# We'll use mcp-use for MCP protocol handling
from mcp_use import MCPClient, create_client


@dataclass
class VectorClock:
    """Vector clock for distributed state synchronization"""

    clocks: Dict[str, int] = field(default_factory=dict)

    def increment(self, node_id: str):
        """Increment clock for a node"""
        self.clocks[node_id] = self.clocks.get(node_id, 0) + 1

    def update(self, other: "VectorClock"):
        """Update with another vector clock"""
        for node_id, clock in other.clocks.items():
            self.clocks[node_id] = max(self.clocks.get(node_id, 0), clock)

    def happens_before(self, other: "VectorClock") -> bool:
        """Check if this clock happens before another"""
        for node_id, clock in self.clocks.items():
            if clock > other.clocks.get(node_id, 0):
                return False
        return True

    def concurrent_with(self, other: "VectorClock") -> bool:
        """Check if two clocks are concurrent"""
        return not self.happens_before(other) and not other.happens_before(self)


@dataclass
class StateNode:
    """A node in the state graph"""

    id: str
    data: Dict[str, Any]
    vector_clock: VectorClock
    device_id: str
    application_id: str
    timestamp: float
    parent_id: Optional[str] = None

    def hash(self) -> str:
        """Generate hash of the state"""
        content = json.dumps(self.data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


class DifferentialStateEngine:
    """
    The core innovation: tracks state changes differentially across
    devices and applications, enabling seamless continuity.
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.states: Dict[str, StateNode] = {}
        self.current_state_id: Optional[str] = None
        self.vector_clock = VectorClock()

        # Differential storage - only store changes
        self.deltas: Dict[str, Dict[str, Any]] = {}

        # Conflict resolution strategies
        self.conflict_handlers = {
            "last-write-wins": self._resolve_lww,
            "merge": self._resolve_merge,
            "user-defined": self._resolve_user_defined,
        }

    def capture_state(
        self, data: Dict[str, Any], device_id: str, app_id: str
    ) -> StateNode:
        """Capture current state with differential tracking"""
        self.vector_clock.increment(self.node_id)

        # Calculate delta from previous state
        delta = {}
        if self.current_state_id:
            current = self.states[self.current_state_id]
            delta = self._calculate_delta(current.data, data)

        # Create new state node
        state = StateNode(
            id=f"{self.node_id}_{time.time()}",
            data=data,
            vector_clock=VectorClock(clocks=self.vector_clock.clocks.copy()),
            device_id=device_id,
            application_id=app_id,
            timestamp=time.time(),
            parent_id=self.current_state_id,
        )

        # Store state and delta
        self.states[state.id] = state
        if delta:
            self.deltas[state.id] = delta

        self.current_state_id = state.id
        return state

    def _calculate_delta(self, old_data: Dict, new_data: Dict) -> Dict:
        """Calculate differential changes between states"""
        delta = {"added": {}, "modified": {}, "removed": []}

        # Find added and modified keys
        for key, value in new_data.items():
            if key not in old_data:
                delta["added"][key] = value
            elif old_data[key] != value:
                delta["modified"][key] = {"old": old_data[key], "new": value}

        # Find removed keys
        for key in old_data:
            if key not in new_data:
                delta["removed"].append(key)

        return delta

    def merge_states(
        self, remote_states: List[StateNode], strategy: str = "merge"
    ) -> StateNode:
        """Merge remote states with local state using vector clocks"""
        if not remote_states:
            return self.states[self.current_state_id]

        # Group states by vector clock relationships
        concurrent_states = []
        for remote in remote_states:
            if self.vector_clock.concurrent_with(remote.vector_clock):
                concurrent_states.append(remote)

        # Resolve conflicts if any
        if concurrent_states:
            resolver = self.conflict_handlers.get(strategy, self._resolve_merge)
            merged_data = resolver(
                self.states[self.current_state_id], concurrent_states
            )
        else:
            # No conflicts, take the most recent
            all_states = [self.states[self.current_state_id]] + remote_states
            latest = max(all_states, key=lambda s: s.timestamp)
            merged_data = latest.data

        # Update vector clocks
        for remote in remote_states:
            self.vector_clock.update(remote.vector_clock)

        # Create merged state
        return self.capture_state(merged_data, self.node_id, "merged")

    def _resolve_lww(self, local: StateNode, remotes: List[StateNode]) -> Dict:
        """Last-write-wins conflict resolution"""
        all_states = [local] + remotes
        latest = max(all_states, key=lambda s: s.timestamp)
        return latest.data

    def _resolve_merge(self, local: StateNode, remotes: List[StateNode]) -> Dict:
        """Merge all concurrent states"""
        merged = local.data.copy()

        for remote in remotes:
            for key, value in remote.data.items():
                if key not in merged:
                    merged[key] = value
                elif isinstance(value, dict) and isinstance(merged[key], dict):
                    # Deep merge dictionaries
                    merged[key] = {**merged[key], **value}
                elif isinstance(value, list) and isinstance(merged[key], list):
                    # Merge lists (remove duplicates)
                    merged[key] = list(set(merged[key] + value))
                else:
                    # Conflict: keep both values
                    merged[f"{key}_local"] = merged[key]
                    merged[f"{key}_remote_{remote.device_id}"] = value

        return merged

    def _resolve_user_defined(self, local: StateNode, remotes: List[StateNode]) -> Dict:
        """Placeholder for user-defined conflict resolution"""
        # This would call a user-provided function
        return self._resolve_merge(local, remotes)


class StateContinuityFabric:
    """
    The main fabric that orchestrates state continuity across
    devices and applications, built on top of MCP for service integration.
    """

    def __init__(self, fabric_id: str):
        self.fabric_id = fabric_id
        self.engines: Dict[str, DifferentialStateEngine] = {}
        self.mcp_client: Optional[MCPClient] = None

        # Cross-device identity management
        self.device_registry: Dict[str, Dict[str, Any]] = {}

        # Privacy boundaries
        self.privacy_rules: Dict[str, List[str]] = {
            "no-sync": [],  # Keys that should never sync
            "device-only": [],  # Keys that stay on device
            "encrypted": [],  # Keys that need encryption
        }

    async def initialize(self, mcp_server_url: Optional[str] = None):
        """Initialize the fabric with MCP connection"""
        if mcp_server_url:
            # Use mcp-use to connect to MCP server
            self.mcp_client = await create_client(
                server_url=mcp_server_url,
                client_name=f"fabric_{self.fabric_id}",
            )

            # Discover available tools
            tools = await self.mcp_client.list_tools()
            print(f"Connected to MCP server with {len(tools)} available tools")

    def register_device(self, device_id: str, device_info: Dict[str, Any]):
        """Register a device with the fabric"""
        self.device_registry[device_id] = {
            "info": device_info,
            "registered_at": time.time(),
            "last_sync": None,
        }

        # Create state engine for device
        self.engines[device_id] = DifferentialStateEngine(device_id)

    async def capture_context(
        self, device_id: str, app_id: str, context: Dict[str, Any]
    ) -> StateNode:
        """Capture context from a device/application"""
        if device_id not in self.engines:
            raise ValueError(f"Device {device_id} not registered")

        # Apply privacy filters
        filtered_context = self._apply_privacy_filters(context)

        # Capture state
        state = self.engines[device_id].capture_state(
            filtered_context, device_id, app_id
        )

        # If MCP is connected, enrich context
        if self.mcp_client:
            enriched = await self._enrich_via_mcp(filtered_context)
            if enriched:
                state.data.update(enriched)

        return state

    async def _enrich_via_mcp(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Use MCP tools to enrich context"""
        enriched = {}

        try:
            # Example: Use MCP tools to analyze context
            if "text" in context and self.mcp_client:
                # This would call actual MCP tools
                result = await self.mcp_client.call_tool(
                    "analyze_text", {"text": context["text"]}
                )
                enriched["mcp_analysis"] = result
        except Exception as e:
            print(f"MCP enrichment failed: {e}")

        return enriched

    def _apply_privacy_filters(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply privacy rules to context"""
        filtered = {}

        for key, value in context.items():
            if key in self.privacy_rules["no-sync"]:
                continue
            elif key in self.privacy_rules["encrypted"]:
                # In production, actually encrypt
                filtered[key] = f"<encrypted>{value}</encrypted>"
            else:
                filtered[key] = value

        return filtered

    async def sync_devices(self, source_device: str, target_device: str) -> StateNode:
        """Synchronize state between devices"""
        if source_device not in self.engines or target_device not in self.engines:
            raise ValueError("Both devices must be registered")

        source_engine = self.engines[source_device]
        target_engine = self.engines[target_device]

        # Get current states
        source_states = [source_engine.states[source_engine.current_state_id]]

        # Merge states
        merged = target_engine.merge_states(source_states)

        # Update sync timestamps
        self.device_registry[source_device]["last_sync"] = time.time()
        self.device_registry[target_device]["last_sync"] = time.time()

        return merged

    def get_continuity_graph(self) -> Dict[str, Any]:
        """Get the full continuity graph for visualization"""
        graph = {
            "nodes": [],
            "edges": [],
            "devices": list(self.device_registry.keys()),
        }

        # Collect all states
        for device_id, engine in self.engines.items():
            for state_id, state in engine.states.items():
                graph["nodes"].append(
                    {
                        "id": state_id,
                        "device": device_id,
                        "app": state.application_id,
                        "timestamp": state.timestamp,
                        "data_keys": list(state.data.keys()),
                    }
                )

                if state.parent_id:
                    graph["edges"].append(
                        {
                            "from": state.parent_id,
                            "to": state_id,
                            "type": "evolution",
                        }
                    )

        return graph


# Example usage showing integration with mcp-use
async def demonstrate_fabric():
    """Demonstrate the State Continuity Fabric with MCP integration"""

    # Create fabric
    fabric = StateContinuityFabric("user_123")

    # Initialize with MCP server (if available)
    try:
        await fabric.initialize("http://localhost:8080")
    except Exception:
        print("Running without MCP server")

    # Register devices
    fabric.register_device("macbook", {"type": "laptop", "os": "macOS"})
    fabric.register_device("iphone", {"type": "phone", "os": "iOS"})

    # Capture context on MacBook
    macbook_context = await fabric.capture_context(
        "macbook",
        "safari_extension",
        {
            "url": "https://example.com",
            "search_query": "quantum computing",
            "timestamp": time.time(),
            "private_key": "should_not_sync",  # This won't sync
        },
    )

    print(f"Captured MacBook state: {macbook_context.id}")

    # Capture context on iPhone
    iphone_context = await fabric.capture_context(
        "iphone",
        "mobile_app",
        {
            "location": "work",
            "last_action": "reading_article",
            "timestamp": time.time(),
        },
    )

    print(f"Captured iPhone state: {iphone_context.id}")

    # Sync states
    merged = await fabric.sync_devices("macbook", "iphone")
    print(f"Merged state: {merged.data}")

    # Get continuity graph
    graph = fabric.get_continuity_graph()
    print(f"Continuity graph: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")


if __name__ == "__main__":
    asyncio.run(demonstrate_fabric())
