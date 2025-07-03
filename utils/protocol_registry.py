# Protocol Registry - Manages protocol discovery and deployment
import json
import os
import hashlib
from datetime import datetime
from utils.logger import log


class ProtocolRegistry:
    """Central registry for protocol management"""

    def __init__(self):
        self.registry_file = "protocols/categories.json"
        self.deployment_config = "deployment_config.json"

    def load_registry(self):
        """Load protocol registry from file"""
        try:
            with open(self.registry_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {"categories": {}, "registry": {}}

    def register_protocol(self, name, category, location="local", metadata=None):
        """Register a new protocol"""
        registry = self.load_registry()

        # Add to category
        if category not in registry["categories"]:
            registry["categories"][category] = {
                "description": f"{category.title()} protocols",
                "protocols": [],
            }

        if name not in registry["categories"][category]["protocols"]:
            registry["categories"][category]["protocols"].append(name)

        # Add protocol metadata
        if "protocols" not in registry:
            registry["protocols"] = {}

        registry["protocols"][name] = {
            "category": category,
            "location": location,
            "registered_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
            "checksum": self._calculate_checksum(name),
        }

        # Update registry metadata
        registry["registry"]["total_protocols"] = sum(
            len(cat["protocols"]) for cat in registry["categories"].values()
        )
        registry["registry"]["last_updated"] = datetime.utcnow().isoformat() + "Z"

        # Save registry
        with open(self.registry_file, "w") as f:
            json.dump(registry, f, indent=2)

        log(f"Protocol {name} registered in category {category}")
        return True

    def _calculate_checksum(self, protocol_name):
        """Calculate checksum for protocol file"""
        protocol_file = f"protocols/{protocol_name}.py"
        if os.path.exists(protocol_file):
            with open(protocol_file, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
        return None

    def deploy_protocol(self, name, target="local", config=None):
        """Deploy protocol to target environment"""
        registry = self.load_registry()

        if name not in registry.get("protocols", {}):
            log(f"Protocol {name} not found in registry")
            return False

        registry["protocols"][name]

        # Deployment logic based on target
        if target == "local":
            # Already deployed locally
            return True
        elif target == "docker":
            # Deploy to Docker container
            return self._deploy_to_docker(name, config)
        elif target == "remote":
            # Deploy to remote worker
            return self._deploy_to_remote(name, config)

        return False

    def _deploy_to_docker(self, name, config):
        """Deploy protocol to Docker container"""
        # Future implementation for Docker deployment
        log(f"Docker deployment for {name} not yet implemented")
        return False

    def _deploy_to_remote(self, name, config):
        """Deploy protocol to remote worker"""
        # Future implementation for remote deployment
        log(f"Remote deployment for {name} not yet implemented")
        return False

    def get_protocol_info(self, name):
        """Get information about a protocol"""
        registry = self.load_registry()
        protocols = registry.get("protocols", {})
        return protocols.get(name, None)

    def list_by_category(self, category=None):
        """List protocols by category"""
        registry = self.load_registry()

        if category:
            cat_info = registry["categories"].get(category, {})
            return cat_info.get("protocols", [])
        else:
            return registry["categories"]

    def search_protocols(self, query):
        """Search protocols by name or metadata"""
        registry = self.load_registry()
        results = []

        query_lower = query.lower()

        # Search in protocol names
        for name, info in registry.get("protocols", {}).items():
            if query_lower in name.lower():
                results.append(
                    {
                        "name": name,
                        "category": info.get("category"),
                        "location": info.get("location"),
                    }
                )

        # Search in categories
        for cat_name, cat_info in registry["categories"].items():
            if (
                query_lower in cat_name.lower()
                or query_lower in cat_info.get("description", "").lower()
            ):
                for protocol in cat_info["protocols"]:
                    if not any(r["name"] == protocol for r in results):
                        results.append(
                            {
                                "name": protocol,
                                "category": cat_name,
                                "location": "local",
                            }
                        )

        return results
