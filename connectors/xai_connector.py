"""
xAI Connector for MCP Integration
=================================

Integrates xAI's Grok models with our MCP infrastructure,
including live search capabilities.
"""

import os
import json
import httpx
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class XAIConnector:
    """
    Production-ready connector for xAI's Grok models with MCP integration.

    Features:
    - Grok-3 model access
    - Live search capabilities
    - Streaming support
    - Material Design 3 compliant responses
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            raise ValueError("XAI API key required")

        self.base_url = "https://api.x.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Material Design 3 color tokens for responses
        self.md3_colors = {
            "primary": "#6750A4",
            "on_primary": "#FFFFFF",
            "primary_container": "#EADDFF",
            "on_primary_container": "#21005D",
            "error": "#BA1A1A",
            "error_container": "#FFDAD6",
            "success": "#006E1C",
            "success_container": "#C6F181",
        }

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "grok-3-latest",
        temperature: float = 0.7,
        stream: bool = False,
        enable_search: bool = False,
    ) -> Dict[str, Any]:
        """
        Send chat completion request to xAI.

        Args:
            messages: List of message dictionaries
            model: Model to use (default: grok-3-latest)
            temperature: Sampling temperature
            stream: Whether to stream responses
            enable_search: Enable live search capabilities

        Returns:
            API response or formatted result
        """
        payload = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "stream": stream,
        }

        # Add search parameters if enabled
        if enable_search:
            payload["search"] = {"enabled": True, "max_results": 5}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=30.0,
                )
                response.raise_for_status()

                result = response.json()

                # Format response with Material Design 3 structure
                return self._format_md3_response(result)

            except httpx.HTTPStatusError as e:
                logger.error(
                    f"xAI API error: {e.response.status_code} - {e.response.text}"
                )
                return {
                    "success": False,
                    "error": str(e),
                    "color": self.md3_colors["error"],
                }
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "color": self.md3_colors["error"],
                }

    def _format_md3_response(
        self, api_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Format API response following Material Design 3 principles.
        """
        content = api_response["choices"][0]["message"]["content"]

        return {
            "success": True,
            "content": content,
            "metadata": {
                "model": api_response["model"],
                "usage": api_response["usage"],
                "sources_used": api_response["usage"].get(
                    "num_sources_used", 0
                ),
                "timestamp": datetime.now().isoformat(),
            },
            "ui": {
                "color_scheme": self.md3_colors,
                "typography": {
                    "headline": "Roboto",
                    "body": "Roboto",
                    "code": "Roboto Mono",
                },
                "elevation": {"card": 1, "modal": 3},
            },
        }

    async def search_and_summarize(self, query: str) -> Dict[str, Any]:
        """
        Use Grok's live search to find and summarize current information.
        """
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant with access to current information. Search for and summarize the requested information.",
            },
            {"role": "user", "content": query},
        ]

        return await self.chat_completion(
            messages=messages,
            enable_search=True,
            temperature=0.3,  # Lower temperature for factual search
        )

    async def code_generation(
        self, task: str, language: str = "python", follow_md3: bool = True
    ) -> Dict[str, Any]:
        """
        Generate code using Grok with optional Material Design 3 compliance.
        """
        system_prompt = f"You are an expert {language} developer."

        if follow_md3 and language in ["javascript", "typescript", "dart"]:
            system_prompt += (
                " Follow Material Design 3 guidelines for any UI components."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        return await self.chat_completion(
            messages=messages, temperature=0.2  # Lower for code generation
        )


class XAIMCPTool:
    """
    MCP Tool wrapper for xAI functionality.
    """

    def __init__(self):
        self.connector = XAIConnector()
        self.name = "xai_grok"
        self.description = "Access xAI's Grok models with live search"

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute xAI tool based on parameters.

        Expected params:
        - action: "chat", "search", "code"
        - query: The user query
        - options: Additional options
        """
        action = params.get("action", "chat")
        query = params.get("query", "")
        options = params.get("options", {})

        if action == "search":
            return await self.connector.search_and_summarize(query)
        elif action == "code":
            return await self.connector.code_generation(
                task=query,
                language=options.get("language", "python"),
                follow_md3=options.get("follow_md3", True),
            )
        else:  # Default chat
            messages = [{"role": "user", "content": query}]
            return await self.connector.chat_completion(
                messages=messages, **options
            )


# Example usage
async def demo_xai_integration():
    """Demonstrate xAI integration with Material Design 3."""

    connector = XAIConnector()

    # Test live search
    print("Testing live search...")
    search_result = await connector.search_and_summarize(
        "What are the latest updates in MCP (Model Context Protocol)?"
    )
    print(f"Search result: {json.dumps(search_result, indent=2)}")

    # Test code generation with MD3
    print("\nTesting MD3-compliant code generation...")
    code_result = await connector.code_generation(
        task="Create a Material Design 3 card component in React",
        language="javascript",
        follow_md3=True,
    )
    print(f"Code result: {json.dumps(code_result, indent=2)}")


if __name__ == "__main__":
    # Set API key for testing
    os.environ["XAI_API_KEY"] = (
        "xai-BEQ4Au7tyMj3NDROcMShfCH8dd1o90upMbBZ8aOjaDeMnjkfESQzznlKDnAQf0anrTLsKrYTdTAQYvLp"
    )
    asyncio.run(demo_xai_integration())
