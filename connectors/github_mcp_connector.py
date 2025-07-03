#!/usr/bin/env python3
"""
GitHub MCP Connector
===================

Real GitHub integration using GitHub API v3 and MCP protocol.
Provides access to repositories, issues, pull requests, and more.
Enhanced with a circuit breaker pattern for resilience.

Features:
- Asynchronous API calls with aiohttp
- Comprehensive action mapping (search, get repo, issues, PRs, etc.)
- Resilient API calls with a Circuit Breaker pattern
- Rate limiting awareness
- MCP protocol compliance
"""

import asyncio
import aiohttp
import logging
import os
import time
from typing import Dict, Any, Optional
from datetime import datetime
import base64

# Assuming the base class is in a file like this.
# Adjust the import if your project structure is different.
from connectors.mcp_base import MCPConnector

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """A simple implementation of the circuit breaker pattern."""

    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failure_count = 0
        self._last_failure_time = 0
        self._state = "CLOSED"  # Can be CLOSED, OPEN, HALF_OPEN

    @property
    def state(self):
        if (
            self._state == "OPEN"
            and time.time() - self._last_failure_time > self.recovery_timeout
        ):
            self._state = "HALF_OPEN"
        return self._state

    def record_failure(self):
        self._failure_count += 1
        if self._failure_count >= self.failure_threshold:
            self._state = "OPEN"
            self._last_failure_time = time.time()

    def record_success(self):
        self._state = "CLOSED"
        self._failure_count = 0
        self._last_failure_time = 0


class GitHubMCPConnector(MCPConnector):
    """
    GitHub MCP Connector for real repository access and integration
    """

    def __init__(self, api_token: Optional[str] = None):
        super().__init__("github_mcp", "version_control")
        self.api_token = api_token or os.environ.get("GITHUB_TOKEN")
        self.base_url = "https://api.github.com"
        self.session = None
        self.connected = False
        self.breaker = CircuitBreaker()  # Add the circuit breaker instance

        # Rate limiting
        self.rate_limit_remaining = 5000
        self.rate_limit_reset = None

    async def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to GitHub API"""
        try:
            self.api_token = config.get("api_token", self.api_token)

            if not self.api_token:
                logger.error(
                    "GitHub API token required. Set GITHUB_TOKEN "
                    "environment variable or pass in config."
                )
                return False

            headers = {
                "Authorization": f"token {self.api_token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "MCP-GitHub-Connector/1.0",
            }

            self.session = aiohttp.ClientSession(headers=headers)

            async with self.session.get(f"{self.base_url}/user") as response:
                if response.status == 200:
                    user_data = await response.json()
                    logger.info(
                        f"Connected to GitHub as: "
                        f"{user_data.get('login', 'Unknown')}"
                    )
                    self.connected = True
                    await self._update_rate_limit()
                    # Connection successful, close breaker
                    self.breaker.record_success()
                    return True
                else:
                    logger.error(
                        f"GitHub API connection failed: {response.status}"
                    )
                    # Connection failed, record failure
                    self.breaker.record_failure()
                    return False

        except Exception as e:
            logger.error(f"Failed to connect to GitHub: {e}")
            self.breaker.record_failure()
            return False

    async def disconnect(self) -> bool:
        """Disconnect from GitHub API"""
        if self.session and not self.session.closed:
            await self.session.close()
        self.connected = False
        return True

    async def get_context(self):
        """Get GitHub context"""
        return self.context

    async def send_context(self, context) -> bool:
        """Send context to GitHub system"""
        self.context = context
        return True

    async def execute_action(
        self, action: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute GitHub action, now protected by a circuit breaker."""
        if self.breaker.state == "OPEN":
            logger.warning(
                f"Circuit breaker is open. Action '{action}' blocked."
            )
            return {
                "error": (
                    "Circuit breaker is open due to repeated failures. "
                    "Please try again later."
                ),
                "action": action,
            }

        if not self.connected or not self.session or self.session.closed:
            logger.error("Action executed while not connected to GitHub API.")
            return {"error": "Not connected to GitHub API"}

        actions = {
            "search_repositories": self.search_repositories,
            "get_repository": self.get_repository,
            "get_issues": self.get_issues,
            "get_pull_requests": self.get_pull_requests,
            "get_file_content": self.get_file_content,
            "get_commits": self.get_commits,
            "get_user_info": self.get_user_info,
            "create_issue": self.create_issue,
            "get_rate_limit": self.get_rate_limit,
        }

        handler = actions.get(action)
        if handler:
            try:
                result = await handler(params)
                # Successful API call, even if GitHub returns a logical
                # error (e.g. "not found")
                # We check for success before resetting the breaker.
                if isinstance(result, dict) and result.get("success") is False:
                    # This indicates a logical failure (e.g. 404 Not Found),
                    # not necessarily a service failure
                    pass
                # Reset breaker on any successful communication
                self.breaker.record_success()
                return result
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.error(
                    f"API call for action '{action}' failed with "
                    f"network error: {e}"
                )
                self.breaker.record_failure()
                return {"success": False, "error": str(e), "action": action}
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred during action "
                    f"'{action}': {e}"
                )
                self.breaker.record_failure()
                return {"success": False, "error": str(e), "action": action}

        return {"success": False, "error": f"Unknown action: {action}"}

    async def search_repositories(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            query = params.get("query", "")
            search_query = query + (
                f" language:{params['language']}"
                if params.get("language")
                else ""
            )
            url = f"{self.base_url}/search/repositories"
            params_dict = {
                "q": search_query,
                "sort": params.get("sort", "stars"),
                "order": params.get("order", "desc"),
                "per_page": params.get("per_page", 10),
            }

            async with self.session.get(url, params=params_dict) as response:
                if response.status == 200:
                    data = await response.json()
                    repositories = [
                        {
                            "name": repo["name"],
                            "full_name": repo["full_name"],
                            "description": repo["description"],
                            "language": repo["language"],
                            "stars": repo["stargazers_count"],
                            "forks": repo["forks_count"],
                            "url": repo["html_url"],
                            "api_url": repo["url"],
                            "created_at": repo["created_at"],
                            "updated_at": repo["updated_at"],
                        }
                        for repo in data.get("items", [])
                    ]
                    return {
                        "success": True,
                        "total_count": data.get("total_count", 0),
                        "repositories": repositories,
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Search failed: {response.status}",
                        "status_code": response.status,
                    }
        except Exception as e:
            logger.error(f"Repository search failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_repository(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            owner, repo = params.get("owner"), params.get("repo")
            if not owner or not repo:
                return {
                    "success": False,
                    "error": "Owner and repo parameters required",
                }
            url = f"{self.base_url}/repos/{owner}/{repo}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {"success": True, "repository": data}
                else:
                    return {
                        "success": False,
                        "error": f"Repository not found: {response.status}",
                        "status_code": response.status,
                    }
        except Exception as e:
            logger.error(f"Get repository failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_issues(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            owner, repo = params.get("owner"), params.get("repo")
            if not owner or not repo:
                return {
                    "success": False,
                    "error": "Owner and repo parameters required",
                }
            url = f"{self.base_url}/repos/{owner}/{repo}/issues"
            params_dict = {
                "state": params.get("state", "open"),
                "per_page": params.get("per_page", 30),
                "labels": params.get("labels", ""),
            }
            async with self.session.get(
                url, params={k: v for k, v in params_dict.items() if v}
            ) as response:
                if response.status == 200:
                    issues_data = await response.json()
                    return {
                        "success": True,
                        "issues": issues_data,
                        "total_count": len(issues_data),
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Failed to get issues: {response.status}",
                        "status_code": response.status,
                    }
        except Exception as e:
            logger.error(f"Get issues failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_pull_requests(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            owner, repo = params.get("owner"), params.get("repo")
            if not owner or not repo:
                return {
                    "success": False,
                    "error": "Owner and repo parameters required",
                }
            url = f"{self.base_url}/repos/{owner}/{repo}/pulls"
            params_dict = {
                "state": params.get("state", "open"),
                "per_page": params.get("per_page", 30),
            }
            async with self.session.get(url, params=params_dict) as response:
                if response.status == 200:
                    prs_data = await response.json()
                    return {
                        "success": True,
                        "pull_requests": prs_data,
                        "total_count": len(prs_data),
                    }
                else:
                    return {
                        "success": False,
                        "error": (
                            f"Failed to get pull requests: {response.status}"
                        ),
                        "status_code": response.status,
                    }
        except Exception as e:
            logger.error(f"Get pull requests failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_file_content(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            owner, repo, path = (
                params.get("owner"),
                params.get("repo"),
                params.get("path"),
            )
            if not all([owner, repo, path]):
                return {
                    "success": False,
                    "error": "Owner, repo, and path parameters required",
                }
            url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}"
            params_dict = {"ref": params.get("ref", "main")}
            async with self.session.get(url, params=params_dict) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("type") == "file":
                        data["decoded_content"] = base64.b64decode(
                            data["content"]
                        ).decode("utf-8")
                    return {"success": True, "file_info": data}
                else:
                    return {
                        "success": False,
                        "error": f"File not found: {response.status}",
                        "status_code": response.status,
                    }
        except Exception as e:
            logger.error(f"Get file content failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_commits(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            owner, repo = params.get("owner"), params.get("repo")
            if not owner or not repo:
                return {
                    "success": False,
                    "error": "Owner and repo parameters required",
                }
            url = f"{self.base_url}/repos/{owner}/{repo}/commits"
            params_dict = {
                "sha": params.get("sha", "main"),
                "per_page": params.get("per_page", 30),
                "since": params.get("since"),
                "until": params.get("until"),
            }
            async with self.session.get(
                url, params={k: v for k, v in params_dict.items() if v}
            ) as response:
                if response.status == 200:
                    commits_data = await response.json()
                    return {
                        "success": True,
                        "commits": commits_data,
                        "total_count": len(commits_data),
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Failed to get commits: {response.status}",
                        "status_code": response.status,
                    }
        except Exception as e:
            logger.error(f"Get commits failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_user_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            username = params.get("username")
            if not username:
                return {
                    "success": False,
                    "error": "Username parameter required",
                }
            url = f"{self.base_url}/users/{username}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    return {"success": True, "user": await response.json()}
                else:
                    return {
                        "success": False,
                        "error": f"User not found: {response.status}",
                        "status_code": response.status,
                    }
        except Exception as e:
            logger.error(f"Get user info failed: {e}")
            return {"success": False, "error": str(e)}

    async def create_issue(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new issue in a repository"""
        try:
            owner, repo, title = (
                params.get("owner"),
                params.get("repo"),
                params.get("title"),
            )
            if not all([owner, repo, title]):
                return {
                    "success": False,
                    "error": "Owner, repo, and title parameters required",
                }

            url = f"{self.base_url}/repos/{owner}/{repo}/issues"
            data = {
                "title": title,
                "body": params.get("body", ""),
                "labels": params.get("labels", []),
                "assignees": params.get("assignees", []),
            }

            async with self.session.post(url, json=data) as response:
                if response.status == 201:  # Successfully created
                    return {"success": True, "issue": await response.json()}
                else:
                    error_details = await response.text()
                    logger.error(
                        f"Failed to create issue. Status: {response.status}, "
                        f"Details: {error_details}"
                    )
                    return {
                        "success": False,
                        "error": f"Failed to create issue: {response.status}",
                        "status_code": response.status,
                        "details": error_details,
                    }
        except Exception as e:
            logger.error(f"Create issue failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_rate_limit(
        self, params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Get GitHub API rate limit information"""
        try:
            url = f"{self.base_url}/rate_limit"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "success": True,
                        "rate_limit": data["resources"]["core"],
                    }
                else:
                    return {
                        "success": False,
                        "error": (
                            f"Failed to get rate limit: {response.status}"
                        ),
                        "status_code": response.status,
                    }
        except Exception as e:
            logger.error(f"Get rate limit failed: {e}")
            return {"success": False, "error": str(e)}

    async def _update_rate_limit(self):
        """Update rate limit information"""
        try:
            rate_limit = await self.get_rate_limit()
            if rate_limit["success"]:
                self.rate_limit_remaining = rate_limit["rate_limit"][
                    "remaining"
                ]
                self.rate_limit_reset = rate_limit["rate_limit"]["reset"]
        except Exception as e:
            logger.warning(f"Failed to update rate limit: {e}")


# Global GitHub connector instance
github_connector = GitHubMCPConnector()


# Example usage
async def demonstrate_github_connector():
    """Demonstrate GitHub MCP connector"""

    print("=== GitHub MCP Connector Demo ===\n")

    # Initialize connector
    config = {"api_token": os.environ.get("GITHUB_TOKEN")}

    connected = await github_connector.connect(config)
    if not connected:
        print("❌ Failed to connect to GitHub API")
        print("   Set GITHUB_TOKEN environment variable to continue")
        return

    print("✅ Connected to GitHub API\n")

    # Demo 1: Search repositories
    print("1. Searching for MCP repositories:")
    search_result = await github_connector.execute_action(
        "search_repositories",
        {
            "query": "model context protocol",
            "language": "python",
            "sort": "stars",
            "per_page": 5,
        },
    )

    if search_result.get("success"):
        print(f"   - Found {search_result.get('total_count')} repositories")
        for repo in search_result.get("repositories", [])[:3]:
            print(f"   - {repo.get('full_name')}: {repo.get('stars')} stars")
    else:
        print(f"   - Error: {search_result.get('error')}")
    print()

    # Demo 2: Get repository info
    print("2. Getting repository information:")
    repo_result = await github_connector.execute_action(
        "get_repository",
        {"owner": "modelcontextprotocol", "repo": "specification"},
    )

    if repo_result.get("success"):
        repo = repo_result.get("repository", {})
        print(f"   - {repo.get('full_name')}")
        print(f"   - Language: {repo.get('language')}")
        print(f"   - Stars: {repo.get('stargazers_count')}")
        print(f"   - Open issues: {repo.get('open_issues_count')}")
    else:
        print(f"   - Error: {repo_result.get('error')}")
    print()

    # Demo 3: Get rate limit
    print("3. Rate limit information:")
    rate_limit_result = await github_connector.execute_action(
        "get_rate_limit", {}
    )
    if rate_limit_result.get("success"):
        rl = rate_limit_result.get("rate_limit", {})
        reset_time = datetime.fromtimestamp(rl.get("reset", 0))
        print(f"   - Remaining requests: {rl.get('remaining')}")
        print(f"   - Limit: {rl.get('limit')}")
        print(f"   - Resets at: {reset_time}")
    else:
        print(f"   - Error: {rate_limit_result.get('error')}")
    print()

    # Disconnect
    await github_connector.disconnect()
    print("✅ GitHub MCP Connector Demo Complete!")


if __name__ == "__main__":
    # Setup basic logging for the demo
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # To run the demo, ensure you have a GITHUB_TOKEN environment variable set
    if not os.environ.get("GITHUB_TOKEN"):
        print(
            "Warning: GITHUB_TOKEN environment variable not set. "
            "Demo may fail."
        )
    asyncio.run(demonstrate_github_connector())
