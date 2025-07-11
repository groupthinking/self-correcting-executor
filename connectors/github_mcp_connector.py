#!/usr/bin/env python3
"""
GitHub MCP Connector

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
from typing import Dict, Any, Optional
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

        # Rate limiting
        self.rate_limit_remaining = 5000
        self.rate_limit_reset = None

    async def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to GitHub API"""
        try:
            # Get API token from config or environment
            headers = {
                "Authorization": f"token {self.api_token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "MCP-GitHub-Connector/1.0",
            }

            self.session = aiohttp.ClientSession(headers=headers)

            # Test connection
                    await self._update_rate_limit()
                    # Connection successful, close breaker
                    self.breaker.record_success()
                    return True
                else:
                    logger.error(
                        f"GitHub API connection failed: {response.status}"
                    )
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
        """Execute GitHub action"""
        if not self.connected:
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
                    f"API call for action '{action}' failed with " f"network error: {e}"
                )
                self.breaker.record_failure()
                return {"success": False, "error": str(e), "action": action}
            except Exception as e:
                return {"error": str(e), "action": action}

        return {"error": f"Unknown action: {action}"}

    async def search_repositories(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Search GitHub repositories

        Args:
            params: Search parameters including query, language, sort, etc.
        """
        try:
            query = params.get("query", "")
            language = params.get("language", "")
            sort = params.get("sort", "stars")
            order = params.get("order", "desc")
            per_page = params.get("per_page", 10)

            # Build search query
            search_query = query
            if language:
                search_query += f" language:{language}"

            url = f"{self.base_url}/search/repositories"
            params_dict = {
                "q": search_query,
                "sort": sort,
                "order": order,
                "per_page": per_page,
            }

            async with self.session.get(url, params=params_dict) as response:
                if response.status == 200:
                    data = await response.json()

                    repositories = []
                    for repo in data.get("items", []):
                        repositories.append(
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
                        )

                    return {
                        "success": True,
                        "total_count": data.get("total_count", 0),
                        "repositories": repositories,
                        "search_query": search_query,
        except Exception as e:
            logger.error(f"Repository search failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_repository(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get detailed repository information

        Args:
            params: Repository parameters (owner, repo)
        """
        try:
            owner = params.get("owner")
            repo = params.get("repo")

            if not owner or not repo:
                return {
                    "success": False,
                    "error": "Owner and repo parameters required",
                }

            url = f"{self.base_url}/repos/{owner}/{repo}"

            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()

                    return {
                        "success": True,
                        "repository": {
                            "name": data["name"],
                            "full_name": data["full_name"],
                            "description": data["description"],
                            "language": data["language"],
                            "stars": data["stargazers_count"],
                            "forks": data["forks_count"],
                            "watchers": data["watchers_count"],
                            "open_issues": data["open_issues_count"],
                            "default_branch": data["default_branch"],
                            "created_at": data["created_at"],
                            "updated_at": data["updated_at"],
                            "pushed_at": data["pushed_at"],
                            "size": data["size"],
                            "topics": data.get("topics", []),
                            "license": data.get("license", {}),
                            "homepage": data.get("homepage"),
                            "url": data["html_url"],
                            "api_url": data["url"],
                        },
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Repository not found: {response.status}",
                        "status_code": response.status,
                    }

            if not owner or not repo:
                return {
                    "success": False,
                    "error": "Owner and repo parameters required",
                }

            url = f"{self.base_url}/repos/{owner}/{repo}/issues"
            params_dict = {"state": state, "per_page": per_page}

            if labels:
                params_dict["labels"] = labels

            async with self.session.get(url, params=params_dict) as response:
                if response.status == 200:
                    issues_data = await response.json()

                    issues = []
                    for issue in issues_data:
                        issues.append(
                            {
                                "number": issue["number"],
                                "title": issue["title"],
                                "body": issue["body"],
                                "state": issue["state"],
                                "labels": [
                                    label["name"] for label in issue["labels"]
                                ],
                                "assignee": (
                                    issue["assignee"]["login"]
                                    if issue["assignee"]
                                    else None
                                ),
                                "created_at": issue["created_at"],
                                "updated_at": issue["updated_at"],
                                "url": issue["html_url"],
                                "api_url": issue["url"],
                            }
                        )

                    return {
                        "success": True,
                        "issues": issues,
                        "total_count": len(issues),
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Failed to get issues: {response.status}",
                        "status_code": response.status,
                    }

            if not owner or not repo:
                return {
                    "success": False,
                    "error": "Owner and repo parameters required",
                }

            url = f"{self.base_url}/repos/{owner}/{repo}/pulls"
            params_dict = {"state": state, "per_page": per_page}

            async with self.session.get(url, params=params_dict) as response:
                if response.status == 200:
                    prs_data = await response.json()

                    pull_requests = []
                    for pr in prs_data:
                        pull_requests.append(
                            {
                                "number": pr["number"],
                                "title": pr["title"],
                                "body": pr["body"],
                                "state": pr["state"],
                                "user": pr["user"]["login"],
                                "created_at": pr["created_at"],
                                "updated_at": pr["updated_at"],
                                "merged_at": pr["merged_at"],
                                "mergeable": pr["mergeable"],
                                "mergeable_state": pr["mergeable_state"],
                                "additions": pr["additions"],
                                "deletions": pr["deletions"],
                                "changed_files": pr["changed_files"],
                                "url": pr["html_url"],
                                "api_url": pr["url"],
                            }
                        )

                    return {
                        "success": True,
                        "pull_requests": pull_requests,
                        "total_count": len(pull_requests),
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
        """
        Get file content from repository

        Args:
            params: File parameters (owner, repo, path, ref)
        """
        try:
            owner = params.get("owner")
            repo = params.get("repo")
            path = params.get("path")
            ref = params.get("ref", "main")

            if not owner or not repo or not path:
                return {
                    "success": False,
                    "error": "Owner, repo, and path parameters required",
                }

            url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}"
            params_dict = {"ref": ref}

            async with self.session.get(url, params=params_dict) as response:
                if response.status == 200:
                    data = await response.json()

                    # Decode content if it's a file
                    content = None
                    if data.get("type") == "file":
                        content = base64.b64decode(data["content"]).decode(
                            "utf-8"
                        )

                    return {
                        "success": True,
                        "file_info": {
                            "name": data["name"],
                            "path": data["path"],
                            "type": data["type"],
                            "size": data["size"],
                            "sha": data["sha"],
                            "url": data["html_url"],
                            "download_url": data.get("download_url"),
                            "content": content,
                        },
                    }
                else:
                    return {
                        "success": False,
                        "error": f"File not found: {response.status}",
                        "status_code": response.status,
                    }

            if not owner or not repo:
                return {
                    "success": False,
                    "error": "Owner and repo parameters required",
                }

            url = f"{self.base_url}/repos/{owner}/{repo}/commits"
            params_dict = {"sha": sha, "per_page": per_page}

            if since:
                params_dict["since"] = since
            if until:
                params_dict["until"] = until

            async with self.session.get(url, params=params_dict) as response:
                if response.status == 200:
                    commits_data = await response.json()

                    commits = []
                    for commit in commits_data:
                        commits.append(
                            {
                                "sha": commit["sha"],
                                "message": commit["commit"]["message"],
                                "author": {
                                    "name": commit["commit"]["author"]["name"],
                                    "email": commit["commit"]["author"][
                                        "email"
                                    ],
                                    "date": commit["commit"]["author"]["date"],
                                },
                                "committer": {
                                    "name": commit["commit"]["committer"][
                                        "name"
                                    ],
                                    "email": commit["commit"]["committer"][
                                        "email"
                                    ],
                                    "date": commit["commit"]["committer"][
                                        "date"
                                    ],
                                },
                                "url": commit["html_url"],
                                "api_url": commit["url"],
                            }
                        )

                    return {
                        "success": True,
                        "commits": commits,
                        "total_count": len(commits),
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Failed to get commits: {response.status}",
                        "status_code": response.status,
                    }

            if not username:
                return {
                    "success": False,
                    "error": "Username parameter required",
                }

            url = f"{self.base_url}/users/{username}"

            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()

                    return {
                        "success": True,
                        "user": {
                            "login": data["login"],
                            "name": data["name"],
                            "email": data.get("email"),
                            "bio": data.get("bio"),
                            "location": data.get("location"),
                            "company": data.get("company"),
                            "blog": data.get("blog"),
                            "public_repos": data["public_repos"],
                            "public_gists": data["public_gists"],
                            "followers": data["followers"],
                            "following": data["following"],
                            "created_at": data["created_at"],
                            "updated_at": data["updated_at"],
                            "avatar_url": data["avatar_url"],
                            "url": data["html_url"],
                        },
                    }
                else:
                    return {
                        "success": False,
                        "error": f"User not found: {response.status}",
                        "status_code": response.status,
                    }

                }

            url = f"{self.base_url}/repos/{owner}/{repo}/issues"
            data = {
                "title": title,
                "body": body,
                "labels": labels,
                "assignees": assignees,
            }

            async with self.session.post(url, json=data) as response:
                if response.status == 201:
                    issue_data = await response.json()

                    return {
                        "success": True,
                        "issue": {
                            "number": issue_data["number"],
                            "title": issue_data["title"],
                            "body": issue_data["body"],
                            "state": issue_data["state"],
                            "labels": [
                                label["name"] for label in issue_data["labels"]
                            ],
                            "assignee": (
                                issue_data["assignee"]["login"]
                                if issue_data["assignee"]
                                else None
                            ),
                            "created_at": issue_data["created_at"],
                            "url": issue_data["html_url"],
                            "api_url": issue_data["url"],
                        },
                    }
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
                        "rate_limit": {
                            "limit": data["resources"]["core"]["limit"],
                            "remaining": data["resources"]["core"][
                                "remaining"
                            ],
                            "reset": data["resources"]["core"]["reset"],
                            "used": data["resources"]["core"]["used"],
                        },
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
    search_result = await github_connector.search_repositories(
        {
            "query": "model context protocol",
            "language": "python",
            "sort": "stars",
            "per_page": 5,
        }
    )

    if search_result["success"]:
        print(f"   - Found {search_result['total_count']} repositories")
        for repo in search_result["repositories"][:3]:
            print(f"   - {repo['full_name']}: {repo['stars']} stars")
    else:
        print(f"   - Error: {search_result.get('error')}")
    print()

    # Demo 2: Get repository info
    print("2. Getting repository information:")
    repo_result = await github_connector.get_repository(
        {"owner": "modelcontextprotocol", "repo": "specification"}
    )

    if repo_result["success"]:
        repo = repo_result["repository"]
        print(f"   - {repo['full_name']}")
        print(f"   - Language: {repo['language']}")
        print(f"   - Stars: {repo['stars']}")
        print(f"   - Open issues: {repo['open_issues']}")
    else:
        print(f"   - Error: {repo_result.get('error')}")
    print()

    # Demo 3: Get rate limit
    print("3. Rate limit information:")
    rate_limit = await github_connector.get_rate_limit()
    if rate_limit["success"]:
        rl = rate_limit["rate_limit"]
        print(f"   - Remaining requests: {rl['remaining']}")
        print(f"   - Used requests: {rl['used']}")
        print(f"   - Total limit: {rl['limit']}")
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
        print("Warning: GITHUB_TOKEN environment variable not set. " "Demo may fail.")
    asyncio.run(demonstrate_github_connector())
