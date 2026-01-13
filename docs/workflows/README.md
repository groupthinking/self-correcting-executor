# Agentic Workflows

This directory contains documentation for agentic workflows that can be added to your repository. These workflows leverage AI agents to automate common development tasks.

## Available Workflows

| Workflow | Description | Trigger |
|----------|-------------|---------|
| [Repo Ask](./repo-ask.md) | Intelligent research assistant for repository questions | `/repo-ask` command |

## How Workflows Work

Agentic workflows are GitHub Actions workflows that are triggered by commands in issue or pull request comments. When you add a comment with a supported command (e.g., `/repo-ask`), the workflow is triggered and an AI agent processes your request.

## Installation

To install a workflow, use the `gh aw` extension:

```bash
# Install the gh aw extension
gh extension install githubnext/gh-aw

# Add a workflow to your repository
gh aw add githubnext/agentics/<workflow-name> --pr
```

This creates a pull request to add the workflow to your repository.

## Configuration

Each workflow can be customized via configuration files in `.github/workflows/agentics/`. After editing configuration files, run `gh aw compile` to update the workflow.

## Security Considerations

⚠️ **Important Security Notes:**

1. **Permissions**: Workflows run with the permissions specified in the workflow file. Review permissions carefully before enabling.

2. **Network Access**: Some workflows may have network access to perform web searches or API calls.

3. **Code Execution**: Workflows may execute bash commands within the GitHub Actions VM.

4. **Triggering**: Only repository admins, maintainers, or users with write permissions should trigger workflows.

## Adding New Workflows

To add documentation for a new workflow:

1. Create a new markdown file in this directory (e.g., `my-workflow.md`)
2. Follow the format of existing workflow documentation
3. Update this README to include the new workflow in the table

## Support

For issues with the `gh aw` extension, visit [githubnext/gh-aw](https://github.com/githubnext/gh-aw).
