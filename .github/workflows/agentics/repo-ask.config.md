# Repo Ask Configuration

This file configures the repo-ask workflow behavior. Edit this file and run `gh aw compile` to apply changes.

## Research Settings

```yaml
# Maximum depth for repository exploration
max_depth: 3

# File types to analyze
file_types:
  - "*.py"
  - "*.js"
  - "*.ts"
  - "*.md"
  - "*.yml"
  - "*.yaml"
  - "*.json"

# Directories to exclude from analysis
exclude_dirs:
  - node_modules
  - .git
  - __pycache__
  - dist
  - build
```

## Response Settings

```yaml
# Response format (markdown, plain)
format: markdown

# Include code snippets in responses
include_code: true

# Maximum response length (characters)
max_length: 4000
```

## Tool Permissions

```yaml
# Allowed bash commands
allowed_commands:
  - find
  - grep
  - ls
  - cat
  - head
  - tail
  - wc

# Allow web search
web_search: true

# Allow API calls to external services
external_apis: false
```

## Custom Instructions

Add any custom instructions for the AI agent below:

```
When answering questions:
1. Always provide code examples when relevant
2. Reference specific files in the repository
3. Include links to relevant documentation
4. Suggest follow-up actions when appropriate
```
