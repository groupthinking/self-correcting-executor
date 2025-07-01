#!/bin/bash
set -euo pipefail

# Configuration
WAIT_DURATION=${CODERABBIT_WAIT_DURATION:-30}
MAX_WAIT_TIME=${CODERABBIT_MAX_WAIT:-300}
POLL_INTERVAL=${CODERABBIT_POLL_INTERVAL:-10}

# check_for_new_commits determines if the remote branch has new commits compared to the local HEAD.
check_for_new_commits() {
    local initial_commit=$(git rev-parse HEAD)
    git fetch origin >/dev/null 2>&1
    local remote_commit=$(git rev-parse origin/$(git branch --show-current))
    
    if [ "$initial_commit" != "$remote_commit" ]; then
        return 0  # New commits found
    else
        return 1  # No new commits
    fi
}

# wait_for_coderabbit polls the remote repository to detect new commits from CodeRabbit, waiting up to a maximum duration before proceeding.
wait_for_coderabbit() {
    local start_time=$(date +%s)
    local initial_commit=$(git rev-parse HEAD)
    
    echo "⏳ Waiting for CodeRabbit analysis (polling every ${POLL_INTERVAL}s, max ${MAX_WAIT_TIME}s)..."
    
    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [ $elapsed -ge $MAX_WAIT_TIME ]; then
            echo "⚠️  Timeout reached (${MAX_WAIT_TIME}s). Proceeding with fallback wait..."
            sleep $WAIT_DURATION
            break
        fi
        
        if check_for_new_commits; then
            echo "✅ New commits detected from CodeRabbit!"
            break
        fi
        
        echo "🔍 No new commits yet... waiting ${POLL_INTERVAL}s (${elapsed}s elapsed)"
        sleep $POLL_INTERVAL
    done
}

# Auto-improve with CodeRabbit - Bulk Apply All
echo "🚀 Starting CodeRabbit auto-improvement..."

# 1. Generate improvements for all files
echo "📝 Generating improvements..."
git add -A
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to add files to Git. Exiting..."
    exit 1
fi
if ! git diff --cached --quiet; then
  git commit -m "feat: prepare for CodeRabbit auto-improvements"
fi
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to commit changes. Exiting..."
    exit 1
fi

# 2. Push to trigger CodeRabbit review
echo "⬆️ Pushing to GitHub for CodeRabbit analysis..."
git push
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to push changes to GitHub. Exiting..."
    exit 1
fi

# 3. Wait for CodeRabbit to process
wait_for_coderabbit

# 4. Pull any auto-applied changes
echo "⬇️ Pulling CodeRabbit improvements..."
git pull

echo "✅ CodeRabbit auto-improvement complete!"
echo "🔍 Check your GitHub PR for any remaining suggestions." 