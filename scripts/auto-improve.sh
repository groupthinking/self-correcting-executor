#!/bin/bash
set -euo pipefail

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
echo "⏳ Waiting for CodeRabbit analysis (30 seconds)..."
sleep 30

# 4. Pull any auto-applied changes
echo "⬇️ Pulling CodeRabbit improvements..."
git pull

echo "✅ CodeRabbit auto-improvement complete!"
echo "🔍 Check your GitHub PR for any remaining suggestions." 