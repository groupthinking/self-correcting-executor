#!/bin/bash

# Auto-improve with CodeRabbit - Bulk Apply All
echo "🚀 Starting CodeRabbit auto-improvement..."

# 1. Generate improvements for all files
echo "📝 Generating improvements..."
git add -A
git commit -m "feat: prepare for CodeRabbit auto-improvements"

# 2. Push to trigger CodeRabbit review
echo "⬆️ Pushing to GitHub for CodeRabbit analysis..."
git push

# 3. Wait for CodeRabbit to process
echo "⏳ Waiting for CodeRabbit analysis (30 seconds)..."
sleep 30

# 4. Pull any auto-applied changes
echo "⬇️ Pulling CodeRabbit improvements..."
git pull

echo "✅ CodeRabbit auto-improvement complete!"
echo "🔍 Check your GitHub PR for any remaining suggestions." 