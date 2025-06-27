#!/bin/bash
# Project Cleanup Script
# Generated from PROJECT_ANALYSIS_REPORT.md findings

echo "🧹 Starting project cleanup..."

# Backup important files first
echo "📦 Creating backup..."
mkdir -p .backup/$(date +%Y%m%d_%H%M%S)
cp -r . .backup/$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || true

# Remove Python cache files
echo "🗑️  Removing Python cache files..."
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true

# Remove empty files (excluding __init__.py for now)
echo "🗑️  Removing empty files..."
find . -type f -empty ! -name "__init__.py" -delete

# Remove specific identified empty/dead files
echo "🗑️  Removing identified dead files..."
rm -f frontend/frontend/src/components/Dashboard.tsx 2>/dev/null || true

# Clean up duplicate test files (keep only the most comprehensive ones)
echo "📋 Listing duplicate test files for review..."
cat << EOF > duplicate_test_files.txt
DUPLICATE TEST FILES FOUND:
- test_mcp_debug_quantum.py (uses mock-gcp-api)
- test_mcp_debug_simple.py (uses mock-gcp-api)
- test_mcp_ecosystem_expansion.py (contains simulated benchmarks)

Consider consolidating these into a single comprehensive test suite.
EOF

# Remove .DS_Store files (Mac)
find . -name ".DS_Store" -delete 2>/dev/null || true

# Clean up venv if it exists (should be in .gitignore)
if [ -d "venv" ]; then
    echo "🗑️  Removing venv directory (should be in .gitignore)..."
    rm -rf venv
fi

# List TODO/FIXME items for tracking
echo "📝 Generating TODO/FIXME report..."
grep -rn "TODO\|FIXME\|HACK\|XXX\|PLACEHOLDER" . \
    --include="*.py" \
    --include="*.yaml" \
    --include="*.yml" \
    --include="*.md" \
    --exclude-dir=".git" \
    --exclude-dir=".backup" \
    --exclude-dir="node_modules" > todo_report.txt || true

# List mock/fake implementations
echo "🎭 Generating mock/fake implementation report..."
grep -rn "mock\|fake\|simulated\|placeholder" . \
    --include="*.py" \
    --include="*.ts" \
    --include="*.tsx" \
    --exclude-dir=".git" \
    --exclude-dir=".backup" \
    --exclude-dir="node_modules" \
    -i > mock_implementations.txt || true

# Check for security issues
echo "🔒 Checking security configuration..."
if grep -q "enabled: false.*TODO" security_config.yaml 2>/dev/null; then
    echo "⚠️  WARNING: Security features are disabled in security_config.yaml"
    echo "   - Authentication is disabled"
    echo "   - Sandboxing is disabled"
    echo "   - Encryption at rest is disabled"
fi

# Create .gitignore if missing common entries
echo "📄 Updating .gitignore..."
cat << EOF >> .gitignore.tmp
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
.pytest_cache/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project specific
.backup/
todo_report.txt
mock_implementations.txt
duplicate_test_files.txt
EOF

# Merge with existing .gitignore
if [ -f .gitignore ]; then
    cat .gitignore >> .gitignore.tmp
    sort -u .gitignore.tmp > .gitignore
    rm .gitignore.tmp
else
    mv .gitignore.tmp .gitignore
fi

# Summary report
echo ""
echo "✅ Cleanup Summary:"
echo "==================="
echo "1. ✓ Removed Python cache files"
echo "2. ✓ Removed empty files (except __init__.py)"
echo "3. ✓ Identified duplicate test files (see duplicate_test_files.txt)"
echo "4. ✓ Generated TODO report (see todo_report.txt)"
echo "5. ✓ Generated mock implementation report (see mock_implementations.txt)"
echo "6. ✓ Updated .gitignore"
echo ""
echo "⚠️  Manual Actions Required:"
echo "============================"
echo "1. Review and consolidate duplicate test files"
echo "2. Enable security features in security_config.yaml"
echo "3. Replace mock implementations with real ones"
echo "4. Address TODO items in todo_report.txt"
echo "5. Consider removing FIND_IT/ directory if analysis is complete"
echo "6. Merge frontend/ and ui/ directories"
echo ""
echo "📊 File counts:"
echo "TODO/FIXME items: $(wc -l < todo_report.txt 2>/dev/null || echo 0)"
echo "Mock implementations: $(wc -l < mock_implementations.txt 2>/dev/null || echo 0)"

echo ""
echo "🎯 Next step: Run 'bash scripts/fix-security.sh' to enable security features"