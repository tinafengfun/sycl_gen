#!/bin/bash
# Pre-submission Checklist for GitHub
# GitHub 提交前检查清单

echo "=========================================="
echo "🚀 GitHub Submission Checklist"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_pass() {
    echo -e "${GREEN}✅${NC} $1"
}

check_fail() {
    echo -e "${RED}❌${NC} $1"
}

check_warn() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

echo "📋 Checking project structure..."
echo ""

# Check directories
dirs=("src" "src/harnesses" "src/core" "tests" "scripts" "docs" "docs/examples" "reports" "logs")
for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        check_pass "Directory: $dir/"
    else
        check_fail "Missing directory: $dir/"
    fi
done

echo ""
echo "📄 Checking essential files..."
echo ""

# Check essential files
files=("pyproject.toml" "requirements.txt" "LICENSE" ".gitignore" "README.md")
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        check_pass "File: $file"
    else
        check_fail "Missing file: $file"
    fi
done

echo ""
echo "🐍 Checking Python files..."
echo ""

# Check Python files
py_files=("src/__init__.py" "src/harnesses/__init__.py" "src/harnesses/all_harnesses.py" "src/harnesses/batch4_harnesses.py")
for file in "${py_files[@]}"; do
    if [ -f "$file" ]; then
        check_pass "Python: $file"
    else
        check_fail "Missing: $file"
    fi
done

echo ""
echo "📊 Statistics:"
echo ""

# Count files
total_files=$(find . -type f -not -path "./__pycache__/*" -not -path "./.git/*" | wc -l)
py_files=$(find . -name "*.py" -not -path "./__pycache__/*" | wc -l)
md_files=$(find . -name "*.md" | wc -l)

echo "  Total files: $total_files"
echo "  Python files: $py_files"
echo "  Markdown files: $md_files"

echo ""
echo "📦 Project size:"
du -sh . 2>/dev/null | awk '{print "  " $1}'

echo ""
echo "=========================================="
echo "✨ Ready for GitHub submission!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. git init"
echo "  2. git add ."
echo "  3. git commit -m 'Initial commit: v0.5.0'"
echo "  4. git remote add origin https://github.com/YOUR_USERNAME/cuda-sycl-harnesses.git"
echo "  5. git push -u origin main"
echo ""
