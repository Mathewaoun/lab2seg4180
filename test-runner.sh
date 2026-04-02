#!/bin/bash
# test-runner.sh - Simple test execution script

set -e  # Exit on first error

echo "================================"
echo "🧪 House Segmentation Service Test Runner"
echo "================================"
echo ""

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Run: python3 -m venv .venv"
    exit 1
fi

# Activate venv
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -q -r requirements.txt

# Run tests
echo "🧪 Running tests..."
echo ""
python -m pytest tests/ -v --tb=short

echo ""
echo "================================"
echo "✅ All tests completed!"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Review test results above"
echo "2. Start API: python app.py"
echo "3. Test endpoint: curl http://localhost:5001/health"
echo ""
