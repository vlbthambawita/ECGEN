#!/bin/bash
# Quick script to serve documentation locally

echo "🚀 Starting ECGEN documentation server..."
echo ""

# Set PYTHONPATH to include src directory
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# Check if ECGEN can be imported
if ! python -c "import ecgen" &> /dev/null; then
    echo "❌ Cannot import ecgen module"
    echo "Make sure you're in the ECGEN directory"
    exit 1
fi

# Check if mkdocs is installed
if ! command -v mkdocs &> /dev/null; then
    echo "⚠️  MkDocs not found. Please install:"
    echo "   pip install mkdocs mkdocs-material 'mkdocstrings[python]'"
    exit 1
fi

echo "✓ ECGEN module found"
echo "📚 Serving documentation at http://127.0.0.1:8000"
echo "Press Ctrl+C to stop"
echo ""

mkdocs serve
