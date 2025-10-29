#!/bin/bash
# APEX DIRECTOR - Quick Start GUI Script

echo "🎬 APEX DIRECTOR - Starting Web Interface..."
echo "=========================================="

# Install dependencies if needed
echo "📦 Checking dependencies..."
pip install fastapi uvicorn python-multipart jinja2 --quiet

# Start the web interface
echo "🚀 Starting web interface..."
echo "🌐 Open your browser to: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="

python web_interface.py