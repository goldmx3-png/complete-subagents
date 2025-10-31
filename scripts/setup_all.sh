#!/bin/bash
# Setup script for complete-subagents project

set -e

echo "========================================="
echo "Complete Subagents - Setup Script"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate || . venv/Scripts/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Setup environment file
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your OPENROUTER_API_KEY"
else
    echo "✓ .env file already exists"
fi

# Start docker services
echo ""
echo "Starting Docker services (Qdrant + PostgreSQL)..."
docker-compose up -d
echo "✓ Docker services started"

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env and add your OPENROUTER_API_KEY"
echo "2. Activate virtual environment: source venv/bin/activate"
echo "3. Start API: python -m uvicorn src.api.routes:app --reload"
echo "4. Visit http://localhost:8000/docs"
echo ""
echo "Services running:"
echo "- Qdrant: http://localhost:6333"
echo "- PostgreSQL: localhost:5432"
echo ""
