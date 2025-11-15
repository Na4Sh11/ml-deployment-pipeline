#!/bin/bash

echo "�� Setting up ML Deployment Pipeline..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create __init__.py files
echo "Creating __init__.py files..."
find src -type d -exec touch {}/__init__.py \;
find tests -type d -exec touch {}/__init__.py \;

# Create logs directory
echo "Creating logs directory..."
mkdir -p logs

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To verify installation, run:"
echo "  python -c \"import fastapi, sklearn, pandas; print('✅ All packages working!')\""
