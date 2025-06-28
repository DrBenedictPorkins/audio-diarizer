#!/bin/bash
# Production deployment script for Ubuntu + RTX 4090

set -e

echo "🚀 Audio Diarization API - Production Deployment"
echo "================================================"

# Check if we're on Ubuntu
if ! grep -q "Ubuntu" /etc/os-release 2>/dev/null; then
    echo "❌ This script is designed for Ubuntu. Detected OS:"
    cat /etc/os-release | grep PRETTY_NAME || echo "Unknown OS"
    exit 1
fi

echo "✅ Ubuntu detected"

# Check for NVIDIA GPU
if ! nvidia-smi &>/dev/null; then
    echo "❌ NVIDIA GPU not detected or drivers not installed"
    echo "Please install NVIDIA drivers first:"
    echo "sudo apt install nvidia-driver-535 nvidia-cuda-toolkit"
    exit 1
fi

echo "✅ NVIDIA GPU detected:"
nvidia-smi --query-gpu=name --format=csv,noheader,nounits

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip
sudo apt install -y ffmpeg redis-server
sudo apt install -y build-essential cmake pkg-config

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
uv sync

# Configure environment
echo "⚙️  Configuring production environment..."
if [ ! -f .env ]; then
    cp .env.production .env
    echo "🔧 Created .env from production template"
    echo "📝 Please edit .env and set your HUGGINGFACE_TOKEN"
else
    echo "⚠️  .env already exists, skipping copy"
fi

# Start and enable Redis
echo "🗄️  Configuring Redis..."
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Create upload directory
mkdir -p uploads
chmod 755 uploads

# Test GPU setup
echo "🧪 Testing GPU setup..."
uv run -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Download models (this will take a while)
echo "🤖 Setting up models (this may take several minutes)..."
uv run scripts/setup_models.py

echo "✅ Production deployment complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and set your HUGGINGFACE_TOKEN"
echo "2. Ensure Ollama is running at http://beefybox.lan:11434"
echo "3. Start the API server: uv run main.py"
echo "4. Start the worker: uv run scripts/start_worker.py"
echo "5. Test: curl http://localhost:8000/health"
echo "6. Test Ollama: curl http://localhost:8000/ollama/status"
echo ""
echo "🔍 Monitor with:"
echo "- nvidia-smi  # GPU usage"
echo "- redis-cli ping  # Redis status"
echo "- systemctl status redis-server  # Redis service"
echo "- curl http://beefybox.lan:11434/api/tags  # Ollama models"