#!/bin/bash
# Development startup script that loads environment variables

set -e

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp .env.example .env
    echo "📝 Please edit .env and set your HUGGINGFACE_TOKEN if needed"
fi

# Load environment variables
echo "🔧 Loading environment variables..."
set -a  # Automatically export all variables
source .env
set +a

# Check Redis
if ! redis-cli ping &>/dev/null; then
    echo "❌ Redis not running. Starting Redis..."
    brew services start redis
    sleep 2
fi

# Check Ollama connectivity
echo "🤖 Testing Ollama connectivity..."
if curl -s "$OLLAMA_HOST/api/tags" &>/dev/null; then
    echo "✅ Ollama server accessible at $OLLAMA_HOST"
else
    echo "⚠️  Ollama server not accessible at $OLLAMA_HOST"
    echo "   LLM analysis will be disabled"
fi

echo "🚀 Starting Audio Diarization API server..."
echo "📍 Server will be available at http://localhost:8000"
echo "📖 API docs available at http://localhost:8000/docs"
echo ""

# Start the server
uv run main.py