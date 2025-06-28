# Audio Diarization API

A high-performance audio diarization API server similar to AssemblyAI, built for GPU-accelerated processing with support for multiple speakers and various output formats.

## Features

- **Speaker Diarization**: Identify who spoke when using state-of-the-art models
- **Speech Transcription**: High-quality speech-to-text with Whisper
- **Multiple Output Formats**: JSON, SRT, VTT, and plain text
- **AI-Powered Analysis**: LLM-enhanced summaries, action items, and topic extraction (production only)
- **Async Processing**: Redis-based job queue for handling multiple requests
- **GPU Accelerated**: Optimized for RTX 4090 with 24GB VRAM
- **RESTful API**: FastAPI-based with automatic documentation
- **Ollama Integration**: Connect to Ollama server for post-processing analysis

## Architecture

The system uses a separate model approach:
- **Diarization**: pyannote.audio for speaker segmentation
- **Transcription**: faster-whisper for speech-to-text
- **LLM Analysis**: Ollama integration for summaries and insights (production)
- **Processing**: Async job queue with Redis
- **API**: FastAPI with automatic OpenAPI docs

## Requirements

### Production Environment (Ubuntu Server)
- **Hardware**: RTX 4090 GPU, 32GB+ RAM
- **OS**: Ubuntu 20.04+ 
- **Python**: 3.11+
- **CUDA**: 11.8+ with PyTorch CUDA support
- **Redis**: Server for job queue
- **FFmpeg**: Audio processing
- **Ollama**: LLM server with llama3.2 model

### Development Environment (Apple Silicon Mac)
- **Hardware**: Apple Silicon Mac
- **OS**: macOS with Apple Silicon
- **Python**: 3.11 (required for ARM64 wheels)
- **Redis**: via Homebrew
- **FFmpeg**: via Homebrew
- **Ollama**: Access to Ollama server

## Installation

### Development Setup (Apple Silicon Mac)

1. **Install system dependencies**:
```bash
brew install ffmpeg redis
```

2. **Clone and install Python dependencies**:
```bash
git clone <repository>
cd audio-diarizer
uv sync  # Uses Python 3.11 for ARM64 compatibility
```

3. **Configure for development**:
```bash
cp .env.example .env
# Edit .env - development settings are already uncommented
```

4. **Start Redis**:
```bash
brew services start redis
```

5. **Start API server** (choose one):
```bash
# Option A: Use convenient startup script
./scripts/start_dev.sh

# Option B: Manual start with environment
source .env && uv run main.py
```

6. **Test setup**:
```bash
uv run scripts/setup_cpu.py
curl http://localhost:8000/ollama/status  # Test Ollama connection
```

### Production Setup (Ubuntu + RTX 4090)

1. **Install system dependencies**:
```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip
sudo apt install -y ffmpeg redis-server
sudo apt install -y nvidia-driver-535 nvidia-cuda-toolkit  # For RTX 4090
```

2. **Install uv and Python dependencies**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repository>
cd audio-diarizer
uv sync
```

3. **Configure for production**:
```bash
cp .env.production .env
# Edit .env with your HuggingFace token
```

4. **Start Redis**:
```bash
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

5. **Verify GPU setup**:
```bash
nvidia-smi  # Should show RTX 4090
uv run scripts/setup_models.py
```

## Usage

### Start the API Server

```bash
uv run main.py
```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

### Start the Worker Process

In a separate terminal:
```bash
uv run scripts/start_worker.py
```

### API Endpoints

#### Submit Audio for Processing
```bash
POST /transcribe
```

**Parameters**:
- `file`: Audio file (required)
- `expected_speakers`: Number of expected speakers (optional, 2-10)
- `response_format`: Output format - `json`, `srt`, `vtt`, or `text` (default: json)
- `enable_llm_analysis`: Enable AI analysis via Ollama (default: false, production only)

**Example**:
```bash
curl -X POST "http://localhost:8000/transcribe" \
     -F "file=@meeting.wav" \
     -F "expected_speakers=3" \
     -F "response_format=json" \
     -F "enable_llm_analysis=true"
```

**Response**:
```json
{
  "job_id": "uuid-string",
  "status": "pending",
  "created_at": "2024-01-01T00:00:00"
}
```

#### Check Job Status
```bash
GET /transcribe/{job_id}
```

**Example**:
```bash
curl "http://localhost:8000/transcribe/your-job-id"
```

**Response** (completed with LLM analysis):
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "created_at": "2024-01-01T00:00:00",
  "completed_at": "2024-01-01T00:05:00",
  "result": {
    "utterances": [
      {
        "speaker": "Speaker A",
        "start": 0.0,
        "end": 3.5,
        "text": "Hello, welcome to our meeting.",
        "confidence": 0.95
      }
    ],
    "audio_duration": 120.5,
    "speakers_detected": 2,
    "llm_enhancements": {
      "summary": "Brief meeting to discuss project timeline and deliverables...",
      "action_items": "• Review budget by Friday\n• Schedule follow-up with client\n• Prepare presentation slides",
      "topics": "• Project timeline\n• Budget review\n• Client communication\n• Next steps"
    }
  }
}
```

#### Delete Job
```bash
DELETE /transcribe/{job_id}
```

### Output Formats

#### JSON (default)
Structured data with speaker labels, timestamps, and confidence scores.

#### SRT Subtitles
```
1
00:00:00,000 --> 00:00:03,500
[Speaker A] Hello, welcome to our meeting.

2
00:00:03,500 --> 00:00:07,200
[Speaker B] Thank you for having me.
```

#### VTT Captions
```
WEBVTT

00:00:00.000 --> 00:00:03.500
[Speaker A] Hello, welcome to our meeting.

00:00:03.500 --> 00:00:07.200
[Speaker B] Thank you for having me.
```

#### Plain Text
```
[00:00:00,000] Speaker A: Hello, welcome to our meeting.
[00:00:03,500] Speaker B: Thank you for having me.
```

## Performance

### Production (Ubuntu + RTX 4090)
- **60-minute audio**: ~20-30 minutes
- **10-minute audio**: ~3-5 minutes  
- **Memory usage**: ~12GB VRAM, ~8GB RAM
- **Concurrent jobs**: 1-2 (limited by GPU memory)

### Development (Apple Silicon Mac)
- **60-minute audio**: ~2-3 hours (CPU only)
- **10-minute audio**: ~20-30 minutes
- **Memory usage**: ~8GB RAM
- **Concurrent jobs**: 1 (CPU intensive)

### Performance Tips
- **Production**: Monitor GPU memory with `nvidia-smi`
- **Development**: Test with short audio files (<5 min) first
- **Both**: Provide `expected_speakers` parameter for better accuracy
- **Both**: Use appropriate models for your environment

## Configuration

### Environment Variables

| Variable | Development (Mac) | Production (Ubuntu) | Description |
|----------|----------------------|-------------------|-------------|
| `DEPLOYMENT_TARGET` | development | production | Target environment |
| `DEVICE` | cpu | cuda | Processing device |
| `TORCH_DTYPE` | float32 | float16 | Model precision |
| `WHISPER_MODEL` | medium | large-v3 | Whisper model size |
| `MAX_FILE_SIZE` | 50MB | 100MB | Max upload size |
| `MAX_AUDIO_DURATION` | 1800s (30min) | 7200s (2hr) | Max audio length |
| `REDIS_HOST` | localhost | localhost | Redis server host |
| `REDIS_PORT` | 6379 | 6379 | Redis server port |
| `HUGGINGFACE_TOKEN` | required | required | HF token for pyannote |
| `OLLAMA_ENABLED` | true | true | Enable Ollama integration |
| `OLLAMA_HOST` | http://localhost:11434 | http://localhost:11434 | Ollama server URL |
| `OLLAMA_MODEL` | llama3.2 | llama3.2 | Ollama model name |

### Model Selection by Environment

**Development (CPU):**
- Whisper: `medium` (faster inference)
- Shorter processing limits for quick testing
- Ollama integration available

**Production (GPU):**
- Whisper: `large-v3` (best accuracy)
- Higher limits for production workloads
- Ollama integration available

## LLM Integration (Development & Production)

Both environments include optional AI-powered post-processing via Ollama:

### Features
- **Meeting Summaries**: Concise overview of key discussion points
- **Action Items**: Extracted tasks and next steps
- **Topic Analysis**: Main themes and subjects discussed

### Configuration
```bash
# In .env
OLLAMA_ENABLED=true
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

### Usage
```bash
# Enable LLM analysis in API request
curl -X POST "http://localhost:8000/transcribe" \
     -F "file=@meeting.wav" \
     -F "enable_llm_analysis=true"
```

### Check Ollama Status
```bash
curl "http://localhost:8000/ollama/status"
```

**Note**: LLM analysis requires Ollama server to be accessible (configure OLLAMA_HOST in .env file).

## Development

### Install dev dependencies:
```bash
uv sync --group dev
```

### Code formatting:
```bash
uv run black src/
uv run isort src/
```

### Type checking:
```bash
uv run mypy src/
```

## Production Deployment (Ubuntu + RTX 4090)

### Automated Deployment

```bash
# Run the automated deployment script
./scripts/deploy_production.sh
```

### Manual Production Setup

1. **Install as systemd services**:
```bash
cd deployment
sudo ./install_services.sh
```

2. **Start services**:
```bash
sudo systemctl start audio-diarizer-api
sudo systemctl start audio-diarizer-worker
```

3. **Monitor services**:
```bash
# Check status
sudo systemctl status audio-diarizer-api
sudo systemctl status audio-diarizer-worker

# View logs
sudo journalctl -u audio-diarizer-api -f
sudo journalctl -u audio-diarizer-worker -f

# Monitor GPU usage
watch nvidia-smi
```

### Production Configuration

The production setup includes:
- **systemd services** for automatic startup/restart
- **GPU optimization** (CUDA, float16 precision)
- **Higher resource limits** (2-hour audio, 100MB files)
- **Performance monitoring** via systemd journals
- **Automatic environment detection**

## Troubleshooting

### Development Issues (Apple Silicon Mac)

1. **sentencepiece build fails**: Use Python 3.11 (`uv python install 3.11`)
2. **ARM64 wheel not found**: Check that you're using Python 3.11
3. **Redis connection failed**: `brew services start redis`
4. **Slow processing**: Expected on CPU, use shorter audio files

### Production Issues (Ubuntu)

1. **CUDA out of memory**: Monitor with `nvidia-smi`, reduce concurrent jobs
2. **pyannote model access denied**: Set `HUGGINGFACE_TOKEN` in `.env`
3. **FFmpeg not found**: `sudo apt install ffmpeg`
4. **Worker crashes**: Check GPU memory and driver compatibility

### Common Issues (Both)

1. **Models not downloading**: Check internet connection and HF token
2. **API connection timeout**: Ensure API server is running and accessible
3. **Job stuck in pending**: Check that worker process is running

### Logs

- **Development**: Logs printed to stdout/stderr
- **Production**: Use `journalctl -u audio-diarizer-api -f` and `journalctl -u audio-diarizer-worker -f`

## License

[Add your license here]

## Contributing

[Add contributing guidelines here]