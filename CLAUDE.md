# Claude Code Memory - Audio Diarization Project

## Project Overview
Audio diarization API with speaker separation and LLM analysis. Built for both development (Apple Silicon Mac) and production (Ubuntu RTX 4090) environments.

## Key Technical Details

### Architecture
- **API Server**: FastAPI with async job processing
- **Worker Process**: RQ SimpleWorker (no subprocess forking to avoid audio library hangs)
- **Queue**: Redis-based job queue
- **Models**: 
  - Diarization: pyannote.audio (speaker-diarization-3.1)
  - Transcription: faster-whisper (medium/large-v3)
  - LLM: Ollama integration (llama3.2)

### Environment Configuration
- Development: Apple Silicon Mac, CPU processing, medium models
- Production: Ubuntu RTX 4090, GPU acceleration, large-v3 models
- Auto-detection via `DEPLOYMENT_TARGET` in .env

### Build System
- **Package Manager**: uv (modern Python package manager)
- **Python Version**: 3.11 (required for Apple Silicon ARM64 wheels)
- **Dependencies**: Managed via pyproject.toml

## Important Troubleshooting

### cuDNN Library Issues on Ubuntu RTX 4090
**Problem**: Worker fails during transcription with:
```
Unable to load any of {libcudnn_cnn.so.9.1.0, libcudnn_cnn.so.9.1, libcudnn_cnn.so.9, libcudnn_cnn.so}
Invalid handle. Cannot load symbol cudnnCreateConvolutionDescriptor
```

**Root Cause**: faster-whisper expects cuDNN 9.1.x but system has cuDNN 8.x loaded by default in linker cache.

**Solution**: Use startup script that sets environment variables:
```bash
./scripts/start_worker_with_cudnn.sh
```

This script sets:
- `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH`
- `CUDNN_PATH=/usr/lib/x86_64-linux-gnu`

**Files Modified**:
- `scripts/start_worker_with_cudnn.sh` - New startup script with cuDNN environment
- `scripts/start_worker.py` - Added debug prints for environment verification
- `.env` - Added cuDNN path documentation

### Worker Configuration
- Uses `SimpleWorker` instead of `Worker` to avoid subprocess forking
- Audio processing libraries can hang in forked processes
- Single-threaded processing but more stable

### GPU Memory Management
- RTX 4090: 24GB VRAM, typically uses ~12GB for large-v3 model
- Monitor with `nvidia-smi`
- Limit concurrent jobs to 1-2 based on available memory

### Model Downloads
- pyannote models require HuggingFace token and terms acceptance
- Models cached in ~/.cache/huggingface/
- faster-whisper models cached in ~/.cache/whisper/

## Development Commands

### Start Development Environment
```bash
# API server
uv run main.py

# Worker (choose based on system)
./scripts/start_worker_with_cudnn.sh  # Ubuntu RTX 4090
uv run scripts/start_worker.py        # Other systems

# Test setup
uv run scripts/setup_cpu.py           # Development
uv run scripts/setup_models.py        # Production
```

### Code Quality
```bash
uv run black src/
uv run isort src/
uv run mypy src/
```

### Testing
```bash
# CLI testing
uv run src/audio_diarizer/cli.py test_audio.wav --format text

# API testing
curl -X POST "http://localhost:8000/transcribe" -F "file=@test.wav"
```

## File Structure
```
src/audio_diarizer/
├── main.py           # FastAPI application
├── worker.py         # RQ worker job handlers
├── config.py         # Environment-based configuration
├── diarization.py    # Speaker diarization (pyannote)
├── transcription.py  # Speech transcription (faster-whisper)
├── audio_processor.py # Audio preprocessing
├── formatters.py     # Output format conversion
├── ollama_client.py  # LLM integration
└── cli.py           # Command-line interface

scripts/
├── start_worker_with_cudnn.sh  # Ubuntu RTX 4090 worker startup
├── start_worker.py             # Standard worker startup
├── setup_models.py             # Production model setup
└── setup_cpu.py               # Development setup

deployment/
├── audio-diarizer-api.service    # systemd service for API
├── audio-diarizer-worker.service # systemd service for worker
└── install_services.sh          # Service installation script
```

## Performance Expectations
- **Production (RTX 4090)**: 60min audio → 20-30min processing
- **Development (Apple Silicon)**: 60min audio → 2-3hr processing
- **Memory**: 12GB VRAM (GPU) / 8GB RAM (CPU)

## Common Debugging Steps
1. Check GPU: `nvidia-smi`
2. Check Redis: `redis-cli ping`
3. Check models: `ls ~/.cache/huggingface/hub/`
4. Check environment: Worker startup shows LD_LIBRARY_PATH
5. Check logs: API stdout, worker stdout, or systemd journals