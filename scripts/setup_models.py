#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.audio_diarizer.config import config
from src.audio_diarizer.diarization import SpeakerDiarizer
from src.audio_diarizer.transcription import SpeechTranscriber


def setup_models():
    """Download and initialize all required models"""
    print("Setting up audio diarization models...")
    
    # Setup Hugging Face token if provided
    if config.HUGGINGFACE_TOKEN:
        print("Using provided Hugging Face token")
    else:
        print("Warning: No Hugging Face token provided. Some models may not be accessible.")
    
    try:
        print(f"Loading speaker diarization model: {config.DIARIZATION_MODEL}")
        diarizer = SpeakerDiarizer()
        print("✓ Speaker diarization model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load speaker diarization model: {e}")
        return False
    
    try:
        print(f"Loading Whisper model: {config.WHISPER_MODEL}")
        transcriber = SpeechTranscriber()
        print("✓ Whisper model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load Whisper model: {e}")
        return False
    
    print("All models loaded successfully!")
    return True


if __name__ == '__main__':
    setup_models()