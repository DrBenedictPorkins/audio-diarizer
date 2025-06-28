#!/usr/bin/env python3
"""CPU-optimized setup for MacBook"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def optimize_for_cpu():
    """Set CPU-optimized environment variables"""
    os.environ.update({
        'DEVICE': 'cpu',
        'TORCH_DTYPE': 'float32',
        'WHISPER_MODEL': 'medium',  # Faster on CPU
        'MAX_AUDIO_DURATION': '1800',  # 30 min limit
        'OMP_NUM_THREADS': '8',  # Optimize for M3 MAX cores
        'MKL_NUM_THREADS': '8',
    })
    
    print("CPU optimization settings applied:")
    for key, value in os.environ.items():
        if key in ['DEVICE', 'TORCH_DTYPE', 'WHISPER_MODEL', 'OMP_NUM_THREADS']:
            print(f"  {key}={value}")

def test_models():
    """Test model loading on CPU"""
    print("\nTesting model loading on CPU...")
    
    try:
        from src.audio_diarizer.diarization import SpeakerDiarizer
        print("✓ Loading diarization model...")
        diarizer = SpeakerDiarizer()
        print("✓ Diarization model loaded successfully")
    except Exception as e:
        print(f"✗ Diarization model failed: {e}")
        return False
    
    try:
        from src.audio_diarizer.transcription import SpeechTranscriber
        print("✓ Loading Whisper model...")
        transcriber = SpeechTranscriber()
        print("✓ Whisper model loaded successfully")
    except Exception as e:
        print(f"✗ Whisper model failed: {e}")
        return False
    
    return True

if __name__ == '__main__':
    optimize_for_cpu()
    success = test_models()
    
    if success:
        print("\n✓ CPU setup successful! Ready for MacBook processing.")
        print("\nRecommended workflow:")
        print("1. Test with short audio files (<5 min) first")
        print("2. Use 'medium' Whisper model for speed")
        print("3. Expect ~4x slower processing than GPU")
    else:
        print("\n✗ Setup failed. Check dependencies.")
        sys.exit(1)