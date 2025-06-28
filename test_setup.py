#!/usr/bin/env python3

import sys
import requests
import time
from pathlib import Path

def test_api_health():
    """Test if the API server is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✓ API server is running")
            return True
        else:
            print(f"✗ API server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API server (is it running on port 8000?)")
        return False
    except Exception as e:
        print(f"✗ Error testing API: {e}")
        return False

def test_redis_connection():
    """Test Redis connection"""
    try:
        import redis
        from src.audio_diarizer.config import config
        
        client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB
        )
        client.ping()
        print("✓ Redis connection successful")
        return True
    except Exception as e:
        print(f"✗ Redis connection failed: {e}")
        return False

def test_dependencies():
    """Test if all dependencies are installed"""
    required_modules = [
        'torch',
        'torchaudio', 
        'faster_whisper',
        'pyannote.audio',
        'librosa',
        'ffmpeg',
        'fastapi',
        'redis',
        'rq'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module} (missing)")
            missing.append(module)
    
    return len(missing) == 0

def main():
    print("Testing Audio Diarization API Setup")
    print("=" * 40)
    
    print("\n1. Testing dependencies...")
    deps_ok = test_dependencies()
    
    print("\n2. Testing Redis connection...")
    redis_ok = test_redis_connection()
    
    print("\n3. Testing API server...")
    api_ok = test_api_health()
    
    print("\n" + "=" * 40)
    if deps_ok and redis_ok and api_ok:
        print("✓ All tests passed! Setup is ready.")
        
        print("\nNext steps:")
        print("1. Start the worker: uv run scripts/start_worker.py")
        print("2. Test with audio file: curl -X POST http://localhost:8000/transcribe -F 'file=@your_audio.wav'")
        print("3. View API docs: http://localhost:8000/docs")
        return 0
    else:
        print("✗ Some tests failed. Please check the setup.")
        if not deps_ok:
            print("  - Install missing dependencies: uv sync")
        if not redis_ok:
            print("  - Start Redis server: redis-server")
        if not api_ok:
            print("  - Start API server: uv run main.py")
        return 1

if __name__ == '__main__':
    sys.exit(main())