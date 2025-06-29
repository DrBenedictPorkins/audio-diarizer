#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rq import SimpleWorker
import redis
from src.audio_diarizer.config import config

def main():
    # Debug info
    from pathlib import Path
    print(f"Worker starting from directory: {os.getcwd()}")
    print(f"src/ directory exists: {Path('src').exists()}")
    print(f"uploads/ directory exists: {Path('uploads').exists()}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
    print(f"CUDNN_PATH: {os.environ.get('CUDNN_PATH', 'Not set')}")
    
    redis_client = redis.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        db=config.REDIS_DB
    )
    
    worker = SimpleWorker(['default'], connection=redis_client)
    print(f"Starting SimpleWorker (no subprocess forking) on {config.REDIS_HOST}:{config.REDIS_PORT}")
    worker.work()

if __name__ == '__main__':
    main()