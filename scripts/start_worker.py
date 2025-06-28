#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rq import Worker
import redis
from src.audio_diarizer.config import config

def main():
    redis_client = redis.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        db=config.REDIS_DB
    )
    
    worker = Worker(['default'], connection=redis_client)
    print(f"Starting RQ worker on {config.REDIS_HOST}:{config.REDIS_PORT}")
    worker.work()

if __name__ == '__main__':
    main()