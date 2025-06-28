import os
from typing import Optional
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    # Deployment target detection
    DEPLOYMENT_TARGET: str = os.getenv("DEPLOYMENT_TARGET", "auto")
    
    # Redis configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    
    # HuggingFace authentication
    HUGGINGFACE_TOKEN: Optional[str] = os.getenv("HUGGINGFACE_TOKEN")
    
    # File handling
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    
    # Model configurations
    DIARIZATION_MODEL: str = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
    
    # Ollama configuration
    OLLAMA_ENABLED: bool = os.getenv("OLLAMA_ENABLED", "false").lower() == "true"
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    
    # Processing settings
    SAMPLE_RATE: int = 16000
    
    def __init__(self):
        # Auto-detect environment if not specified
        if self.DEPLOYMENT_TARGET == "auto":
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                # RTX 4090 has compute capability 8.9
                self.DEPLOYMENT_TARGET = "production"
            else:
                self.DEPLOYMENT_TARGET = "development"
        
        # Environment-specific defaults
        if self.DEPLOYMENT_TARGET == "production":
            # Ubuntu + RTX 4090 defaults
            self.DEVICE = os.getenv("DEVICE", "cuda")
            self.TORCH_DTYPE = os.getenv("TORCH_DTYPE", "float16")
            self.WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
            self.MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "100000000"))  # 100MB
            self.MAX_AUDIO_DURATION = int(os.getenv("MAX_AUDIO_DURATION", "7200"))  # 2 hours
        else:
            # MacBook Pro M3 MAX defaults
            self.DEVICE = os.getenv("DEVICE", "cpu")
            self.TORCH_DTYPE = os.getenv("TORCH_DTYPE", "float32")
            self.WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
            self.MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "50000000"))  # 50MB
            self.MAX_AUDIO_DURATION = int(os.getenv("MAX_AUDIO_DURATION", "1800"))  # 30 minutes
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.DEPLOYMENT_TARGET == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.DEPLOYMENT_TARGET == "development"


config = Config()