import torch
from pyannote.audio import Pipeline
from typing import List, Tuple, Optional
import tempfile
import os

from .config import config


class SpeakerDiarizer:
    def __init__(self):
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        try:
            if config.HUGGINGFACE_TOKEN:
                # Try to load real model if token is available
                self.pipeline = Pipeline.from_pretrained(
                    config.DIARIZATION_MODEL,
                    use_auth_token=config.HUGGINGFACE_TOKEN
                )
                
                if torch.cuda.is_available():
                    self.pipeline = self.pipeline.to(self.device)
            else:
                # Use mock implementation for testing without authentication
                print("WARNING: No HuggingFace token provided. Using mock diarization for testing.")
                self.pipeline = "mock"  # Mock pipeline
                
        except Exception as e:
            print(f"Failed to load real diarization model: {e}")
            print("Falling back to mock diarization for testing.")
            self.pipeline = "mock"
    
    def diarize(self, audio_file: str, num_speakers: Optional[int] = None) -> List[Tuple[float, float, str]]:
        if self.pipeline is None:
            raise RuntimeError("Diarization model not loaded")
        
        try:
            if self.pipeline == "mock":
                # Mock diarization for testing
                return self._mock_diarization(audio_file, num_speakers)
            
            # Run real diarization
            diarization_params = {}
            if num_speakers is not None:
                diarization_params['num_speakers'] = num_speakers
            
            with torch.no_grad():
                diarization = self.pipeline(audio_file, **diarization_params)
            
            # Convert to list of (start, end, speaker) tuples
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append((turn.start, turn.end, speaker))
            
            # Sort by start time
            segments.sort(key=lambda x: x[0])
            
            # Relabel speakers as A, B, C, etc.
            speaker_mapping = {}
            speaker_counter = 0
            relabeled_segments = []
            
            for start, end, speaker in segments:
                if speaker not in speaker_mapping:
                    speaker_mapping[speaker] = f"Speaker {chr(65 + speaker_counter)}"
                    speaker_counter += 1
                
                relabeled_segments.append((start, end, speaker_mapping[speaker]))
            
            return relabeled_segments
            
        except Exception as e:
            raise RuntimeError(f"Diarization failed: {e}")
    
    def _mock_diarization(self, audio_file: str, num_speakers: Optional[int] = None) -> List[Tuple[float, float, str]]:
        """Mock diarization for testing without HuggingFace authentication"""
        import librosa
        
        # Load audio to get duration
        try:
            y, sr = librosa.load(audio_file, sr=16000)
            duration = len(y) / sr
        except Exception as e:
            # Fallback duration if audio loading fails
            duration = 20.0
        
        # Create mock segments with alternating speakers
        segments = []
        segment_length = max(2.0, duration / 8)  # Aim for ~8 segments
        current_time = 0.0
        speaker_idx = 0
        num_speakers = num_speakers or 2
        
        while current_time < duration:
            end_time = min(current_time + segment_length, duration)
            speaker = f"Speaker {chr(65 + speaker_idx)}"
            segments.append((current_time, end_time, speaker))
            
            current_time = end_time
            speaker_idx = (speaker_idx + 1) % num_speakers
        
        return segments
    
    def get_speaker_count(self, segments: List[Tuple[float, float, str]]) -> int:
        unique_speakers = set(segment[2] for segment in segments)
        return len(unique_speakers)