import librosa
import numpy as np
import ffmpeg
from pathlib import Path
from typing import Tuple, Optional
import tempfile
import os

from .config import config


class AudioProcessor:
    def __init__(self):
        self.sample_rate = config.SAMPLE_RATE
        
    def preprocess_audio(self, file_path: str) -> Tuple[str, float]:
        input_path = Path(file_path)
        output_path = input_path.parent / f"processed_{input_path.name}"
        
        try:
            # Get audio info
            probe = ffmpeg.probe(str(input_path))
            duration = float(probe['streams'][0]['duration'])
            
            if duration > config.MAX_AUDIO_DURATION:
                raise ValueError(f"Audio duration {duration}s exceeds maximum {config.MAX_AUDIO_DURATION}s")
            
            # Convert to 16kHz mono WAV with normalization
            (
                ffmpeg
                .input(str(input_path))
                .filter('loudnorm')
                .output(
                    str(output_path),
                    acodec='pcm_s16le',
                    ac=1,
                    ar=self.sample_rate,
                    f='wav'
                )
                .overwrite_output()
                .run(quiet=True)
            )
            
            return str(output_path), duration
            
        except ffmpeg.Error as e:
            raise RuntimeError(f"FFmpeg processing failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Audio preprocessing failed: {e}")
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            return audio, sr
        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {e}")
    
    def apply_vad(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Simple VAD using energy-based detection"""
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        # Compute RMS energy
        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # Threshold based on median energy
        threshold = np.median(rms) * 0.1
        
        # Create mask
        frames_to_keep = rms > threshold
        
        # Convert frame indices to sample indices
        sample_indices = []
        for i, keep in enumerate(frames_to_keep):
            if keep:
                start_sample = i * hop_length
                end_sample = min(start_sample + frame_length, len(audio))
                sample_indices.extend(range(start_sample, end_sample))
        
        if not sample_indices:
            return audio  # Return original if no speech detected
        
        return audio[sample_indices]
    
    def segment_audio(self, audio: np.ndarray, segments: list, sr: int, padding: float = 0.15) -> list:
        """Extract audio segments with padding"""
        audio_segments = []
        padding_samples = int(padding * sr)
        
        for start, end, speaker in segments:
            start_sample = max(0, int(start * sr) - padding_samples)
            end_sample = min(len(audio), int(end * sr) + padding_samples)
            
            segment_audio = audio[start_sample:end_sample]
            audio_segments.append({
                'audio': segment_audio,
                'start': start,
                'end': end,
                'speaker': speaker,
                'start_sample': start_sample,
                'end_sample': end_sample
            })
        
        return audio_segments