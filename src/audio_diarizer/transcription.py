import torch
from faster_whisper import WhisperModel
from typing import List, Dict, Any, Optional
import numpy as np
import tempfile
import librosa
import soundfile as sf
import os

from .config import config


class SpeechTranscriber:
    def __init__(self):
        self.device = config.DEVICE if torch.cuda.is_available() else "cpu"
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            compute_type = config.TORCH_DTYPE if self.device == "cuda" else "int8"
            
            self.model = WhisperModel(
                config.WHISPER_MODEL,
                device=self.device,
                compute_type=compute_type
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}")
    
    def transcribe_segments(self, audio_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("Whisper model not loaded")
        
        transcribed_segments = []
        
        for segment in audio_segments:
            try:
                # Save segment to temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    sf.write(temp_file.name, segment['audio'], config.SAMPLE_RATE)
                    temp_path = temp_file.name
                
                # Transcribe segment
                segments, info = self.model.transcribe(
                    temp_path,
                    beam_size=5,
                    word_timestamps=True,
                    language="en"  # TODO: Add language detection
                )
                
                # Combine all segments into one text
                full_text = ""
                word_level_timestamps = []
                
                for whisper_segment in segments:
                    full_text += whisper_segment.text
                    
                    if hasattr(whisper_segment, 'words') and whisper_segment.words:
                        for word in whisper_segment.words:
                            word_level_timestamps.append({
                                'word': word.word,
                                'start': segment['start'] + word.start,
                                'end': segment['start'] + word.end,
                                'probability': word.probability
                            })
                
                transcribed_segments.append({
                    'speaker': segment['speaker'],
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': full_text.strip(),
                    'confidence': info.language_probability if hasattr(info, 'language_probability') else None,
                    'words': word_level_timestamps
                })
                
                # Clean up temporary file
                os.unlink(temp_path)
                
            except Exception as e:
                print(f"Warning: Failed to transcribe segment {segment['start']}-{segment['end']}: {e}")
                transcribed_segments.append({
                    'speaker': segment['speaker'],
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': "[Transcription failed]",
                    'confidence': 0.0,
                    'words': []
                })
        
        return transcribed_segments
    
    def merge_consecutive_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not segments:
            return segments
        
        merged = []
        current = segments[0].copy()
        
        for segment in segments[1:]:
            # If same speaker and segments are close (within 2 seconds)
            if (segment['speaker'] == current['speaker'] and 
                segment['start'] - current['end'] <= 2.0):
                
                # Merge segments
                current['end'] = segment['end']
                current['text'] += " " + segment['text']
                
                # Merge word timestamps
                if 'words' in current and 'words' in segment:
                    current['words'].extend(segment['words'])
                
                # Average confidence
                if current['confidence'] and segment['confidence']:
                    current['confidence'] = (current['confidence'] + segment['confidence']) / 2
                
            else:
                # Start new segment
                merged.append(current)
                current = segment.copy()
        
        merged.append(current)
        return merged