from typing import List, Dict, Any, Optional
from datetime import timedelta

from .models import LLMEnhancements


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS,mmm format"""
    td = timedelta(seconds=seconds)
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    secs = td.total_seconds() % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")


def format_timestamp_vtt(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format for VTT"""
    td = timedelta(seconds=seconds)
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    secs = td.total_seconds() % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


class ResponseFormatter:
    @staticmethod
    def format_json(
        segments: List[Dict[str, Any]], 
        audio_duration: float, 
        speakers_detected: int,
        llm_enhancements: Optional[LLMEnhancements] = None
    ) -> Dict[str, Any]:
        utterances = []
        for segment in segments:
            utterances.append({
                "speaker": segment["speaker"],
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "confidence": segment.get("confidence")
            })
        
        result = {
            "utterances": utterances,
            "audio_duration": audio_duration,
            "speakers_detected": speakers_detected
        }
        
        if llm_enhancements:
            result["llm_enhancements"] = {
                "summary": llm_enhancements.summary,
                "action_items": llm_enhancements.action_items,
                "topics": llm_enhancements.topics
            }
        
        return result
    
    @staticmethod
    def format_srt(segments: List[Dict[str, Any]]) -> str:
        srt_content = []
        
        for i, segment in enumerate(segments, 1):
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            
            srt_content.append(f"{i}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(f"[{segment['speaker']}] {segment['text']}")
            srt_content.append("")  # Empty line between subtitles
        
        return "\n".join(srt_content)
    
    @staticmethod
    def format_vtt(segments: List[Dict[str, Any]]) -> str:
        vtt_content = ["WEBVTT", ""]
        
        for segment in segments:
            start_time = format_timestamp_vtt(segment["start"])
            end_time = format_timestamp_vtt(segment["end"])
            
            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(f"[{segment['speaker']}] {segment['text']}")
            vtt_content.append("")  # Empty line between captions
        
        return "\n".join(vtt_content)
    
    @staticmethod
    def format_text(segments: List[Dict[str, Any]]) -> str:
        text_content = []
        
        for segment in segments:
            timestamp = format_timestamp(segment["start"])
            text_content.append(f"[{timestamp}] {segment['speaker']}: {segment['text']}")
        
        return "\n".join(text_content)