import json
import os
import traceback
import asyncio
from datetime import datetime
from typing import Dict, Any

import redis
import torch

from .config import config
from .audio_processor import AudioProcessor
from .diarization import SpeakerDiarizer
from .transcription import SpeechTranscriber
from .formatters import ResponseFormatter
from .models import JobStatus, ResponseFormat, LLMEnhancements
from .ollama_client import ollama_client


redis_client = redis.Redis(
    host=config.REDIS_HOST,
    port=config.REDIS_PORT,
    db=config.REDIS_DB,
    decode_responses=True
)


def update_job_status(job_id: str, status: JobStatus, **kwargs):
    """Update job status in Redis"""
    updates = {"status": status.value}
    updates.update(kwargs)
    redis_client.hset(f"job:{job_id}", mapping=updates)


def process_audio(job_data: Dict[str, Any]):
    """Main worker function to process audio diarization"""
    job_id = job_data["job_id"]
    file_path = job_data["file_path"]
    expected_speakers = job_data.get("expected_speakers")
    response_format = ResponseFormat(job_data.get("response_format", "json"))
    enable_llm_analysis = job_data.get("enable_llm_analysis", False)
    
    try:
        print(f"Starting job {job_id} for file {file_path}")
        update_job_status(job_id, JobStatus.PROCESSING)
        
        # Initialize processors
        audio_processor = AudioProcessor()
        diarizer = SpeakerDiarizer()
        transcriber = SpeechTranscriber()
        
        # Step 1: Preprocess audio
        print(f"Job {job_id}: Preprocessing audio...")
        processed_file, duration = audio_processor.preprocess_audio(file_path)
        
        # Step 2: Load audio
        print(f"Job {job_id}: Loading audio...")
        audio, sr = audio_processor.load_audio(processed_file)
        
        # Step 3: Speaker diarization
        print(f"Job {job_id}: Running speaker diarization...")
        diarization_segments = diarizer.diarize(
            processed_file, 
            num_speakers=expected_speakers
        )
        
        if not diarization_segments:
            raise ValueError("No speakers detected in audio")
        
        speakers_detected = diarizer.get_speaker_count(diarization_segments)
        print(f"Job {job_id}: Detected {speakers_detected} speakers")
        
        # Step 4: Segment audio for transcription
        print(f"Job {job_id}: Segmenting audio...")
        audio_segments = audio_processor.segment_audio(audio, diarization_segments, sr)
        
        # Step 5: Transcribe segments
        print(f"Job {job_id}: Transcribing {len(audio_segments)} segments...")
        transcribed_segments = transcriber.transcribe_segments(audio_segments)
        
        # Step 6: Merge consecutive same-speaker segments
        print(f"Job {job_id}: Merging consecutive segments...")
        final_segments = transcriber.merge_consecutive_segments(transcribed_segments)
        
        # Step 7: Generate LLM enhancements (if enabled and available)
        llm_enhancements = None
        if enable_llm_analysis:
            print(f"Job {job_id}: Checking Ollama availability...")
            try:
                # Run async functions in sync context
                ollama_available = asyncio.run(ollama_client.is_available())
                if ollama_available:
                    print(f"Job {job_id}: Generating LLM analysis...")
                    enhancements = asyncio.run(ollama_client.enhance_transcript(final_segments))
                    if enhancements:
                        llm_enhancements = LLMEnhancements(**enhancements)
                        print(f"Job {job_id}: LLM analysis completed")
                    else:
                        print(f"Job {job_id}: LLM analysis failed or returned empty")
                else:
                    print(f"Job {job_id}: LLM analysis requested but Ollama unavailable at {config.OLLAMA_HOST}")
            except Exception as e:
                print(f"Job {job_id}: LLM analysis error: {e}")
                # Don't fail the job if LLM analysis fails

        # Step 8: Format response
        print(f"Job {job_id}: Formatting response...")
        if response_format == ResponseFormat.JSON:
            result = ResponseFormatter.format_json(
                final_segments, duration, speakers_detected, llm_enhancements
            )
        elif response_format == ResponseFormat.SRT:
            result = ResponseFormatter.format_srt(final_segments)
        elif response_format == ResponseFormat.VTT:
            result = ResponseFormatter.format_vtt(final_segments)
        elif response_format == ResponseFormat.TEXT:
            result = ResponseFormatter.format_text(final_segments)
        else:
            result = ResponseFormatter.format_json(
                final_segments, duration, speakers_detected, llm_enhancements
            )
        
        # Save result
        completed_at = datetime.utcnow().isoformat()
        update_job_status(
            job_id,
            JobStatus.COMPLETED,
            completed_at=completed_at,
            result=json.dumps(result) if isinstance(result, dict) else result
        )
        
        print(f"Job {job_id}: Completed successfully")
        
        # Cleanup files
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            if os.path.exists(processed_file):
                os.remove(processed_file)
        except OSError:
            pass
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        error_msg = f"Job {job_id} failed: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        
        update_job_status(
            job_id,
            JobStatus.FAILED,
            error=error_msg,
            completed_at=datetime.utcnow().isoformat()
        )
        
        # Cleanup files on error
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            processed_file = file_path.replace(os.path.basename(file_path), f"processed_{os.path.basename(file_path)}")
            if os.path.exists(processed_file):
                os.remove(processed_file)
        except OSError:
            pass
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        raise