from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ResponseFormat(str, Enum):
    JSON = "json"
    SRT = "srt"
    VTT = "vtt"
    TEXT = "text"


class TranscribeRequest(BaseModel):
    expected_speakers: Optional[int] = Field(
        None, ge=2, le=10, description="Expected number of speakers (2-10)"
    )
    response_format: ResponseFormat = Field(
        ResponseFormat.JSON, description="Format of the response"
    )
    enable_llm_analysis: bool = Field(
        False, description="Enable AI-powered analysis (summary, action items, topics) via Ollama"
    )


class SpeakerUtterance(BaseModel):
    speaker: str = Field(description="Speaker identifier (e.g., 'Speaker A')")
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    text: str = Field(description="Transcribed text")
    confidence: Optional[float] = Field(None, description="Confidence score")


class LLMEnhancements(BaseModel):
    summary: Optional[str] = Field(None, description="AI-generated summary of the conversation")
    action_items: Optional[str] = Field(None, description="Extracted action items and next steps")
    topics: Optional[str] = Field(None, description="Main topics discussed")


class TranscriptionResult(BaseModel):
    utterances: List[SpeakerUtterance] = Field(description="List of speaker utterances")
    audio_duration: float = Field(description="Total audio duration in seconds")
    speakers_detected: int = Field(description="Number of speakers detected")
    llm_enhancements: Optional[LLMEnhancements] = Field(None, description="AI-powered analysis (when Ollama is available)")


class JobResponse(BaseModel):
    job_id: str = Field(description="Unique job identifier")
    status: JobStatus = Field(description="Current job status")
    created_at: str = Field(description="Job creation timestamp")
    completed_at: Optional[str] = Field(None, description="Job completion timestamp")
    result: Optional[TranscriptionResult] = Field(None, description="Transcription result")
    error: Optional[str] = Field(None, description="Error message if job failed")


class JobCreate(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.PENDING
    created_at: str