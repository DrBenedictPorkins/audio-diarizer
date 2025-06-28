import uuid
import os
import json
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

import redis.asyncio as redis
import redis as sync_redis
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from rq import Queue

from .models import (
    TranscribeRequest, 
    JobResponse, 
    JobStatus, 
    ResponseFormat,
    JobCreate
)
from .config import config
from .ollama_client import ollama_client

app = FastAPI(
    title="Audio Diarization API",
    description="Speaker diarization and transcription service similar to AssemblyAI",
    version="0.1.0"
)

async def get_redis():
    return redis.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        db=config.REDIS_DB,
        decode_responses=True
    )

# Keep sync Redis for RQ queue (RQ doesn't support async yet)
sync_redis_client = sync_redis.Redis(
    host=config.REDIS_HOST,
    port=config.REDIS_PORT,
    db=config.REDIS_DB,
    decode_responses=True
)

queue = Queue(connection=sync_redis_client)

os.makedirs(config.UPLOAD_DIR, exist_ok=True)


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    health_status = {
        "status": "healthy", 
        "service": "audio-diarizer",
        "environment": config.DEPLOYMENT_TARGET,
        "device": config.DEVICE,
        "ollama_enabled": config.OLLAMA_ENABLED,
        "ollama_available": await ollama_client.is_available() if config.OLLAMA_ENABLED else False
    }
    return health_status


@app.get("/ollama/status")
async def ollama_status() -> Dict[str, Any]:
    """Check Ollama server status and capabilities"""
    if not config.OLLAMA_ENABLED:
        return {
            "enabled": False,
            "message": "Ollama integration is disabled"
        }
    
    available = await ollama_client.is_available()
    return {
        "enabled": True,
        "available": available,
        "host": config.OLLAMA_HOST,
        "model": config.OLLAMA_MODEL,
        "message": "Ollama server is available" if available else "Ollama server is not responding"
    }


@app.post("/transcribe", response_model=JobResponse)
async def create_transcription_job(
    file: UploadFile = File(...),
    expected_speakers: int = Form(None),
    response_format: ResponseFormat = Form(ResponseFormat.JSON),
    enable_llm_analysis: bool = Form(False)
) -> JobResponse:
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=400, 
            detail="File must be an audio file"
        )
    
    if file.size and file.size > config.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds maximum allowed size of {config.MAX_FILE_SIZE} bytes"
        )
    
    job_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    
    # Sanitize filename to prevent path traversal
    safe_filename = Path(file.filename or "unknown").name[:100]
    file_path = Path(config.UPLOAD_DIR) / f"{job_id}_{safe_filename}"
    
    try:
        content = await file.read()
        
        # Check actual file size
        if len(content) > config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File size {len(content)} exceeds maximum allowed size of {config.MAX_FILE_SIZE} bytes"
            )
        
        # Write file in threadpool to avoid blocking
        await run_in_threadpool(file_path.write_bytes, content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    job_data = {
        "job_id": job_id,
        "file_path": str(file_path),
        "expected_speakers": expected_speakers,
        "response_format": response_format.value,
        "enable_llm_analysis": enable_llm_analysis,
        "created_at": created_at
    }
    
    redis_conn = await get_redis()
    try:
        await redis_conn.hset(f"job:{job_id}", mapping={
            "status": JobStatus.PENDING.value,
            "created_at": created_at,
            "file_path": str(file_path),
            "expected_speakers": expected_speakers or "",
            "response_format": response_format.value,
            "enable_llm_analysis": str(enable_llm_analysis).lower()
        })
        # Set TTL of 24 hours for job data
        await redis_conn.expire(f"job:{job_id}", 86400)
    except redis.RedisError as e:
        raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")
    finally:
        await redis_conn.aclose()
    
    queue.enqueue("audio_diarizer.worker.process_audio", job_data)
    
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        created_at=created_at
    )


@app.get("/transcribe/{job_id}", response_model=JobResponse)
async def get_transcription_job(job_id: str) -> JobResponse:
    redis_conn = await get_redis()
    try:
        job_data = await redis_conn.hgetall(f"job:{job_id}")
    except redis.RedisError as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job: {str(e)}")
    finally:
        await redis_conn.aclose()
    
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status = JobStatus(job_data["status"])
    
    response = JobResponse(
        job_id=job_id,
        status=status,
        created_at=job_data["created_at"],
        completed_at=job_data.get("completed_at"),
        error=job_data.get("error"),
        progress=job_data.get("progress"),
        progress_percent=int(job_data["progress_percent"]) if job_data.get("progress_percent") else None
    )
    
    if status == JobStatus.COMPLETED and "result" in job_data:
        try:
            result_data = json.loads(job_data["result"])
            response.result = result_data
        except json.JSONDecodeError:
            response.error = "Failed to parse result data"
            response.status = JobStatus.FAILED
    
    return response


@app.delete("/transcribe/{job_id}")
async def delete_transcription_job(job_id: str) -> Dict[str, str]:
    redis_conn = await get_redis()
    try:
        job_data = await redis_conn.hgetall(f"job:{job_id}")
        
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if "file_path" in job_data and os.path.exists(job_data["file_path"]):
            try:
                os.remove(job_data["file_path"])
            except OSError:
                pass
        
        await redis_conn.delete(f"job:{job_id}")
    except redis.RedisError as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete job: {str(e)}")
    finally:
        await redis_conn.aclose()
    
    return {"message": f"Job {job_id} deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)