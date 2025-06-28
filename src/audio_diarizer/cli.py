#!/usr/bin/env python3
"""
CLI client for Audio Diarization API
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional

import httpx


class AudioDiarizationClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=300.0)  # 5 minute timeout
    
    async def submit_job(
        self,
        audio_file: Path,
        expected_speakers: Optional[int] = None,
        response_format: str = "json",
        enable_llm_analysis: bool = False
    ) -> str:
        """Submit audio file for processing and return job_id"""
        
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        files = {"file": (audio_file.name, audio_file.open("rb"), "audio/wav")}
        data = {
            "response_format": response_format,
            "enable_llm_analysis": str(enable_llm_analysis).lower()
        }
        
        if expected_speakers:
            data["expected_speakers"] = expected_speakers
        
        response = await self.client.post(f"{self.base_url}/transcribe", files=files, data=data)
        response.raise_for_status()
        
        result = response.json()
        return result["job_id"]
    
    async def get_job_status(self, job_id: str) -> dict:
        """Get job status and result"""
        response = await self.client.get(f"{self.base_url}/transcribe/{job_id}")
        response.raise_for_status()
        return response.json()
    
    async def wait_for_completion(self, job_id: str, poll_interval: int = 5) -> dict:
        """Wait for job to complete and return result"""
        print(f"Waiting for job {job_id} to complete...")
        
        while True:
            result = await self.get_job_status(job_id)
            status = result["status"]
            
            if status == "completed":
                print("✓ Job completed successfully")
                return result
            elif status == "failed":
                error = result.get("error", "Unknown error")
                raise Exception(f"Job failed: {error}")
            elif status in ["pending", "processing", "preprocessing", "diarizing", "transcribing", "llm_analysis", "formatting"]:
                # Show detailed progress
                progress = result.get("progress", status.replace("_", " ").title())
                progress_percent = result.get("progress_percent")
                if progress_percent is not None:
                    print(f"  [{progress_percent:3d}%] {progress}")
                else:
                    print(f"  Status: {progress}")
                await asyncio.sleep(poll_interval)
            else:
                raise Exception(f"Unknown status: {status}")
    
    async def check_health(self) -> dict:
        """Check API server health"""
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


async def main():
    parser = argparse.ArgumentParser(
        description="CLI client for Audio Diarization API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transcription
  uv run python -m src.audio_diarizer.cli meeting.wav

  # With expected speakers and LLM analysis
  uv run python -m src.audio_diarizer.cli meeting.wav --speakers 3 --llm-analysis

  # Save to specific output file
  uv run python -m src.audio_diarizer.cli meeting.wav --output transcript.json

  # Get SRT format
  uv run python -m src.audio_diarizer.cli meeting.wav --format srt --output subtitles.srt

  # Check server health
  uv run python -m src.audio_diarizer.cli --health
        """
    )
    
    parser.add_argument("audio_file", nargs="?", type=Path, help="Path to audio file")
    parser.add_argument("--output", "-o", type=Path, help="Output file path (default: auto-generated)")
    parser.add_argument("--speakers", "-s", type=int, help="Expected number of speakers (2-10)")
    parser.add_argument("--format", "-f", choices=["json", "srt", "vtt", "text"], default="json", help="Output format")
    parser.add_argument("--llm-analysis", action="store_true", help="Enable LLM analysis (summary, action items, topics)")
    parser.add_argument("--server", default="http://localhost:8000", help="API server URL")
    parser.add_argument("--poll-interval", type=int, default=5, help="Polling interval in seconds")
    parser.add_argument("--health", action="store_true", help="Check server health and exit")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode - minimal output")
    
    args = parser.parse_args()
    
    client = AudioDiarizationClient(args.server)
    
    try:
        # Health check mode
        if args.health:
            health = await client.check_health()
            if not args.quiet:
                print("Server Health:")
                print(json.dumps(health, indent=2))
            sys.exit(0)
        
        # Require audio file for transcription
        if not args.audio_file:
            parser.error("Audio file is required (or use --health to check server)")
        
        # Validate audio file
        if not args.audio_file.exists():
            print(f"Error: Audio file not found: {args.audio_file}", file=sys.stderr)
            sys.exit(1)
        
        # Generate output filename if not provided
        if not args.output:
            stem = args.audio_file.stem
            ext = "json" if args.format == "json" else args.format
            args.output = Path(f"{stem}_transcript.{ext}")
        
        if not args.quiet:
            print(f"Submitting audio file: {args.audio_file}")
            print(f"Expected speakers: {args.speakers or 'auto-detect'}")
            print(f"Output format: {args.format}")
            print(f"LLM analysis: {'enabled' if args.llm_analysis else 'disabled'}")
            print(f"Output file: {args.output}")
            print()
        
        # Submit job
        job_id = await client.submit_job(
            args.audio_file,
            expected_speakers=args.speakers,
            response_format=args.format,
            enable_llm_analysis=args.llm_analysis
        )
        
        if not args.quiet:
            print(f"Job submitted: {job_id}")
        
        # Wait for completion
        result = await client.wait_for_completion(job_id, args.poll_interval)
        
        # Save result
        if args.format == "json":
            output_content = json.dumps(result, indent=2)
        else:
            # For non-JSON formats, the result should be in text format
            if "result" in result and isinstance(result["result"], str):
                output_content = result["result"]
            else:
                # Fallback to JSON if format conversion failed
                output_content = json.dumps(result, indent=2)
        
        args.output.write_text(output_content, encoding="utf-8")
        
        if not args.quiet:
            print(f"✓ Transcription saved to: {args.output}")
            
            # Show summary info
            if "result" in result and isinstance(result["result"], dict):
                res = result["result"]
                if "audio_duration" in res:
                    print(f"  Audio duration: {res['audio_duration']:.1f}s")
                if "speakers_detected" in res:
                    print(f"  Speakers detected: {res['speakers_detected']}")
                if "utterances" in res:
                    print(f"  Utterances: {len(res['utterances'])}")
                if "llm_enhancements" in res:
                    print("  ✓ LLM analysis included")
    
    except KeyboardInterrupt:
        print("\nCancelled by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())