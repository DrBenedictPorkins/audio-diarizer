import json
import httpx
from typing import Dict, List, Optional, Any
import logging

from .config import config


logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self):
        self.base_url = config.OLLAMA_HOST
        self.model = config.OLLAMA_MODEL
        self.enabled = config.OLLAMA_ENABLED
        
        if self.enabled:
            logger.info(f"Ollama client initialized: {self.base_url} with model {self.model}")
        else:
            logger.info("Ollama integration disabled")
    
    async def is_available(self) -> bool:
        """Check if Ollama server is available"""
        if not self.enabled:
            return False
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/tags", timeout=5)
                return response.status_code == 200
        except (httpx.RequestError, httpx.TimeoutException):
            return False
    
    async def generate_completion(self, prompt: str, temperature: float = 0.3) -> Optional[str]:
        """Generate completion using Ollama"""
        if not self.enabled or not await self.is_available():
            return None
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=60
                )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.error(f"Ollama API error: {response.status_code} {response.text}")
                return None
                
        except (httpx.RequestError, httpx.TimeoutException) as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return None
        except Exception as e:
            logger.error(f"Ollama completion error: {e}")
            return None
    
    async def summarize_transcript(self, segments: List[Dict[str, Any]]) -> Optional[str]:
        """Generate a summary of the diarized transcript"""
        if not self.enabled:
            return None
        
        # Format transcript for summarization
        transcript_text = self._format_transcript_for_llm(segments)
        
        prompt = f"""Please provide a concise summary of this meeting transcript. Focus on:
1. Key topics discussed
2. Main decisions made
3. Action items or next steps
4. Important points raised by each speaker

Transcript:
{transcript_text}

Summary:"""

        return await self.generate_completion(prompt, temperature=0.1)
    
    async def extract_action_items(self, segments: List[Dict[str, Any]]) -> Optional[str]:
        """Extract action items from the transcript"""
        if not self.enabled:
            return None
        
        transcript_text = self._format_transcript_for_llm(segments)
        
        prompt = f"""Extract all action items, tasks, and next steps from this meeting transcript. Format as a bullet-point list.

Transcript:
{transcript_text}

Action Items:"""

        return await self.generate_completion(prompt, temperature=0.1)
    
    async def identify_topics(self, segments: List[Dict[str, Any]]) -> Optional[str]:
        """Identify main topics discussed"""
        if not self.enabled:
            return None
        
        transcript_text = self._format_transcript_for_llm(segments)
        
        prompt = f"""Identify the main topics and themes discussed in this meeting transcript. List them as bullet points.

Transcript:
{transcript_text}

Main Topics:"""

        return await self.generate_completion(prompt, temperature=0.2)
    
    async def enhance_transcript(self, segments: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
        """Generate enhanced analysis of the transcript"""
        if not self.enabled:
            return None
        
        try:
            # Generate multiple enhancements in parallel could be added here
            # For now, do them sequentially to avoid overwhelming Ollama
            
            summary = await self.summarize_transcript(segments)
            action_items = await self.extract_action_items(segments)
            topics = await self.identify_topics(segments)
            
            return {
                "summary": summary,
                "action_items": action_items,
                "topics": topics
            }
            
        except Exception as e:
            logger.error(f"Error enhancing transcript: {e}")
            return None
    
    def _format_transcript_for_llm(self, segments: List[Dict[str, Any]], max_length: int = 8000) -> str:
        """Format transcript segments for LLM processing"""
        formatted_lines = []
        current_length = 0
        
        for segment in segments:
            line = f"{segment['speaker']}: {segment['text']}"
            
            if current_length + len(line) > max_length:
                break
            
            formatted_lines.append(line)
            current_length += len(line) + 1  # +1 for newline
        
        return "\n".join(formatted_lines)


# Global instance
ollama_client = OllamaClient()