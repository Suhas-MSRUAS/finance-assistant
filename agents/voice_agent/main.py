from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks, Form, File
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from faster_whisper import WhisperModel
import pyttsx3
import asyncio
import tempfile
import os
import logging
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import uuid
import shutil
from pathlib import Path
import io
import wave
from concurrent.futures import ThreadPoolExecutor
import json

# Audio processing libraries
try:
    import librosa
    import soundfile as sf
    import numpy as np
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False
    print("Audio processing libraries not available. Install librosa and soundfile for advanced features.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced Voice Agent", version="2.0.0")

# Configuration
TEMP_DIR = Path("temp_audio")
TEMP_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
SUPPORTED_FORMATS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
MAX_DURATION = 600  # 10 minutes

class TranscriptionRequest(BaseModel):
    language: Optional[str] = Field(None, description="Language code (e.g., 'en', 'es', 'fr')")
    task: str = Field("transcribe", description="'transcribe' or 'translate'")
    model_size: str = Field("base", description="Whisper model size")
    temperature: float = Field(0.0, ge=0.0, le=1.0)
    beam_size: int = Field(5, ge=1, le=10)
    best_of: int = Field(5, ge=1, le=10)
    no_speech_threshold: float = Field(0.6, ge=0.0, le=1.0)

class TranscriptionResponse(BaseModel):
    text: str
    language: Optional[str] = None
    language_probability: Optional[float] = None
    segments: List[Dict[str, Any]] = []
    duration: Optional[float] = None
    processing_time_ms: float
    model_used: str
    confidence_score: Optional[float] = None

class TTSRequest(BaseModel):
    text: str = Field(..., max_length=5000, description="Text to convert to speech")
    voice_id: Optional[str] = Field(None, description="Voice identifier")
    rate: int = Field(200, ge=50, le=400, description="Speech rate (words per minute)")
    volume: float = Field(0.9, ge=0.0, le=1.0, description="Volume level")
    pitch: int = Field(0, ge=-50, le=50, description="Pitch adjustment")
    output_format: str = Field("wav", description="Output audio format")

class TTSResponse(BaseModel):
    status: str
    audio_file: Optional[str] = None
    duration_estimate: Optional[float] = None
    processing_time_ms: float
    voice_used: Optional[str] = None

class VoiceAgent:
    def __init__(self):
        self.whisper_models = {}
        self.tts_engine = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.available_voices = []
        self._init_tts()
        self._load_whisper_model("base")

    def _init_tts(self):
        """Initialize TTS engine"""
        try:
            self.tts_engine = pyttsx3.init()
            # Get available voices
            voices = self.tts_engine.getProperty('voices')
            if voices:
                self.available_voices = [
                    {
                        "id": voice.id, 
                        "name": getattr(voice, 'name', 'Unknown'),
                        "age": getattr(voice, 'age', None),
                        "gender": getattr(voice, 'gender', None)
                    }
                    for voice in voices
                ]
            logger.info(f"TTS initialized with {len(self.available_voices)} voices")
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {e}")
            self.tts_engine = None
            self.available_voices = []

    def _load_whisper_model(self, model_size: str):
        """Load Whisper model if not already loaded"""
        if model_size not in self.whisper_models:
            try:
                logger.info(f"Loading Whisper model: {model_size}")
                self.whisper_models[model_size] = WhisperModel(model_size)
                logger.info(f"Whisper model {model_size} loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper model {model_size}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    def _validate_audio_file(self, file: UploadFile) -> Dict[str, Any]:
        """Validate uploaded audio file"""
        # Reset file pointer to get accurate size
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        # Check file size
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB")

        # Check file extension
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported format. Supported: {', '.join(SUPPORTED_FORMATS)}"
            )

        return {"extension": file_ext, "size": file_size}

    async def _preprocess_audio(self, file_path: Path, target_sr: int = 16000) -> Path:
        """Preprocess audio file for optimal transcription"""
        if not AUDIO_PROCESSING_AVAILABLE:
            return file_path

        try:
            # Load audio
            audio, sr = librosa.load(str(file_path), sr=target_sr)
            
            # Check duration
            duration = len(audio) / sr
            if duration > MAX_DURATION:
                raise HTTPException(status_code=400, detail=f"Audio too long. Maximum: {MAX_DURATION} seconds")

            # Apply noise reduction and normalization
            audio = librosa.effects.preemphasis(audio)
            audio = librosa.util.normalize(audio)

            # Save preprocessed audio
            processed_path = file_path.with_suffix('.processed.wav')
            sf.write(str(processed_path), audio, target_sr)
            
            return processed_path

        except Exception as e:
            logger.warning(f"Audio preprocessing failed: {e}")
            return file_path

    async def transcribe_audio(self, file: UploadFile, request: TranscriptionRequest) -> TranscriptionResponse:
        """Transcribe audio file to text"""
        start_time = datetime.now()
        
        # Validate file
        file_info = self._validate_audio_file(file)
        
        # Load appropriate model
        self._load_whisper_model(request.model_size)
        model = self.whisper_models[request.model_size]

        # Create temporary file
        temp_file = TEMP_DIR / f"{uuid.uuid4()}{file_info['extension']}"
        processed_file = temp_file
        
        try:
            # Save uploaded file
            content = await file.read()
            with open(temp_file, "wb") as buffer:
                buffer.write(content)

            # Preprocess audio if libraries available
            processed_file = await self._preprocess_audio(temp_file)

            # Run transcription in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._run_transcription,
                str(processed_file),
                request
            )

            # Process results
            segments = []
            full_text = ""
            
            # Handle different result types from faster-whisper
            if hasattr(result, '__iter__'):
                # Result is a tuple (segments, info)
                segments_iter, info = result
                language = getattr(info, 'language', None)
                language_probability = getattr(info, 'language_probability', None)
                duration = getattr(info, 'duration', None)
                
                for segment in segments_iter:
                    segment_dict = {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text,
                        "avg_logprob": getattr(segment, 'avg_logprob', None),
                        "no_speech_prob": getattr(segment, 'no_speech_prob', None)
                    }
                    segments.append(segment_dict)
                    full_text += segment.text
            else:
                # Fallback for different result format
                language = getattr(result, 'language', None)
                language_probability = getattr(result, 'language_probability', None)
                duration = getattr(result, 'duration', None)
                
                if hasattr(result, 'segments'):
                    for segment in result.segments:
                        segment_dict = {
                            "start": segment.start,
                            "end": segment.end,
                            "text": segment.text,
                            "avg_logprob": getattr(segment, 'avg_logprob', None),
                            "no_speech_prob": getattr(segment, 'no_speech_prob', None)
                        }
                        segments.append(segment_dict)
                        full_text += segment.text

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            return TranscriptionResponse(
                text=full_text.strip(),
                language=language,
                language_probability=language_probability,
                segments=segments,
                duration=duration,
                processing_time_ms=processing_time,
                model_used=request.model_size,
                confidence_score=self._calculate_confidence(segments)
            )

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
        
        finally:
            # Cleanup temporary files
            for temp_path in [temp_file, processed_file]:
                if temp_path.exists() and temp_path != temp_file:
                    try:
                        temp_path.unlink()
                    except:
                        pass
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except:
                    pass

    def _run_transcription(self, file_path: str, request: TranscriptionRequest):
        """Run transcription in thread pool"""
        model = self.whisper_models[request.model_size]
        
        return model.transcribe(
            file_path,
            language=request.language,
            task=request.task,
            temperature=request.temperature,
            beam_size=request.beam_size,
            best_of=request.best_of,
            no_speech_threshold=request.no_speech_threshold
        )

    def _calculate_confidence(self, segments: List[Dict]) -> Optional[float]:
        """Calculate average confidence score from segments"""
        if not segments:
            return None
        
        try:
            valid_segments = [seg for seg in segments if seg.get('avg_logprob') is not None]
            if not valid_segments:
                return None
                
            total_logprob = sum(seg['avg_logprob'] for seg in valid_segments)
            avg_logprob = total_logprob / len(valid_segments)
            # Convert log probability to confidence score (0-1)
            confidence = min(1.0, max(0.0, (avg_logprob + 1.0)))
            return round(confidence, 3)
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return None

    async def text_to_speech(self, request: TTSRequest) -> TTSResponse:
        """Convert text to speech"""
        if not self.tts_engine:
            raise HTTPException(status_code=503, detail="TTS engine not available")

        start_time = datetime.now()
        
        try:
            # Generate unique filename
            audio_id = str(uuid.uuid4())
            output_file = TEMP_DIR / f"tts_{audio_id}.{request.output_format}"

            # Run TTS in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self._run_tts,
                request,
                str(output_file)
            )

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            # Estimate duration (rough calculation)
            words = len(request.text.split())
            duration_estimate = (words / request.rate) * 60 if request.rate > 0 else 0

            return TTSResponse(
                status="completed",
                audio_file=f"tts_{audio_id}.{request.output_format}",
                duration_estimate=duration_estimate,
                processing_time_ms=processing_time,
                voice_used=request.voice_id
            )

        except Exception as e:
            logger.error(f"TTS failed: {e}")
            raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

    def _run_tts(self, request: TTSRequest, output_file: str):
        """Run TTS in thread pool"""
        try:
            engine = pyttsx3.init()
            
            # Configure voice settings
            if request.voice_id and any(v['id'] == request.voice_id for v in self.available_voices):
                engine.setProperty('voice', request.voice_id)
            
            engine.setProperty('rate', request.rate)
            engine.setProperty('volume', request.volume)
            
            # Save to file
            engine.save_to_file(request.text, output_file)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            logger.error(f"TTS engine error: {e}")
            raise

    def get_voices(self) -> List[Dict[str, Any]]:
        """Get available TTS voices"""
        return self.available_voices

    def get_models(self) -> List[str]:
        """Get available Whisper models"""
        return ["tiny", "base", "small", "medium", "large"]

# Global voice agent
voice_agent = VoiceAgent()

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    task: str = Form("transcribe"),
    model_size: str = Form("base"),
    temperature: float = Form(0.0),
    beam_size: int = Form(5),
    best_of: int = Form(5),
    no_speech_threshold: float = Form(0.6)
):
    """Transcribe audio file to text with advanced options"""
    request = TranscriptionRequest(
        language=language,
        task=task,
        model_size=model_size,
        temperature=temperature,
        beam_size=beam_size,
        best_of=best_of,
        no_speech_threshold=no_speech_threshold
    )
    
    return await voice_agent.transcribe_audio(file, request)

@app.post("/speak", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest):
    """Convert text to speech with voice customization"""
    return await voice_agent.text_to_speech(request)

@app.post("/speak/simple", response_model=TTSResponse)
async def speak_simple(
    text: str = Form(...),
    voice_id: Optional[str] = Form(None),
    rate: int = Form(200)
):
    """Simple text-to-speech endpoint"""
    request = TTSRequest(text=text, voice_id=voice_id, rate=rate)
    return await voice_agent.text_to_speech(request)

@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """Download generated audio file"""
    # Validate filename to prevent path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    file_path = TEMP_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    # Determine media type based on file extension
    ext = file_path.suffix.lower()
    media_type_map = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.ogg': 'audio/ogg',
        '.flac': 'audio/flac'
    }
    media_type = media_type_map.get(ext, 'audio/wav')
    
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/voices")
async def get_available_voices():
    """Get list of available TTS voices"""
    return {
        "voices": voice_agent.get_voices(),
        "total_count": len(voice_agent.get_voices())
    }

@app.get("/models")
async def get_available_models():
    """Get list of available Whisper models"""
    return {
        "models": voice_agent.get_models(),
        "loaded_models": list(voice_agent.whisper_models.keys())
    }

@app.post("/batch_transcribe")
async def batch_transcribe(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None,
    model_size: str = Form("base")
):
    """Transcribe multiple audio files"""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
    
    results = {}
    
    for file in files:
        try:
            request = TranscriptionRequest(model_size=model_size)
            result = await voice_agent.transcribe_audio(file, request)
            results[file.filename or f"file_{len(results)}"] = {
                "success": True,
                "transcription": result
            }
        except Exception as e:
            results[file.filename or f"file_{len(results)}"] = {
                "success": False,
                "error": str(e)
            }
    
    return {"results": results, "total_processed": len(files)}

@app.delete("/cleanup")
async def cleanup_temp_files():
    """Clean up temporary audio files"""
    try:
        files_deleted = 0
        if TEMP_DIR.exists():
            for file_path in TEMP_DIR.glob("*"):
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        files_deleted += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")
        
        return {
            "status": "completed",
            "files_deleted": files_deleted,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@app.get("/stats")
async def get_statistics():
    """Get voice agent statistics"""
    temp_files = []
    total_size = 0
    
    if TEMP_DIR.exists():
        temp_files = list(TEMP_DIR.glob("*"))
        total_size = sum(f.stat().st_size for f in temp_files if f.is_file())
    
    return {
        "loaded_models": list(voice_agent.whisper_models.keys()),
        "available_voices": len(voice_agent.get_voices()),
        "temp_files_count": len(temp_files),
        "temp_storage_mb": round(total_size / (1024 * 1024), 2),
        "audio_processing_available": AUDIO_PROCESSING_AVAILABLE,
        "tts_available": voice_agent.tts_engine is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "whisper_models_loaded": len(voice_agent.whisper_models),
        "tts_available": voice_agent.tts_engine is not None,
        "temp_directory_exists": TEMP_DIR.exists(),
        "timestamp": datetime.now().isoformat()
    }

# Graceful shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    try:
        voice_agent.executor.shutdown(wait=True)
        if voice_agent.tts_engine:
            try:
                voice_agent.tts_engine.stop()
            except:
                pass
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)