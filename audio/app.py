import uvicorn
import redis.asyncio as redis
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from datetime import datetime
import os
from faster_whisper import WhisperModel
import logging
from typing import Optional
from pydantic import BaseModel
from googletrans import Translator  # For translation functionality

# Configure logging
LOG_PATH = './logs'
LOGFILE_CONTAINER = f'{LOG_PATH}/logfile_container_audio.log'
os.makedirs(os.path.dirname(LOGFILE_CONTAINER), exist_ok=True)
logging.basicConfig(filename=LOGFILE_CONTAINER, level=logging.INFO, 
                   format='[%(asctime)s - %(name)s - %(levelname)s - %(message)s]')
logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] started logging in {LOGFILE_CONTAINER}')

# Global variables
current_model = None
redis_connection = None
translator = Translator()  # Google Translate instance

# Pydantic models for request validation
class TranslationRequest(BaseModel):
    text: str
    source_lang: str = 'auto'
    target_lang: str = 'en'

def load_audio(req_audio_model, req_device, req_compute_type):
    try:
        global current_model
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [load_audio] trying to start WhisperModel with req_audio_model: {req_audio_model}')
        if current_model is None:
            current_model = WhisperModel(req_audio_model, device=req_device, compute_type=req_compute_type)
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [load_audio] [success] WhisperModel started!')
    
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [load_audio] [error] Failed to load WhisperModel')
        raise

def format_time(seconds: float) -> str:
    """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
    milliseconds = int((seconds - int(seconds))) * 1000
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def generate_srt(segments, output_path: str = "subtitles.srt") -> str:
    """Generate SRT file from transcription segments"""
    srt_content = []
    for i, segment in enumerate(segments, start=1):
        start_time = format_time(segment.start)
        end_time = format_time(segment.end)
        text = segment.text.strip()
        srt_content.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(srt_content))
    
    return output_path

def transcribe_audio(audio_model, audio_path, device, compute_type, generate_srt_file: bool = False):
    try:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [transcribe_audio] trying to load WhisperModel audio_model: {audio_model} device: {device} compute_type: {compute_type} audio_path: {audio_path} ...')
        load_audio(audio_model, device, compute_type)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [transcribe_audio] WhisperModel loaded!')
        
        start_time = time.time()
        
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [transcribe_audio] trying to transcribe path: {audio_path} ...')
        segments, info = current_model.transcribe(audio_path, word_timestamps=True)
        
        # Convert segments to list since we might need to iterate multiple times
        segments_list = list(segments)
        full_text = "\n".join([segment.text for segment in segments_list])
        processing_time = time.time() - start_time
        
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [transcribe_audio] finished transcribing audio from {audio_path}! lang found: {info.language} len text_length: {len(full_text)} in {processing_time:.2f}s ...')
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [transcribe_audio] full_text {full_text}')
        
        result = {
            "language": info.language,
            "text": full_text,
            "processing_time": f"{processing_time:.2f}s",
            "srt_path": None
        }
        
        if generate_srt_file:
            srt_path = generate_srt(segments_list)
            result["srt_path"] = srt_path
        
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [transcribe_audio] returning ...')
        return result
        
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [transcribe_audio] [error]: {e}')
        return {"error": str(e)}

def translate_text(text: str, source_lang: str = 'auto', target_lang: str = 'en') -> str:
    """Translate text from source language to target language"""
    try:
        translation = translator.translate(text, src=source_lang, dest=target_lang)
        return translation.text
    except Exception as e:
        print(f'Translation error: {e}')
        raise HTTPException(status_code=400, detail=f"Translation failed: {str(e)}")

def start_redis(req_redis_port):
    try:
        r = redis.Redis(host="redis", port=req_redis_port, db=0)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [start_redis] Redis started successfully.')
        return r
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [start_redis] Failed to start Redis on port {req_redis_port}: {e}')
        raise

app = FastAPI()
llm_instance = None

@app.get("/")
async def root():
    return 'Hello from audio server!'

@app.post("/a")
async def fnaudio2(request: Request):
    try:
        req_data = await request.json()
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [audio] req_data > {req_data}')
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [audio] req_data > {req_data}')

        if req_data["method"] == "status":
            return JSONResponse({"result_status": 200, "result_data": f'ok'})

        if req_data["method"] == "transcribe":
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [{req_data["method"]}] trying to transcribe ...')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [{req_data["method"]}] trying to transcribe ...')
            res_transcribe = transcribe_audio(req_data["audio_model"],req_data["audio_path"],req_data["device"],req_data["compute_type"])
            return JSONResponse({"result_status": 200, "result_data": f'{res_transcribe}'})


    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return JSONResponse({"result_status": 500, "result_data": f'{e}'})


@app.post("/t")
async def fnaudio(request: Request):
    try:
        req_data = await request.json()
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [audio] req_data > {req_data}')
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [audio] req_data > {req_data}')

        if req_data["method"] == "status":
            return JSONResponse({"result_status": 200, "result_data": "ok"})

        if req_data["method"] == "transcribe":
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [{req_data["method"]}] trying to transcribe ...')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [{req_data["method"]}] trying to transcribe ...')
            
            generate_srt = req_data.get("generate_srt", False)
            res_transcribe = transcribe_audio(
                req_data["audio_model"],
                req_data["audio_path"],
                req_data["device"],
                req_data["compute_type"],
                generate_srt
            )
            
            if "error" in res_transcribe:
                return JSONResponse({"result_status": 500, "result_data": res_transcribe["error"]})
            
            response_data = {
                "language": res_transcribe["language"],
                "text": res_transcribe["text"],
                "processing_time": res_transcribe["processing_time"]
            }
            
            if generate_srt and res_transcribe["srt_path"]:
                response_data["srt_download_url"] = f"/download_srt?srt_path={res_transcribe['srt_path']}"
            
            return JSONResponse({"result_status": 200, "result_data": response_data})

    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return JSONResponse({"result_status": 500, "result_data": str(e)})

@app.get("/download_srt")
async def download_srt(srt_path: str):
    if not os.path.exists(srt_path):
        raise HTTPException(status_code=404, detail="SRT file not found")
    return FileResponse(srt_path, media_type="application/x-subrip", filename="subtitles.srt")

@app.post("/translate")
async def translate_text_endpoint(request: TranslationRequest):
    try:
        translated_text = translate_text(request.text, request.source_lang, request.target_lang)
        return JSONResponse({
            "result_status": 200,
            "result_data": {
                "original_text": request.text,
                "translated_text": translated_text,
                "source_lang": request.source_lang,
                "target_lang": request.target_lang
            }
        })
    except Exception as e:
        return JSONResponse({"result_status": 500, "result_data": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=f'{os.getenv("AUDIO_IP")}', port=int(os.getenv("AUDIO_PORT")))