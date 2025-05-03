import asyncio
import base64
from contextlib import asynccontextmanager
import json
import threading
from fastapi import FastAPI, HTTPException, WebSocket, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import numpy as np
from openai import OpenAI
import logging
import uvicorn
import sys
from RealtimeSTT import AudioToTextRecorder
from scipy.signal import resample
import websockets

if __name__ == '__main__':
    app = FastAPI()
    gpt = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    prompt = "Based off the image, how correct is the following exact verbal recitation of the material? Assume that the student can only speak the material, meaning all symbols are pronounced. Please give an accurate score between 0 and 10, with 0 being the least accurate and 10 being perfect. Penalize errors in important information more heavily than other errors. Please also provide corrections if necessary. " 

    # Allow frontend access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Change "*" to your frontend URL if needed
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

   

    # Store active WebSocket connections
    active_connections = []
    ALLOWED_EXTENSIONS = ["pdf", "jpg", "jpeg", "png"]
    currentFile = None
    filetype = None
    filename = None
    ws = None
    mainloop = None

    def encode64():
        image_data = base64.b64encode(currentFile).decode('utf-8')
        return image_data

    def messageConstruction(filecontent, data):
        messages = [
                        {"role": "system", "content": "You are a teacher."},
                        {"role": "user", "content": [
                                {"type": "text", "text": prompt + data},
                                filecontent
                            ]}
                        ]
        return messages

    def decode_and_resample(audio_data, original_sample_rate, target_sample_rate):
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            num_original_samples = len(audio_np)
            num_target_samples = int(num_original_samples * target_sample_rate / original_sample_rate)
            resampled_audio = resample(audio_np, num_target_samples)
            return resampled_audio.astype(np.int16).tobytes()
        except Exception as e:
            print(f"Error in resampling: {e}")
            return audio_data


    def text_detected(text):
        print(text)
        asyncio.run_coroutine_threadsafe(send_to_client(json.dumps({
                    'type': 'realtime',
                    'text': text})), mainloop)

    recorder_config = {
             'spinner': False,
                # 'model': 'small.en', # or large-v2 or deepdml/faster-whisper-large-v3-turbo-ct2 or ...
                'realtime_model_type': 'tiny.en', # or small.en or distil-small.en or ...
                'language': 'en',
                'silero_sensitivity': 0.05,
                'webrtc_sensitivity': 3,
                'post_speech_silence_duration': 0.7,
                'min_length_of_recording': 0,        
                'min_gap_between_recordings': 0,                
                'enable_realtime_transcription': True,
                'realtime_processing_pause': 0,
                'silero_deactivity_detection': True,
                'early_transcription_on_silence': 0,
                'beam_size': 5,
                'beam_size_realtime': 3,
                'no_log_file': True,
                'initial_prompt': (
                    "End incomplete sentences with ellipses.\n"
                    "Examples:\n"
                    "Complete: The sky is blue.\n"
                    "Incomplete: When the sky...\n"
                    "Complete: She walked home.\n"
                    "Incomplete: Because he...\n"
                ),
                'use_microphone': False,
                'on_realtime_transcription_update': text_detected,
        }
    recorder = AudioToTextRecorder(**recorder_config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global main_loop
        main_loop = asyncio.get_running_loop()
        yield
        quit()
    
    @app.websocket("/ws")
    async def handle_Client(websocket: WebSocket):
        global ws, recorder
        await websocket.accept()
        active_connections.append(websocket)
        ws = websocket
        try:
            while True:
                message = await websocket.receive()
                # print(message)
                if(message["type"] == "websocket.disconnect"):
                    break
                if "bytes" in message:
                    await handle_STT(websocket, message['bytes'])
                elif 'text' in message:
                    await handle_AI(websocket, message['text'])
        finally:
            active_connections.remove(websocket)
            
    async def handle_AI(websocket: WebSocket, data):
        try:
            if (filetype == 'png'):
                filecontent = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode64()}"}}
                messages = messageConstruction(filecontent, data)
            elif (filetype == 'jpg' or filetype == 'jpeg'):
                filecontent = {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{encode64()}"}}
                messages = messageConstruction(filecontent, data)
            elif(filetype == 'pdf'):
                filecontent = {"type": "file","file":{"filename":filename, "file_data": f"data:application/pdf;base64,{encode64()}"}}
                messages = messageConstruction(filecontent, data)
            else:
                messages = [
                    {"role": "user", "content": "Provide a filetype error message"}
                    ]
            response = gpt.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                stream=True
            )
            # Stream each chunk to the client
            for chunk in response:
                token = chunk.choices[0].delta.content
                if token:
                    await websocket.send_text(json.dumps({"type":"AI", "text":token}))
                    await asyncio.sleep(0.05)  # Yield control to handle other tasks
        except Exception:
            active_connections.remove(websocket)
            recorder.stop()
            recorder.shutdown()

    async def handle_STT(websocket, data):
        try: 
            # Read the metadata length (first 4 bytes)
            metadata_length = int.from_bytes(data[:4], byteorder='little')
            # Get the metadata JSON string
            metadata_json = data[4:4+metadata_length].decode('utf-8')
            metadata = json.loads(metadata_json)
            sample_rate = metadata['sampleRate']
            # Get the audio chunk following the metadata
            chunk = data[4+metadata_length:]
            resampled_chunk = decode_and_resample(chunk, sample_rate, 16000)
            recorder.feed_audio(resampled_chunk)
            # full_sentence = recorder.text()
            # if full_sentence:
            #     await send_to_client(json.dumps({'type': 'fullSentence', 'text': full_sentence}))
            # print(full_sentence)
        except Exception as e:
                print(f"Error: {e}" )
    

    async def send_to_client(message):
        print(message)
        await ws.send_text(message)

    @app.post("/upload/")
    async def upload_file(file: UploadFile = File(...)):
        global currentFile, filetype, filename
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Invalid file type! Please upload a supported file.")

        uploaded = await file.read()
        filetype = file_extension
        filename = file.filename
        currentFile = uploaded
        return {"message": f"The memorization material has been set to '{file.filename}'"}
    
    uvicorn.run(app, host="0.0.0.0", port=8000)


