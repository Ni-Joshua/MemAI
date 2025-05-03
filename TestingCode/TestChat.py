# server.py
import asyncio
import websockets
import struct
import json
import numpy as np
from RealtimeSTT import AudioToTextRecorder
from scipy.signal import resample
import openai

if __name__ == '__main__':
    openai.api_key = "YOUR_OPENAI_API_KEY"

    recorder = AudioToTextRecorder()
    clients = set()

    async def handle_client(websocket):
        clients.add(websocket)
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # Decode metadata and audio chunk
                    meta_len = struct.unpack("<I", message[:4])[0]
                    meta = json.loads(message[4:4+meta_len])
                    audio_data = message[4+meta_len:]

                    # Convert and resample audio
                    audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    if meta['sampleRate'] != 16000:
                        audio = resample(audio, int(len(audio) * 16000 / meta['sampleRate']))
                    recorder.push_audio(audio)

                else:
                    data = json.loads(message)
                    if data['type'] == 'text':
                        await handle_ai_chat(websocket, data['message'])
        finally:
            clients.remove(websocket)

    async def handle_ai_chat(websocket, prompt):
        await websocket.send(json.dumps({"type": "aiChunk", "text": "🧠 Thinking..."}))
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            for chunk in response:
                if 'choices' in chunk:
                    delta = chunk['choices'][0]['delta'].get('content')
                    if delta:
                        await websocket.send(json.dumps({"type": "aiChunk", "text": delta}))
            await websocket.send(json.dumps({"type": "aiDone"}))
        except Exception as e:
            await websocket.send(json.dumps({"type": "aiChunk", "text": f"Error: {e}"}))

    async def stt_monitor():
        prev_text = ""
        while True:
            text = recorder.text()
            if text and text != prev_text:
                prev_text = text
                for client in clients:
                    await client.send(json.dumps({"type": "fullSentence", "text": text}))
            await asyncio.sleep(0.3)

    async def main():
        asyncio.create_task(stt_monitor())
        async with websockets.serve(handle_client, "0.0.0.0", 8001):
            await asyncio.Future()

    asyncio.run(main())
