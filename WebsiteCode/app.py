import asyncio
import base64
import json
from fastapi import FastAPI, HTTPException, WebSocket, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
from openai import OpenAI

import uvicorn

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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            print(filetype)
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
                    await websocket.send_text(token)
                    await asyncio.sleep(0.05)  # Yield control to handle other tasks
            await websocket.close()


    except Exception:
        active_connections.remove(websocket)

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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


