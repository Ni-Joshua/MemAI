import google.generativeai as genai
import anthropic
from openai import OpenAI
import json
from llamaapi import LlamaAPI
import os
import time
import pandas as pd
import numpy as np
import base64

#Models
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
claude = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)
gpt = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
deepseek = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/v1")
llama = OpenAI(api_key= os.environ.get("LLAMA_API_KEY"), base_url = "https://api.llama-api.com")
gemini = genai.GenerativeModel(model_name="models/gemini-2.0-flash")

modelNames = ["Gemini", "Claude", "GPT", "DeepSeek", "Llama"]
columns = ["Response", "Ground_Truth", "Time", "Model", "Recitation"]

instruction = "You are a teacher. "
# prompt = "You are a teacher. Based off the image, how correct is the following understandng of the material? Assume that the student can only speak the material meaning all symbols are pronounced. Please give an accurate score between 0 and 10 before your response, with 0 being the least accurate and 10 being perfect. Penalize errors in important information more heavily than other errors. Please also provide corrections if necessary."
prompt = "Based off the image, how correct is the following exact verbal recitation of the material? Assume that the student can only speak the material, meaning all symbols are pronounced. Please give an accurate score between 0 and 10, with 0 being the least accurate and 10 being perfect. Penalize errors in important information more heavily than other errors. Please also provide corrections if necessary. "


geminiFileMap = {}
base64FileMap = {}
def geminiUpload(filePath, mime_type):
    sampleFile = genai.upload_file(
        path=filePath,
        mime_type=mime_type
    )
    print(f"Uploaded file '{sampleFile.display_name}' as: {sampleFile.uri}")
    geminiFileMap[filePath] = sampleFile

def uploadAll(testMaterials):
    groundTruths = set(testMaterials["Ground_Truth"])
    for filePath in groundTruths:
        geminiUpload(filePath, "image/png")
        image_data = encodeImage(filePath)
        base64FileMap[filePath] = image_data


def encodeImage(filePath):
    with open(filePath, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    return image_data


def callGemini(currentTest):
    file = geminiFileMap[currentTest["Ground_Truth"]]
    response = gemini.generate_content(
        contents=[instruction, prompt, currentTest["Recitation"], file]
    )
    return response

def callClaude(currentTest):
    currentImage = base64FileMap[currentTest["Ground_Truth"]]
    response = claude.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", 
                           "content": [
            {"type": "text", "text": instruction + prompt + currentTest["Recitation"]},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": currentImage}}]}]
            )
    return response

def callGPT(currentTest):
    currentImage = base64FileMap[currentTest["Ground_Truth"]]
    response = gpt.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": instruction + prompt + currentTest["Recitation"]},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{currentImage}"}}
                        ]
                    }
                ],
                max_tokens=2000
            )
    return response

# def callDeepSeek(currentTest):
#     currentImage = base64FileMap[currentTest["Ground_Truth"]]
#     response = deepseek.chat.completions.create(
#                 model="deepseek-reasoner",
#                  messages=[
#                     {
#                         "role": "user",
#                         "content": [
#                             {"type": "text", "text": instruction + prompt + currentTest["Recitation"]},
#                             {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{currentImage}"}}
#                         ]
#                     }
#                 ],
#                 max_tokens=2000
#             )   
#     return response

# def callLlama(currentTest): 
#     currentImage = base64FileMap[currentTest["Ground_Truth"]]
#     response = llama.chat.completions.create(
#                 model="llama3.2-11b-vision",
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": [
#                             {"type": "text", "text": instruction + prompt + currentTest["Recitation"]},
#                             {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{currentImage}"}}
#                         ]
#                     }
#                 ],
#                 max_tokens=2000
#             )
#     return response

def test(modelName, responses, testMaterials):
    for i in range(0, len(testMaterials)):
        currentTest = testMaterials.iloc[i]
        info = []
        start_time = time.time()
        if (modelName == "Gemini"):
            info.append(' '.join(callGemini(currentTest).text.splitlines()))
        elif (modelName == "Claude"):
            info.append(' '.join(callClaude(currentTest).content[0].text.splitlines())) 
        elif (modelName == "GPT"):
            info.append(' '.join(callGPT(currentTest).choices[0].message.content.splitlines()))
        # elif (modelName == "DeepSeek"):
        #     info.append(' '.join(callDeepSeek(currentTest).choices[0].message.content.splitlines()))
        # elif (modelName == "Llama"):
        #     info.append(' '.join(callLlama(currentTest).choices[0].message.content.splitlines()))
        end_time = time.time()

        info.append(currentTest["Ground_Truth"])
        info.append(str(end_time - start_time) + " seconds")
        info.append(modelName)
        info.append(currentTest['Recitation'])
        print(info)
        responses.append(info)

def testAll(testMaterials):
    responses = []
    uploadAll(testMaterials)
    test("Gemini", responses, testMaterials)
    test("Claude", responses, testMaterials)
    test("GPT", responses, testMaterials)
    return responses

frame = testAll(pd.read_csv("Code\TestMaterials\Tests.csv"))
frame = np.array(frame)
print(frame.shape)
frame = pd.DataFrame(frame, columns=columns)
frame.to_csv("Code/TestResults/TestResult.csv", index= None)