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
deepseek = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
llama = LlamaAPI(os.environ.get("LLAMA_API_KEY"))
gemini = genai.GenerativeModel(model_name="models/gemini-2.0-flash")

modelNames = ["Gemini", "Claude", "GPT", "DeepSeek", "Llama"]
# subjects = ["History", "Math", "Physics", "Linguistics"]
columns = ["Response", "Ground_Truth", "Time", "Model"]

instruction = "You are a teacher. "
# prompt = "You are a teacher. Based off the image, how correct is the following understandng of the material? Assume that the student can only speak the material. Please give an accurate score between 0 and 10 before your response, with 0 being the least accurate and 10 being perfect. Penalize errors in important information more heavily than other errors. Please also provide corrections if necessary."
prompt = "Based off the image, how correct is the following exact verbal recitation of the material? Assume that the student can only speak the material. Please give an accurate score between 0 and 10, with 0 being the least accurate and 10 being perfect. Penalize errors in important information more heavily than other errors. Please also provide corrections if necessary."


geminiFileMap = {}
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


def test(modelName, responses, testMaterials):
    for i in range(0, len(testMaterials)):
        currentTest = testMaterials.iloc[i]
        info = []
        with open(currentTest["Ground_Truth"], "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')

        start_time = time.time()
        if (modelName == "Gemini"):
            file = geminiFileMap[currentTest["Ground_Truth"]]
            response = gemini.generate_content(
                contents=[instruction, prompt, currentTest["Recitation"], file]
            )
            info.append(' '.join(response.text.splitlines()))
        elif (modelName == "Claude"):
            response = claude.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", 
                           "content": [
            {"type": "text", "text": [instruction, prompt, currentTest["Recitation"]]},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_data}}]}]
            )
            info.append(response.content)
        elif (modelName == "GPT"):
            response = gpt.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": [instruction, prompt, currentTest["Recitation"]]},
                            {"type": "image_url", "image_url": f"data:image/png;base64,{image_data}"}
                        ]
                    }
                ],
                max_tokens=2000
            )
            info.append(response.choices[0].message.content)

        elif (modelName == "DeepSeek"):
            response = deepseek.chat.completions.create(
                model="deepseek-chat",
                 messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": [instruction, prompt, currentTest["Recitation"]]},
                            {"type": "image_url", "image_url": f"data:image/png;base64,{image_data}"}
                        ]
                    }
                ],
                max_tokens=2000
            )   
            info.append(response.choices[0].message.content)

        elif (modelName == "Llama"):
            api_request_json = {
            "model": "llama3.1-70b",
            "messages": [
                {"role": "user", "content":  [
                            {"type": "text", "text": [instruction, prompt, currentTest["Recitation"]]},
                            {"type": "image_url", "image_url": f"data:image/png;base64,{image_data}"}
                        ]},
            ],
            "max_tokens": 2000
            }
            response = llama.run(api_request_json)
            info.append(response.choices[0].message.content)

        end_time = time.time()
        info.append(currentTest["Ground_Truth"])
        info.append(str(end_time - start_time) + " seconds")
        info.append(modelName)
        print(info)
        responses.append(info)

def testAll(testMaterials):
    responses = []
    uploadAll(testMaterials)
    test("GPT", responses, testMaterials)

    return responses

frame = testAll(pd.read_csv("Code\TestMaterials\Tests.csv"))
frame = np.array(frame)
print(frame.shape)
frame = pd.DataFrame(frame, columns=columns)
frame.to_csv("Code/TestResults/TestResult.csv", index= None)


# message = claude.messages.create(
#     model="claude-3-5-sonnet-20241022",
#     max_tokens=1024,
#     messages=[
#         {"role": "user", "content": "Explain how AI works"}
#     ]
# )
# print(message)





# response = deepseek.chat.completions.create(
#     model="deepseek-chat",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant"},
#         {"role": "user", "content": "Hello"},
#     ],
#     stream=False
# )

# print(response.choices[0].message.content)


# api_request_json = {
#     "model": "llama3.1-70b",
#     "messages": [
#         {"role": "user", "content": "What is the weather like in Boston?"},
#     ],
#     "functions": [
#         {
#             "name": "get_current_weather",
#             "description": "Get the current weather in a given location",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "location": {
#                         "type": "string",
#                         "description": "The city and state, e.g. San Francisco, CA",
#                     },
#                     "days": {
#                         "type": "number",
#                         "description": "for how many days ahead you wants the forecast",
#                     },
#                     "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
#                 },
#             },
#             "required": ["location", "days"],
#         }
#     ],
#     "stream": False,
#     "function_call": "get_current_weather",
# }

# # Execute the Request
# response = llama.run(api_request_json)
# print(json.dumps(response.json(), indent=2))