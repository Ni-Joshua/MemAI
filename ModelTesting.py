from google import genai
import anthropic
from openai import OpenAI
import json
from llamaapi import LlamaAPI
import os
import time

#Models

gemini = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
claude = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)
gpt = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
deepseek = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
llama = LlamaAPI(os.environ.get("LLAMA_API_KEY"))

start_time = time.time()

print(time.time() - start_time, "seconds")

# response = gemini.models.generate_content(
#     model="gemini-2.0-flash",
#     contents="Explain how AI works",
# )
# print(response.text)

# message = claude.messages.create(
#     model="claude-3-5-sonnet-20241022",
#     max_tokens=1024,
#     messages=[
#         {"role": "user", "content": "Explain how AI works"}
#     ]
# )
# print(message.content)


# completion = gpt.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {
#             "role": "user",
#             "content": "Write a haiku about recursion in programming."
#         }
#     ]
# )
# print(completion.choices[0].message)


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