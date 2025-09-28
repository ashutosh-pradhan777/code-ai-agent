import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import sys
from functions.get_files_info import get_files_info, schema_get_files_info
from functions.get_file_content import get_file_content, schema_get_file_content
from functions.write_file import write_file, schema_write_file
from functions.run_python_file import run_python_file, schema_run_python_file
from call_function import call_function

def main():


    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    system_prompt = """
You are a helpful AI coding agent.

When a user asks a question or makes a request, make a function call plan. You can perform the following operations:

- List files and directories
- Read the content of a file
- Write to a file (create or update)
- Run a python file with optional arguments

All paths you provide should be relative to the working directory. You do not need to specify the working directory in your function calls as it is automatically injected for security reasons.
"""

    if len(sys.argv) < 2:
        print("We need a prompt")
        sys.exit(1)

    verboseflag = False
    if len(sys.argv) == 3 and sys.argv[2] == "--verbose":
        verboseflag = True
    prompt = sys.argv[1]

    messages = [
    types.Content(role="user", parts=[types.Part(text=prompt)]),
]
    
    available_functions = types.Tool(
    function_declarations=[
        schema_get_files_info,
        schema_run_python_file,
        schema_write_file,
        schema_get_file_content
    ]
)
    
    config=types.GenerateContentConfig(
    tools=[available_functions], system_instruction=system_prompt
)

    response = client.models.generate_content(
        model="gemini-2.0-flash-001", 
        contents=messages,
        config=config
    )

    if response is None or response.usage_metadata is None:
        print("response is malformed")
        return
    
    if verboseflag:
        print(f"User prompt: {prompt}")
        print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
        print(f"Response tokens: {response.usage_metadata.candidates_token_count}")

    if response.function_calls:
        for function_call_part in response.function_calls:
            result = call_function(function_call_part)
            print(result)
    else:
        print(response.text)
    
    

main()