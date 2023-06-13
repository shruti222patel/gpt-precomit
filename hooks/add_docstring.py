# Import necessary libraries
from __future__ import annotations

import os
import sys
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import argparse
import openai
from redbaron import RedBaron

# Set OpenAI API key
from tqdm import tqdm

openai.api_key = os.getenv("OPENAI_API_KEY")

system_prompt = dict(
    {
        "role": "system",
        "content": "You are a world-class Python developer and technical writer specializing in understanding code and explaining it clearly and concisely. The user will send you python functions, You must read and understand the code and respond with the corresponding docstring for the given function. The docstring must be accurate, clear, and concise. ONLY respond with the docstring in Google Style Multi-Line format. DO NOT respond with an explanation.",
    }
)
history = [
    system_prompt,
]


def add_docstring(filename):
    # Open the Python file using RedBaron library
    with open(filename, "r", encoding="utf-8") as file:
        code = RedBaron(file.read())

    # Loop through all functions in the Python file
    for node in code.find_all("def"):
        # Check if function already has a docstring
        if not node.value[0].type == "string":
            # Extract the function code
            function_code = node.dumps()

            # Send the function code to ChatGPT API for generating docstring (offcourse use GPT4 API if you hace access to it)
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0.2,
                messages=[
                    *history,
                    {"role": "user", "content": function_code},
                ],
            )

            # Extract the generated docstring from the OpenAI response
            docstring = response.choices[0].message.content

            # Remove the quotes from the generated docstring if present
            if docstring.startswith('"""') or docstring.startswith("'''"):
                docstring = docstring[3:-3]
            if docstring.startswith('"'):
                docstring = docstring[1:-1]

            # Add the function code and generated docstring to history
            history.append({"role": "user", "content": function_code})
            history.append(
                {
                    "role": "assistant",
                    "content": docstring,
                }
            )

            # Insert the generated docstring to the Function node
            if node.next and node.next.type == "comment":
                node.next.insert_after(f'"""\n{docstring}\n"""')
            else:
                node.value.insert(0, f'"""\n{docstring}\n"""')

    # Write the modified Python file back to disk
    with open(filename, "w", encoding="utf-8") as file:
        file.write(code.dumps())


def main(argv: Sequence[str] | None = None) -> int:
    print("Adding docstrings to your Python files...")
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='*', help='Filenames to check.')
    args = parser.parse_args(argv)

    python_files = [ filename for filename in args.filenames if filename.endswith(".py") ]

    if len(python_files) <= 3:
        for filename in tqdm(python_files):
            add_docstring(filename)
    else:
        # We will use a ThreadPoolExecutor to manage our threads
        with ThreadPoolExecutor() as executor:
            # Here we map each filename to the add_docstring function
            # Each call to add_docstring will run in a separate thread
            list(tqdm(executor.map(add_docstring, python_files), total=len(python_files)))

if __name__ == "__main__":
    sys.exit(main())