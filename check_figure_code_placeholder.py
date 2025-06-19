import os
from openai import OpenAI
from pathlib import Path
import subprocess
import time
import json
from itertools import islice

class FigErrors: ...


FIG_QC_PROMPT = """
You will be given code from multiple Python files. For each file, extract the following information:
- file name
- plot title
- x-axis and y-axis titles
- x-axis scale
- y-axis scale
- legend
- font size
- colors
- figure size
- how the figure is saved
- whether the figure is matplotlib or altair

If anything doesn't exist, return N/A.

The files are separated by lines like: # --- <file_name> ---

Return your answer as a JSON list, where each element is a dictionary for one file.

{fig_code}
"""


def call_ai(fig_qc_prompt: str, fig_code: str, model_name: str):
    prompt = fig_qc_prompt.format(fig_code=fig_code)
    # Get API key from environment variable
    #OPENROUTER_API_KEY = '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'
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "axis titles",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "file_name": {
                            "type": "string",
                            "description": "the name of the file",
                        },
                        "Title": {
                            "type": "string",
                            "description": "plot title",
                        },
                        "x-axis title": {
                            "type": "string",
                            "description": "the title of the x-axis",
                        },
                        "y-axis title": {
                            "type": "string",
                            "description": "the title of the y-axis",
                        },
                        "x-axis scale": {
                            "type": "string",
                            "description": "the scale of the x-axis",
                        },
                        "y-axis scale": {
                            "type": "string",
                            "description": "the scale of the y-axis",
                        },
                        "legend": {
                            "type": "string",
                            "description": "the legend of the figure",
                        },
                        "font size": {
                            "type": "string",
                            "description": "the font size of the figure",
                        },
                        "colors": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "description": "the colors of the figure",
                            },
                        },
                        "figure size": {
                            "type": "string",
                            "description": "the size of the figure",
                        },
                        "saving the figure": {
                            "type": "string",
                            "description": "how to save the figure",
                        },
                        "matplotlib vs. altair": {
                            "type": "string",
                            "description": "the library used to create the figure",
                        },                        
                    },
                    "required": ["file_name", "Title", "x-axis title", "y-axis title", 'font size', 'colors', 'figure size', 'saving the figure', 'matplotlib vs. altair'],
                    "additionalProperties": False,
                },
            },
        },
    )

    print(completion.choices[0].message.content)
    # parse the response
    ...


def check_code(fig_code: str) -> FigErrors:
    call_ai(FIG_QC_PROMPT, fig_code, model_name="deepseek/deepseek-chat-v3-0324:free")

"""
def get_changed_py_files():
    Returns a list of changed .py files in the last commit.
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    changed_files = [line.strip() for line in result.stdout.splitlines()]
    py_files = [Path(f) for f in changed_files if f.endswith(".py") and Path(f).exists()]
    return py_files

"""
def get_all_py_files(directory: str = ".") -> list[Path]:
    """Returns a list of all .py files in the given directory and its subdirectories."""
    py_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                py_files.append(Path(root) / file)
    return py_files

def batch_files(files, batch_size):
    it = iter(files)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch


if __name__ == "__main__":
    py_files = get_all_py_files("llm_ontology/llm_ontology")
    if not py_files:
        print("No Python files found.")
    else:
        print(f"Found {len(py_files)} Python files to process.")
        for batch in batch_files(py_files, 3):
            batch_code = ""
            for file in batch:
                try:
                    with open(file, "r") as f:
                        code = f.read()
                        batch_code += f"\n# --- {file} ---\n{code}\n"
                except Exception as e:
                    print(f"Error processing {file}: {e}")
                    continue
            if batch_code:
                check_code(batch_code)
            time.sleep(10)  
       # for file in changed_py_files:
            #print(f"\nProcessing: {file}")
            #with open(file, "r") as f:
             #   fig_code = f.read()
              #  check_code(fig_code) 