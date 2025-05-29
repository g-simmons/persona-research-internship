import os
from openai import OpenAI
from pathlib import Path
import subprocess


class FigErrors: ...


FIG_QC_PROMPT = """
What are the plot title, x-axis and y-axis titles,
x-axis scale, y-axis scale, legend, font size, colors, figure size,
how we saved the figure, and whether the figure is matplotlib or altair
if they exist? 
If anything doesn't exist, return N/A.
{fig_code}
"""


def call_ai(fig_qc_prompt: str, fig_code: str, model_name: str):
    prompt = fig_qc_prompt.format(fig_code=fig_code)
    # Get API key from environment variable
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
                    "required": ["Title", "x-axis title", "y-axis title", 'font size'],
                    "additionalProperties": False,
                },
            },
        },
    )

    print(completion.choices[0].message.content)
    # parse the response
    ...


def check_code(fig_code: str) -> FigErrors:
    call_ai(FIG_QC_PROMPT, fig_code, model_name="meta-llama/llama-4-maverick:free")


def get_changed_py_files():
    """Returns a list of changed .py files in the last commit."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    changed_files = [line.strip() for line in result.stdout.splitlines()]
    py_files = [Path(f) for f in changed_files if f.endswith(".py") and Path(f).exists()]
    return py_files


if __name__ == "__main__":
    changed_py_files = get_changed_py_files()
    
    if not changed_py_files:
        print("No changed .py files in the latest commit.")
    else:
        for file in changed_py_files:
            print(f"\nProcessing: {file}")
            with open(file, "r") as f:
                fig_code = f.read()
                check_code(fig_code) 