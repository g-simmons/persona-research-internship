import os
import sys
from openai import OpenAI
from pathlib import Path
import subprocess
import json
import pandas as pd
from colorama import Fore

# None

class FigErrors: ...


# y-axis title
# y-axis ticks
# y-axis scale
# x-axis title
# x-axis ticks
# x-axis scale
# legend
# font size
# colors
# figure size
# saving the figure
# matplotlib vs. altair?

FIG_QC_PROMPT = """
Provide the items asked for in valid JSON format.
What are the plot type(e.g. scatterplot, bar chart, line chart, etc.), plot title, x-axis and y-axis titles,
x-axis scale, y-axis scale, legend, font size, colors, figure size,
how we saved the figure, and whether the figure is matplotlib, altair, or seaborn, or
if they exist? If anything doesn't exist, return N/A. 

IMPORTANT: If no figure code is detected in the provided code (no matplotlib, seaborn, altair, plotly imports or plotting functions), 
set all fields to "NO_FIGURE_DETECTED" and return an empty array for missing code suggestion.

Also include a list of line numbers in the code (1-indexed) where figure-related code appears.
These should be lines where plotting or figure creation functions are called, such as .plot(), .scatter(), .bar(), etc.

For anything returning N/A (other than NO_FIGURE_DETECTED), provide a code snippet suggestion to fix it. If there is a suggestion for matplotlib,
refer to these rules:
<matplotlib-rules>
#Never use global matplotlib plt methods

description: Never use global matplotlib `plt` methods for figure creation or modification. Instead, always use object-oriented matplotlib interface with `fig, ax = plt.subplots()` and call methods on the figure or axes objects.

Examples:
```python
# BAD - Don't use global plt methods
import matplotlib.pyplot as plt
plt.figure()
plt.plot(x, y)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Title')
plt.show()

# GOOD - Use object-oriented interface
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_title('Title')
plt.show()  # Only global method allowed for display

# For multiple subplots
fig, axes = plt.subplots(2, 2)
axes[0, 0].plot(x, y)
axes[0, 1].scatter(x, y)
</matplotlib-rules>
If there are no code suggestions, return an empty array.
{fig_code}

"""


def call_ai(fig_qc_prompt: str, fig_code: str, model_name: str) -> bool:
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
        #   extra_headers={
        #     "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
        #     "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
        #   },
        #max_tokens=1000,
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
                        "Plot type":{
                            "type": "string",
                            "description": "the type of graph",
                            "example": "scatterplot, bar chart, line chart, etc.",
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
                            "type": "string",
                            "description": "the colors of the figure",                           
                        },
                        "figure size": {
                            "type": "string",
                            "description": "the size of the figure",
                        },
                        "saving the figure": {
                            "type": "string",
                            "description": "how to save the figure",
                        },
                        "matplotlib vs. altair vs. seaborn": {
                            "type": "string",
                            "description": "the library used to create the figure",
                        },
                        "missing code suggestion": {
                            "type": "array",
                            "description": "any code snippet suggestions to fix the figure",
                            "items": {
                                "type": "string",
                                "description": "a code snippet suggestion to fix the figure",
                            }
                        },
                        "figure line numbers": {
                            "type": "array",
                            "description": "line numbers in the code where figure generation or plotting happens",
                            "items": {
                                "type": "integer"
                            }
                        }                        
                    },
                    "required": ["Title", "Plot type", "x-axis title", "y-axis title", 'font size', 'colors', 'figure size', 'saving the figure', 'matplotlib vs. altair vs. seaborn', 'missing code suggestion', 'figure line numbers'],
                    "additionalProperties": False,
                },
            },
        },
    )

    message = completion.choices[0].message.content
    jsonform = json.loads(message)
    #keys=["Title", "x-axis title", "y-axis title", "font size", "colors", "figure size", "saving the figure", "matplotlib vs. altair"]
    #print([message.get(key) for key in keys])
    # parse the response
    """print("Title: " + jsonform["Title"])
    print("x-axis title: " + jsonform["x-axis title"])
    print("y-axis title: " + jsonform["y-axis title"])
    print("font size: " + jsonform["font size"])
    print("colors: ", jsonform["colors"])
    print("figure size: " + jsonform["figure size"])
    print("saving the figure: " + jsonform["saving the figure"])
    print("matplotlib vs. altair: " + jsonform["matplotlib vs. altair"])"""
    
    df = pd.json_normalize(jsonform)
    pd.set_option('display.max_colwidth', None)
    mydf = df.transpose()
    mydf.columns = mydf.iloc[0]
    dfnew = mydf[1:]
    lines = dfnew.to_string().splitlines()
    
    # Check if any lines contain N/A
    has_na_values = any("N/A" in line for line in lines)
    
    # Check if no figure was detected
    has_no_figure = any("NO_FIGURE_DETECTED" in line for line in lines)
    
    colored_lines = [
        (Fore.YELLOW if "NO_FIGURE_DETECTED" in line else (Fore.RED if "N/A" in line else Fore.GREEN)) + line
        for line in lines
    ]
    print("\n".join(colored_lines))
    
    # If no figure was detected, return False (no issues)
    if has_no_figure:
        return False
    
    return has_na_values, jsonform


def check_code(fig_code: str):
    return call_ai(FIG_QC_PROMPT, fig_code, model_name="deepseek/deepseek-chat-v3-0324:free")



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

def get_blame_info(file_path: Path, line_numbers: list[int]) -> set:
    authors = set()
    for line_num in set(line_numbers):
        try:
            result = subprocess.run(
                ["git", "blame", "-L", f"{line_num},{line_num}", "--line-porcelain", str(file_path)],
                stdout=subprocess.PIPE,
                text=True,
                check=True
            )
            for line in result.stdout.splitlines():
                if line.startswith("author "):
                    author = line.replace("author ", "").strip()
                    authors.add(author)
                    break
        except subprocess.CalledProcessError as e:
            print(Fore.RED + f"Failed to get git blame for line {line_num} in {file_path}: {e}")
    return authors



if __name__ == "__main__":
    changed_py_files = get_changed_py_files()
    has_failures = False
    
    if not changed_py_files:
        print("No changed .py files in the latest commit.")
    else:
        for file in changed_py_files:
            print(changed_py_files)
            print(f"\nProcessing: {file}")
            with open(file, "r") as f:
                fig_code = f.read()
                has_na, ai_response = check_code(fig_code)
                if has_na:
                    print(f"❌ Found N/A values in {file}")
                    line_numbers = ai_response.get("figure line numbers", [])
                    authors = get_blame_info(file, line_numbers)
                    if authors:
                        print(Fore.YELLOW + f"🔍 Responsible figure authors: {', '.join(authors)}")
                    else:
                        print(Fore.YELLOW + "⚠️  Could not determine figure authors.")
                else:
                    print(f"✅ No issues found in {file}")
    
    # Exit with nonzero code if there were any failures
    if has_failures:
        print("\n❌ Figure quality check failed - found N/A values that need attention")
        sys.exit(1)
    else:
        print("\n✅ All figure quality checks passed")

    # TODO: run from GitHub Actions
    # - Add the API key as a repository secret @gabe
    # - How do we inform the user about the results
    # - Extend the prompt to check for more things