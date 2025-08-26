import os
from openai import OpenAI
import json

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
#print(OPENROUTER_API_KEY)  # None


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


# FIG_QC_PROMPT = """
# What are the x-axis and y-axis titles if they exist?

# {fig_code}
# """
def build_prompt(fig_code: str) -> str:
    '''
    Use the prompt structure from the file prompt_structure.txt
    '''
    with open('./prompts/prompt_structure.txt', "r", encoding="utf-8") as f:
        fig_gc_prompt = f.read()
    return fig_gc_prompt.replace("{fig_code}", fig_code)

def call_ai(fig_qc_prompt: str, fig_code: str, model_name: str):
    #prompt = fig_qc_prompt.format(fig_code=fig_code)
    prompt = fig_qc_prompt
    # read in an API key from the environment
    # how to set an environment variable in a github action?
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    # call an AI model
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    completion = client.chat.completions.create(
        #   extra_headers={
        #     "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
        #     "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
        #   },
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
                        "x-axis title": {
                            "type": "string",
                            "description": "the title of the x-axis",
                        },
                        "y-axis title": {
                            "type": "string",
                            "description": "the title of the y-axis",
                        },
                        "x-axis ticks": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "description": "the ticks on the x-axis",
                            },
                        },
                        "y-axis ticks": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "description": "the ticks on the y-axis",
                            },
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
                        "location": {
                            "type": "string",
                            "description": "the location of the figure in the code",
                        },
                    },
                    "required": ["location", "temperature", "conditions"],
                    "additionalProperties": False,
                },
            },
        },
    )

    #print(completion.choices[0].message.content)

    # Parse the response
    content = completion.choices[0].message.content.strip()
    content_dict = json.loads(content)

    # Save the response to a file
    output_path = os.makedirs("./model_output", exist_ok=True)
    out_file = os.path.join("./model_output", f"fig_qc.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(content_dict, f, indent=2, ensure_ascii=False)

    print(f"Model's response saved to {os.path.abspath(out_file)}")

    return content_dict
    

def check_code(fig_code: str) -> FigErrors:
    #call_ai(FIG_QC_PROMPT, fig_code, model_name="meta-llama/llama-4-maverick:free")
    call_ai(build_prompt(fig_code), fig_code, model_name="meta-llama/llama-4-maverick:free")


if __name__ == "__main__":
    # get the string contents of a file
    # change this so it runs all files not just a singular one
    with open("frequency_score_scatterplots.py", "r") as f:
        fig_code = f.read()


    check_code(fig_code)

    # TODO: run from GitHub Actions
    # - Add the API key as a repository secret @gabe
    # - How do we inform the user about the results
    # - Extend the prompt to check for more things
