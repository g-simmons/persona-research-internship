import os
from openai import OpenAI

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
print(OPENROUTER_API_KEY)  # None


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
What are the x-axis and y-axis titles if they exist?

{fig_code}
"""


def call_ai(fig_qc_prompt: str, fig_code: str, model_name: str):
    prompt = fig_qc_prompt.format(fig_code=fig_code)
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
                    },
                    "required": ["location", "temperature", "conditions"],
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


if __name__ == "__main__":
    # get the string contents of a file

    with open("frequency_score_scatterplots.py", "r") as f:
        fig_code = f.read()


    check_code(fig_code)

    # TODO: run from GitHub Actions
    # - Add the API key as a repository secret @gabe
    # - How do we inform the user about the results
    # - Extend the prompt to check for more things
