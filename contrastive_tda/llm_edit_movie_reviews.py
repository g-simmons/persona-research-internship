import json
from typing import Tuple
import openai
from contrastive_tda.schemas import TargetSentimentLevel, LLMEditedReview
from itertools import product
from contrastive_tda.catalog import Catalog
from loguru import logger
from dotenv import load_dotenv

cat = Catalog()
logs_path = cat.logs_path
logger.add(logs_path / "gpt_edit_movie_reviews.log", format="{time} {level} {message}", level="DEBUG")

load_dotenv()

PROMPT_TEMPLATE = """

Please edit the following movie review to match the target sentiment level. 

The movie review consists of three components: 
- the first component describes the acting in the movie
- the second component describes the direction in the movie
- the third component describes the cinematography in the movie

Please ONLY EDIT the {edited_component} component of the movie review.

Here are the movie review components:

ACTING:
```
{acting}
```

DIRECTION:
```
{direction}
```

CINEMATOGRAPHY:
```
{cinematography}
```

Please edit only the {edited_component} component of the movie review to match the target sentiment level. 

Sentiment levels are defined as follows:

SENTIMENT LEVELS:
```
-1: Extremely negative
-0.75: Very negative
-0.5: Negative
-0.25: Slightly negative
0: Neutral
+0.25: Slightly positive
+0.5: Positive
+0.75: Very positive
+1: Extremely positive
```

TARGET SENTIMENT LEVEL: {target_sentiment_level}

"""

def get_edited_review(
    original_acting: str,
    original_direction: str,
    original_cinematography: str,
    edited_component: str,
    target_sentiment_level: TargetSentimentLevel,
) -> Tuple[str, str]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "submit_edited_component",
                "description": "Submit an edited version of a component of the movie review",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "edited_text": {
                            "type": "string",
                            "description": "The edited text of the component of the movie review",
                        },
                        "edited_component": {
                            "type": "string",
                            "enum": ["acting", "direction", "cinematography"],
                        },
                    },
                    "required": ["edited_text", "edited_component"],
                },
            },
        }
    ]
    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(
                    edited_component=edited_component,
                    acting=original_acting,
                    direction=original_direction,
                    cinematography=original_cinematography,
                    target_sentiment_level=target_sentiment_level,
                ),
            },
        ],
        max_tokens = max([len(original_acting) + len(original_direction) + len(original_cinematography)]) + 25,
        tools=tools,  # type: ignore
        tool_choice={
            "type": "function",
            "function": {"name": "submit_edited_component"},
        },
    )

    tool_calls = completion.choices[0].message.tool_calls[0].function.arguments  # type: ignore
    tool_calls = json.loads(tool_calls)
    edited_component = tool_calls["edited_component"]
    edited_text = tool_calls["edited_text"]

    return edited_component, edited_text


def main():
    catalog = Catalog()
    original_acting, original_direction, original_cinematography = catalog.load_original_movie_review_components()

    replicates = range(100)

    # erik suggested 10k examples
    # we have 9 sentiment levels, 3 components
    # so want roughly 10k/27 = 370 examples per sentiment level per component
    # lets try 10 replicates for now, plot it, then scale up if it looks like it's working.

    components = ["acting", "direction", "cinematography"]

    for target_sentiment_level, target_component, replicate in product(TargetSentimentLevel, components, replicates):
        logger.info(f"Generating examples for sentiment {target_sentiment_level.value}, component {target_component}, example {replicate}")

        edited_component, edited_text = get_edited_review(
            original_acting=original_acting,
            original_direction=original_direction,
            original_cinematography=original_cinematography,
            edited_component=target_component,
            target_sentiment_level=target_sentiment_level,
        )

        output = LLMEditedReview(
            original_acting=original_acting,
            original_direction=original_direction,
            original_cinematography=original_cinematography,
            edited_component=edited_component,
            edited_acting=edited_text if edited_component == "acting" else original_acting,
            edited_direction=edited_text if edited_component == "direction" else original_direction,
            edited_cinematography=edited_text if edited_component == "cinematography" else original_cinematography,
            target_sentiment_level=target_sentiment_level.value,
            edit_model_name="gpt-4",
        )
        logger.debug(output)
        catalog.append_llm_edited_review(output)


if __name__ == "__main__":
    main()
