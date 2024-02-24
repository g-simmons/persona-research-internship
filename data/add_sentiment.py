import pandas as pd
from textblob import TextBlob

# TODO: use catalog for loading the file
data = pd.read_json("./llm_edited_reviews_embedded.jsonl",lines=True) #lines is for jsonl

def textblob_get_sentiment(x: str) -> float:
    """
    Using textblob, return sentiment score of x.

    Args:
        - x (str): prompt
    """
    return TextBlob(x).sentiment.polarity # type: ignore

data["full_prompt_sentiment"] = data["full_prompt"].apply(textblob_get_sentiment)

# TODO: use catalog for saving the file
# validate with schema

data.to_json("../llm_edited_reviews_embedded_with_sentiment/llm_edited_reviews_embedded_with_sentiment.jsonl",lines=True,orient="records")

# import sys

# print(sys.prefix)