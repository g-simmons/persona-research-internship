#!/usr/bin/env python3

import pandas as pd
import openai
import jsonlines
from dotenv import load_dotenv
import os
from pathlib import Path

embedding_model = "text-embedding-3-small"
max_tokens = 8000


def get_dataframe_embeddings(model: str, max_tokens: int, subsample_size: int) -> pd.DataFrame:
    # loading hackaprompt dataset
    df = pd.read_parquet("hf://datasets/hackaprompt/hackaprompt-dataset/hackaprompt.parquet")
    df = df[["user_input", "token_count"]]
    df = df.dropna()


    # subsample to 1k most recent reviews and remove samples that are too long
    df = df.head(subsample_size * 2)  # first cut to first 2k entries, assuming less than half will be filtered out


    # omit reviews that are too long to embed
    df = df[df.token_count <= max_tokens].head(subsample_size)


    load_dotenv()
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_embedding(text, model = model):
        text = text.replace("\n", " ")
        return client.embeddings.create(input = [text], model=model).data[0].embedding

    df["embedding"] = df.user_input.apply(lambda x: get_embedding(x, model=model))

    return df



def make_json(df: pd.DataFrame):
    outpath = Path("data/hackaprompt_embeddings/embeddings.jsonl")
    if not outpath.parent.exists():
        outpath.parent.mkdir(exist_ok=True)

    df.to_json(lines=True, orient="records", path_or_buf="data/hackaprompt_embeddings/embeddings.jsonl")


df = get_dataframe_embeddings(embedding_model, max_tokens, 2)
make_json(df)