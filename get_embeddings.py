#!/usr/bin/env python3

import pandas as pd
import openai
import jsonlines
from dotenv import load_dotenv
import os
from pathlib import Path

embedding_model = "text-embedding-3-small"
embedding_encoding = "cl100k_base"
max_tokens = 8000


# loading hackaprompt dataset
df = pd.read_parquet("hf://datasets/hackaprompt/hackaprompt-dataset/hackaprompt.parquet")
df = df[["user_input", "token_count"]]
df = df.dropna()


# subsample to 1k most recent reviews and remove samples that are too long
top_n = 3
df = df.head(top_n * 2)  # first cut to first 2k entries, assuming less than half will be filtered out


# omit reviews that are too long to embed
df = df[df.token_count <= max_tokens].head(top_n)


load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text, model = embedding_model):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

df["embedding"] = df.user_input.apply(lambda x: get_embedding(x, model=embedding_model))
# df.to_csv("hackaprompts.csv")




outpath = Path("data/hackaprompt_embeddings/embeddings.jsonl")
if not outpath.parent.exists():
   outpath.parent.mkdir(exist_ok=True)

df.to_json(lines=True, orient="records", path_or_buf="data/hackaprompt_embeddings/embeddings.jsonl")
