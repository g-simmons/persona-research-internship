import cohere
import pandas as pd
from llama_index.embeddings import HuggingFaceEmbedding
from dotenv import load_dotenv
import os
from cohere.responses.embeddings import Embeddings
from typing import Callable

load_dotenv()

cohere_key = os.getenv("COHERE_API_KEY")

if cohere_key is None:
    raise ValueError("COHERE_API_KEY is not set")

co = cohere.Client(cohere_key)


def local_embedding(prompt) -> list:
    """
    Get text embeddings using HuggingFace BERT model

    Args:
        - prompt: prompt that need to be embedded
    """
    embed_model = HuggingFaceEmbedding(model_name="bert-base-uncased")
    prompt_embedding = embed_model.get_text_embedding(prompt)
    return prompt_embedding


def cohere_embedding(li) -> Embeddings:
    """
    Get text embeddings using Cohere
    """
    response = co.embed(li, model='small')
    return response

def embed_df(df: pd.DataFrame, embed_column: str, embedding_method: Callable) -> pd.DataFrame:
    df = df.copy()
    embedded_column = embed_column + "_embedding"
    df[embedded_column] = df[embed_column].apply(embedding_method)
    return df


def get_embedding_method(embedding_method: str) -> Callable:

    embedding_methods = {
        "local": local_embedding,
        "cohere": cohere_embedding,
    }
    try:
        embedding_method = embedding_methods[embedding_method]
    except KeyError:
        raise ValueError(f"Invalid embedding method: {embedding_method}")
    
    return embedding_method # type: ignore