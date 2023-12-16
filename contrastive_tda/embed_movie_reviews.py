import cohere
import pandas as pd
import click
# from schemas import expected_input_schema, expected_output_schema
from llama_index.embeddings import HuggingFaceEmbedding
from catalog import Catalog
from dotenv import load_dotenv
import os
from cohere.responses.embeddings import Embeddings
from typing import Callable

load_dotenv()

cohere_key = os.getenv("COHERE_API_KEY")

if cohere_key is None:
    raise ValueError("COHERE_API_KEY is not set")

co = cohere.Client(cohere_key)


def process_data(file_name) -> pd.DataFrame:
    """
    Process an Excel file into a Pandas DataFrame

    Args:
        file_path (str): Path to the Excel file.
    """
    df = pd.read_excel(file_name)
    return df


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

def _embed_df(df: pd.DataFrame, embed_column: str, embedding_method: Callable) -> pd.DataFrame:
    df = df.copy()
    embedded_column = embed_column + "_embedding"
    df[embedded_column] = df[embed_column].apply(embedding_method)
    return df

def embed_movie_reviews(embedding_method: Callable, df: pd.DataFrame) -> pd.DataFrame:
    """
    Store prompt embedding data into a new Excel file

    Args:
     - embedding_method: Local/Cloud/OpenAI Embedding
    """
    df = _embed_df(df, "full_prompt", embedding_method)

    direction_edits = df.iloc[:33]
    acting_edits = df.iloc[33:67]
    cinematography_edits = df.iloc[67:]

    for key, data in [
        ("direction", direction_edits),
        ("acting", acting_edits),
        ("cinematography", cinematography_edits),
    ]:
        data = _embed_df(data, f"edited_{key}", embedding_method)
        df.update(data)

    return df

@click.command()
@click.option("--em", type=str, help="Embedding method (local,cohere,openai)")
def main(embedding_method):
    catalog = Catalog()

    embedding_methods = {
        "local": local_embedding,
        "cohere": cohere_embedding,
    }
    try:
        embedding_method = embedding_methods[embedding_method]
    except KeyError:
        raise ValueError(f"Invalid embedding method: {embedding_method}")

    try:
        # Return a DataFrame after processing data
        movie_reviews = catalog.load_movie_reviews_manual_edits()
        embedded_reviews = embed_movie_reviews(embedding_method, movie_reviews)
        catalog.save_embedded_reviews(embedded_reviews)
        click.echo(f"Finished processing")
    except Exception as e:
        click.echo(f"Error processing file: {str(e)}", err=True)


if __name__ == '__main__':
    main()