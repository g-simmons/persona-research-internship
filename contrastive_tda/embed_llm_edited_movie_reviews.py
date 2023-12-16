import pandas as pd
import click
from contrastive_tda.embedding_utils import embed_df, get_embedding_method
from contrastive_tda.catalog import Catalog
from contrastive_tda.schemas import LLMEditedReview
from typing import Callable, List
from catalog import Catalog

def embed_llm_edited_movie_reviews(embedding_method: Callable, edited_reviews: List[LLMEditedReview]) -> pd.DataFrame:
    df = pd.DataFrame([review.model_dump() for review in edited_reviews])
    df["edited_review"] = df["edited_acting"] + df["edited_direction"] + df["edited_cinematography"]

    df = embed_df(df, "edited_review", embedding_method)

    return df

@click.command()
@click.option("--embedding-method", type=str, help="Embedding method (local,cohere,openai)")
def main(embedding_method):
    catalog = Catalog()
    embedding_method = get_embedding_method(embedding_method)
    movie_reviews = catalog.load_llm_edited_reviews()

    try:
        embedded_reviews = embed_llm_edited_movie_reviews(embedding_method, movie_reviews)
        click.echo(f"Finished processing")
        catalog.save_embedded_llm_edited_reviews(embedded_reviews)
    except Exception as e:
        click.echo(f"Error processing file: {str(e)}", err=True)

if __name__ == '__main__':
    main()