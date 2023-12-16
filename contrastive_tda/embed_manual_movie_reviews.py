import pandas as pd
import click
from contrastive_tda.embedding_utils import embed_df, get_embedding_method
from catalog import Catalog
from typing import Callable

def embed_movie_reviews(embedding_method: Callable, df: pd.DataFrame) -> pd.DataFrame:
    df = embed_df(df, "full_prompt", embedding_method)

    direction_edits = df.iloc[:33]
    acting_edits = df.iloc[33:67]
    cinematography_edits = df.iloc[67:]

    for key, data in [
        ("direction", direction_edits),
        ("acting", acting_edits),
        ("cinematography", cinematography_edits),
    ]:
        data = embed_df(data, f"edited_{key}", embedding_method)
        df.update(data)

    return df

@click.command()
@click.option("--em", type=str, help="Embedding method (local,cohere,openai)")
def main(embedding_method):
    catalog = Catalog()
    embedding_method = get_embedding_method(embedding_method)
    movie_reviews = catalog.load_movie_reviews_manual_edits()

    try:
        embedded_reviews = embed_movie_reviews(embedding_method, movie_reviews)
        click.echo(f"Finished processing")
        catalog.save_embedded_manual_edited_reviews(embedded_reviews)
    except Exception as e:
        click.echo(f"Error processing file: {str(e)}", err=True)

if __name__ == '__main__':
    main()