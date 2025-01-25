from textblob import TextBlob
import vader
import click
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import seaborn as sns
from catalog import Catalog
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import PCA # Principal component Analysis
from sklearn.manifold import TSNE # t-Stochastic Neighbor Embedding

    
def textblob_get_sentiment(x) -> float:
    """
    Using textblob, return sentiment score of x.

    Args:
        - x (str): prompt
    """
    return TextBlob(x).sentiment.polarity # type: ignore

def vader_get_sentiment(x) -> float:
    """
    Using Vader, return sentiment score of x.
    Args:
         - x (str): prompt
    """
    # Create a SentimentIntensityAnalyzer Object
    sid_obj = SentimentIntensityAnalyzer()
    return sid_obj.polarity_scores(x)["compound"]

def get_full_prompt_sentiment(sentiment_anl: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Get sentiment score for full prompt. Adds a column `full_prompt_sentiment` to the DataFrame.
    """
    if(sentiment_anl == "vader"):
        df["full_prompt_sentiment"] = df.full_prompt.apply(vader_get_sentiment)
    elif(sentiment_anl == "textblob"):
        df["full_prompt_sentiment"] = df.full_prompt.apply(textblob_get_sentiment)
    return df
    
def _get_row_edited_column(row: pd.Series) -> str:
    acting_edited = direction_edited = cinematography_edited = False
    if row["original_direction"] != row["edited_direction"]:
        direction_edited = True
    if row["original_acting"] != row["edited_acting"]:
        acting_edited = True
    if row["original_cinematography"] != row["edited_cinematography"]:
        cinematography_edited = True
    
    assert direction_edited or acting_edited or cinematography_edited, "No columns were edited"
    assert sum([direction_edited, acting_edited, cinematography_edited]) == 1, "More than one column was edited"

    if acting_edited:
        return "acting"
    elif direction_edited:
        return "direction"
    else:
        return "cinematography"
        
    

def get_edited_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the edited part of the original prompt
    """
    df["edited_col"] = df.apply(lambda x: _get_row_edited_column(x), axis=1)
    return df


def make_pca_figure(df: pd.DataFrame) -> Axes:
    pca = PCA(n_components = 2)
    embedding_pca = pca.fit_transform(np.array(df.full_prompt_embedding.tolist()))
    embedding_pca_df = pd.DataFrame(embedding_pca, columns=["PC1", "PC2"])
    pca_plot = sns.scatterplot(data=embedding_pca_df, x="PC1", y="PC2", hue=df.edited_col,sizes=df.full_prompt_sentiment)
    return pca_plot

def make_tsne_figure(df: pd.DataFrame) -> Axes:
    tsne = TSNE(n_components=2, perplexity=80, learning_rate=100,n_iter=1000)
    embedding_tsne = tsne.fit_transform(np.array(df.full_prompt_embedding.tolist()))
    embedding_tsne_df = pd.DataFrame(embedding_tsne, columns=["TSNE1", "TSNE2"])
    tsne_plot = sns.scatterplot(data=embedding_tsne_df,x="TSNE1", y="TSNE2", hue=df.edited_col,sizes=df.full_prompt_sentiment)
    return tsne_plot

@click.command()
# @click.option("--data", type=str, help="Path to jsonl file")
# @click.option("--dim-reduce",type=click.Choice(['pca','tsne']),help="Dimensional Reduction Method", required=False)
# @click.option("--sent-analysis",type=click.Choice(['vader','textblob']),help="Sentimental Analysis Method", required=False)
def main():
    catalog = Catalog()
    df = catalog.load_llm_edited_movie_reviews_embedded(as_df=True)
    df = get_edited_column(df)
    assert isinstance(df, pd.DataFrame)
    for sent_analysis in ["vader", "textblob"]:
        df_ = df.copy()
        df_ = get_full_prompt_sentiment(sent_analysis, df_) 
        for dim_reduce_name, figure_method in zip(["pca", "tsne"], [make_pca_figure, make_tsne_figure]):
            fig = figure_method(df_)
            sns.move_legend(fig, "upper left", bbox_to_anchor=(1, 1))
            fig.set_title(f"{dim_reduce_name}_{sent_analysis}")
            catalog.savefig(fig, f"llm_edited_movie_reviews_dim_reduced/{dim_reduce_name}_{sent_analysis}")

    # figure.sentiment_figure()


if __name__ == '__main__':
    main()
