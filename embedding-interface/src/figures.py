import textblob
import vader
import click
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from catalog import Catalog
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ast import literal_eval
from sklearn.decomposition import PCA # Principal component Analysis
from sklearn.manifold import TSNE # t-Stochastic Neighbor Embedding

class Sentiment:
    def __init__(self, sentiment_anl, df):
        self.sentiment_anl = sentiment_anl.lower()
        self.df = df
    def textblob_get_sentiment(x) -> float:
        """
        Using textblob, return sentiment score of x.

        Args:
            - x (str): prompt
        """
        return TextBlob(x).sentiment.polarity

    def vader_get_sentiment(x) -> float:
        """
        Using Vader, return sentiment score of x.
        Args:
            - x (str): prompt
        """
        # Create a SentimentIntensityAnalyzer Object
        sid_obj = SentimentIntensityAnalyzer()
        return sid_obj.polarity_scores(x)["compound"]
    def get_full_prompt_sentiment(self):
        """
        Get sentiment score for full prompt
        """
        if(self.sentiment_anl == "vader"):
            self.df["full_prompt_sentiment"] = self.df.full_prompt.apply(self.vader_get_sentiment)
        elif(self.sentiment_anl == "textblob"):
            self.df["full_prompt_sentiment"] = self.df.full_prompt.apply(self.textblob_get_sentiment)
        return self.df
    
def get_edited_column(x: pd.Series):
    """"
    Return the edited part of the original prompt

    """
    if x["is_acting_edited"]:
        return "acting"
    elif x["is_direction_edited"]:
        return "direction"
    elif x["is_cinematography_edited"]:
        return "cinematography"

#TODO: Import Figure function from notebook and parameterize accordingly    
class Figure:
    def __init__(self, embeddings, df, dim_red):
        # Assign generated embeddings from the DataFrame
        self.embeddings = np.array(df.full_prompt_embedding[1:].apply(literal_eval).values)





@click.command()
@click.option("--e", type=str, help="Name of Excel file in excel_output")
@click.option("--em", type=str, help="Embedding method")
@click.option("--d", type=str, help="Dimensional Reduction Method")
@click.option("--s", type=str, help="Sentimental Analysis Method")
def main(e, em, d, s):
    catalog = Catalog()
    # Assign FINALOUTPUT Excel file into DataFrame
    df = catalog._process_excel()
    sentiment = Sentiment(df, s)
    # Return full prompt sentiment and store it in df DataFrame
    df = sentiment.get_full_prompt_sentiment()


if __name__ == main:
    main()
