from textblob import TextBlob
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
def get_full_prompt_sentiment(sentiment_anl,df):
    """
    Get sentiment score for full prompt
    """
    if(sentiment_anl == "vader"):
        df["full_prompt_sentiment"] = df.full_prompt.apply(vader_get_sentiment)
    elif(sentiment_anl == "textblob"):
        df["full_prompt_sentiment"] = df.full_prompt.apply(textblob_get_sentiment)
    return df
    
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
    def __init__(self, df, dim_red, sentiment_anl):
        # Assign generated embeddings from the DataFrame
        self.df = df
        self.embeddings = df.full_prompt_embedding[1:].apply(literal_eval).values
        self.embeddings = np.array(list(self.embeddings))
        self.dim_red = dim_red
        self.sentiment_anl = sentiment_anl.lower()
        # Mapping - True if category is edited, False if not
        self.df.is_acting_edited = self.df.is_acting_edited.map({"Yes": True, "No": False})
        self.df.is_cinematography_edited = self.df.is_cinematography_edited.map({"Yes": True, "No": False})
        self.df.is_direction_edited = self.df.is_direction_edited.map({"Yes": True, "No": False}) 
        # Get edited columns
        self.edited_col = self.df.apply(lambda x: get_edited_column(x), axis=1)[1:]

    def save_to_png(self, plot, file_name):
        """
        Save the figure into figures/

        Args:
        - plot: Type of plot ('pca','tsne','text blob')
        - file_name : Name of the file (based on the type of figure)
        """
        # Set output directory to figures/
        output_dir = Catalog().figures
        output_file_path = output_dir / file_name

        # Get the underlying Matplotlib Figure object from the Seaborn plot
        figure = plot.get_figure()
        plt.tight_layout()

        # Save the Matplotlib Figure to a file
        figure.savefig(output_file_path)
        print(f"Save to {output_dir}")
    def pca_figure(self):
        """
        Generate PCA figure
        """
        pca = PCA(n_components = 2)
        # Store pca embeddings in a seperate variable
        embedding_pca = pca.fit_transform(self.embeddings)
        # DataFrame for embedding_pca
        self.embedding_pca_df = pd.DataFrame(embedding_pca, columns=["PC1", "PC2"])
        # self.pca_plot = sns.scatterplot(data=self.embedding_pca_df, x="PC1", y="PC2", hue=self.edited_col)
        # # Set the title as PCA for the plot
        # self.pca_plot.set_title("PCA")
        # # Save figure to a file using save_to_png method
        # self.save_to_png(self.pca_plot,"PCA")
    
    def tsne_figure(self):
        """
        Gerenate TSNE figure
        """
        tsne = TSNE(n_components=2, perplexity=80, learning_rate=100,n_iter=1000)
        self.embedding_tsne = tsne.fit_transform(self.embeddings)
        self.embedding_tsne_df = pd.DataFrame(self.embedding_tsne, columns=["TSNE1", "TSNE2"])
        # self.tsne_plot = sns.scatterplot(data=self.embedding_tsne_df,x="TSNE1", y="TSNE2", hue=self.edited_col)
        # # Save the title as TSNE for the plot
        # self.tsne_plot.set_title("TSNE")
        # # Save figure to a file using save_to_png method
        # self.save_to_png(self.tsne_plot, "TSNE")
    def dim_red_data(self):
        """  
        Return the corresponding data_x and data_y for figure based on dim_red 
        """
        self.dim_red = self.dim_red.lower()
        if (self.dim_red == 'pca'):
            self.data_plot = self.embedding_pca_df
            self.data_x = "PC1"
            self.data_y = "PC2"
        elif (self.dim_red == 'tsne'):
            self.data_plot = self.embedding_tsne_df
            self.data_x = "TSNE1"
            self.data_y = "TSNE2"
    # Function to produce TextBlob figure with the corresponding dimensional reduction method
    def sentiment_figure(self):
        """
        Generate figure with a specified dimensional reduction and sentiment analysis method
        Parameter:
        - dim_red: Specify the type of dimensional reduction method
        """
        # Return data for dim_red
        self.dim_red_data()

        # Plot
        self.sentiment_plot = sns.scatterplot(data=self.data_plot, x=self.data_x, y=self.data_y, hue=self.edited_col, size=self.df.full_prompt_sentiment[1:])
        sns.move_legend(self.sentiment_plot, "upper left", bbox_to_anchor=(1, 1))
        self.sentiment_plot.set_title(f"{self.dim_red}_{self.sentiment_anl}")
        # Save figure to a file using save_to_png method
        self.save_to_png(self.sentiment_plot, f"{self.dim_red}_{self.sentiment_anl}")





@click.command()
@click.option("--e", type=str, help="Name of Excel file in excel_output")
@click.option("--d", type=str, help="Dimensional Reduction Method")
@click.option("--s", type=str, help="Sentimental Analysis Method")

def main(e, d, s):
    catalog = Catalog()
    # Assign FINALOUTPUT Excel file into DataFrame
    df = catalog._process_excel(e)
    # Return full prompt sentiment and store it in df DataFrame
    df = get_full_prompt_sentiment(s, df)
    # Assign figure as a variable
    figure = Figure(df=df, dim_red=d, sentiment_anl=s)
    # Get dimensional reduction data
    if d.lower() == 'pca':
        figure.pca_figure()
    elif d.lower() == 'tsne':
        figure.tsne_figure()
    # Get figure
    figure.sentiment_figure()


if __name__ == '__main__':
    main()
