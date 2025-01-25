import pandas as pd
from pathlib import Path
from contrastive_tda.schemas import ManualEditedMovieReview, LLMEditedMovieReview, LLMEditedMovieReviewEmbedded
from typing import List, Union
import json
from pathlib import Path
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.transforms import Bbox
from seaborn.axisgrid import FacetGrid
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Tuple, Optional, Literal, Union, Dict, Any

from itertools import combinations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.text import Annotation
from matplotlib.transforms import Transform
import seaborn as sns



# kedro

# pd.read_csv()
# fix the columns
# from catalog import Catalog
# cat = Catalog()
# cat.load("file")
class Catalog: # job is to load and save data
    def __init__(self):
        root = Path(__file__).parent.parent
        self.root = root
        self.data = self.root / "data"
        self.src = self.root / "src"
        self.figures = self.root / "figures"
        self.ipynb = self.root / "ipynb"
        self.prompt_in = self.root / "excel_prompt/excel_input"
        self.prompt_out = self.root / "excel_prompt/excel_output"
        self.manual_movie_reviews_path = self.data / "manual_edited_movie_reviews/Movie_Reviews_Manual_Edits.xlsx"
        self.llm_movie_reviews_path = self.data / "llm_edited_reviews/llm_edited_reviews.jsonl"
        self.llm_movie_reviews_embedded_path = self.data / "llm_edited_reviews_embedded/llm_edited_reviews_embedded.jsonl"
    
    def load_manual_edited_movie_reviews(self,) -> List[ManualEditedMovieReview]:
        """
        Load movie reviews data
        """
        df = pd.read_excel(self.manual_movie_reviews_path)
        return [ManualEditedMovieReview(**row) for row in df.to_dict(orient="records")] # type: ignore
    
    def load_llm_edited_movie_reviews(self,as_df=False) -> Union[List[LLMEditedMovieReview],pd.DataFrame]:
        """
        as_df: if True, return a Pandas DataFrame, else return a list of LLMEditedMovieReview objects
        """
        df = pd.read_json(self.llm_movie_reviews_path, lines=True)
        df["full_prompt"] = df.edited_direction + ' ' + df.edited_acting + ' ' + df.edited_cinematography
        if as_df:
            return pd.DataFrame([LLMEditedMovieReview(**row).model_dump() for row in df.to_dict(orient="records")]) # type: ignore
        else:
            return [LLMEditedMovieReview(**row) for row in df.to_dict(orient="records")] # type: ignore
    
    def save_llm_edited_movie_reviews_embedded(self, data: Union[pd.DataFrame,List[LLMEditedMovieReviewEmbedded]]) -> None:
        if not self.llm_movie_reviews_embedded_path.parent.exists():
            self.llm_movie_reviews_embedded_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, pd.DataFrame):
            # data validation
            df.apply(lambda x: LLMEditedMovieReviewEmbedded(**x), axis=1)
            df.to_json(self.llm_movie_reviews_embedded_path, orient="records", lines=True)
        else:
            # data validation
            [LLMEditedMovieReviewEmbedded(**row.model_dump()) for row in data]
            with open(self.llm_movie_reviews_embedded_path, 'w') as f:
                for row in data:
                    json.dump(row.model_dump(), f)
                    f.write('\n')
    
    def load_llm_edited_movie_reviews_embedded(self, as_df=False) -> Union[List[LLMEditedMovieReviewEmbedded],pd.DataFrame]:
        """
        as_df: if True, return a Pandas DataFrame, else return a list of LLMEditedMovieReviewEmbedded objects
        """
        df = pd.read_json(self.llm_movie_reviews_embedded_path, lines=True)
        items = [LLMEditedMovieReviewEmbedded(**row) for row in df.to_dict(orient="records")] # type: ignore
        return pd.DataFrame([i.model_dump() for i in items]) if as_df else items

        

    def _process_excel(self, input_path: str) -> pd.DataFrame:
        """
        Process Excel data and return a Pandas DataFrame 

        Args:
            - name : name of Excel file in excel/excel_output folder
        """
        excel_file_path = self.prompt_out / input_path
        df =  pd.read_excel(excel_file_path)
        return df
    
    def savefig(
        self,
        fig,
        figure_name,
        formats: List[str] = ["svg", "png"],
        dpi=600,
        bbox_inches="tight",
        overwrite=True,
        # data=pd.DataFrame,
    ):
        while not isinstance(fig, Figure):
            if isinstance(fig, SubFigure):
                fig = fig.figure
            elif isinstance(fig, FacetGrid):
                fig = fig.figure
            elif isinstance(fig, Axes):
                fig = fig.get_figure()

        fig.tight_layout()

        outfolder = Path(self.figures / figure_name)
        if not outfolder.exists() or overwrite:
            for format in formats:
                outpath = outfolder / f"{Path(figure_name).stem}.{format}"
                if not outpath.parent.exists():
                    outpath.parent.mkdir(exist_ok=True, parents=True)
                fig.savefig(outpath, format=format, bbox_inches=bbox_inches, dpi=dpi)

