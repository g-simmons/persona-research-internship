import pandas as pd
from pathlib import Path
from contrastive_tda.schemas import ManualEditedMovieReview, LLMEditedMovieReview
from typing import List

class Catalog:
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
    
    def load_manual_edited_movie_reviews(self,) -> List[ManualEditedMovieReview]:
        """
        Load movie reviews data
        """
        df = pd.read_excel(self.manual_movie_reviews_path)
        return [ManualEditedMovieReview(**row) for row in df.to_dict(orient="records")] # type: ignore
    
    def load_llm_edited_movie_reviews(self,as_df=False) -> List[LLMEditedMovieReview]:
        df = pd.read_json(self.llm_movie_reviews_path, lines=True)
        if as_df:
            return pd.DataFrame([LLMEditedMovieReview(**row).model_dump() for row in df.to_dict(orient="records")]) # type: ignore
        else:
            return [LLMEditedMovieReview(**row) for row in df.to_dict(orient="records")] # type: ignore
        

    def _process_excel(self, input_path: str) -> pd.DataFrame:
        """
        Process Excel data and return a Pandas DataFrame 

        Args:
            - name : name of Excel file in excel/excel_output folder
        """
        excel_file_path = self.prompt_out / input_path
        df =  pd.read_excel(excel_file_path)
        return df
