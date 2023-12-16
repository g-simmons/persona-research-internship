import json
from typing import List, Union, Optional, Tuple
from pathlib import Path
import pandas as pd
from loguru import logger
import pandas as pd
from pathlib import Path

def iter_validate(iterable, validator):
    for item in iterable:
        try:
            validator(item)
        except Exception as e:
            print(validator)
            print(item)
            print(e)
            print("Validation failed")


class Catalog:
    def __init__(self, base_path: Optional[Path] = None):
        if base_path is None:
            # Set base_path to the parent directory of the directory where catalog.py is located
            base_path = Path(__file__).resolve().parent.parent
        self.base_path = base_path
        self.data_path = base_path / "data"
        self.figures_path = base_path / "figures"
        self.ipynb_path = base_path / "ipynb"
        self.logs_path = base_path / "logs"
        self.llm_edited_reviews_path = self.data_path / "llm_edited_reviews/llm_edited_reviews.jsonl"
        self.embedded_reviews_path = self.data_path / "embedded_reviews/embedded_reviews.jsonl"
    
    def load_movie_reviews_manual_edits(self) -> pd.DataFrame:
        original_data = pd.read_excel(self.data_path / "movie_reviews_manual_edits.xlsx")
        return original_data
    
    def load_original_movie_review_components(self) -> Tuple[str, str, str]:
        movie_reviews = self.load_movie_reviews_manual_edits()
        original_acting = movie_reviews.loc[1,"original_acting"]
        original_direction = movie_reviews.loc[1,"original_direction"]
        original_cinematography = movie_reviews.loc[1,"original_cinematography"]

        assert isinstance(original_acting, str)
        assert isinstance(original_direction, str)
        assert isinstance(original_cinematography, str)

        return original_acting, original_direction, original_cinematography
    
    def append_llm_edited_review(self, edited_review: dict):
        if not self.llm_edited_reviews_path.exists():
            self.llm_edited_reviews_path.parent.mkdir(parents=True, exist_ok=True)
            self.llm_edited_reviews_path.touch()

        with open(self.llm_edited_reviews_path, "a") as f:
            json.dump(edited_review, f)
            f.write("\n")
    
    def load_embedded_reviews(self):
        pass
    
    def save_embedded_reviews(self, embedded_reviews: pd.DataFrame):
        embedded_reviews.to_json(self.embedded_reviews_path, orient="records", lines=True)
        pass
