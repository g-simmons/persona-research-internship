#TODO: Add PATH
import pandas as pd
from pathlib import Path

class Catalog:
    def __init__(self):
        # Access the root of the directory
        root = Path(__file__).parent.parent
        self.root = root
        # Data folders
        self.data = self.root / "data"
        # Python folder
        self.src = self.root / "src"
        # Figures folder
        self.figures = self.root / "figures"
        # Notebook folder
        self.ipynb = self.root / "ipynb"
        # Prompt folder - initial
        self.prompt_in = self.root / "excel_prompt/excel_input"
        # Output excel prompt file (contain embeddings)
        self.prompt_out = self.root / "excel_prompt/excel_output"

    def _process_excel(self, name) -> pd.DataFrame:
        """
        Process Excel data and return a Pandas DataFrame 

        Args:
            - name : name of Excel file in excel/excel_output folder
        """
        excel_file_path = self.prompt_out / name
        df =  pd.read_excel(excel_file_path)
        return df
