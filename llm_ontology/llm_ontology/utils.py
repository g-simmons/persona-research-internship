from pathlib import Path
import pandas as pd

import pandas as pd

from typing import List, Callable, Optional, Union, Literal
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from seaborn.axisgrid import FacetGrid
import altair as alt
import logging

# Jaxtyping imports
from jaxtyping import Float, Int
import torch
import numpy as np

# Set up logger
logger = logging.getLogger(__name__)

# Create handlers
file_handler = logging.FileHandler('utils.log')
stdout_handler = logging.StreamHandler()

# Create formatters and add it to handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

# Set log level
logger.setLevel(logging.INFO)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"


def figname_from_fig_metadata(metadata: dict) -> str:
    """
    This function generates a string that represents the name of a figure based on its metadata.

    Parameters:
    metadata (dict): A dictionary containing the metadata of the figure.

    Returns:
    str: A string that represents the name of the figure. The string is formed by joining the key-value pairs in the metadata dictionary, separated by an underscore.
    """
    name = "_".join([f"{key[0:3]}={value}" for key, value in metadata.items()])
    path = "/".join([f"{key[0:3]}={value}" for key, value in metadata.items()])
    return path + "/" + name

def savefig(
    fig: Union[Figure, SubFigure, FacetGrid, Axes, alt.Chart, alt.LayerChart],
    figures_dir: Union[str, Path],
    figure_name: str,
    formats: list[str] = ["svg", "png"],
    dpi=600,
    bbox_inches="tight",
    overwrite=True,
    data: Optional[pd.DataFrame] = None,
    tight_layout=True,
) -> None:
    """
    This function saves a figure in specified formats. It supports both matplotlib and altair charts.

    Parameters:
    fig (Union[Figure, SubFigure, FacetGrid, Axes, alt.Chart]): The figure to be saved.
    figures_dir (Union[str, Path]): The directory where the figure will be saved.
    figure_name (str): The name of the figure file.
    formats (list[str], optional): The formats in which the figure will be saved. Defaults to ["svg", "png"].
    dpi (int, optional): The resolution in dots per inch. Defaults to 600.
    bbox_inches (str, optional): The bounding box in inches. Defaults to "tight".
    overwrite (bool, optional): If True, overwrite the existing file. Defaults to True.
    data (pd.DataFrame, optional): The data to be saved along with the figure. Defaults to None.
    tight_layout (bool, optional): If True, use tight layout. Defaults to True.
    """
    figures_dir = Path(figures_dir)
    outfolder = figures_dir / figure_name

    if isinstance(fig, alt.Chart) or isinstance(fig, alt.LayerChart):
        # Handle Altair charts
        if not outfolder.exists() or overwrite:
            logger.info(f"Writing to {outfolder}")
            for format in formats:
                outpath = outfolder / f"{(figures_dir/figure_name).stem}.{format}"

                if not outpath.parent.exists():
                    outpath.parent.mkdir(exist_ok=True, parents=True)

                logger.info(f"Saving figure {outpath}")
                # Altair only supports specific formats
                if format in ['json', 'html', 'png', 'svg', 'pdf']:
                    fig.save(str(outpath))
                else:
                    logger.warning(f"Format {format} not supported for Altair charts. Skipping.")
    else:
        # Handle matplotlib-based figures
        assert isinstance(fig, (Figure, SubFigure, FacetGrid, Axes))
        while not isinstance(fig, Figure):
            if isinstance(fig, SubFigure):
                fig = fig.figure
            elif isinstance(fig, FacetGrid):
                fig = fig.figure
            elif isinstance(fig, Axes):
                fig = fig.get_figure()

        if tight_layout:
            fig.tight_layout()

        if not outfolder.exists() or overwrite:
            logger.info(f"Writing to {outfolder}")
            for format in formats:
                outpath = outfolder / f"{(figures_dir/figure_name).stem}.{format}"

                if not outpath.parent.exists():
                    outpath.parent.mkdir(exist_ok=True, parents=True)

                logger.info(f"Saving figure {outpath}")
                fig.savefig(str(outpath), format=format, bbox_inches=bbox_inches, dpi=dpi)

    if data is not None:
        logger.info(f"Saving data to {outfolder}")
        data.to_csv(outfolder / f"{Path(figure_name).stem}.csv")

def read_olmo_model_names() -> list[str]:
    """
    Read the model names from the olmo_7B_model_names.txt file.
    """
    with open(DATA_DIR / "olmo_7B_model_names.txt", "r") as a:
        steps = a.readlines()

    steps = list(map(lambda x: x[:-1], steps))
    steps.sort(key=lambda x: int(x.split("-")[0].split("p")[1]))

    return steps

def sample_from_steps(steps: list[str]) -> list[str]:
    """
    Sample every 15th step from the list of steps.
    """
    newsteps = []
    for i in range(len(steps)):
        if i % 15 == 0:
            newsteps.append(steps[i])

    return newsteps