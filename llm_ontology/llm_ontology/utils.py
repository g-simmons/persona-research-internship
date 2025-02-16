from pathlib import Path
import pandas as pd

import pandas as pd

from loguru import logger

from typing import List, Callable, Optional
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

from pathlib import Path
from seaborn.axisgrid import FacetGrid

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
    fig: Figure or SubFigure or FacetGrid or Axes,
    figures_dir: str or Path,
    figure_name: str,
    formats: list[str] = ["svg", "png"],
    dpi=600,
    bbox_inches="tight",
    overwrite=True,
    data: Optional[pd.DataFrame] = None,
    tight_layout=True,
):
    """
    This function saves a figure in specified formats.

    Parameters:
    fig (Figure or SubFigure or FacetGrid or Axes): The figure to be saved.
    figures_dir (str or Path): The directory where the figure will be saved.
    figure_name (str): The name of the figure file.
    formats (list[str], optional): The formats in which the figure will be saved. Defaults to ["svg", "png"].
    dpi (int, optional): The resolution in dots per inch. Defaults to 600.
    bbox_inches (str, optional): The bounding box in inches. Defaults to "tight".
    overwrite (bool, optional): If True, overwrite the existing file. Defaults to True.
    data (pd.DataFrame, optional): The data to be saved along with the figure. Defaults to None.
    tight_layout (bool, optional): If True, use tight layout. Defaults to True.
    """
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

    figures_dir = Path(figures_dir)
    outfolder = figures_dir / figure_name

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