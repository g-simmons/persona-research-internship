#!/usr/bin/env python3


import altair as alt
import pandas as pd
import torch
import numpy as np
import logging
import pathlib
from utils import savefig, figname_from_fig_metadata
from .ontology_scores import (
    causal_sep_score_simple as causal_sep_score,
    hierarchy_score_simple as hierarchy_score,
    linear_rep_score_simple as linear_rep_score
)
from utils import read_olmo_model_names, sample_from_steps
# Jaxtyping imports
from jaxtyping import Float, Int

# Global paths
BIGSTORAGE_DIR = pathlib.Path("/mnt/bigstorage")

# Set up logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def load_scores(
    parameter_models: list[str],
    steps: list[str],
    model_name: str,
    score: str
) -> list[list[float]]:
    scores = []     # each element is a list of scores for a parameter model

    for parameter_model in parameter_models:
        temp_scores = []
        for step in steps:
            # These heatmaps are all multi word
            if model_name == "pythia":
                adj = torch.load(str(BIGSTORAGE_DIR / f"raymond/heatmaps-pythia/{parameter_model}/{parameter_model}-{step}-1.pt"),weights_only=False)
                cos = torch.load(str(BIGSTORAGE_DIR / f"raymond/heatmaps-pythia/{parameter_model}/{parameter_model}-{step}-2.pt"),weights_only=False)
                hier = torch.load(str(BIGSTORAGE_DIR / f"raymond/heatmaps-pythia/{parameter_model}/{parameter_model}-{step}-3.pt"),weights_only=False)
                linear = torch.load(str(BIGSTORAGE_DIR / f"raymond/heatmaps-pythia/{parameter_model}/{parameter_model}-{step}-4.pt"),weights_only=False)
            if model_name == "olmo":
                adj = torch.load(str(BIGSTORAGE_DIR / f"raymond/heatmaps-olmo/{parameter_model}/{step}-1.pt"),weights_only=False)
                cos = torch.load(str(BIGSTORAGE_DIR / f"raymond/heatmaps-olmo/{parameter_model}/{step}-2.pt"),weights_only=False)
                hier = torch.load(str(BIGSTORAGE_DIR / f"raymond/heatmaps-olmo/{parameter_model}/{step}-3.pt"),weights_only=False)

            if score == "causal_sep":
                temp_scores.append(causal_sep_score(adj, cos))
            elif score == "hierarchy":
                temp_scores.append(hierarchy_score(hier))
            elif score == "linear":
                temp_scores.append(linear_rep_score(linear))

        scores.append(temp_scores)

    return scores

def save_plot(score: str, output_dir: str, model_name: str, parameter_models, steps) -> None:

    if score == "causal_sep":
        title = "Causal Separability Scores Multi Word"
        y_title = "causal-sep-score"
    elif score == "hierarchy":
        title = "Hierarchy Scores Multi Word"
        y_title = "hierarchy-score"
    elif score == "linear":
        title = "Linear Representation Scores Multi Word"
        y_title = "linear-rep-score"
    
    scores = load_scores(parameter_models, steps, model_name, score)

    # Convert scores to numpy array and transpose
    new_scores = np.array(scores).T

    if model_name == "pythia":
        steps_nums = [int(step.split("p")[1]) for step in steps]
        df = pd.DataFrame(new_scores, columns=parameter_models, index=pd.Index(steps_nums, name="Step"))
        df = df.reset_index().melt("Step", var_name="Model Size", value_name="Score")

    if model_name == "olmo":
        steps_nums = [int(step.split('-')[0].split('p')[1]) for step in steps]
        df = pd.DataFrame(new_scores, 
                         columns=parameter_models, 
                         index=pd.Index(steps_nums, name="Step"))
        logger.info(df)
        df = df.reset_index().melt("Step", var_name="Model Size", value_name="Score")
        logger.info(df)

    nearest = alt.selection_point(nearest=True, on="pointerover",
                                fields=["Step"], empty=False)
    vis_idea = {
        "x": "Step:Q",
        "y": "Score:Q",
        "color": alt.Color("Model Size:N", sort=parameter_models),
        "tooltip": ["Step", "Score", "Model Size"]
    }
    vis_ideas = [vis_idea]

    for vis_idea in vis_ideas:

        line = alt.Chart(df).mark_line(interpolate="linear").encode(**vis_idea).interactive()

        # Transparent selectors across the chart. This is what tells us
        # the x-value of the cursor
        selectors = alt.Chart(df).mark_point().encode(
            # TODO collect these args into a dict
            x=alt.X('Step:Q', title='Steps', scale=alt.Scale(nice=False)),
            opacity=alt.value(0),
        ).add_params(
            nearest
        )
        when_near = alt.when(nearest)

        # Draw points on the line, and highlight based on selection
        points = line.mark_point().encode(
            opacity=when_near.then(alt.value(1)).otherwise(alt.value(0))
        )

        # Draw text labels near the points, and highlight based on selection
        text = line.mark_text(align="left", dx=5, dy=-5).encode(
            text=when_near.then("Score:Q").otherwise(alt.value(" "))
        )

        # Draw a rule at the location of the selection
        rules = alt.Chart(df).mark_rule(color="gray").encode(
            x=alt.X('Step:Q', title='Steps', scale=alt.Scale(nice=False)),
        ).transform_filter(
            nearest
        )
        
        properties_dict = {
            'width': 400,
            'height': 300,
            'title': title
        }

        # Put the five layers into a chart and bind the data
        final_chart = alt.layer(
            line, selectors, points, rules, text
        ).properties(**properties_dict).interactive()

        metadata = {
            "score": score,
            "title": title,
            "y_title": y_title
        }
        figure_name = figname_from_fig_metadata(metadata)

        savefig(
            fig=final_chart,
            figures_dir=output_dir,
            figure_name=figure_name,
            formats=["png", "html"],
            data=df
        )

def main():
    model_name = "olmo"
    script_dir = pathlib.Path(__file__).parent
    figures_dir = script_dir.parent / "figures"

    if model_name == "pythia":
        # stuff
        steps = [f"step{i}" for i in range(1000, 145000, 2000)]
        parameter_models = ["70M", "160M", "1.4B", "2.8B", "12B"]

        pythia_dir = figures_dir / "model_score_plots_pythia_multi"
        plot1 = save_plot("causal_sep", str(pythia_dir / "causal_sep_scores"), model_name, parameter_models, steps)
        plot2 = save_plot("hierarchy", str(pythia_dir / "hierarchy_scores"), model_name, parameter_models, steps)
        plot3 = save_plot("linear", str(pythia_dir / "linear_rep_scores"), model_name, parameter_models, steps)

        combined = alt.hconcat(plot1, plot2, plot3)
        combined.save(str(pythia_dir / "combined_scores.html"))
        combined.save(str(pythia_dir / "combined_scores.png"))

    if model_name == "olmo":
        # stuff
        data_path = script_dir.parent / "data" / "olmo_7B_model_names.txt"
        steps = read_olmo_model_names()
        logger.info(f"Number of steps: {len(steps)}")
        newsteps = sample_from_steps(steps)
        logger.info(f"Selected steps: {newsteps}")
        parameter_models = ["7B"]

        # saving plots
        olmo_dir = figures_dir / "model_score_plots_olmo_multi"
        plot1 = save_plot("causal_sep", str(olmo_dir / "causal_sep_scores"), model_name, parameter_models, newsteps)
        plot2 = save_plot("hierarchy", str(olmo_dir / "hierarchy_scores"), model_name, parameter_models, newsteps)
        # plot3 = save_plot("linear", str(olmo_dir / "linear_rep_scores"), model_name, parameter_models, newsteps)

        combined = alt.hconcat(plot1, plot2)
        combined.save(str(olmo_dir / "combined_scores.html"))
        combined.save(str(olmo_dir / "combined_scores.png"))

if __name__ == "__main__":
    main()