#!/usr/bin/env python3

import altair as alt
import pandas as pd
import torch
import numpy as np
import logging
import pathlib
import os

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

# SCORES
def causal_sep_score(adj_mat: np.ndarray, cos_mat: np.ndarray) -> float:
    size = cos_mat.shape

    # 0_diag Hadamard product equivalent
    for i in range(size[0]):
        cos_mat[i][i] = 0

    new_mat = cos_mat - adj_mat

    # Frobenius norm
    return np.linalg.norm(new_mat, ord = "fro")

def hierarchy_score(cos_mat: np.ndarray) -> float:
    size = cos_mat.shape

    # 0_diag Hadamard product equivalent
    for i in range(size[0]):
        cos_mat[i][i] = 0

    # Frobenius norm
    return np.linalg.norm(cos_mat, ord = "fro")

def linear_rep_score(values: np.ndarray) -> float:
    sum = 0
    for i in range(len(values)):
        sum += values[i].item()
    return sum / len(values)


def save_plot(score: str, output_dir: str, model_name: str, parameter_models, steps):

    if score == "causal_sep":
        title = "Causal Separability Scores Multi Word"
        y_title = "causal-sep-score"
    elif score == "hierarchy":
        title = "Hierarchy Scores Multi Word"
        y_title = "hierarchy-score"
    elif score == "linear":
        title = "Linear Representation Scores Multi Word"
        y_title = "linear-rep-score"

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

    new_scores = []  # each element is a list of scores for a step, formatted for df.DataFrame
    for i in range(len(steps)):
        new_scores.append([score_list[i] for score_list in scores])

    if model_name == "pythia":
        # df = pd.DataFrame(new_scores, columns=parameter_models, index=pd.RangeIndex(start = 1000, stop = 145000, step = 2000, name="Step"))
        steps_nums = [int(step.split("p")[1]) for step in steps]
        df = pd.DataFrame(new_scores, columns=parameter_models, index=pd.Index(steps_nums, name="Step"))
        df = df.reset_index().melt("Step", var_name="Model Size", value_name="Score")

    if model_name == "olmo":
        steps_nums = [int(step.split('-')[0].split('p')[1]) for step in steps]
        df = pd.DataFrame(new_scores, columns=parameter_models, index=pd.Index(steps_nums, name="Step"))
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

    line = alt.Chart(df).mark_line(interpolate="linear").encode(**vis_idea).interactive()
    #TODO parameterize the visualization ideas

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

    # TODO extend savefig from https://github.com/g-simmons/persona-research-internship/issues/230 function to handle altair charts
    # TODO get a filename from get_figname_from_fig_metadata
    # TODO call savefig with the chart and filename
    final_chart.save(f'{output_dir}.png')
    final_chart.save(f'{output_dir}.html')
    
    return final_chart


model_name = "pythia"
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

    steps = []
    for path in os.listdir("/mnt/bigstorage/raymond/heatmaps-olmo/7B"):
        if int(path.split("-")[2][0]) == 1:
            steps.append(path.split('.')[0][:-2])
    newsteps = sorted(steps, key=lambda x: int(x.split('-')[0].split('p')[1]))
    print(newsteps)

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