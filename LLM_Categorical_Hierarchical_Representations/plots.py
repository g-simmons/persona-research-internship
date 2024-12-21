#!/usr/bin/env python3

import altair as alt
import pandas as pd
import torch
import numpy as np


# stuff
steps = [f"step{i}" for i in range(1000, 145000, 2000)]
steps_nums = [i for i in range(1000, 145000, 2000)]
parameter_models = ["70M", "160M", "1.4B", "2.8B"]      # add 12B once ready


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


def save_plot(score: str, output_dir: str):

    if score == "causal_sep":
        title = "Causal Separability Scores"
        y_title = "causal-sep-score"
    elif score == "hierarchy":
        title = "Hierarchy Scores"
        y_title = "hierarchy-score"

    scores = []     # each element is a list of scores for a parameter model

    for parameter_model in parameter_models:
        temp_scores = []
        for step in steps:
            adj = torch.load(f"/mnt/bigstorage/raymond/heatmaps/{parameter_model}/{parameter_model}-{step}-1.pt")
            cos = torch.load(f"/mnt/bigstorage/raymond/heatmaps/{parameter_model}/{parameter_model}-{step}-2.pt")
            hier = torch.load(f"/mnt/bigstorage/raymond/heatmaps/{parameter_model}/{parameter_model}-{step}-3.pt")

            if score == "causal_sep":
                temp_scores.append(causal_sep_score(adj, cos))
            elif score == "hierarchy":
                temp_scores.append(hierarchy_score(hier))

        scores.append(temp_scores)

    new_scores = []  # each element is a list of scores for a step, formatted for df.DataFrame
    for i in range(len(steps)):
        new_scores.append([score_list[i] for score_list in scores])

    df = pd.DataFrame(new_scores, columns=parameter_models, index=pd.RangeIndex(start = 1000, stop = 145000, step = 2000, name="Step"))
    print(df)
    df = df.reset_index().melt("Step", var_name="Model Size", value_name="Score")
    print(df)

    nearest = alt.selection_point(nearest=True, on="pointerover",
                                fields=["Step"], empty=False)

    # The basic line
    line = alt.Chart(df).mark_line(interpolate="basis").encode(
        x=alt.X('Step:Q', title='Steps', scale=alt.Scale(nice=False)),
        y=alt.Y('Score:Q', title=y_title),
        color="Model Size:N"
    )

    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(df).mark_point().encode(
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

    # Put the five layers into a chart and bind the data
    final_chart = alt.layer(
        line, selectors, points, rules, text
    ).properties(
        width=400,
        height=300,
        title=title
    )

    final_chart.save(f'{output_dir}.png')
    final_chart.save(f'{output_dir}.html')



save_plot("causal_sep", "model_score_plots/causal_sep_scores")
save_plot("hierarchy", "model_score_plots/hierarchy_scores")
