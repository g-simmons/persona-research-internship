#!/usr/bin/env python3

import altair as alt
import pandas as pd
import torch
import numpy as np

# GENERATING NON-MULTI WORD PLOTS (old plots)
with open("scores_70M_old.txt", "r") as f:
    scores = f.readlines()

scores = [line.strip().split(",") for line in scores]
df_70M = pd.DataFrame(scores, columns=["linear", "causal_sep", "hierarchy"])


with open("scores_160M.txt", "r") as f:
    scores = f.readlines()
scores = [line.strip().split(",") for line in scores]
df_160M = pd.DataFrame(scores, columns=["linear", "causal_sep", "hierarchy"])


with open("scores_1.4B.txt", "r") as f:
    scores = f.readlines()
scores = [line.strip().split(",") for line in scores]
df_14B = pd.DataFrame(scores, columns=["linear", "causal_sep", "hierarchy"])


with open("scores_2.8B_old.txt", "r") as f:
    scores = f.readlines()
scores = [line.strip().split(",") for line in scores]
df_28B = pd.DataFrame(scores, columns=["linear", "causal_sep", "hierarchy"])

with open("scores_12B.txt", "r") as f:
    scores = f.readlines()
scores = [line.strip().split(",") for line in scores]
df_12B = pd.DataFrame(scores, columns=["linear", "causal_sep", "hierarchy"])

dataframes = [df_70M, df_160M, df_14B, df_28B, df_12B]



steps = [f"step{i}" for i in range(1000, 145000, 2000)]
steps_nums = [i for i in range(1000, 145000, 2000)]
parameter_models = ["70M", "160M", "1.4B", "2.8B", "12B"]


def save_plot(score, output_dir):
    if score == "causal_sep":
        title = "Causal Separability Scores Non-Multi Word"
        y_title = "causal-sep-score"
    elif score == "hierarchy":
        title = "Hierarchy Scores Non-Multi Word"
        y_title = "hierarchy-score"
    elif score == "linear":
        title = "Linear Representation Scores Non-Multi Word"
        y_title = "linear-rep-score"

    newscores = []
    for i in range(len(steps)):
        newscores.append([df.loc[i, score] for df in dataframes])

    newdf = pd.DataFrame(newscores, columns=parameter_models, index=pd.RangeIndex(start = 1000, stop = 145000, step = 2000, name="Step"))
    print(newdf)
    newdf = newdf.reset_index().melt("Step", var_name="Model Size", value_name="Score")
    print(newdf)

    nearest = alt.selection_point(nearest=True, on="pointerover",
                                fields=["Step"], empty=False)

    # The basic line
    line = alt.Chart(newdf).mark_line(interpolate="basis").encode(
        x=alt.X('Step:Q', title='Steps', scale=alt.Scale(nice=False)),
        y=alt.Y('Score:Q', title=y_title),
        color=alt.Color('Model Size:N', sort=["70M", "160M", "1.4B", "2.8B"])
    )

    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(newdf).mark_point().encode(
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
    rules = alt.Chart(newdf).mark_rule(color="gray").encode(
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
    ).interactive()

    final_chart.save(f'{output_dir}.png')
    final_chart.save(f'{output_dir}.html')



save_plot("causal_sep", "model_score_plots_nonmulti/causal_sep_scores")
save_plot("hierarchy", "model_score_plots_nonmulti/hierarchy_scores")
save_plot("linear", "model_score_plots_nonmulti/linear_rep_scores")