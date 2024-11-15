#!/usr/bin/env python3

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Assuming get_numpy_arrs is defined elsewhere in your code
def get_numpy_arrs(txt_dir: str):
    with open(txt_dir, "r") as f:
        scores = f.readlines()
        scores = list(map(lambda x: x[:-1], scores))
        scores = list(map(lambda x: x.split(", "), scores))
        scores = list(map(lambda x: list(map(float, x)), scores))
        
        linear_scores = []
        causal_sep_score = []
        hierarchy_score = []
        for score in scores:
            linear_scores.append(score[0])
            causal_sep_score.append(score[1])
            hierarchy_score.append(score[2])
    
    return np.array(linear_scores), np.array(causal_sep_score), np.array(hierarchy_score)

# Init
steps = np.array([i for i in range(1000, 145000, 2000)])
parameter_models = ["70M", "1.4B"]

linear_scores_70M, causal_sep_score_70M, hierarchy_score_70M = get_numpy_arrs("scores_70M.txt")
linear_scores_14B, causal_sep_score_14B, hierarchy_score_14B = get_numpy_arrs("scores_1.4B.txt")        #1.4B, not 14B

# Prepare data for seaborn
data = {
    "Steps": np.tile(steps, 2),
    "Scores": np.concatenate([linear_scores_70M, linear_scores_14B]),
    "Model Size": np.repeat(parameter_models, len(steps))
}

# Create a DataFrame
import pandas as pd
df = pd.DataFrame(data)

# Linear rep
sns.lineplot(data=df, x="Steps", y="Scores", hue="Model Size")

plt.xlabel("Steps")
plt.ylabel("idk")
plt.title("Linear Representation Scores")
plt.legend(title="Model Size", loc="upper right")

plt.savefig("plots_sns/linear_rep.png")
plt.clf()