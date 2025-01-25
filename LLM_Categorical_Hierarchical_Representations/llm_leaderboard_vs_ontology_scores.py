import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

model_sizes = ["160M", "1.4B", "2.8B", "12B"]
scores = {}

for size in model_sizes:
    with open(f"scores_{size}.txt", "r") as f:
        last_line = f.readlines()[-1]  # read last line
        linear, causal, hierarchy = map(float, last_line.strip().split(","))
        scores[size] = {"linear": linear, "causal": causal, "hierarchy": hierarchy}

# Read Pythia leaderboard scores
with open("./data/open_llm_pythia_scores_102724.json", "r") as f:
    pythia_data = json.load(f)
    pythia_scores = {
        model["name"]
        .split("/")[-1]
        .split("-")[-1]: {
            "average": model["average"],
            "ifeval": model["ifeval"],
            "bbh": model["bbh"],
            "math_lvl_5": model["math_lvl_5"],
            "gpqa": model["gpqa"],
            "musr": model["musr"],
            "mmlu_pro": model["mmlu_pro"],
        }
        for model in pythia_data["models"]
    }

# Create dataframe for plotting
plot_data = []
for size in model_sizes:
    model_name = size.lower()
    if model_name in pythia_scores:
        for benchmark in ["mmlu_pro", "ifeval", "bbh", "math_lvl_5", "gpqa", "musr"]:
            for score_type in ["linear_score", "causal_score", "hierarchy_score"]:
                plot_data.append(
                    {
                        "model": size,
                        "benchmark": benchmark,
                        "benchmark_score": pythia_scores[model_name][benchmark],
                        "score_type": score_type,
                        "ontology_score": scores[size][
                            score_type.replace("_score", "")
                        ],
                    }
                )

df = pd.DataFrame(plot_data)

# Define markers for each model size
markers = {"160M": "o", "1.4B": "s", "2.8B": "^", "12B": "D"}

# Create figure with faceted plots
fig = plt.figure(figsize=(15, 20))

# Plot each ontology score vs benchmarks
g = sns.FacetGrid(
    df,
    row="benchmark",
    col="score_type",
    height=3,
    aspect=1.2,
    sharey=False,
    sharex=False,
)

def scatter_with_markers(data, x, y, **kwargs):
    for size in model_sizes:
        mask = data["model"] == size
        plt.scatter(
            data[mask][x], 
            data[mask][y],
            marker=markers[size],
            label=size,
            alpha=0.5
        )
    # Add regression line
    sns.regplot(
        data=data,
        x=x,
        y=y,
        scatter=False,
        color='gray',
        line_kws={'alpha': 0.5}
    )

g.map_dataframe(
    scatter_with_markers,
    x="ontology_score",
    y="benchmark_score"
)

# Add legend to the first subplot
g.axes[0,0].legend(title="Model Size")

# Customize titles and labels
for ax in g.axes.flat:
    ax.set_xlabel("Ontology Score")
    ax.set_ylabel("Benchmark Score")

g.fig.suptitle("Benchmark Scores vs Ontology Scores", y=1.02, fontsize=16)

# Add row and column labels
g.fig.text(0.01, 0.5, "Benchmarks", va="center", rotation=90, fontsize=14)
g.fig.text(0.5, 0.01, "Ontology Types", ha="center", fontsize=14)

# Adjust layout
plt.tight_layout()

# Save plots
g.fig.savefig("figures/benchmark_vs_ontology_scores.png", dpi=300, bbox_inches="tight")
