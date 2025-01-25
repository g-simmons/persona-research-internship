This codebase extends the work from Park, et al (2024).

# Setup

Install the `uv` tool.

Then create a virtual environment and install the dependencies.

```bash
uv venv --python 3.12.0
source .venv/bin/activate
uv pip install -r requirements.txt
```

# Experiments

## Ontology Scores vs. Number of Training Steps
We evaluate the ontology scores vs. training steps for models from the Pythia family.

To run these experiments:

```bash
python parallel_heatmaps.py
python plot_scores_vs_training_steps_from_heatmaps.py
```

## Ontology Scores vs. Term Depth
We evaluate whether ontology scores vary with term depth

To run these experiments:

```bash
python obo_depth_scatterplots.py
```

## Ontology Scores vs. Term Frequency

We evaluate how ontology scores change with term frequency

```bash
python frequencies.py
python frequency_score_scatterplots.py
```

