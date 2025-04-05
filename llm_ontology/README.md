This codebase extends the work from Park, et al (2024).

# Environment Setup

First, check if `uv` is installed:
```bash
uv --version
```

If not installed, install the `uv` tool:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create and activate a virtual environment with Python 3.12:
```bash
uv venv --python 3.12.0
source .venv/bin/activate
```

Install the project dependencies:
```bash
uv pip install -r requirements.txt
```

Configure Hugging Face cache location:
```bash
export HF_HOME="/mnt/bigstorage/$USER/huggingface_cache"
export HF_HUB_CACHE="/mnt/bigstorage/$USER/huggingface_cache"
```


# Setup for Experiments

```bash
python llm_ontology/store_matrices.py
```

# Experiments

## Ontology Scores vs. Number of Training Steps
We evaluate the ontology scores vs. training steps for models from the Pythia family.

To run these experiments:

```bash
# saves the unembedding matrices for ontology terms
python llm_ontology/parallel_heatmaps.py
# create line plots show ontology scores vs. training steps, split by model size
python llm_ontology/plot_scores_vs_training_steps_from_heatmaps.py
```

## Ontology Scores vs. Term Depth
We evaluate whether ontology scores vary with term depth

To run these experiments:

```bash
python llm_ontology/obo_depth_scatterplots.py
```

## Ontology Scores vs. Pretraining Term Frequency

We evaluate how ontology scores change with term frequency

```bash
python llm_ontology/frequencies.py
python llm_ontology/frequency_score_scatterplots.py
```

