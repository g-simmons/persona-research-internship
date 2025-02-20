.PHONY: setup env experiments training-steps depth frequency clean

# Environment setup
env:
	uv venv --python 3.12.0
	. .venv/bin/activate && uv pip install -r requirements.txt
	@echo "Remember to set HF_HOME and HF_HUB_CACHE environment variables"

# Initial setup
setup: env
	python llm_ontology/store_matrices.py

# Experiment targets
training-steps:
	python llm_ontology/parallel_heatmaps.py
	python llm_ontology/plot_scores_vs_training_steps_from_heatmaps.py

depth:
	python llm_ontology/obo_depth_scatterplots.py

frequency:
	python llm_ontology/frequencies.py
	python llm_ontology/frequency_score_scatterplots.py

# Run all experiments
experiments: training-steps depth frequency

# Clean up
clean:
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

