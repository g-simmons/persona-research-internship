# LLM Persona Research Internship Fall 2023

- `ipynb`: Jupyter Notebooks
- `DualAlpha`: Code from the Carlsson paper
- `data`: data for the project
- `contrastive_tda`: Python source files for the project

## Installation

After cloning the repository, run the following command inside the repository folder to install the `contrastive_tda` package:

```bash
pip install -e .
```

Install dependencies using poetry:

```bash
poetry install
```

## Adding Data to DVC

DVC is used to track data and models. To add data to DVC, run the following command:

```bash
dvc add data/<data_file>
dvc push
```

The same way that git tracks changes to files, and stores copies of the files in a remote repository, DVC tracks changes to data files, and stores copies of the data files in a remote repo. To view remote repositories for this project, run the following command:

```bash
dvc remote list
```    

## Adding a DVC Remote

To add a remote to DVC, run the following command:

```bash
dvc remote add -d <remote_name> <remote_url>
```

Where `<remote_url>` is the URL of the remote repository. 
