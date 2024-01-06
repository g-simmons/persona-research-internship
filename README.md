# LLM Persona Research Internship Fall 2023

- `ipynb`: Jupyter Notebooks
- `DualAlpha`: Code from the Carlsson paper
- `data`: data for the project
- `contrastive_tda`: Python source files for the project

## Installation

Install dependencies using poetry:

```bash
poetry install
```

After cloning the repository, run the following command inside the repository folder to install the `contrastive_tda` package:

```bash
pip install -e .
```

Installing development hooks

```bash
make install_hooks
```

## Adding Dependencies to Poetry

To add a dependency to poetry, run the following command:

```bash
poetry add <package_name>
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

## Environment Variables

Environment variables should be stored in a .env file in the root directory of the repository. 
The .env file should NOT be tracked by git. 
To add an environment variable, add a line to the .env file in the following format:

```
export COHERE_API_KEY = <api_key>
```

## Adding a DVC Remote

To add a remote to DVC, run the following command:

```bash
dvc remote add -d <remote_name> <remote_url>
```

Where `<remote_url>` is the URL of the remote repository. 

## Organization Conventions

The ideal data processing script has 
- 1 function that performs the atomic data processing operation. This is the data processing with the minimum size of arguments.

For example, the minimal data processing operation for the embed_reviews function is to embed a single review. in this case there should be a function called embed_review that takes a single review as an argument and returns the embedded review.

Some exceptions:
- Considerable efficiency gain processing multiple things at once
- Requirement to process many things at once (many-to-one mapping)
- Requirement to process all things simultaneously

These exceptions are not rare, but they are the minority of cases.

## Naming Conventions

Functions should generally start with a verb.

Examples of name changes I would suggest:
`embedding_reviews` -> `embed_reviews`
