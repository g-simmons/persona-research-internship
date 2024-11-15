# LLM Persona Research Internship

## Installation

After cloning the repository, run the following command inside the repository folder to install the `contrastive_tda` package:

```bash
pip install -e .
```

Installing development hooks

```bash
make install_hooks
```

This will install several "hooks" - scripts that run automatically when certain git commands are run.
These are generally useful for maintaining code quality and consistency. 

To run the commands without the hooks, you can use the `--no-verify` flag. For example, to commit without running the hooks, run the following command:

```bash
git commit --no-verify
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
