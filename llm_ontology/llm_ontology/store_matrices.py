#!/usr/bin/env python3

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPTNeoXForCausalLM
from tqdm import tqdm
import os
import pathlib
from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast
  # pip install ai2-olmo


# ### load model ###
# device = torch.device("cuda:0")
# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
# model = AutoModelForCausalLM.from_pretrained("google/gemma-2b",
#                                              torch_dtype=torch.float32,
#                                              device_map="auto")

# gamma = model.get_output_embeddings().weight.detach()
# W, d = gamma.shape
# gamma_bar = torch.mean(gamma, dim = 0)
# centered_gamma = gamma - gamma_bar

# ### compute Cov(gamma) and tranform gamma to g ###
# Cov_gamma = centered_gamma.T @ centered_gamma / W
# eigenvalues, eigenvectors = torch.linalg.eigh(Cov_gamma)
# inv_sqrt_Cov_gamma = eigenvectors @ torch.diag(1/torch.sqrt(eigenvalues)) @ eigenvectors.T
# sqrt_Cov_gamma = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T
# g = centered_gamma @ inv_sqrt_Cov_gamma


# ## Use this PATH to load g in the notebooks=
# torch.save(g, f"FILE_PATH")

def generate_unembedding_matrix(parameter_model: str, step: str, output_dir: str):
    # TODO parameterize instead of comment

    # model = GPTNeoXForCausalLM.from_pretrained(
    # f"EleutherAI/pythia-{parameter_model}-deduped",
    # revision=f"{step}",
    # cache_dir=f"/mnt/bigstorage/raymond/huggingface_cache/pythia-{parameter_model}-deduped/{step}",
    # )
    # tokenizer = AutoTokenizer.from_pretrained(
    # f"EleutherAI/pythia-{parameter_model}-deduped",
    # revision=f"{step}",
    # cache_dir=f"/mnt/bigstorage/raymond/huggingface_cache/pythia-{parameter_model}-deduped/{step}",
    # )

    
    # TODO
    # load only the last layer of the model instead of the entire model
    # 1. Profile a run of store_matrices.py to confirm that whole model loading takes a long time
    # 2. If it takes a long time, find way to load only the last layer of the model
    # This might require working in Pytorch instead of relying on huggingface
    model = OLMoForCausalLM.from_pretrained(f"allenai/OLMo-{parameter_model}", revision=step)

    tokenizer = OLMoTokenizerFast.from_pretrained(f"allenai/OLMo-{parameter_model}", revision=step)

    ### load unembdding vectors ###
    gamma = model.get_output_embeddings().weight.detach()
    W, d = gamma.shape
    gamma_bar = torch.mean(gamma, dim = 0)
    centered_gamma = gamma - gamma_bar

    ### compute Cov(gamma) and tranform gamma to g ###
    Cov_gamma = centered_gamma.T @ centered_gamma / W

    # eigenvalues, eigenvectors = torch.linalg.eigh(Cov_gamma)

    # NOTE: 
    #     Intel oneMKL ERROR: Parameter 8 was incorrect on entry to SSYEVD.
    # Traceback (most recent call last):
    #   File "/home/gabe/persona-research-internship/llm_ontology/llm_ontology/store_matrices.py", line 101, in <module>
    #     generate_unembedding_matrix(parameter_model, step, str(folder))
    #   File "/home/gabe/persona-research-internship/llm_ontology/llm_ontology/store_matrices.py", line 62, in generate_unembedding_matrix
    #     eigenvalues, eigenvectors = torch.linalg.eigh(Cov_gamma)
    # RuntimeError: false INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/native/BatchLinearAlgebra.cpp":1601, please report a bug to PyTorch. linalg.eigh: Argument 8 has illegal value. Most certainly there is a bug in the implementation calling the backend library.

    eigenvalues, eigenvectors = np.linalg.eigh(Cov_gamma)
    eigenvalues = torch.from_numpy(eigenvalues)
    eigenvectors = torch.from_numpy(eigenvectors)

    inv_sqrt_Cov_gamma = eigenvectors @ torch.diag(1/torch.sqrt(eigenvalues)) @ eigenvectors.T
    sqrt_Cov_gamma = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T
    g = centered_gamma @ inv_sqrt_Cov_gamma


    ## Use this PATH to load g in the notebooks=
    torch.save(g, f"{output_dir}/{step}")

def main() -> None:
    parameter_models = ["7B"]

    SCRIPT_DIR = pathlib.Path(__file__).parent
    DATA_DIR = SCRIPT_DIR / "../data"
    BIGSTORAGE_DIR = pathlib.Path("/mnt/bigstorage")

    with open(DATA_DIR / "olmo_7B_model_names.txt", "r") as a:
        steps = a.readlines()

    steps = list(map(lambda x: x[:-1], steps))
    steps.sort(key=lambda x: int(x.split("-")[0].split("p")[1]))

    print(len(steps))

    newsteps = steps[1:15]

    print(newsteps)
    print(len(newsteps))

    for parameter_model in parameter_models:
        folder = BIGSTORAGE_DIR / "raymond" / "olmo" / f"{parameter_model}-unembeddings"
        folder.mkdir(parents=True, exist_ok=True)
        for step in newsteps:
            generate_unembedding_matrix(parameter_model, step, str(folder))

if __name__ == "__main__":
    main()



# model = OLMoForCausalLM.from_pretrained("allenai/OLMo-7B", revision="step1000-tokens4B")

# tokenizer = OLMoTokenizerFast.from_pretrained("allenai/OLMo-7B", revision="step1000-tokens4B")
