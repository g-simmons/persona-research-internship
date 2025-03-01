#!/usr/bin/env python3

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import GPTNeoXForCausalLM, PretrainedConfig
from tqdm import tqdm
import os
import pathlib
import logging
from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast
from huggingface_hub import hf_hub_download
import json
import psutil
import gc
from pathlib import Path
from safetensors.torch import load_file
import inspect
import torch.nn.functional as F


  # pip install ai2-olmo

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
file_handler = logging.FileHandler('store_matrices.log')
stdout_handler = logging.StreamHandler()

# Create formatters and add it to handlers
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(log_format)
stdout_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

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

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
    return memory_gb

def load_olmo_last_layer(model_name):
    """Load only the last layer"""
    initial_memory = get_memory_usage()
    
    # Get the model config first
    config = AutoConfig.from_pretrained(model_name)
    last_layer_idx = config.num_hidden_layers - 1
    
    # Define last layer weight keys
    last_layer_keys = [
        f"model.layers.{last_layer_idx}.self_attn.q_proj",
        f"model.layers.{last_layer_idx}.self_attn.k_proj",
        f"model.layers.{last_layer_idx}.self_attn.v_proj",
        f"model.layers.{last_layer_idx}.self_attn.o_proj",
        f"model.layers.{last_layer_idx}.mlp.gate_proj",
        f"model.layers.{last_layer_idx}.mlp.up_proj",
        f"model.layers.{last_layer_idx}.mlp.down_proj",
        f"model.layers.{last_layer_idx}.input_layernorm",
        f"model.layers.{last_layer_idx}.post_attention_layernorm"
    ]
    
    # Load index file
    index_file = hf_hub_download(model_name, "pytorch_model.bin.index.json")
    with open(index_file, 'r') as f:
        index = json.load(f)
    
    last_layer_weights = {}
    
    # Load only last layer weights
    for key in last_layer_keys:
        for weight_key, filename in index['weight_map'].items():
            if key in weight_key:
                checkpoint_file = hf_hub_download(model_name, filename)
                weights = torch.load(checkpoint_file, map_location='cpu')
                last_layer_weights[key] = weights[key]
                break
    
    final_memory = get_memory_usage()
    return last_layer_weights, final_memory - initial_memory

def generate_unembedding_matrix(parameter_model: str, step: str, output_dir: str):
    """
    Apply the causal inner product to the unembedding matrix and save it.
    The causal inner product is estimated as the product of the square root of the 
    covariance matrix of the unembedding vectors and the centered unembedding vectors.
    """
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
    
    model_name = "allenai/OLMo-7B"
    cache_dir = f"/mnt/bigstorage/raymond/huggingface_cache/OLMo-{parameter_model}/{step}"

    model = OLMoForCausalLM.from_pretrained(f"allenai/OLMo-{parameter_model}", revision=step)

    GPU = 0

    ### load unembdding vectors ###
    gamma = model.get_output_embeddings().weight.detach()
    if GPU:
        gamma = gamma.to('cuda')
    # import code
    # code.interact(local=dict(globals(), **locals()))
    W, d = gamma.shape

    gamma_bar = torch.mean(gamma, dim = 0)
    centered_gamma = gamma - gamma_bar

    ### compute Cov(gamma) and tranform gamma to g ###
    Cov_gamma = centered_gamma.T @ centered_gamma / W
    # import code
    # code.interact(local=dict(globals(), **locals()))

    if GPU:
        eigenvalues, eigenvectors = torch.linalg.eigh(Cov_gamma) # for hermitian matrices
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(Cov_gamma)
    # eigenvalues, eigenvectors = torch.linalg.eig(Cov_gamma) # for not necessarily hermitian matrices

    # NOTE: 
    #     Intel oneMKL ERROR: Parameter 8 was incorrect on entry to SSYEVD.
    # Traceback (most recent call last):
    #   File "/home/gabe/persona-research-internship/llm_ontology/llm_ontology/store_matrices.py", line 101, in <module>
    #     generate_unembedding_matrix(parameter_model, step, str(folder))
    #   File "/home/gabe/persona-research-internship/llm_ontology/llm_ontology/store_matrices.py", line 62, in generate_unembedding_matrix
    #     eigenvalues, eigenvectors = torch.linalg.eigh(Cov_gamma)
    # RuntimeError: false INTERNAL ASSERT FAILED at "/pytorch/aten/src/ATen/native/BatchLinearAlgebra.cpp":1601, please report a bug to PyTorch. linalg.eigh: Argument 8 has illegal value. Most certainly there is a bug in the implementation calling the backend library.

    # eigenvalues, eigenvectors = np.linalg.eigh(Cov_gamma)
    if not GPU:
        eigenvalues = torch.from_numpy(eigenvalues)
        eigenvectors = torch.from_numpy(eigenvectors)

    inv_sqrt_Cov_gamma = eigenvectors @ torch.diag(1/torch.sqrt(eigenvalues)) @ eigenvectors.T
    # sqrt_Cov_gamma = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T
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

    logger.info(f"Total number of steps: {len(steps)}")

    newsteps = steps[1:15]

    logger.info(f"Selected steps: {newsteps}")
    logger.info(f"Number of selected steps: {len(newsteps)}")

    for parameter_model in parameter_models:
        folder = BIGSTORAGE_DIR / "raymond" / "olmo" / f"{parameter_model}-unembeddings"
        folder.mkdir(parents=True, exist_ok=True)
        step = steps[1]
        # for step in newsteps:
        logger.info(f"Processing model {parameter_model} at step {step}")
        generate_unembedding_matrix(parameter_model, step, str(folder))

def run_single_step():
    parameter_models = ["7B"]

    SCRIPT_DIR = pathlib.Path(__file__).parent
    DATA_DIR = SCRIPT_DIR / "../data"
    BIGSTORAGE_DIR = pathlib.Path("/mnt/bigstorage")

    with open(DATA_DIR / "olmo_7B_model_names.txt", "r") as a:
        steps = a.readlines()

    steps = list(map(lambda x: x[:-1], steps))
    steps.sort(key=lambda x: int(x.split("-")[0].split("p")[1]))

    logger.info(f"Total number of steps: {len(steps)}")

    newstep = steps[1]

    logger.info(f"Selected step: {newstep}")
    logger.info(f"Number of selected step: {1}")

    for parameter_model in parameter_models:
        folder = BIGSTORAGE_DIR / "raymond" / "olmo" / f"{parameter_model}-unembeddings"
        folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Processing model {parameter_model} at step {newstep}")
        generate_unembedding_matrix(parameter_model, newstep, str(folder))


if __name__ == "__main__":
    main()



# model = OLMoForCausalLM.from_pretrained("allenai/OLMo-7B", revision="step1000-tokens4B")

# tokenizer = OLMoTokenizerFast.from_pretrained("allenai/OLMo-7B", revision="step1000-tokens4B")
