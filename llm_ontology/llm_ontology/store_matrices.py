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

def load_olmo_last_layer(parameter_model: str, step: str, cache_dir: str = None):
    """
    Load only the last layer of OLMo model from a specific step in HF cache.
    
    Args:
        parameter_model: Model size (e.g., "7B")
        step: Training step (e.g., "step1000-tokens4B")
        cache_dir: Base cache directory
    """
    # Construct the path using HF cache structure
    model_path = Path(f"/mnt/bigstorage/raymond/huggingface_cache/models--allenai--OLMo-{parameter_model}")
    step_path = model_path / "refs" / step
    snapshots_path = model_path / "snapshots"
    
    print(f"Looking in step path: {step_path}")
    
    # Read the commit hash from the step file
    if not step_path.exists():
        raise FileNotFoundError(f"Step file not found at {step_path}")
        
    with open(step_path, 'r') as f:
        commit_hash = f.read().strip()
    
    print(f"Found commit hash: {commit_hash}")
    
    # Construct the path to the actual model files
    model_files_path = snapshots_path / commit_hash
    
    print(f"Looking for model files in: {model_files_path}")

    config_path = model_files_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    config = PretrainedConfig(**config_dict)

    num_hidden_layers = config.n_layers
    last_layer_idx = num_hidden_layers - 1
    penultimate_layer_idx = num_hidden_layers - 2

    checkpoint_paths = list(model_files_path.glob("*.safetensors"))
    if not checkpoint_paths:
        checkpoint_paths = list(model_files_path.glob("*.bin"))

    # import code
    # code.interact(local=dict(globals(), **locals()))

    last_layer_weights = {}
    penultimate_layer_activations = None
    for checkpoint_path in checkpoint_paths:
        print(f"Loading from {checkpoint_path.name}")
        if checkpoint_path.suffix == '.safetensors':
            weights = load_file(str(checkpoint_path))
        else:
            weights = torch.load(str(checkpoint_path), map_location='cpu')
        
        if checkpoint_path.suffix == '.safetensors':#
            checkpoint = load_file(str(checkpoint_path))#
        else: #
            checkpoint = torch.load(str(checkpoint_path), map_location='cpu') #

        layer_weights = {
            k: v for k, v in weights.items()
            if f"model.transformer.blocks.{last_layer_idx}." in k  # Corrected line
        }
        print(f"Found {len(layer_weights)} weights for last layer in this file")
        last_layer_weights.update(layer_weights)

    # Extract penultimate layer activations
        for key, value in checkpoint.items():
            # *** IMPORTANT: Adjust this key pattern based on the actual keys in your checkpoint file ***
            if f"hidden_states.{penultimate_layer_idx}" in key:  
                penultimate_layer_activations = value
                break  # Assuming only one set of activations per checkpoint

    print(f"Loaded weights: {list(last_layer_weights.keys())}")

    return last_layer_weights, config, penultimate_layer_activations

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
    
    model_name = "allenai/OLMo-7B"
    cache_dir = f"/mnt/bigstorage/raymond/huggingface_cache/OLMo-{parameter_model}/{step}"
    # last_layer_weights, config, penultimate_activations = load_olmo_last_layer(parameter_model, step, cache_dir)
    # print("Loaded weights:", list(last_layer_weights.keys()))
    # print('pen', penultimate_activations)
    
    # for elem in list(last_layer_weights.keys()):
    #     print(last_layer_weights[elem])
    # print(inspect.getsource(OLMoForCausalLM.forward))
    #print("config", config)
    # """config={
    #     "d_model": 4096,
    #     "vocab_size": 50280,
    #     "n_layers": 32,
    #     "n_heads": 32,
    #     "mlp_hidden_size": 22016
    # }"""

    # att_proj_weight = last_layer_weights['model.transformer.blocks.31.att_proj.weight']
    # attn_out_weight = last_layer_weights['model.transformer.blocks.31.attn_out.weight']
    # ff_proj_weight = last_layer_weights['model.transformer.blocks.31.ff_proj.weight']
    # ff_out_weight = last_layer_weights['model.transformer.blocks.31.ff_out.weight']

    # def last_layer_forward(x):
    #     # Attention block
    #     x_attn = F.linear(x, att_proj_weight, bias=None)
    #     x_attn = F.gelu(x_attn)  # Example activation function
    #     x_attn = F.linear(x_attn, attn_out_weight, bias=None)
    #     x = x + x_attn  # Residual connection
    #     x = layer_norm1(x)  # Layer normalization

    #     # Feedforward block
    #     x_ff = F.linear(x, ff_proj_weight, bias=None)
    #     x_ff = F.gelu(x_ff)  # Example activation function
    #     x_ff = F.linear(x_ff, ff_out_weight, bias=None)
    #     x = x + x_ff  # Residual connection
    #     x = layer_norm2(x)  # Layer normalization

    #     return x

    model = OLMoForCausalLM.from_pretrained(f"allenai/OLMo-{parameter_model}", revision=step)

    # tokenizer = OLMoTokenizerFast.from_pretrained(f"allenai/OLMo-{parameter_model}", revision=step)

    # x = torch.randn((config.vocab_size, config.d_model))
    # #x = torch.randn((config["vocab_size"], config["d_model"]))
    # gamma_approx = last_layer_forward(x).detach()  # Approximate output embedding

    # # Get dimensions (W = vocab size, d = embedding size)
    # W, d = gamma_approx.shape  

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
    #print("g shape:", g.shape)
    #print("First few values of g:", g[:5])  # Print first 5 rows


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
    print(newstep)

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
