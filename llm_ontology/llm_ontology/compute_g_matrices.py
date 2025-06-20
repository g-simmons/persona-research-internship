import torch
import random
import requests
from joblib import Parallel, delayed
import numpy as np
from transformers import GPTNeoXForCausalLM, AutoConfig
import os
import pathlib
import logging
import argparse
from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast
from huggingface_hub import HfApi, get_safetensors_metadata
from huggingface_hub.utils._safetensors import TensorInfo
from joblib_progress import joblib_progress
from safetensors import safe_open


def setup_logger() -> logging.Logger:
    script_dir = pathlib.Path(__file__).parent
    logs_dir = script_dir / "../logs"
    logs_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(logs_dir / "compute_g_matrices.log")
    stdout_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    return logger

logger = setup_logger()

model_name = "allenai/OLMo-1.4B"

BIGSTORAGE_DIR = pathlib.Path("/mnt/bigstorage")


def get_current_user() -> str:
    return os.environ["USER"]


def save_gamma_matrix(model_name: str, revision: str, user: str, fast: bool = True, cache_dir: pathlib.Path = None) -> None:
    """

    The park paper uses model.get_output_embeddings().weight.detach()

    The definition of get_output_embeddings is:

        def get_output_embeddings(self):
            if self.config.weight_tying:
                return self.model.transformer.wte
            else:
                return self.model.transformer.ff_out

    To avoid downloading the whole model (fast mode), we can use the config to check for weight_tying, then get the appropriate embedding matrix from the safetensors metadata.
    """
    if cache_dir is None:
        cache_dir = BIGSTORAGE_DIR
    
    if fast:
        config = AutoConfig.from_pretrained(model_name, revision=revision)

        if config.weight_tying:
            matrix_name = "model.transformer.wte.weight"
        else:
            matrix_name = "model.transformer.ff_out.weight"

        metadata = get_safetensors_metadata(model_name, revision=revision)

        file = metadata.weight_map[matrix_name]
        tensor_info: TensorInfo = metadata.files_metadata[file].tensors[matrix_name]
        # tensor_info is something like TensorInfo(dtype='F32', shape=[50304, 2048], data_offsets=(4294967296, 4707057664), parameter_count=103022592)

        # Get the safetensors header size to calculate correct byte offsets
        # Safetensors metadata offsets are relative to start of tensor data section, not file start
        url = f"https://huggingface.co/{model_name}/resolve/{revision}/model.safetensors"
        
        # Read first 8 bytes to get header size
        header_response = requests.get(url, headers={"Range": "bytes=0-7"})
        header_response.raise_for_status()
        header_size = int.from_bytes(header_response.content, byteorder='little')
        
        # Calculate actual byte positions in file
        header_overhead = 8 + header_size  # 8 bytes for header size + JSON metadata
        actual_start = header_overhead + tensor_info.data_offsets[0]
        actual_end = header_overhead + tensor_info.data_offsets[1] - 1
        
        # Download tensor data with corrected offsets
        headers = {
            "Range": f"bytes={actual_start}-{actual_end}"
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.content
        gamma = torch.frombuffer(data, dtype=torch.float32).reshape(tensor_info.shape).clone()
    
    else:
        if "OLMo" in model_name:
            model = OLMoForCausalLM.from_pretrained(
                model_name,
                revision=revision,
                cache_dir=cache_dir / user / "huggingface_cache",
            )
        elif "pythia" in model_name:
            model = GPTNeoXForCausalLM.from_pretrained(
                model_name,
                revision=revision,
                cache_dir=cache_dir / user / "huggingface_cache",
            )
        else:
            raise ValueError(f"Model {model_name} not supported")

        gamma = model.get_output_embeddings().weight.detach()  # type: ignore

    suffix = "" if fast else "-slow"
    torch.save(gamma, f"{model_name.split('/')[-1]}-{revision}{suffix}.pt")


def generate_unembedding_matrix(
    model_name: str, step: str, output_dir: str, use_gpu: bool = False
) -> None:
    """
    Apply the causal inner product to the unembedding matrix and save it.
    The causal inner product is estimated as the product of the square root of the
    covariance matrix of the unembedding vectors and the centered unembedding vectors.
    """

    logger.info(f"Loading model {model_name} at step {step}")
    model = OLMoForCausalLM.from_pretrained(model_name, revision=step)

    # load unembdding vectors
    gamma: Float[Tensor, "vocab_size embedding_dim"] = model.get_output_embeddings().weight.detach()  # type: ignore

    if use_gpu:
        logger.info("Moving gamma to CUDA")
        gamma = gamma.to("cuda")

    W, d = gamma.shape
    logger.info(f"gamma shape: {gamma.shape}")

    gamma_bar = torch.mean(gamma, dim=0)
    centered_gamma = gamma - gamma_bar

    # compute Cov(gamma) and tranform gamma to g
    Cov_gamma = centered_gamma.T @ centered_gamma / W

    if use_gpu:
        logger.info("Computing eigenvalues/eigenvectors with torch.linalg.eigh")
        eigenvalues, eigenvectors = torch.linalg.eigh(
            Cov_gamma
        )  # for hermitian matrices
    else:
        logger.info("Computing eigenvalues/eigenvectors with np.linalg.eigh")
        eigenvalues, eigenvectors = np.linalg.eigh(Cov_gamma)

    if not use_gpu:
        eigenvalues = torch.from_numpy(eigenvalues)
        eigenvectors = torch.from_numpy(eigenvectors)

    if not isinstance(eigenvalues, torch.Tensor):
        eigenvalues = torch.from_numpy(eigenvalues)

    inv_sqrt_Cov_gamma = (
        eigenvectors @ torch.diag(1 / torch.sqrt(eigenvalues)) @ eigenvectors.T
    )

    g = centered_gamma @ inv_sqrt_Cov_gamma

    output_path = f"{output_dir}/{step}"
    logger.info(f"Saving g matrix to {output_path}")
    torch.save(g, output_path)


def get_output_path(model_name: str, step: str, user: str, cache_dir: pathlib.Path = None) -> str:
    if cache_dir is None:
        cache_dir = BIGSTORAGE_DIR
    return str(cache_dir / user / "g_matrices" / f"{model_name}-unembeddings")


def get_olmo_model_revisions(model_name: str) -> list[str]:
    api = HfApi()
    refs = api.list_repo_refs(model_name)
    revisions = []
    for branch in refs.branches:
        revision = branch.name
        revisions.append(revision)
    return revisions


def get_olmo_model_names() -> list[str]:
    olmo_model_names = ["allenai/OLMo-1B", "allenai/OLMo-7B"]
    return olmo_model_names


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute gamma matrices for OLMo models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model-names",
        type=str,
        nargs="+",
        help="Model names to process (e.g., allenai/OLMo-1B). If not specified, uses all OLMo models"
    )
    
    parser.add_argument(
        "--revisions",
        type=str,
        nargs="+",
        help="Specific revisions/steps to process. If not specified, processes all revisions"
    )
    
    parser.add_argument(
        "--user",
        type=str,
        default=None,
        help="User name for storage paths. Defaults to current user"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel processing"
    )
    
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Number of parallel jobs"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode - only process 3 random revisions per model"
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        default=True,
        help="Use fast mode for save_gamma_matrix (downloads only required weights)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for saving matrices"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for model downloads. Defaults to /mnt/bigstorage"
    )
    
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for computations"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    user = args.user or get_current_user()
    model_names = args.model_names or get_olmo_model_names()
    cache_dir = pathlib.Path(args.cache_dir) if args.cache_dir else BIGSTORAGE_DIR
    
    if args.revisions:
        # Process specific revisions
        revisions_to_process = args.revisions
    else:
        # Get all revisions for each model
        revisions_to_process = []
        for model_name in model_names:
            model_revisions = get_olmo_model_revisions(model_name)
            if args.debug:
                model_revisions = random.sample(model_revisions, min(3, len(model_revisions)))
            revisions_to_process.extend(model_revisions)
    
    # Create all model-revision combinations
    model_revision_pairs = []
    for model_name in model_names:
        for revision in revisions_to_process:
            model_revision_pairs.append((model_name, revision))
    
    if args.parallel:
        with joblib_progress(
            "Calculating gamma matrices...",
            total=len(model_revision_pairs),
        ):
            Parallel(n_jobs=args.n_jobs)(
                delayed(save_gamma_matrix)(model_name, revision, user, fast=args.fast, cache_dir=cache_dir)
                for model_name, revision in model_revision_pairs
            )
    else:
        for model_name, revision in model_revision_pairs:
            save_gamma_matrix(model_name, revision, user, fast=args.fast, cache_dir=cache_dir)
            logger.info(f"Saved gamma matrix for {model_name} at revision {revision}")


if __name__ == "__main__":
    main()
