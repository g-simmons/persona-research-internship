#!/usr/bin/env python3

import time
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import logging
from store_matrices import generate_unembedding_matrix

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stdout_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)


def test_unembedding_speed(
    parameter_model: str = "7B",
    step: str = "step0-tokens0B", 
    output_dir: str | None = None,
    num_trials: int = 5
):
    """Test speed comparison between GPU and non-GPU unembedding matrix generation."""
    
    BIGSTORAGE_DIR = pathlib.Path("/mnt/bigstorage")
    
    if output_dir is None:
        folder = BIGSTORAGE_DIR / "raymond" / "olmo" / f"{parameter_model}-unembeddings"
    else:
        folder = pathlib.Path(output_dir)
    
    folder.mkdir(parents=True, exist_ok=True)
    
    non_gpu_times = []
    gpu_times = []
    
    logger.info(f"Running {num_trials} trials for speed comparison")
    
    for i in range(num_trials):
        # Test non-GPU
        logger.info(f"Trial {i+1}/{num_trials} - Testing non-GPU")
        start = time.perf_counter()
        generate_unembedding_matrix(parameter_model, step, str(folder), use_gpu=False)
        end = time.perf_counter()
        non_gpu_time = end - start
        non_gpu_times.append(non_gpu_time)
        logger.info(f"Non-GPU time {i+1}: {non_gpu_time:.2f}s")
        
        # Test GPU
        logger.info(f"Trial {i+1}/{num_trials} - Testing GPU")
        start = time.perf_counter()
        generate_unembedding_matrix(parameter_model, step, str(folder), use_gpu=True)
        end = time.perf_counter()
        gpu_time = end - start
        gpu_times.append(gpu_time)
        logger.info(f"GPU time {i+1}: {gpu_time:.2f}s")
    
    logger.info(f"Non-GPU times: {non_gpu_times}")
    logger.info(f"GPU times: {gpu_times}")
    
    ratio_times = [non / gpu for non, gpu in zip(non_gpu_times, gpu_times)]
    
    # Create visualizations
    trials = np.arange(len(non_gpu_times))
    trial_labels = [i + 1 for i in trials]
    width = 0.35
    
    # Grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(trials - width / 2, non_gpu_times, width, label="Non-GPU")
    ax.bar(trials + width / 2, gpu_times, width, label="GPU")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("GPU vs. Non-GPU Time Comparison")
    ax.set_xticks(trials)
    ax.set_xticklabels(trial_labels)
    ax.legend()
    fig.savefig("gpu_vs_nongpu_times.png")
    logger.info("Saved gpu_vs_nongpu_times.png")
    plt.close(fig)
    
    # Ratio chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(trials, ratio_times, width, label="Non-GPU/GPU")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Ratio (Non-GPU/GPU times)")
    ax.set_title("GPU, Non-GPU Time Comparison ratio")
    ax.set_xticks(trials)
    ax.set_xticklabels(trial_labels)
    ax.legend()
    fig.savefig("gpu_vs_nongpu_ratio_times.png")
    logger.info("Saved gpu_vs_nongpu_ratio_times.png")
    plt.close(fig)
    
    # Print summary statistics
    avg_non_gpu = np.mean(non_gpu_times)
    avg_gpu = np.mean(gpu_times)
    avg_ratio = np.mean(ratio_times)
    
    logger.info(f"Average non-GPU time: {avg_non_gpu:.2f}s")
    logger.info(f"Average GPU time: {avg_gpu:.2f}s")
    logger.info(f"Average speedup ratio: {avg_ratio:.2f}x")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test speed of unembedding matrix generation with GPU vs non-GPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--parameter-model",
        type=str,
        default="7B",
        help="Parameter model to test (e.g., 7B, 1B)"
    )
    
    parser.add_argument(
        "--step", 
        type=str,
        default="step0-tokens0B",
        help="Model step/revision to test"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for matrices. Defaults to /mnt/bigstorage/raymond/olmo/{model}-unembeddings"
    )
    
    parser.add_argument(
        "--num-trials",
        type=int,
        default=5,
        help="Number of trials to run for each method"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    test_unembedding_speed(
        parameter_model=args.parameter_model,
        step=args.step,
        output_dir=args.output_dir,
        num_trials=args.num_trials
    )


if __name__ == "__main__":
    main()