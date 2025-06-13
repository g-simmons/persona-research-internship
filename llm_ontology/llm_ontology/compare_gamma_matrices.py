import torch
import argparse
import logging

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stdout_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)


def compare_gamma_matrices(slow_path: str, fast_path: str):
    """Compare gamma matrices loaded via slow and fast methods."""
    logger.info(f"Loading slow matrix from {slow_path}")
    slow = torch.load(slow_path)
    
    logger.info(f"Loading fast matrix from {fast_path}")
    fast = torch.load(fast_path)
    
    logger.info(f"Slow shape: {slow.shape}, Fast shape: {fast.shape}")
    logger.info(f"Matrices are equal: {torch.allclose(slow, fast)}")
    
    if not torch.allclose(slow, fast):
        diff = torch.abs(slow - fast)
        logger.info(f"Max difference: {torch.max(diff)}")
        logger.info(f"Mean difference: {torch.mean(diff)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare gamma matrices loaded via different methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--slow-path",
        type=str,
        default="OLMo-1B-main-slow.pt",
        help="Path to gamma matrix loaded via slow method"
    )
    
    parser.add_argument(
        "--fast-path",
        type=str,
        default="OLMo-1B-main.pt",
        help="Path to gamma matrix loaded via fast method"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    compare_gamma_matrices(args.slow_path, args.fast_path)


if __name__ == "__main__":
    main()