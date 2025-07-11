import torch
import logging
import os
from pathlib import Path
from llm_ontology.compute_g_matrices import save_gamma_matrix

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stdout_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

def test_fast_slow_equivalence():
    """Test that fast and slow methods return tensors with identical shape, dtype, and content."""
    
    model_name = "allenai/OLMo-1B"
    revision = "step100000-tokens419B"
    user = os.environ["USER"]
    
    # Use a temp directory for cache in CI, otherwise use default
    if os.getenv("GITHUB_ACTIONS"):
        cache_dir = Path.home() / ".cache" / "llm_ontology_test"
        cache_dir.mkdir(parents=True, exist_ok=True)
    else:
        cache_dir = None
    
    logger.info(f"Testing fast vs slow method equivalence for {model_name} at {revision}")
    
    # Generate matrices using both methods
    logger.info("Generating matrix with FAST method...")
    save_gamma_matrix(model_name, revision, user, fast=True, cache_dir=cache_dir)
    fast_filename = f"{model_name.split('/')[-1]}-{revision}.pt"
    
    logger.info("Generating matrix with SLOW method...")
    save_gamma_matrix(model_name, revision, user, fast=False, cache_dir=cache_dir)
    slow_filename = f"{model_name.split('/')[-1]}-{revision}-slow.pt"
    
    # Load and compare
    logger.info("Loading matrices for comparison...")
    fast_matrix = torch.load(fast_filename)
    slow_matrix = torch.load(slow_filename)
    
    # Test shape equivalence
    logger.info(f"Fast matrix shape: {fast_matrix.shape}")
    logger.info(f"Slow matrix shape: {slow_matrix.shape}")
    shape_match = fast_matrix.shape == slow_matrix.shape
    logger.info(f"Shapes match: {shape_match}")
    
    # Test dtype equivalence
    logger.info(f"Fast matrix dtype: {fast_matrix.dtype}")
    logger.info(f"Slow matrix dtype: {slow_matrix.dtype}")
    dtype_match = fast_matrix.dtype == slow_matrix.dtype
    logger.info(f"Dtypes match: {dtype_match}")
    
    # Test content equivalence
    content_match = torch.allclose(fast_matrix, slow_matrix)
    logger.info(f"Contents match: {content_match}")
    
    if content_match:
        logger.info(f"First 3x3 elements (both methods):")
        logger.info(fast_matrix[:3, :3])
    else:
        logger.info(f"Fast method first 3x3:")
        logger.info(fast_matrix[:3, :3])
        logger.info(f"Slow method first 3x3:")
        logger.info(slow_matrix[:3, :3])
        
        diff = torch.abs(fast_matrix - slow_matrix)
        logger.info(f"Max difference: {torch.max(diff)}")
        logger.info(f"Mean difference: {torch.mean(diff)}")
    
    # Clean up test files
    try:
        os.remove(fast_filename)
        os.remove(slow_filename)
        logger.info("Cleaned up test files")
    except Exception as e:
        logger.warning(f"Could not clean up test files: {e}")
    
    # Assert that all comparisons pass
    assert shape_match, f"Shapes don't match: fast={fast_matrix.shape}, slow={slow_matrix.shape}"
    assert dtype_match, f"Dtypes don't match: fast={fast_matrix.dtype}, slow={slow_matrix.dtype}"
    assert content_match, f"Contents don't match (max diff: {torch.max(torch.abs(fast_matrix - slow_matrix)) if not content_match else 'N/A'})"
    
    logger.info("🎉 SUCCESS! Fast and slow methods produce identical tensors")
    logger.info("✅ Shape, dtype, and content all match perfectly")

if __name__ == "__main__":
    test_fast_slow_equivalence()