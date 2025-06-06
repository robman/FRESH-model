# model_configs/llama_config.py
"""
Model-specific configuration for Llama models (including Llama 3, 3.1, 3.3)
using Hugging Face Transformers. Assumes standard HF implementation structure.
"""
import logging
from transformers import LlamaForCausalLM # Import the specific class

logger = logging.getLogger(__name__)

def get_model_class():
    """Returns the Hugging Face model class for Llama models."""
    logger.debug("Returning LlamaForCausalLM model class.")
    return LlamaForCausalLM

def get_layer_list_path(model_name):
    """
    Returns the attribute path to access the list of transformer layers.
    Args:
        model_name (str): The full Hugging Face model name (e.g., "meta-llama/Meta-Llama-3.1-70B").
                          Included for potential future overrides.
    Returns:
        str: Dot-separated path string (e.g., "model.layers").
    """
    # Standard path for HF Llama implementations
    path = "model.layers"
    logger.debug(f"Returning layer list path for {model_name}: {path}")
    # Add conditional logic here if specific Llama versions/sizes differ
    return path

def get_activation_target_paths(model_name):
    """
    Returns a dictionary mapping standard activation keys to model-specific
    attribute paths within a single transformer layer module.

    Args:
        model_name (str): The full Hugging Face model name.
                          Included for potential future overrides.
    Returns:
        dict: Mapping like {"attn_output": "path.to.attn_out", ...}
              Use "" as path to hook the entire layer module (for residual_stream).
    """
    # Standard paths for HF Llama implementations
    paths = {
        "attn_output": "self_attn.o_proj",     # Output projection of attention block
        "mlp_output": "mlp.down_proj",        # Output projection of MLP block (down projection)
        "residual_stream": ""                 # Hook the entire layer output for residual stream
    }
    logger.debug(f"Returning activation target paths for {model_name}: {paths}")
    # Add conditional logic here if specific Llama versions/sizes differ
    return paths


