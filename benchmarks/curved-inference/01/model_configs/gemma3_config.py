# model_configs/gemma3_config.py
"""
Model-specific configuration for Gemma 3 models using Hugging Face Transformers.
"""
import logging
from transformers import Gemma3ForCausalLM # Import the specific class

logger = logging.getLogger(__name__)

def get_model_class():
    """Returns the Hugging Face model class for Gemma 3."""
    logger.debug("Returning Gemma3ForCausalLM model class.")
    return Gemma3ForCausalLM

def get_layer_list_path(model_name):
    """
    Returns the attribute path to access the list of transformer layers.
    Args:
        model_name (str): The full Hugging Face model name (e.g., "google/gemma-3-1b-pt").
                          Included for potential future overrides based on size.
    Returns:
        str: Dot-separated path string (e.g., "model.model.layers").
    """
    # Based on introspection of google/gemma-3-1b-pt
    path = "model.model.layers"
    logger.debug(f"Returning layer list path for {model_name}: {path}")
    # Add conditional logic here if different Gemma 3 sizes use different paths
    # if "some-larger-gemma" in model_name:
    #     return "some.other.path"
    return path

def get_activation_target_paths(model_name):
    """
    Returns a dictionary mapping standard activation keys to model-specific
    attribute paths within a single transformer layer module.

    Args:
        model_name (str): The full Hugging Face model name.
                          Included for potential future overrides based on size.
    Returns:
        dict: Mapping like {"attn_output": "path.to.attn_out", ...}
              Use "" as path to hook the entire layer module (for residual_stream).
    """
    # Based on introspection of google/gemma-3-1b-pt
    paths = {
        "attn_output": "self_attn.o_proj",     # Output projection of attention block
        "mlp_output": "mlp.down_proj",        # Output projection of MLP block
        "residual_stream": ""                 # Hook the entire layer output for residual stream
    }
    logger.debug(f"Returning activation target paths for {model_name}: {paths}")
    # Add conditional logic here if different Gemma 3 sizes use different paths
    # if "some-larger-gemma" in model_name:
    #     paths["mlp_output"] = "ffn.output_layer"
    return paths


