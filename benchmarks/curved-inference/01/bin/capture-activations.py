import argparse
import importlib # For dynamic module loading
import json
import logging
import os
import sys # Added sys import
import time
import uuid
from functools import partial
from pathlib import Path # Use Path for directory creation

import h5py
import numpy as np
import torch
from transformers import AutoTokenizer # Keep AutoTokenizer for flexibility

# --- Global Configuration ---
# Configure logging
# Added check to prevent duplicate handlers if script is reloaded in interactive session
log_handlers = [logging.StreamHandler(sys.stdout)]
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=log_handlers
    )
logger = logging.getLogger(__name__)


# Dictionary to hold activations for the *current* prompt being processed
current_prompt_activations = {}
hook_handles = [] # To store hook handles for removal

# --- Helper Function ---
def get_nested_attr(obj, attr_path):
    """Accesses nested attributes using a dot-separated path string."""
    try:
        attributes = attr_path.split('.')
        current_obj = obj
        for attr in attributes:
            current_obj = getattr(current_obj, attr)
        return current_obj
    except AttributeError:
        logger.error(f"Could not find attribute path '{attr_path}' in object {type(obj)}")
        return None

# --- Hook Function ---
def capture_activation_hook(
    module,
    input_tensors,
    output_tensor,
    layer_index,
    activation_type, # The key name (e.g., "mlp_output")
    storage_dict
):
    """
    PyTorch forward hook to capture activation tensors.
    Now uses activation_type directly.
    """
    try:
        output_to_capture = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
        if not isinstance(output_to_capture, torch.Tensor):
            logger.warning(f"Hook for L{layer_index}_{activation_type} received non-tensor output: {type(output_to_capture)}. Skipping.")
            return

        if output_to_capture.dim() == 3 and output_to_capture.shape[0] == 1:
            activation_data = output_to_capture[0].detach().cpu().to(torch.float32).numpy()
        elif output_to_capture.dim() == 2:
            activation_data = output_to_capture.detach().cpu().to(torch.float32).numpy()
            logger.warning(f"Captured 2D tensor for L{layer_index}_{activation_type}. Shape: {activation_data.shape}")
        else:
            logger.error(f"Unexpected tensor shape for L{layer_index}_{activation_type}: {output_to_capture.shape}. Skipping.")
            return

        if layer_index not in storage_dict:
            storage_dict[layer_index] = {}
        storage_dict[layer_index][activation_type] = activation_data
        # logger.debug(f"Captured L{layer_index}_{activation_type}, Shape: {activation_data.shape}")

    except Exception as e:
        logger.error(f"Error in hook for L{layer_index}_{activation_type}: {e}", exc_info=True)


# --- Core Capture Function ---
def run_capture(config_path):
    """
    Loads configuration, model (dynamically based on architecture),
    runs prompts, captures activations, and saves to HDF5.
    """
    global current_prompt_activations, hook_handles # Allow modification

    logger.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load or parse config {config_path}: {e}")
        return

    # --- Configuration Validation (Basic) ---
    required_keys = ['model_name', 'architecture_type', 'prompts_file', 'output_hdf5_file', 'capture_points']
    if not all(key in config for key in required_keys):
        logger.error(f"Config file missing one or more required keys: {required_keys}")
        return
    if not isinstance(config.get('capture_points'), dict):
        logger.error("'capture_points' in config must be a dictionary.")
        return

    model_name = config['model_name']
    architecture_type = config['architecture_type'] # e.g., "gemma3", "llama3.3"
    prompts_file = config['prompts_file']
    output_hdf5_file = config['output_hdf5_file']
    capture_points = config['capture_points']
    batch_size = config.get('batch_size', 1)

    if batch_size != 1:
        logger.error("Currently, only batch_size=1 is supported.")
        return

    # --- Load Model-Specific Configuration ---
    try:
        # Use architecture_type directly for module name
        # Replace '.' with '_' for valid module names (e.g., llama3.3 -> llama3_3)
        module_name_safe = architecture_type.replace('.', '_')
        module_path = f"model_configs.{module_name_safe}_config"

        logger.info(f"Attempting to load model config module: {module_path}")
        model_config_module = importlib.import_module(module_path)

        # Get model-specific details by calling functions from the module
        # Pass the full HF model_name to the functions
        ModelClass = model_config_module.get_model_class()
        layer_list_path = model_config_module.get_layer_list_path(model_name)
        activation_target_paths = model_config_module.get_activation_target_paths(model_name)

        logger.info(f"Using Model Class: {ModelClass.__name__}")
        logger.info(f"Layer List Path: {layer_list_path}")
        logger.info(f"Activation Target Paths: {activation_target_paths}")

    except ImportError:
        logger.error(f"Could not import model config module '{module_path}'. Make sure 'model_configs/{module_name_safe}_config.py' exists.")
        return
    except AttributeError as e:
         logger.error(f"Model config module '{module_path}' is missing a required function (get_model_class, get_layer_list_path, or get_activation_target_paths): {e}")
         return
    except Exception as e:
        logger.error(f"Error loading model-specific configuration: {e}", exc_info=True)
        return

    logger.info(f"Model: {model_name}")
    logger.info(f"Architecture Type specified in config: {architecture_type}")
    logger.info(f"Output HDF5: {output_hdf5_file}")
    logger.info(f"Activations to capture: {', '.join(k for k, v in capture_points.items() if v)}")

    # Ensure output directory exists
    Path(output_hdf5_file).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured output directory exists: {Path(output_hdf5_file).parent}")

    # --- Load Prompts ---
    try:
        with open(prompts_file, 'r') as f:
            prompts_data = json.load(f)
        if not isinstance(prompts_data, list):
             raise ValueError("Prompts file should contain a JSON list of prompt objects.")
        logger.info(f"Loaded {len(prompts_data)} prompts from {prompts_file}")
    except Exception as e:
        logger.error(f"Error reading or parsing prompts file {prompts_file}: {e}")
        return

    # --- Device Setup ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        compute_dtype = torch.float32
    else:
        device = torch.device("cpu")
        compute_dtype = torch.float32
    logger.info(f"Target device type: {device.type}")
    logger.info(f"Using compute dtype: {compute_dtype}")

    # --- Load Tokenizer & Model ---
    try:
        logger.info("Loading tokenizer...")
        # Use AutoTokenizer, assuming compatibility across models
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("Loading model...")
        # Load using the dynamically determined ModelClass
        model = ModelClass.from_pretrained(
            model_name,
            torch_dtype=compute_dtype,
            device_map="auto" # Use accelerate
        )
        model.eval()
        model_device = next(model.parameters()).device
        logger.info(f"Model loaded successfully. Primary device: {model_device}")

        # Get layers using the dynamic path
        layers = get_nested_attr(model, layer_list_path)
        if layers is None or not isinstance(layers, torch.nn.ModuleList):
             logger.error(f"Could not retrieve layer list using path: {layer_list_path}")
             return
        num_layers = len(layers)
        logger.info(f"Model has {num_layers} transformer layers.")

    except Exception as e:
        logger.error(f"Failed to load model or tokenizer: {e}", exc_info=True)
        return

    # --- HDF5 File Initialization ---
    logger.info(f"Opening HDF5 file for appending: {output_hdf5_file}")
    try:
        hdf5_file = h5py.File(output_hdf5_file, 'a')
    except Exception as e:
        logger.error(f"Failed to open HDF5 file {output_hdf5_file}: {e}")
        return

    # --- Process Prompts ---
    total_prompts = len(prompts_data)
    for i, prompt_info in enumerate(prompts_data):
        prompt_text = prompt_info.get('text')
        prompt_id = prompt_info.get('prompt_id', str(uuid.uuid4()))

        if not prompt_text:
             logger.warning(f"Skipping prompt {i+1}/{total_prompts} due to missing 'text' field.")
             continue

        logger.info(f"Processing prompt {i+1}/{total_prompts} (ID: {prompt_id})...")
        start_prompt_time = time.time()

        if prompt_id in hdf5_file:
            logger.warning(f"Prompt ID {prompt_id} already exists in HDF5 file. Skipping.")
            continue

        current_prompt_activations.clear()
        hook_handles.clear()

        try:
            # --- Register Hooks ---
            logger.debug(f"Registering hooks for prompt {prompt_id}...")
            layers = get_nested_attr(model, layer_list_path) # Re-get layers just in case
            if layers is None: raise RuntimeError("Layer list became unavailable.")

            for layer_idx in range(num_layers):
                layer = layers[layer_idx]

                for act_type, should_capture in capture_points.items():
                    if not should_capture: continue

                    # Get the specific path for this activation type from the model config
                    submodule_path = activation_target_paths.get(act_type)
                    if submodule_path is None:
                        logger.warning(f"No path defined for activation type '{act_type}' in model config. Skipping.")
                        continue

                    # Find the target module
                    if submodule_path == "": # Special case: hook the layer itself
                        target_module = layer
                    else:
                        target_module = get_nested_attr(layer, submodule_path)

                    if target_module is None:
                        logger.error(f"Could not find target module for L{layer_idx}_{act_type} using path '{submodule_path}'. Skipping hook.")
                        continue

                    # Register hook using functools.partial
                    hook_func_with_context = partial(
                        capture_activation_hook,
                        layer_index=layer_idx,
                        activation_type=act_type, # Pass the key name
                        storage_dict=current_prompt_activations
                    )
                    handle = target_module.register_forward_hook(hook_func_with_context)
                    hook_handles.append(handle)
            logger.debug(f"Registered {len(hook_handles)} hooks.")

            # --- Prepare Input & Run Forward Pass ---
            logger.debug("Tokenizing prompt...")
            # Use model's config for max length if available, otherwise fallback
            max_length = getattr(model.config, 'max_position_embeddings', 512)
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = inputs.to(model_device) # Move inputs to model device
            logger.debug(f"Input tensor device: {inputs['input_ids'].device}")

            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]
            logger.debug(f"Input shape: {input_ids.shape}")

            logger.debug("Running forward pass...")
            forward_start_time = time.time()
            with torch.no_grad():
                outputs = model(**inputs) # Pass dict
            logger.debug(f"Forward pass took {time.time() - forward_start_time:.2f}s")

        except Exception as e:
            logger.error(f"Error during forward pass or hook registration for prompt {prompt_id}: {e}", exc_info=True)
            # Ensure hooks removed even if error occurs
            for handle in hook_handles: handle.remove()
            hook_handles.clear()
            continue
        finally:
            # --- Remove Hooks ---
            logger.debug("Removing hooks...")
            for handle in hook_handles: handle.remove()
            logger.debug(f"Removed {len(hook_handles)} hooks.")
            hook_handles.clear()

        # --- Save Captured Activations ---
        if not current_prompt_activations:
             logger.warning(f"No activations were captured for prompt {prompt_id}. Skipping HDF5 save.")
             continue

        logger.debug(f"Saving activations for prompt {prompt_id} to HDF5...")
        try:
            prompt_group = hdf5_file.create_group(prompt_id)
            # Save prompt metadata
            for key, value in prompt_info.items():
                 try:
                     if value is None: value = 'None'
                     elif isinstance(value, (list, tuple)): value = str(value)
                     elif not isinstance(value, (int, float, str, bytes, np.generic)): value = str(value)
                     prompt_group.attrs[key] = value
                 except TypeError as attr_err:
                     logger.warning(f"Could not save attribute '{key}' (value: {value}, type: {type(value)}) for prompt {prompt_id}. Error: {attr_err}. Saving as string.")
                     prompt_group.attrs[key] = str(value)

            prompt_group.attrs['sequence_length'] = seq_len
            # Save token IDs
            prompt_group.create_dataset('token_ids', data=input_ids[0].cpu().numpy(), compression="gzip")
            # Save activations
            for layer_idx, activations in sorted(current_prompt_activations.items()):
                layer_group = prompt_group.create_group(f'layer_{layer_idx}')
                for act_type, data in activations.items():
                    layer_group.create_dataset(
                        act_type, data=data, compression="gzip", dtype='float32'
                    )
            hdf5_file.flush()
            logger.info(f"Successfully saved activations for prompt {prompt_id}.")

        except Exception as e:
            logger.error(f"Error saving activations for prompt {prompt_id} to HDF5: {e}", exc_info=True)
            if prompt_id in hdf5_file:
                 try: del hdf5_file[prompt_id]; logger.info(f"Removed potentially incomplete group {prompt_id}.")
                 except Exception as del_e: logger.error(f"Failed to remove incomplete group {prompt_id}: {del_e}")

        end_prompt_time = time.time()
        logger.info(f"Finished processing prompt {prompt_id} in {end_prompt_time - start_prompt_time:.2f} seconds.")

    # --- Cleanup ---
    logger.info("Closing HDF5 file.")
    hdf5_file.close()
    logger.info("Activation capture process complete.")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture LLM activations using PyTorch hooks (Generalised).")
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the JSON configuration file."
    )
    args = parser.parse_args()

    run_capture(args.config_file)

