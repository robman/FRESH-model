import argparse
import json
import logging
import os
import re
import sys
import time

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize as sk_normalize
from umap import UMAP  # Use umap-learn library

# --- Configuration ---
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def parse_layer_indices(layer_spec, max_layers):
    """Parses layer specification string (e.g., "0-25", "0,5,10") into a list of indices."""
    indices = set()
    if not layer_spec or layer_spec.lower() == 'all':
        return list(range(max_layers))

    parts = layer_spec.split(',')
    for part in parts:
        part = part.strip()
        if not part: continue
        if '-' in part:
            # Handle range
            try:
                start, end = map(int, part.split('-'))
                if start < 0 or end >= max_layers or start > end:
                    raise ValueError(f"Invalid layer range: {part}")
                indices.update(range(start, end + 1))
            except ValueError as e:
                logger.error(f"Error parsing layer range '{part}': {e}")
                return None
        else:
            # Handle single number
            try:
                idx = int(part)
                if idx < 0 or idx >= max_layers:
                     raise ValueError(f"Invalid layer index: {part}")
                indices.add(idx)
            except ValueError as e:
                logger.error(f"Error parsing layer index '{part}': {e}")
                return None
    if not indices:
         logger.error(f"No valid layer indices found in spec: '{layer_spec}'")
         return None
    return sorted(list(indices))

def save_config_to_hdf5_attrs(hdf5_group, config_dict):
    """Saves configuration dictionary items as attributes to an HDF5 group."""
    for key, value in config_dict.items():
        try:
            # Convert lists/dicts to JSON strings for storage as attributes
            if isinstance(value, (list, dict)):
                value_str = json.dumps(value)
                hdf5_group.attrs[key] = value_str
            elif value is None:
                 hdf5_group.attrs[key] = 'None' # h5py doesn't support None attribute
            else:
                hdf5_group.attrs[key] = value
        except TypeError as e:
            logger.warning(f"Could not save config key '{key}' (value: {value}) as HDF5 attribute: {e}. Skipping.")


# --- Main Reduction Function ---

def run_reduction(config_path):
    """
    Loads configuration, raw activations, performs dimensionality reduction,
    and saves embeddings and metadata to a new HDF5 file.
    """
    logger.info(f"Loading reduction configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from configuration file: {config_path}")
        return

    # --- Get Configuration Parameters ---
    input_hdf5_file = config.get('input_hdf5_file')
    output_hdf5_file = config.get('output_hdf5_file')
    prompt_ids_spec = config.get('prompt_ids', "__ALL__") # List or "__ALL__"
    activation_types = config.get('activation_types', []) # List of strings
    layer_indices_spec = config.get('layer_indices', "all") # String like "0-25" or "0,5,10"
    normalize_method = config.get('normalize', None) # e.g., "L2" or None
    use_pca = config.get('use_pca', True)
    pca_components = config.get('pca_components', 50)
    reduction_method = config.get('reduction_method', 'UMAP').upper()
    final_components = config.get('final_components', 2)
    # UMAP params
    umap_neighbors = config.get('umap_neighbors', 15)
    umap_min_dist = config.get('umap_min_dist', 0.1)
    # t-SNE params
    tsne_perplexity = config.get('tsne_perplexity', 30.0)
    tsne_learning_rate = config.get('tsne_learning_rate', 'auto') # scikit-learn default
    tsne_n_iter = config.get('tsne_n_iter', 1000) # scikit-learn default

    # --- Basic Config Validation ---
    if not all([input_hdf5_file, output_hdf5_file, activation_types]):
        logger.error("Config missing required fields: 'input_hdf5_file', 'output_hdf5_file', 'activation_types'")
        return
    if reduction_method not in ['UMAP', 'TSNE']:
         logger.error(f"Invalid reduction_method: '{reduction_method}'. Choose 'UMAP' or 'TSNE'.")
         return
    if normalize_method and normalize_method.upper() not in ['L2']: # Add other norms if needed
         logger.warning(f"Unsupported normalize method: '{normalize_method}'. Normalization will be skipped.")
         normalize_method = None

    logger.info(f"Input HDF5: {input_hdf5_file}")
    logger.info(f"Output HDF5: {output_hdf5_file}")
    logger.info(f"Processing Prompts: {prompt_ids_spec}")
    logger.info(f"Processing Activation Types: {activation_types}")
    logger.info(f"Processing Layers: {layer_indices_spec}")
    logger.info(f"Normalization: {normalize_method}")
    logger.info(f"PCA: {use_pca}, Components: {pca_components}")
    logger.info(f"Reduction Method: {reduction_method}, Components: {final_components}")
    if reduction_method == 'UMAP':
        logger.info(f"UMAP Params: n_neighbors={umap_neighbors}, min_dist={umap_min_dist}")
    else: # t-SNE
        logger.info(f"t-SNE Params: perplexity={tsne_perplexity}, learning_rate={tsne_learning_rate}, n_iter={tsne_n_iter}")


    # --- Prepare Output File ---
    output_dir = os.path.dirname(output_hdf5_file)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        except OSError as e:
             logger.error(f"Failed to create output directory {output_dir}: {e}")
             return

    try:
        # Open output in append mode to allow resuming
        output_f = h5py.File(output_hdf5_file, 'a')
        # Store config used for this reduction run as top-level attributes
        if 'config' not in output_f.attrs: # Only save config if not already present
            save_config_to_hdf5_attrs(output_f, config)
            output_f.flush()
            logger.info(f"Saved reduction configuration to attributes of {output_hdf5_file}")
        else:
             logger.info(f"Output file {output_hdf5_file} already exists with configuration attributes.")

    except Exception as e:
        logger.error(f"Failed to open or write config to output HDF5 file {output_hdf5_file}: {e}")
        return

    # --- Open Input File and Process ---
    try:
        with h5py.File(input_hdf5_file, 'r') as input_f:
            # Determine which prompts to process
            all_prompt_ids_in_file = list(input_f.keys())
            if not all_prompt_ids_in_file:
                 logger.error(f"Input HDF5 file {input_hdf5_file} contains no prompts.")
                 output_f.close()
                 return

            if prompt_ids_spec == "__ALL__":
                prompts_to_process = all_prompt_ids_in_file
            elif isinstance(prompt_ids_spec, list):
                prompts_to_process = [pid for pid in prompt_ids_spec if pid in input_f]
                missing_prompts = set(prompt_ids_spec) - set(prompts_to_process)
                if missing_prompts:
                    logger.warning(f"Specified prompt IDs not found in input HDF5: {missing_prompts}")
            else:
                logger.error("Invalid 'prompt_ids' format in config. Use list or '__ALL__'.")
                output_f.close()
                return

            if not prompts_to_process:
                 logger.error("No valid prompts selected for processing.")
                 output_f.close()
                 return

            logger.info(f"Will process {len(prompts_to_process)} prompts.")

            # Determine layer indices (needs max layer count from first valid prompt)
            max_layers_found = 0
            for pid in prompts_to_process:
                try:
                    layer_keys = [k for k in input_f[pid].keys() if k.startswith('layer_')]
                    if layer_keys:
                        max_layers_found = max(max_layers_found, max(int(k.split('_')[-1]) for k in layer_keys) + 1)
                except Exception:
                    logger.warning(f"Could not determine layer count for prompt {pid}. Skipping.")
            if max_layers_found == 0:
                logger.error("Could not determine number of layers from input file.")
                output_f.close()
                return

            layer_indices = parse_layer_indices(layer_indices_spec, max_layers_found)
            if layer_indices is None:
                logger.error("Failed to parse layer indices. Exiting.")
                output_f.close()
                return
            logger.info(f"Target layer indices: {layer_indices}")


            # --- Main Processing Loop ---
            total_processed_count = 0
            for i, prompt_id in enumerate(prompts_to_process):
                logger.info(f"--- Processing Prompt {i+1}/{len(prompts_to_process)}: {prompt_id} ---")
                prompt_group_in = input_f[prompt_id]

                # Create prompt group in output file if it doesn't exist
                if prompt_id not in output_f:
                    prompt_group_out = output_f.create_group(prompt_id)
                    # Copy attributes (metadata) from input prompt group
                    for key, value in prompt_group_in.attrs.items():
                        try:
                            prompt_group_out.attrs[key] = value
                        except TypeError:
                            prompt_group_out.attrs[key] = str(value) # Fallback
                else:
                    prompt_group_out = output_f[prompt_id]
                    logger.info(f"Prompt group {prompt_id} already exists in output file.")


                for act_type in activation_types:
                    logger.info(f"Processing activation type: {act_type}")

                    # --- Check if already processed in output file ---
                    if act_type in prompt_group_out:
                        logger.warning(f"'{act_type}' embeddings for prompt '{prompt_id}' already exist in output file. Skipping.")
                        continue

                    # --- Load, Stack, and Prepare Metadata ---
                    start_load_stack_time = time.time()
                    stacked_activations = []
                    metadata_rows = [] # List of dicts
                    sequence_length = prompt_group_in.attrs.get('sequence_length')
                    shift_idx = prompt_group_in.attrs.get('shift_token_idx', -1)
                    prompt_text = prompt_group_in.attrs.get('text', '')

                    if sequence_length is None:
                        logger.error(f"Missing 'sequence_length' for prompt {prompt_id}. Cannot process {act_type}.")
                        continue

                    valid_load = True
                    for layer_idx in layer_indices:
                        layer_name = f'layer_{layer_idx}'
                        if layer_name not in prompt_group_in or act_type not in prompt_group_in[layer_name]:
                            logger.error(f"Missing data for {layer_name}/{act_type} in prompt {prompt_id}. Skipping this activation type.")
                            valid_load = False
                            break
                        # Load data: shape (sequence_length, hidden_size)
                        layer_data = prompt_group_in[layer_name][act_type][:]
                        stacked_activations.append(layer_data)
                        # Append metadata for each token in this layer
                        for token_idx in range(sequence_length):
                            metadata_rows.append({
                                "layer": layer_idx,
                                "token_idx": token_idx,
                                # Add other potentially useful metadata here later if needed
                            })

                    if not valid_load or not stacked_activations:
                        continue # Skip to next activation type or prompt

                    # Combine layer data: shape (num_layers * sequence_length, hidden_size)
                    combined_data = np.vstack(stacked_activations)
                    metadata_df = pd.DataFrame(metadata_rows) # For easier handling if needed, though we save arrays

                    logger.info(f" -> Loaded and stacked {combined_data.shape[0]} vectors in {time.time() - start_load_stack_time:.2f}s")

                    # --- Preprocessing: Normalization ---
                    if normalize_method == 'L2':
                        logger.info(" -> Applying L2 normalization...")
                        combined_data = sk_normalize(combined_data, norm='l2', axis=1)

                    # --- Reduction Step 1: PCA ---
                    if use_pca:
                        logger.info(f" -> Applying PCA ({pca_components} components)...")
                        start_pca_time = time.time()
                        # Ensure n_components is not larger than samples or features
                        n_samples, n_features = combined_data.shape
                        current_pca_components = min(pca_components, n_samples, n_features)
                        if current_pca_components < pca_components:
                             logger.warning(f"Reducing PCA components from {pca_components} to {current_pca_components} due to data shape.")

                        if current_pca_components <= 0:
                             logger.error("Cannot apply PCA with 0 or fewer components. Skipping reduction.")
                             continue

                        pca = PCA(n_components=current_pca_components, random_state=42)
                        reduced_data = pca.fit_transform(combined_data)
                        explained_variance = np.sum(pca.explained_variance_ratio_)
                        logger.info(f" -> PCA finished in {time.time() - start_pca_time:.2f}s. Explained Var: {explained_variance:.4f}")
                    else:
                        logger.info(" -> Skipping PCA.")
                        reduced_data = combined_data # Pass original (maybe normalized) data to next step
                        explained_variance = None

                    # --- Reduction Step 2: UMAP / t-SNE ---
                    logger.info(f" -> Applying {reduction_method} ({final_components} components)...")
                    start_final_red_time = time.time()
                    if reduction_method == 'UMAP':
                        reducer = UMAP(
                            n_components=final_components,
                            n_neighbors=umap_neighbors,
                            min_dist=umap_min_dist,
                            random_state=42,
                            n_jobs=1 # Often more stable
                        )
                    else: # t-SNE
                         # Adjust perplexity if needed (must be < n_samples)
                        current_perplexity = min(tsne_perplexity, reduced_data.shape[0] - 1)
                        if current_perplexity < tsne_perplexity:
                             logger.warning(f"Reducing t-SNE perplexity from {tsne_perplexity} to {current_perplexity:.1f} due to data shape.")
                        if current_perplexity <= 0:
                             logger.error("Cannot apply t-SNE with perplexity <= 0. Skipping reduction.")
                             continue

                        reducer = TSNE(
                            n_components=final_components,
                            perplexity=current_perplexity,
                            learning_rate=tsne_learning_rate,
                            n_iter=tsne_n_iter,
                            init='pca', # Use PCA init for stability
                            random_state=42,
                            n_jobs=-1 # Use all available cores
                        )

                    final_embeddings = reducer.fit_transform(reduced_data)
                    logger.info(f" -> {reduction_method} finished in {time.time() - start_final_red_time:.2f}s.")

                    # --- Save Results to Output HDF5 ---
                    logger.info(f" -> Saving results for {act_type}...")
                    try:
                        act_type_group = prompt_group_out.create_group(act_type)
                        # Save embeddings (shape: num_vectors, final_components)
                        act_type_group.create_dataset(
                            'embeddings',
                            data=final_embeddings.astype(np.float32), # Store as float32
                            compression="gzip"
                        )
                        # Save corresponding metadata as separate arrays
                        act_type_group.create_dataset(
                            'layer_indices',
                            data=metadata_df['layer'].values,
                            dtype='int16', # Smaller int type
                            compression="gzip"
                        )
                        act_type_group.create_dataset(
                            'token_indices',
                            data=metadata_df['token_idx'].values,
                            dtype='int16',
                            compression="gzip"
                        )
                        # Save PCA variance if calculated
                        if explained_variance is not None:
                            act_type_group.attrs['pca_explained_variance'] = explained_variance

                        output_f.flush() # Write to disk
                        logger.info(f" -> Successfully saved embeddings and metadata for {act_type}.")
                        total_processed_count += 1

                    except Exception as e:
                         logger.error(f" -> Failed to save results for {prompt_id}/{act_type}: {e}", exc_info=True)
                         # Clean up potentially partially created group
                         if act_type in prompt_group_out:
                             del prompt_group_out[act_type]


    except FileNotFoundError:
        logger.error(f"Input HDF5 file not found: {input_hdf5_file}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during processing: {e}", exc_info=True)
    finally:
        # --- Cleanup ---
        if 'output_f' in locals() and output_f:
            logger.info(f"Closing output HDF5 file: {output_hdf5_file}")
            output_f.close()

    logger.info(f"--- Reduction Process Finished ---")
    logger.info(f"Successfully processed and saved embeddings for {total_processed_count} prompt/activation combinations.")

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform dimensionality reduction on captured LLM activations."
    )
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the JSON configuration file for the reduction process."
    )
    args = parser.parse_args()

    run_reduction(args.config_file)

