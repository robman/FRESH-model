import argparse
import json
import logging
import math
import os
import re
import sys
import time
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Define metric output filenames
METRIC_FILENAMES = {
    "trajectory_length": "{base}-trajectory-length.csv",
    "local_curvature": "{base}-local-curvature.csv",
    "layerwise_deviation": "{base}-layerwise-deviation.csv",
    "direction_deviation": "{base}-direction-deviation.csv",
    "cosine_similarity": "{base}-cosine-similarity.csv",
}

# --- Helper Functions ---

def get_config_value(config, key, default=None):
    """Safely gets a value from the config dictionary."""
    return config.get(key, default)

def get_hdf5_attr(hdf5_object, attr_name, default=None):
     """Safely gets an attribute from an HDF5 object."""
     try:
         value = hdf5_object.attrs.get(attr_name, default)
         if isinstance(value, (bytes, str)):
             try:
                 value_str = value.decode('utf-8') if isinstance(value, bytes) else value
                 if value_str.startswith('[') or value_str.startswith('{'):
                    # Try decoding potential JSON strings (like lists)
                    return json.loads(value_str)
                 elif value_str == 'None': return None
                 elif value_str.lower() == 'true': return True
                 elif value_str.lower() == 'false': return False
                 else: return value_str # Return string if not JSON/None/Bool
             except (json.JSONDecodeError, UnicodeDecodeError):
                 return value # Return original value if decoding fails
         # Handle numpy types explicitly
         elif isinstance(value, np.bool_): return bool(value)
         elif isinstance(value, (np.int64, np.int32, np.int16)): return int(value)
         elif isinstance(value, (np.float64, np.float32)): return float(value)
         # Return other types directly
         return value
     except Exception as e:
         logger.warning(f"Could not read attribute '{attr_name}'. Error: {e}. Using default: {default}")
         return default

def parse_prompt_ids(ids_str):
    """Parses a comma-separated string of prompt IDs into a list."""
    if not ids_str:
        return None
    return [pid.strip() for pid in ids_str.split(',') if pid.strip()] # Ensure non-empty strings

def find_cs_ctrl_pairs(prompt_ids):
    """
    Identify (cs, ctrl) prompt pairs by matching the core ID segment before variation types.
    Supports IDs like 'prefix_03_ctrl' vs 'prefix_03_pos_mod_cs'.
    """
    pairs = []
    #grouped_prompts = defaultdict(dict)
    grouped_prompts = defaultdict(lambda: {'cs': [], 'ctrl': []})

    # Regex to extract base like 'emotional_analytical_03'
    base_pattern = re.compile(r'^(.*?_\d+)_')

    for pid in prompt_ids:
        base_match = base_pattern.match(pid)
        if not base_match:
            logger.warning(f"Could not extract base from prompt ID '{pid}'")
            continue

        base = base_match.group(1)
        if pid.endswith('_cs'):
            grouped_prompts[base]['cs'].append(pid)
        elif pid.endswith('_ctrl'):
            grouped_prompts[base]['ctrl'].append(pid)

    for base, group in grouped_prompts.items():
        cs_list = group.get('cs', [])
        ctrl_list = group.get('ctrl', [])

        if not cs_list or not ctrl_list:
            logger.warning(f"No valid pair found for base '{base}'")
            continue

        for cs in cs_list:
            # Match any one ctrl from same group — if multiple, this can be refined
            pairs.append((cs, ctrl_list[0]))

    return pairs

def calculate_angle(v1, v2, epsilon=1e-9):
    """Calculates the angle between two vectors in degrees."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < epsilon or norm2 < epsilon:
        return np.nan # Cannot compute angle if one vector is zero

    # Clip dot product to avoid floating point errors outside [-1, 1]
    dot_product = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    return np.degrees(angle_rad)

def append_to_csv(filepath, data_dict_list, expected_headers):
    """Appends a list of dictionaries to a CSV file, writing headers if needed."""
    if not data_dict_list: # Don't do anything if list is empty
        return
    file_exists = os.path.isfile(filepath)
    is_empty = not file_exists or os.path.getsize(filepath) == 0
    try:
        # Use pandas for robust CSV writing
        df = pd.DataFrame(data_dict_list)
        # Reorder columns to match expected headers, adding missing columns with None
        # Ensure all expected headers are present before writing
        for header in expected_headers:
            if header not in df.columns:
                df[header] = None # Add missing column initialized to None/NaN
        df = df[expected_headers] # Enforce column order

        df.to_csv(filepath, mode='a', header=is_empty, index=False, lineterminator='\n') # Specify lineterminator
        # logger.debug(f"Appended {len(data_dict_list)} rows to {filepath}")
    except Exception as e:
        logger.error(f"Failed to append data to CSV file {filepath}: {e}", exc_info=True)


# --- Metric Calculation Functions ---

def calculate_trajectory_length(embeddings):
    """Calculates the Euclidean length of a trajectory."""
    if embeddings.shape[0] < 2: return 0.0
    diffs = np.diff(embeddings, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return np.sum(segment_lengths)

def calculate_local_curvature(embeddings, window=3):
    """Calculates average local curvature (angle change) over a sliding window."""
    if embeddings.shape[0] < window: return np.nan
    angles = []
    for i in range(embeddings.shape[0] - (window - 1)):
        p1, p2, p3 = embeddings[i:i+window]
        v1 = p2 - p1
        v2 = p3 - p2
        angle = calculate_angle(v1, v2)
        if not np.isnan(angle):
            angles.append(angle)
    return np.mean(angles) if angles else np.nan

# --- Main Analysis Function ---

def run_metric_analysis(config_path, output_base_name, specific_prompt_ids=None):
    """
    Loads reduction config, embeddings, raw activations, calculates metrics,
    and appends results to CSV files.

    Args:
        config_path (str): Path to the reduction JSON config file.
        output_base_name (str): Base name/path for the output CSV metric files.
        specific_prompt_ids (list[str], optional): List of specific prompt IDs to process.
                                                  If None, processes all prompts found.
    """
    logger.info(f"Loading reduction configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            reduction_config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load or parse reduction config {config_path}: {e}")
        return

    # --- Extract parameters and file paths ---
    embeddings_hdf5_path = get_config_value(reduction_config, 'output_hdf5_file')
    capture_hdf5_path = get_config_value(reduction_config, 'input_hdf5_file') # Path to original activations
    activation_types_reduced = get_config_value(reduction_config, 'activation_types', []) # Types used for reduction

    if not embeddings_hdf5_path: logger.error("Config missing 'output_hdf5_file'."); return
    if not capture_hdf5_path: logger.error("Config missing 'input_hdf5_file' (for capture data)."); return
    if not activation_types_reduced: logger.error("Config missing 'activation_types'."); return

    # Assume only one activation type was reduced per file
    reduced_activation_type = activation_types_reduced[0]
    logger.info(f"Analysing embeddings for activation type: {reduced_activation_type}")
    logger.info(f"Using embeddings file: {embeddings_hdf5_path}")
    logger.info(f"Using capture file (for cosine sim): {capture_hdf5_path}")

    # --- Prepare Output Files ---
    output_files = {
        metric: path.format(base=output_base_name)
        for metric, path in METRIC_FILENAMES.items()
    }
    # Ensure output directory exists for metric files
    output_dir = os.path.dirname(output_base_name)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logger.info(f"Created output directory for metrics: {output_dir}")
        except OSError as e:
             logger.error(f"Failed to create output directory {output_dir}: {e}")
             # Continue, maybe files can be written to current dir

    # Define headers for each CSV
    base_headers = ["config_basename", "prompt_id", "activation_type_reduced"] # Common context
    config_param_headers = ["reduction_method", "final_components", "pca_components", "norm_method", "umap_neighbors", "umap_min_dist", "tsne_perplexity"] # Add more reduction params if needed

    csv_headers = {
        "trajectory_length": base_headers + ["token_idx", "trajectory_length"] + config_param_headers,
        "local_curvature": base_headers + ["token_idx", "avg_local_curvature"] + config_param_headers,
        # CORRECTED: Added prompt_id column (populated by cs_id)
        "layerwise_deviation": [base_headers[0], "prompt_id", "prompt_id_cs", "prompt_id_ctrl", "layer", "avg_euclidean_distance_layer", base_headers[2]] + config_param_headers,
        "direction_deviation": [base_headers[0], "prompt_id", "prompt_id_cs", "prompt_id_ctrl", "layer_transition", "avg_angle_degrees_layer", base_headers[2]] + config_param_headers,
        "cosine_similarity": [base_headers[0], "prompt_id", "prompt_id_cs", "prompt_id_ctrl", "layer", "avg_cosine_similarity_layer", "activation_type_capture"] + config_param_headers,
    }
    config_basename = os.path.splitext(os.path.basename(config_path))[0]

    # --- Load Data and Config from Embeddings File ---
    try:
        with h5py.File(embeddings_hdf5_path, 'r') as f_emb:
            all_prompt_ids_in_file = list(f_emb.keys())
            if not all_prompt_ids_in_file: logger.error("Embeddings file contains no prompts."); return

            # --- Determine Prompts to Process ---
            if specific_prompt_ids:
                prompts_to_process = [pid for pid in specific_prompt_ids if pid in all_prompt_ids_in_file]
                missing = set(specific_prompt_ids) - set(prompts_to_process)
                if missing:
                    logger.warning(f"Specified prompt IDs not found in embeddings file: {missing}")
                logger.info(f"Processing specified prompts: {prompts_to_process}")
            else:
                prompts_to_process = all_prompt_ids_in_file
                logger.info("Processing all prompts found in the embeddings file.")

            if not prompts_to_process:
                 logger.error("No prompts selected or found to process.")
                 return

            # Load reduction config attributes from embeddings file
            stored_config = {key: get_hdf5_attr(f_emb, key) for key in f_emb.attrs.keys()}
            logger.info(f"Embeddings file generated with stored config: {stored_config}")
            # Extract relevant params for CSV context
            config_context = {
                "reduction_method": stored_config.get('reduction_method', 'UNKNOWN'),
                "final_components": stored_config.get('final_components', 'N/A'),
                "pca_components": stored_config.get('pca_components', 'N/A') if stored_config.get('use_pca', False) else "None",
                "norm_method": stored_config.get('normalize', 'None'),
                "umap_neighbors": stored_config.get('umap_neighbors', 'N/A'),
                "umap_min_dist": stored_config.get('umap_min_dist', 'N/A'),
                "tsne_perplexity": stored_config.get('tsne_perplexity', 'N/A'),
            }

            # --- Loop 1: General Metrics (Length, Curvature) ---
            logger.info(f"Calculating general metrics for {len(prompts_to_process)} prompts...")
            trajectory_length_results = []
            local_curvature_results = []

            for prompt_id in prompts_to_process: # Use the filtered list
                # Skip paired metrics calculation in this loop if it's a _cs or _ctrl prompt
                if prompt_id.endswith("_cs") or prompt_id.endswith("_ctrl"):
                    logger.debug(f"Skipping general metrics for paired prompt: {prompt_id}")
                    continue

                logger.debug(f"Processing general metrics for prompt: {prompt_id}")
                if reduced_activation_type not in f_emb[prompt_id]:
                    logger.warning(f"Reduced activation type '{reduced_activation_type}' not found for prompt '{prompt_id}'. Skipping general metrics.")
                    continue

                act_group = f_emb[prompt_id][reduced_activation_type]
                embeddings = act_group['embeddings'][:]
                layer_indices = act_group['layer_indices'][:]
                token_indices = act_group['token_indices'][:]

                # Group embeddings by token index
                embeddings_by_token = defaultdict(list)
                for i, token_idx in enumerate(token_indices):
                    embeddings_by_token[token_idx].append((layer_indices[i], embeddings[i]))

                # Sort layers within each token trajectory
                token_trajectories = {}
                max_layers_in_prompt = 0
                for token_idx, layer_emb_pairs in embeddings_by_token.items():
                    sorted_pairs = sorted(layer_emb_pairs, key=lambda x: x[0])
                    if sorted_pairs:
                         token_trajectories[token_idx] = np.array([emb for layer, emb in sorted_pairs])
                         max_layers_in_prompt = max(max_layers_in_prompt, sorted_pairs[-1][0] + 1)


                # Calculate metrics per token
                for token_idx, trajectory in token_trajectories.items():
                    length = calculate_trajectory_length(trajectory)
                    curvature = calculate_local_curvature(trajectory)

                    base_result = {"config_basename": config_basename, "prompt_id": prompt_id, "activation_type_reduced": reduced_activation_type, **config_context}
                    trajectory_length_results.append({**base_result, "token_idx": token_idx, "trajectory_length": length})
                    local_curvature_results.append({**base_result, "token_idx": token_idx, "avg_local_curvature": curvature})

            # Append results to CSV
            append_to_csv(output_files["trajectory_length"], trajectory_length_results, csv_headers["trajectory_length"])
            append_to_csv(output_files["local_curvature"], local_curvature_results, csv_headers["local_curvature"])
            logger.info(f"Finished calculating general metrics.")


            # --- Loop 2: Paired Metrics (Deviation, Angle, Cosine Sim) ---
            logger.info("Identifying CS/CTRL pairs from the set of prompts to process...")
            # Find pairs ONLY within the selected prompts_to_process list
            cs_ctrl_pairs = find_cs_ctrl_pairs(prompts_to_process)
            logger.info(f"Found {len(cs_ctrl_pairs)} CS/CTRL pairs for paired metrics calculation.")

            if not cs_ctrl_pairs:
                logger.info("No CS/CTRL pairs found in the selected prompts. Skipping paired metrics.")
                # Close the embeddings file before returning (already handled by outer try/except)
                # return # Exit if no pairs

            else:
                # Define lists to hold results for appending *after* the loop
                all_layerwise_deviation_results = []
                all_direction_deviation_results = []
                all_cosine_similarity_results = []

                # Need to open the capture file for cosine similarity
                try:
                    with h5py.File(capture_hdf5_path, 'r') as f_cap:
                        logger.info("Calculating paired metrics...")
                        for cs_id, ctrl_id in cs_ctrl_pairs:
                            logger.debug(f"Processing pair: {cs_id} vs {ctrl_id}")

                            # --- Load Embeddings for Pair ---
                            if reduced_activation_type not in f_emb[cs_id] or reduced_activation_type not in f_emb[ctrl_id]:
                                logger.warning(f"Reduced activation type '{reduced_activation_type}' missing for pair {cs_id}/{ctrl_id}. Skipping pair.")
                                continue

                            emb_cs = f_emb[cs_id][reduced_activation_type]['embeddings'][:]
                            layers_cs = f_emb[cs_id][reduced_activation_type]['layer_indices'][:]
                            tokens_cs = f_emb[cs_id][reduced_activation_type]['token_indices'][:]

                            emb_ctrl = f_emb[ctrl_id][reduced_activation_type]['embeddings'][:]
                            layers_ctrl = f_emb[ctrl_id][reduced_activation_type]['layer_indices'][:]
                            tokens_ctrl = f_emb[ctrl_id][reduced_activation_type]['token_indices'][:]

                            # --- Load Raw Activations for Pair ---
                            raw_activations_cs = {}
                            raw_activations_ctrl = {}
                            # Determine available activation types from capture file safely
                            capture_activation_types = []
                            if cs_id in f_cap and 'layer_0' in f_cap[cs_id]:
                                capture_activation_types = list(f_cap[cs_id]['layer_0'].keys())
                            elif ctrl_id in f_cap and 'layer_0' in f_cap[ctrl_id]:
                                 capture_activation_types = list(f_cap[ctrl_id]['layer_0'].keys())

                            if not capture_activation_types:
                                 logger.warning(f"Could not determine capture activation types for pair {cs_id}/{ctrl_id}. Skipping cosine similarity.")


                            max_layer = max(np.max(layers_cs) if len(layers_cs)>0 else -1,
                                            np.max(layers_ctrl) if len(layers_ctrl)>0 else -1)
                            min_seq_len = min(get_hdf5_attr(f_emb[cs_id], 'sequence_length', 0),
                                              get_hdf5_attr(f_emb[ctrl_id], 'sequence_length', 0))

                            if max_layer < 0 or min_seq_len == 0:
                                logger.warning(f"Invalid layer/sequence length for pair {cs_id}/{ctrl_id}. Skipping.")
                                continue


                            # Load raw data layer by layer for memory efficiency
                            for layer_idx in range(max_layer + 1):
                                 layer_name = f"layer_{layer_idx}"
                                 if layer_name in f_cap.get(cs_id, {}):
                                     raw_activations_cs[layer_idx] = {atype: f_cap[cs_id][layer_name][atype][:min_seq_len]
                                                                      for atype in capture_activation_types if atype in f_cap[cs_id][layer_name]}
                                 if layer_name in f_cap.get(ctrl_id, {}):
                                     raw_activations_ctrl[layer_idx] = {atype: f_cap[ctrl_id][layer_name][atype][:min_seq_len]
                                                                        for atype in capture_activation_types if atype in f_cap[ctrl_id][layer_name]}

                            # --- Reconstruct Trajectories from Embeddings ---
                            traj_cs = defaultdict(lambda: np.empty((0, emb_cs.shape[1])))
                            traj_ctrl = defaultdict(lambda: np.empty((0, emb_ctrl.shape[1])))

                            temp_cs = defaultdict(list)
                            for i, token_idx in enumerate(tokens_cs):
                                 if token_idx < min_seq_len: temp_cs[token_idx].append((layers_cs[i], emb_cs[i]))
                            for token_idx, pairs in temp_cs.items():
                                 sorted_pairs = sorted(pairs, key=lambda x: x[0])
                                 if sorted_pairs: traj_cs[token_idx] = np.array([emb for layer, emb in sorted_pairs])

                            temp_ctrl = defaultdict(list)
                            for i, token_idx in enumerate(tokens_ctrl):
                                 if token_idx < min_seq_len: temp_ctrl[token_idx].append((layers_ctrl[i], emb_ctrl[i]))
                            for token_idx, pairs in temp_ctrl.items():
                                 sorted_pairs = sorted(pairs, key=lambda x: x[0])
                                 if sorted_pairs: traj_ctrl[token_idx] = np.array([emb for layer, emb in sorted_pairs])


                            # --- Calculate Layer-wise Metrics ---
                            # CORRECTED: Add prompt_id (using cs_id) to base result
                            base_result_pair = {"config_basename": config_basename, "prompt_id": cs_id, "prompt_id_cs": cs_id, "prompt_id_ctrl": ctrl_id, "activation_type_reduced": reduced_activation_type, **config_context}

                            for layer_idx in range(max_layer + 1): # Iterate through layers
                                # Euclidean Deviation
                                dist_layer = [euclidean(traj_cs[tok][layer_idx], traj_ctrl[tok][layer_idx])
                                              for tok in range(min_seq_len)
                                              if layer_idx < len(traj_cs.get(tok,[])) and layer_idx < len(traj_ctrl.get(tok,[]))]
                                avg_dist_layer = np.mean(dist_layer) if dist_layer else np.nan
                                all_layerwise_deviation_results.append({**base_result_pair, "layer": layer_idx, "avg_euclidean_distance_layer": avg_dist_layer})

                                # Direction Deviation (L -> L+1)
                                if layer_idx < max_layer:
                                    angles_layer = []
                                    for tok in range(min_seq_len):
                                         # Check if data exists for L and L+1 for this token in both trajectories
                                         if (layer_idx + 1 < len(traj_cs.get(tok,[])) and
                                             layer_idx + 1 < len(traj_ctrl.get(tok,[]))):
                                             vec_cs = traj_cs[tok][layer_idx+1] - traj_cs[tok][layer_idx]
                                             vec_ctrl = traj_ctrl[tok][layer_idx+1] - traj_ctrl[tok][layer_idx]
                                             angle = calculate_angle(vec_cs, vec_ctrl)
                                             if not np.isnan(angle):
                                                 angles_layer.append(angle)
                                    avg_angle_layer = np.mean(angles_layer) if angles_layer else np.nan
                                    all_direction_deviation_results.append({**base_result_pair, "layer_transition": f"L{layer_idx}_L{layer_idx+1}", "avg_angle_degrees_layer": avg_angle_layer})

                                # Cosine Similarity (Raw Activations)
                                for capture_act_type in capture_activation_types:
                                    if layer_idx in raw_activations_cs and layer_idx in raw_activations_ctrl and \
                                       capture_act_type in raw_activations_cs[layer_idx] and capture_act_type in raw_activations_ctrl[layer_idx]:

                                        raw_cs = raw_activations_cs[layer_idx][capture_act_type] # Shape (min_seq_len, hidden_size)
                                        raw_ctrl = raw_activations_ctrl[layer_idx][capture_act_type]

                                        if raw_cs.shape[0] != min_seq_len or raw_ctrl.shape[0] != min_seq_len:
                                             logger.warning(f"Shape mismatch in raw activations for layer {layer_idx}, type {capture_act_type}, pair {cs_id}/{ctrl_id}. Skipping cosine sim.")
                                             continue
                                        if raw_cs.shape[0] == 0: continue # Skip if no tokens

                                        # Calculate cosine sim row-wise (per token) then average
                                        # Ensure float64 for precision in intermediate steps
                                        sims = cosine_similarity(raw_cs.astype(np.float64), raw_ctrl.astype(np.float64)).diagonal()
                                        avg_sim_layer = np.mean(sims) if len(sims) > 0 else np.nan

                                        # Add result, note the activation type is the *capture* type
                                        # CORRECTED: Include prompt_id (using cs_id)
                                        all_cosine_similarity_results.append({
                                            "config_basename": config_basename,
                                            "prompt_id": cs_id, # Added
                                            "prompt_id_cs": cs_id,
                                            "prompt_id_ctrl": ctrl_id,
                                            "layer": layer_idx,
                                            "avg_cosine_similarity_layer": avg_sim_layer,
                                            "activation_type_capture": capture_act_type, # Specify which raw activation
                                            **config_context # Add full reduction context
                                        })
                                    else:
                                         logger.debug(f"Skipping cosine similarity for layer {layer_idx}, type {capture_act_type} due to missing data in pair {cs_id}/{ctrl_id}")


                        # Append paired results to CSVs *after* iterating through all pairs
                        append_to_csv(output_files["layerwise_deviation"], all_layerwise_deviation_results, csv_headers["layerwise_deviation"])
                        append_to_csv(output_files["direction_deviation"], all_direction_deviation_results, csv_headers["direction_deviation"])
                        append_to_csv(output_files["cosine_similarity"], all_cosine_similarity_results, csv_headers["cosine_similarity"])
                        logger.info(f"Finished calculating paired metrics.")

                except FileNotFoundError:
                     logger.error(f"Capture HDF5 file not found: {capture_hdf5_path}. Skipping cosine similarity.")
                except Exception as e:
                     logger.error(f"Error during paired metric calculation: {e}", exc_info=True)


    except FileNotFoundError:
        logger.error(f"Embeddings HDF5 file not found: {embeddings_hdf5_path}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

    logger.info(f"--- Metric Analysis Finished ---")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate metrics from reduced LLM activation embeddings."
    )
    parser.add_argument(
        "reduction_config_file",
        type=str,
        help="Path to the JSON reduction configuration file used to generate the embeddings."
    )
    parser.add_argument(
        "--output_base",
        type=str,
        required=True,
        help="Base name/path for the output CSV metric files (e.g., 'analysis/full/metrics_umap_residual')."
    )
    # Add --prompt_ids argument for filtering
    parser.add_argument(
        "--prompt_ids",
        type=str,
        default=None,
        help="Optional comma-separated list of specific prompt IDs to process (e.g., 'id1,id2'). If None, processes all prompts."
    )

    args = parser.parse_args()

    # Parse optional prompt IDs
    prompt_ids_to_process = parse_prompt_ids(args.prompt_ids)
    if args.prompt_ids is not None and prompt_ids_to_process is None:
         logger.error("Invalid format provided for --prompt_ids.")
         sys.exit(1)

    run_metric_analysis(
        config_path=args.reduction_config_file,
        output_base_name=args.output_base,
        specific_prompt_ids=prompt_ids_to_process # Pass the parsed list or None
    )

