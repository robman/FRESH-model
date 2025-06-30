import argparse
import json
import logging
import os
import sys
from pathlib import Path
import numpy as np
import h5py
import pandas as pd

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Activation Type Constants
# -----------------------------------------------------------------------------
RESIDUAL_STREAM_TYPE = "residual_stream"
ATTENTION_OUTPUT_TYPE = "attn_output"
MLP_OUTPUT_TYPE = "mlp_output"

# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
def gdot(x: np.ndarray, y: np.ndarray, G: np.ndarray) -> np.ndarray:
    return np.einsum("...i,ij,...j->...", x, G, y)

# Note: cumulative_arc_length is not directly used for this salience definition,
# but gdot is. Salience here is per-segment norm, not cumulative.

# -----------------------------------------------------------------------------
# CSV helper
# -----------------------------------------------------------------------------
def append_to_csv(filepath: str, rows: list, expected_headers: list[str]):
    if not rows: return
    file_exists = os.path.isfile(filepath)
    is_empty = not file_exists or os.path.getsize(filepath) == 0
    df = pd.DataFrame(rows)
    for h in expected_headers:
        if h not in df.columns: df[h] = pd.NA
    df = df.reindex(columns=expected_headers)
    try:
        df.to_csv(filepath, mode='a', header=is_empty, index=False, lineterminator='\n')
    except Exception as e:
        logger.error(f"Error writing to CSV {filepath}: {e}")
        error_csv_path = filepath.replace(".csv", f"_error_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.csv")
        try: df.to_csv(error_csv_path, mode='w', header=True, index=False, lineterminator='\n')
        except Exception as e2: logger.error(f"Could not save data to fallback CSV {error_csv_path}: {e2}")

# -----------------------------------------------------------------------------
# Helper function to safely get activation for the "current" token
# -----------------------------------------------------------------------------
def get_current_token_activation(run_group, hdf5_layer_group_name: str, act_type: str,
                                 step_idx: int, pid: str, run_id: str):
    dataset_path = f"{hdf5_layer_group_name}/{act_type}/step_{step_idx}"
    try:
        matrix = run_group[dataset_path][:]
        if matrix.ndim == 2 and matrix.shape[0] >= 1:
            return matrix[0, :].astype(np.float64)
        elif matrix.ndim == 1 and matrix.shape[0] > 0 :
             logger.debug(f"      GenStep {step_idx}, {dataset_path}: Loaded 1D array (shape {matrix.shape}). Assuming current token. (pid: {pid}, run: {run_id})")
             return matrix.astype(np.float64)
        else:
            logger.debug(f"      GenStep {step_idx}, {dataset_path}: Activation tensor has unexpected shape "
                         f"{matrix.shape} or is empty. Cannot get current token. (pid: {pid}, run: {run_id})")
            return None
    except KeyError:
        logger.debug(f"      GenStep {step_idx}, {dataset_path}: Dataset not found. "
                     f"(pid: {pid}, run: {run_id})")
        return None
    except Exception as e:
        logger.error(f"      GenStep {step_idx}, {dataset_path}: Unexpected error loading slice: {e}. "
                      f"(pid: {pid}, run: {run_id})")
        return None

# -----------------------------------------------------------------------------
# Unnormalized trajectory reconstruction
# -----------------------------------------------------------------------------
def reconstruct_unnormalized_trajectory(run_group, num_layers: int, step_idx: int, pid: str, run_id: str):
    """
    Reconstruct the unnormalized trajectory x_unnorm^0, x_unnorm^1, ..., x_unnorm^L
    where x_unnorm^{l+1} = x_unnorm^l + attention_output^l + mlp_output^l
    """
    # Get x^0 (embedding layer output)
    x0 = get_current_token_activation(run_group, "layer_0", RESIDUAL_STREAM_TYPE, step_idx, pid, run_id)
    if x0 is None:
        logger.debug(f"      GenStep {step_idx}: Could not get x^0 for unnormalized trajectory reconstruction")
        return None
    
    trajectory = [x0.copy()]
    current_x = x0.copy()
    
    # Reconstruct x^1 through x^L
    for layer_idx in range(num_layers):
        # Get attention and MLP outputs for this layer
        attn_output = get_current_token_activation(run_group, f"layer_{layer_idx}", ATTENTION_OUTPUT_TYPE, step_idx, pid, run_id)
        mlp_output = get_current_token_activation(run_group, f"layer_{layer_idx}", MLP_OUTPUT_TYPE, step_idx, pid, run_id)
        
        if attn_output is None or mlp_output is None:
            logger.debug(f"      GenStep {step_idx}: Missing attention or MLP output at layer {layer_idx} for unnormalized trajectory")
            return None
        
        # Update: x^{l+1} = x^l + attention^l + mlp^l
        current_x = current_x + attn_output + mlp_output
        trajectory.append(current_x.copy())
    
    return np.array(trajectory, dtype=np.float64)

# -----------------------------------------------------------------------------
# Core analysis routine
# -----------------------------------------------------------------------------
def run_path_salience_analysis_ci02_style(reduction_config_path: str,
                                          output_base_name: str,
                                          specific_prompt_ids: list[str] | None = None,
                                          dump_salience_series: bool = False,
                                          use_unnormalized: bool = False):

    logger.info(f"Loading config for CI02-style path salience: {reduction_config_path}")
    try:
        with open(reduction_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load or parse config {reduction_config_path}: {e}")
        return

    hdf5_path_str = config.get('input_hdf5_file')
    unembed_path_str = config.get('unembedding_npz_file')
    # activation_type is fixed to RESIDUAL_STREAM_TYPE for this script
    activation_type_for_display = RESIDUAL_STREAM_TYPE 

    if not all([hdf5_path_str, unembed_path_str]):
        logger.error("Config missing 'input_hdf5_file' or 'unembedding_npz_file'.")
        return
    
    hdf5_path = Path(hdf5_path_str)
    unembed_path = Path(unembed_path_str)

    if not hdf5_path.exists():
        logger.error(f"HDF5 file not found: {hdf5_path}")
        return
    if not unembed_path.exists():
        logger.error(f"Unembedding NPZ file not found: {unembed_path}")
        return
        
    try:
        with np.load(unembed_path) as data:
            if 'G' not in data:
                logger.error(f"Metric 'G' not found in unembedding NPZ file: {unembed_path}")
                return
            G = data['G']
        logger.info(f"Loaded pull‑back metric G from {unembed_path}, shape: {G.shape}")
    except Exception as e:
        logger.error(f"Error loading metric G from {unembed_path}: {e}")
        return

    # Update output filename to indicate trajectory type
    trajectory_suffix = "unnormalized" if use_unnormalized else "residual"
    output_csv_file = Path(f"{output_base_name}-{trajectory_suffix}-path-salience.csv")
    output_csv_file.parent.mkdir(parents=True, exist_ok=True)

    csv_headers = [
        "config_basename", "prompt_id", "run_id", "generation_step_idx",
        "activation_trajectory_type", # Will denote this is residual stream salience
        "mean_salience", "max_salience", "layer_idx_of_max_salience", "total_salience",
        "num_segments_in_salience_series", "trajectory_num_points" # L segments for L+1 points
    ]
    if dump_salience_series:
        csv_headers.append("salience_series") # Series of ||x^(l+1)-x^(l)||_G

    config_basename = Path(reduction_config_path).stem
    results_buffer = []
    
    # Keep trajectory type name consistent for downstream compatibility
    TRAJECTORY_TYPE_NAME = "residual_stream_salience_per_gen_step"
    
    if use_unnormalized:
        logger.info("Using UNNORMALIZED trajectory reconstruction (x^0 + cumulative attention/MLP outputs)")
    else:
        logger.info("Using RESIDUAL STREAM trajectory (layer input states)")

    try:
        with h5py.File(hdf5_path, 'r') as f_hdf5:
            prompt_ids_to_process = []
            if specific_prompt_ids:
                for pid_spec in specific_prompt_ids:
                    if pid_spec in f_hdf5:
                        prompt_ids_to_process.append(pid_spec)
                    else:
                        logger.warning(f"Specified prompt_id '{pid_spec}' not found in HDF5 file. Skipping.")
            else:
                prompt_ids_to_process = list(f_hdf5.keys())
            
            if not prompt_ids_to_process:
                logger.warning("No prompt IDs to process. Exiting.")
                return
            logger.info(f"Processing {len(prompt_ids_to_process)} prompt(s) for path salience.")

            for pid_idx, pid in enumerate(prompt_ids_to_process):
                if pid not in f_hdf5:
                    logger.warning(f"Prompt ID {pid} not found in HDF5. Skipping.")
                    continue
                logger.info(f"Starting prompt {pid_idx + 1}/{len(prompt_ids_to_process)}: {pid}")
                prompt_group = f_hdf5[pid]
                run_ids = list(prompt_group.keys())

                for run_idx_iter, run_id in enumerate(run_ids):
                    logger.info(f"  Processing run {run_idx_iter + 1}/{len(run_ids)}: {run_id}")
                    run_group = prompt_group[run_id]

                    # Determine number of transformer layers
                    num_transformer_layers = -1 # This is L
                    try:
                        if use_unnormalized:
                            # For unnormalized, we need attention and MLP outputs
                            attn_layer_keys = [k for k in run_group.keys() if k.startswith('layer_') and ATTENTION_OUTPUT_TYPE in run_group[k]]
                            mlp_layer_keys = [k for k in run_group.keys() if k.startswith('layer_') and MLP_OUTPUT_TYPE in run_group[k]]
                            if not attn_layer_keys or not mlp_layer_keys:
                                logger.warning(f"    Missing attention or MLP outputs for unnormalized trajectory in {pid}/{run_id}. Skipping run.")
                                continue
                            max_attn_layer = max(int(k.split('_')[1]) for k in attn_layer_keys)
                            max_mlp_layer = max(int(k.split('_')[1]) for k in mlp_layer_keys)
                            num_transformer_layers = min(max_attn_layer, max_mlp_layer) + 1  # +1 because we use 0-indexed layers
                        else:
                            # For residual stream, we need residual stream inputs
                            res_layer_keys = [k for k in run_group.keys() if k.startswith('layer_') and RESIDUAL_STREAM_TYPE in run_group[k]]
                            if not res_layer_keys:
                                logger.warning(f"    No layers with '{RESIDUAL_STREAM_TYPE}' found for {pid}/{run_id}. Skipping run.")
                                continue
                            # HDF5 layers for residual stream are 0 (x0) to L (xL)
                            max_hdf5_layer_idx_for_residual = max(int(k.split('_')[1]) for k in res_layer_keys)
                            num_transformer_layers = max_hdf5_layer_idx_for_residual # L = max index
                        
                        if num_transformer_layers < 0: # Need at least x0 and x1 for one delta
                            logger.warning(f"    Inferred L={num_transformer_layers}. Need L>=0 for at least one segment. Skipping run.")
                            continue
                    except Exception as e:
                        logger.error(f"    Could not determine num_transformer_layers for {pid}/{run_id}: {e}. Skipping run.")
                        continue
                    
                    num_trajectory_points = num_transformer_layers + 1 # x0 to xL
                    logger.debug(f"    Run {pid}/{run_id}: Expecting {num_trajectory_points} points (x0 to xL={num_transformer_layers}) in trajectory.")

                    # Determine number of captured generation steps
                    max_step_k_captured = -1
                    layer0_res_path = f"layer_0/{RESIDUAL_STREAM_TYPE}"
                    if layer0_res_path in run_group:
                        step_keys = [k for k in run_group[layer0_res_path].keys() if k.startswith("step_")]
                        if step_keys:
                            try: 
                                max_step_k_captured = max(int(k.split('_')[1]) for k in step_keys)
                            except ValueError:
                                logger.warning(f"    Could not parse step indices from '{layer0_res_path}' for {pid}/{run_id}. Assuming 0 steps.")
                                max_step_k_captured = -1
                    
                    if max_step_k_captured == -1:
                        logger.warning(f"    Could not determine number of captured generation steps for {pid}/{run_id}. Skipping run.")
                        continue
                    num_gen_steps_captured = max_step_k_captured + 1

                    for k_gen_step in range(num_gen_steps_captured):
                        # Get trajectory based on method
                        if use_unnormalized:
                            x_vals_np = reconstruct_unnormalized_trajectory(run_group, num_transformer_layers, k_gen_step, pid, run_id)
                            if x_vals_np is None:
                                logger.debug(f"      GenStep {k_gen_step}: Failed to reconstruct unnormalized trajectory for {pid}/{run_id}")
                                continue
                        else:
                            # Original residual stream method
                            x_states_for_current_token = []
                            valid_trajectory_for_step = True
                            for l_hdf5_idx in range(num_trajectory_points): # 0 to L (for x^0 to x^L)
                                current_x_vec = get_current_token_activation(run_group, f"layer_{l_hdf5_idx}", 
                                                                             RESIDUAL_STREAM_TYPE, k_gen_step,
                                                                             pid, run_id)
                                if current_x_vec is None:
                                    logger.debug(f"      GenStep {k_gen_step}, State x^{l_hdf5_idx}: Data missing. Salience trajectory for this step incomplete.")
                                    valid_trajectory_for_step = False
                                    break
                                x_states_for_current_token.append(current_x_vec)
                            
                            if not valid_trajectory_for_step or len(x_states_for_current_token) < 2: # Need at least 2 points for one delta
                                if valid_trajectory_for_step: # Log only if not already broken due to missing data
                                    logger.debug(f"      Not enough points for salience calculation for {pid}/{run_id} GenStep {k_gen_step} (found {len(x_states_for_current_token)}). Need >=2. Skipping step.")
                                continue
                            
                            x_vals_np = np.array(x_states_for_current_token, dtype=np.float64) # (L+1, d_model)

                        # Need at least 2 points for one delta
                        if len(x_vals_np) < 2:
                            logger.debug(f"      Not enough points for salience calculation for {pid}/{run_id} GenStep {k_gen_step} (found {len(x_vals_np)}). Need >=2. Skipping step.")
                            continue
                        
                        deltas = x_vals_np[1:] - x_vals_np[:-1] # (L, d_model)
                        if deltas.shape[0] == 0: # Should be caught by len < 2 check, but defensive
                            logger.debug(f"      GenStep {k_gen_step}: No deltas to calculate salience from. Skipping.")
                            continue

                        salience_sq_series = gdot(deltas, deltas, G) # (L,)
                        salience_series_np = np.sqrt(np.maximum(0, salience_sq_series))

                        if salience_series_np.size == 0:
                             logger.debug(f"      GenStep {k_gen_step}: Empty salience series. Skipping.")
                             continue
                        
                        mean_salience = np.nanmean(salience_series_np) if salience_series_np.size > 0 else 0.0
                        max_salience_val = np.nanmax(salience_series_np) if salience_series_np.size > 0 else 0.0
                        
                        idx_of_max_salience = -1
                        if salience_series_np.size > 0 and not np.all(np.isnan(salience_series_np)):
                            idx_of_max_salience = int(np.nanargmax(salience_series_np))
                            # This index is into the deltas array (0 to L-1).
                            # layer_of_max_salience corresponds to the layer *producing* that delta,
                            # i.e., delta_l is x^(l+1) - x^(l), associated with layer l's processing.
                            # So, index `d` in salience_series corresponds to layer `d`.
                        
                        total_salience_val = np.nansum(salience_series_np)

                        record = {
                            "config_basename": config_basename,
                            "prompt_id": pid,
                            "run_id": run_id,
                            "generation_step_idx": k_gen_step,
                            "activation_trajectory_type": TRAJECTORY_TYPE_NAME,
                            "mean_salience": float(mean_salience),
                            "max_salience": float(max_salience_val),
                            "layer_idx_of_max_salience": idx_of_max_salience, # Index of segment l (0 to L-1)
                            "total_salience": float(total_salience_val),
                            "num_segments_in_salience_series": len(salience_series_np),
                            "trajectory_num_points": len(x_vals_np)
                        }

                        if dump_salience_series:
                            salience_list = [None if np.isnan(s) else float(s) for s in salience_series_np]
                            record["salience_series"] = json.dumps(salience_list)
                        
                        results_buffer.append(record)
                        
                        if len(results_buffer) >= 500:
                            logger.info(f"    Writing {len(results_buffer)} records to CSV...")
                            append_to_csv(output_csv_file, results_buffer, csv_headers)
                            results_buffer.clear()

            if results_buffer:
                logger.info(f"Writing final {len(results_buffer)} records to CSV...")
                append_to_csv(output_csv_file, results_buffer, csv_headers)
                results_buffer.clear()

    except FileNotFoundError: 
        logger.error(f"Input HDF5 file not found: {hdf5_path}")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        if results_buffer:
            logger.error(f"Attempting to save {len(results_buffer)} buffered results due to error...")
            append_to_csv(output_csv_file, results_buffer, csv_headers)
        return

    logger.info(f"Path salience analysis (CI02-style) complete. Results saved to {output_csv_file}")

# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Residual stream path salience analysis for 'current token' trajectories during generation.")
    parser.add_argument("reduction_config_file", type=str, help="Path to JSON config (HDF5 input, G matrix).")
    parser.add_argument("--output_base", type=str, required=True, help="Prefix for output CSV.")
    parser.add_argument("--prompt_ids", type=str, default=None, help="Comma-separated subset of prompt ids to analyse. If None, all prompts are processed.")
    parser.add_argument("--dump_salience_series", action="store_true", help="Include full salience series (||x^(l+1)-x^(l)||_G) as JSON in CSV.")
    parser.add_argument("--use_unnormalized", action="store_true", help="Use unnormalized trajectory reconstruction (x^0 + cumulative attention/MLP) instead of residual stream states.")

    args = parser.parse_args()
    pid_list = [p.strip() for p in args.prompt_ids.split(',')] if args.prompt_ids else None

    run_path_salience_analysis_ci02_style(
        reduction_config_path=args.reduction_config_file,
        output_base_name=args.output_base,
        specific_prompt_ids=pid_list,
        dump_salience_series=args.dump_salience_series,
        use_unnormalized=args.use_unnormalized
    )
