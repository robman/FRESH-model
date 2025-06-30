import argparse
import json
import logging
import os
import sys
from pathlib import Path

import h5py
import numpy as np
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

def cumulative_arc_length(x: np.ndarray, G: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        return np.zeros(len(x))
    deltas = x[1:] - x[:-1]
    seg_len_sq = gdot(deltas, deltas, G)
    seg_len = np.sqrt(np.maximum(seg_len_sq, 1e-24))
    return np.concatenate([[0.0], np.cumsum(seg_len)])

# -----------------------------------------------------------------------------
# Curvature calculations
# -----------------------------------------------------------------------------
def calculate_discrete_curvature_3pt(s: np.ndarray,
                                     x: np.ndarray,
                                     G: np.ndarray,
                                     epsilon: float = 1e-12) -> np.ndarray:
    n = len(x)
    if n < 3:
        return np.full(n, np.nan)
    kappa = np.full(n, np.nan)
    if len(s) != n:
        logger.warning(f"s length ({len(s)}) != x length ({n}) in 3pt curvature. Using unit steps for s.")
        s = np.arange(n, dtype=float) # Fallback if s is not appropriate

    for i in range(1, n - 1):
        ds1 = s[i] - s[i - 1]
        ds2 = s[i + 1] - s[i]
        if ds1 <= epsilon or ds2 <= epsilon:
            logger.debug(f"Skipping 3-point curvature at trajectory point {i} due to zero/small step size in s: ds1={ds1}, ds2={ds2}")
            continue
        v = (x[i + 1] - x[i - 1]) / (ds1 + ds2)
        a = 2 * (ds1 * (x[i + 1] - x[i]) - ds2 * (x[i] - x[i - 1])) / (ds1 * ds2 * (ds1 + ds2))
        norm_v_sq = gdot(v, v, G)
        norm_a_sq = gdot(a, a, G)
        dot_av = gdot(a, v, G)
        num_sq = max(0.0, norm_a_sq * norm_v_sq - dot_av ** 2)
        denom_cubed = max(epsilon, norm_v_sq) ** 1.5
        if denom_cubed < epsilon: # Avoid division by zero or very small numbers
            kappa[i] = np.nan
        else:
            kappa[i] = np.sqrt(num_sq) / denom_cubed
    return kappa

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
                                 step_idx: int,
                                 pid: str, run_id: str):
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
def run_3point_curvature_analysis(reduction_config_path: str,
                                  output_base_name: str,
                                  specific_prompt_ids: list[str] | None = None,
                                  dump_3point_series: bool = False,
                                  use_unnormalized: bool = False):

    logger.info(f"Loading config: {reduction_config_path}")
    try:
        with open(reduction_config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load or parse config {reduction_config_path}: {e}")
        return

    hdf5_path = config.get('input_hdf5_file')
    unembed_path = config.get('unembedding_npz_file')

    if not all([hdf5_path, unembed_path]):
        logger.error("Config missing 'input_hdf5_file' or 'unembedding_npz_file'.")
        return

    try:
        with np.load(unembed_path) as data:
            if 'G' not in data:
                logger.error(f"Metric 'G' not found in unembedding file: {unembed_path}")
                return
            G = data['G']
        logger.info(f"Loaded pull‑back metric G from {unembed_path}, shape: {G.shape}")
    except Exception as e:
        logger.error(f"Error loading unembedding NPZ file {unembed_path}: {e}")
        return

    # Update output filename to indicate trajectory type
    trajectory_suffix = "unnormalized" if use_unnormalized else "residual"
    output_csv_file = f"{output_base_name}-3point-curvature-{trajectory_suffix}.csv"
    Path(output_base_name).parent.mkdir(parents=True, exist_ok=True)

    csv_headers = [
        "config_basename", "prompt_id", "run_id", "generation_step_idx",
        "activation_trajectory_type",
        "mean_3point_curvature", "max_3point_curvature", "idx_of_max_3point_curvature",
        "trajectory_num_points"
    ]
    if dump_3point_series:
        csv_headers.append("kappa_3point_series")

    config_basename = Path(reduction_config_path).stem
    results_buffer = []
    
    # Keep trajectory type name consistent for downstream compatibility
    TRAJECTORY_TYPE_NAME = "residual_stream_x_states_3pt"
    
    if use_unnormalized:
        logger.info("Using UNNORMALIZED trajectory reconstruction (x^0 + cumulative attention/MLP outputs)")
    else:
        logger.info("Using RESIDUAL STREAM trajectory (layer input states)")

    try:
        with h5py.File(hdf5_path, 'r') as f_hdf5:
            prompt_ids_to_process = specific_prompt_ids or list(f_hdf5.keys())
            logger.info(f"Processing {len(prompt_ids_to_process)} prompt(s) for 3-point curvature.")

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
                    num_transformer_layers = -1
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
                            max_res_layer_idx = max(int(k.split('_')[1]) for k in res_layer_keys)
                            num_transformer_layers = max_res_layer_idx 
                        
                        if num_transformer_layers < 0:
                            logger.warning(f"    Inferred non-positive L ({num_transformer_layers}) for {pid}/{run_id}. Skipping run.")
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
                            max_step_k_captured = max(int(k.split('_')[1]) for k in step_keys)
                    
                    if max_step_k_captured == -1:
                        logger.warning(f"    Could not determine number of captured generation steps from '{layer0_res_path}' for {pid}/{run_id}. Skipping run.")
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
                                    logger.debug(f"      GenStep {k_gen_step}, State x^{l_hdf5_idx}: Data missing. Trajectory for this step incomplete.")
                                    valid_trajectory_for_step = False
                                    break
                                x_states_for_current_token.append(current_x_vec)
                            
                            if not valid_trajectory_for_step or len(x_states_for_current_token) < 3:
                                if valid_trajectory_for_step:
                                    logger.debug(f"      Not enough points in residual trajectory for {pid}/{run_id} GenStep {k_gen_step} (found {len(x_states_for_current_token)}). Expected {num_trajectory_points} but need >=3 for curvature. Skipping step.")
                                continue
                            
                            x_vals_np = np.array(x_states_for_current_token, dtype=np.float64)

                        # Need at least 3 points for 3-point curvature
                        if len(x_vals_np) < 3:
                            logger.debug(f"      Not enough points in trajectory for {pid}/{run_id} GenStep {k_gen_step} (found {len(x_vals_np)}). Need >=3 for curvature. Skipping step.")
                            continue

                        s_param = cumulative_arc_length(x_vals_np, G)
                        if len(s_param) < 3 or np.allclose(s_param[-1], 0.0): # Need s_param for 3 points, and non-zero total length
                            logger.debug(f"      Not enough arc-length points or zero total arc-length for trajectory {pid}/{run_id} GenStep {k_gen_step}. Skipping step.")
                            continue
                        
                        # Calculate 3-point curvature
                        kappa_3point_values = calculate_discrete_curvature_3pt(s_param, x_vals_np, G)
                        
                        mean_kappa_3pt = np.nanmean(kappa_3point_values)
                        max_kappa_3pt_val = np.nan
                        idx_max_kappa_3pt = np.nan

                        if not np.all(np.isnan(kappa_3point_values)):
                            max_kappa_3pt_val = np.nanmax(kappa_3point_values)
                            idx_max_kappa_3pt = np.nanargmax(kappa_3point_values)
                        
                        record = {
                            "config_basename": config_basename,
                            "prompt_id": pid,
                            "run_id": run_id,
                            "generation_step_idx": k_gen_step,
                            "activation_trajectory_type": TRAJECTORY_TYPE_NAME,
                            "mean_3point_curvature": float(mean_kappa_3pt) if not np.isnan(mean_kappa_3pt) else None,
                            "max_3point_curvature": float(max_kappa_3pt_val) if not np.isnan(max_kappa_3pt_val) else None,
                            "idx_of_max_3point_curvature": int(idx_max_kappa_3pt) if not np.isnan(idx_max_kappa_3pt) else None,
                            "trajectory_num_points": len(x_vals_np)
                        }
                        
                        if dump_3point_series:
                            record["kappa_3point_series"] = json.dumps([None if np.isnan(k) else float(k) for k in kappa_3point_values.tolist()])
                        
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

    logger.info(f"3-point curvature analysis complete. Results saved to {output_csv_file}")

# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Residual stream trajectory 3-point curvature analysis during generation.")
    parser.add_argument("reduction_config_file", type=str, help="Path to JSON config (contains HDF5 input, unembed G matrix path).")
    parser.add_argument("--output_base", type=str, required=True, help="Prefix for output CSV (e.g., 'metrics/model_name/prompt_set_name').")
    parser.add_argument("--prompt_ids", type=str, default=None, help="Comma-separated subset of prompt ids to analyse. If None, all prompts are processed.")
    parser.add_argument("--dump_3point_series", action="store_true", help="Include full 3-point kappa vector as JSON in CSV.")
    parser.add_argument("--use_unnormalized", action="store_true", help="Use unnormalized trajectory reconstruction (x^0 + cumulative attention/MLP) instead of residual stream states.")

    args = parser.parse_args()
    pid_list = [p.strip() for p in args.prompt_ids.split(',')] if args.prompt_ids else None

    run_3point_curvature_analysis(
        reduction_config_path=args.reduction_config_file,
        output_base_name=args.output_base,
        specific_prompt_ids=pid_list,
        dump_3point_series=args.dump_3point_series,
        use_unnormalized=args.use_unnormalized
    )
