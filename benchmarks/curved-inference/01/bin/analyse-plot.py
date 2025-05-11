import argparse
import json
import logging
import os
import sys
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting

# --- Configuration ---
SHIFT_POINT_MARKER = '*' # Marker for the shift token
SHIFT_POINT_COLOR = 'red' # Color for the shift token marker
SHIFT_POINT_SIZE = 100 # Size for the shift token marker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def get_global_axis_limits(hdf5_path, activation_type='residual_stream'):
    import h5py
    import numpy as np

    all_coords = []

    with h5py.File(hdf5_path, 'r') as f:
        for prompt_id in f:
            prompt_group = f[prompt_id]
            if activation_type not in prompt_group:
                continue
            act_group = prompt_group[activation_type]
            if 'embeddings' not in act_group:
                continue
            embeddings = act_group['embeddings'][:]  # shape: (num_points, 3)
            all_coords.append(embeddings)

    if not all_coords:
        raise ValueError("No valid embeddings found for global axis limits.")

    combined = np.vstack(all_coords)
    x_min, y_min, z_min = combined.min(axis=0)
    x_max, y_max, z_max = combined.max(axis=0)

    return (x_min, x_max), (y_min, y_max), (z_min, z_max)

def parse_token_indices(indices_str):
    """Parses a comma-separated string of token indices into a list of integers."""
    if not indices_str:
        return []
    try:
        return [int(i.strip()) for i in indices_str.split(',')]
    except ValueError:
        logger.error(f"Invalid token indices format: '{indices_str}'. Please use comma-separated integers.")
        return None

def parse_prompt_ids(ids_str):
    """Parses a comma-separated string of prompt IDs into a list."""
    if not ids_str:
        return None
    return [pid.strip() for pid in ids_str.split(',')]

def get_config_value(config, key, default=None):
    """Safely gets a value from the config dictionary."""
    return config.get(key, default)

def get_hdf5_attr(hdf5_object, attr_name, default=None):
     """Safely gets an attribute from an HDF5 object."""
     try:
         value = hdf5_object.attrs.get(attr_name, default)
         # Handle JSON-encoded lists/dicts stored as strings
         if isinstance(value, (bytes, str)): # Check bytes too
             try:
                 # Decode bytes to string if necessary
                 value_str = value.decode('utf-8') if isinstance(value, bytes) else value

                 # Try decoding potential JSON strings (like lists)
                 if value_str.startswith('[') or value_str.startswith('{'):
                    json_decoded = json.loads(value_str)
                    return json_decoded
                 # Handle 'None' string
                 elif value_str == 'None':
                      return None
                 # Handle boolean strings
                 elif value_str.lower() == 'true':
                      return True
                 elif value_str.lower() == 'false':
                      return False
                 else:
                     # Return the string itself if not JSON/None/Bool
                     return value_str
             except (json.JSONDecodeError, UnicodeDecodeError):
                 # If not JSON/decodable, return the original value (might be string/bytes)
                 return value
         # Handle numpy bool explicitly before returning other types
         elif isinstance(value, np.bool_):
             return bool(value)
         return value
     except Exception as e:
         logger.warning(f"Could not read attribute '{attr_name}'. Error: {e}. Using default: {default}")
         return default

# --- Main Plotting Function ---

def run_analysis_plot(
    reduction_config_path,
    specific_prompt_ids=None, # Optional list of prompt IDs to plot
    plot_token_indices=None, # Optional list of token indices for trajectories
    output_plot_file=None
):
    """
    Loads reduced embeddings based on a reduction config file and generates plots.

    Args:
        reduction_config_path (str): Path to the reduction JSON config file.
        specific_prompt_ids (list[str], optional): List of specific prompt IDs to plot.
                                                  If None, plots all prompts found.
        plot_token_indices (list[int], optional): List of token indices for trajectories.
        output_plot_file (str, optional): Path to save the plot image. Displays if None.
    """
    logger.info(f"Loading reduction configuration from: {reduction_config_path}")
    try:
        with open(reduction_config_path, 'r') as f:
            reduction_config = json.load(f)
    except FileNotFoundError:
        logger.error(f"Reduction configuration file not found: {reduction_config_path}")
        return
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from reduction config file: {reduction_config_path}")
        return

    # --- Extract parameters from reduction config ---
    embeddings_hdf5_path = get_config_value(reduction_config, 'output_hdf5_file')
    activation_types = get_config_value(reduction_config, 'activation_types', [])
    reduction_method = get_config_value(reduction_config, 'reduction_method', 'UNKNOWN').upper()
    final_components = get_config_value(reduction_config, 'final_components', 2)
    normalize_method = get_config_value(reduction_config, 'normalize', None)
    use_pca_config = get_config_value(reduction_config, 'use_pca', False) # Get raw value
    pca_components = get_config_value(reduction_config, 'pca_components', 'N/A') if use_pca_config else "None"
    umap_neighbors = get_config_value(reduction_config, 'umap_neighbors', 'N/A')
    umap_min_dist = get_config_value(reduction_config, 'umap_min_dist', 'N/A')
    tsne_perplexity = get_config_value(reduction_config, 'tsne_perplexity', 'N/A')

    if not embeddings_hdf5_path:
        logger.error("Config file does not specify 'output_hdf5_file'.")
        return
    if not activation_types:
        logger.error("Config file does not specify 'activation_types'.")
        return
    # Assuming only one activation type was processed per file, as planned
    activation_type = activation_types[0]
    logger.info(f"Analyzing embeddings for activation type: {activation_type}")
    logger.info(f"Loading embeddings from: {embeddings_hdf5_path}")

    (x_min, x_max), (y_min, y_max), (z_min, z_max) = get_global_axis_limits(embeddings_hdf5_path, activation_type)

    if final_components not in [2, 3]:
        logger.error(f"Unsupported number of final_components ({final_components}). Only 2 or 3 are supported for plotting.")
        return

    # --- Load Embeddings and Metadata ---
    all_embeddings = []
    metadata_list = []
    prompt_metadata_store = {} # Store attributes per prompt

    try:
        with h5py.File(embeddings_hdf5_path, 'r') as f:
            # Read reduction config stored in the embeddings file (for verification/title)
            # Use the safe getter for attributes
            stored_reduction_config = {key: get_hdf5_attr(f, key) for key in f.attrs.keys()}
            logger.info(f"Embeddings file generated with config: {stored_reduction_config}")
            # Re-read necessary params from stored config if needed, ensures consistency
            use_pca = get_hdf5_attr(f, 'use_pca', False)
            pca_components = get_hdf5_attr(f, 'pca_components', 'N/A') if use_pca else "None"
            reduction_method = get_hdf5_attr(f, 'reduction_method', 'UNKNOWN').upper()
            final_components = get_hdf5_attr(f, 'final_components', 2)
            normalize_method = get_hdf5_attr(f, 'normalize', None)
            umap_neighbors = get_hdf5_attr(f, 'umap_neighbors', 'N/A')
            umap_min_dist = get_hdf5_attr(f, 'umap_min_dist', 'N/A')
            tsne_perplexity = get_hdf5_attr(f, 'tsne_perplexity', 'N/A')


            # Determine which prompts to load
            all_prompt_ids_in_file = list(f.keys())
            if specific_prompt_ids:
                prompts_to_load = [pid for pid in specific_prompt_ids if pid in all_prompt_ids_in_file]
                missing = set(specific_prompt_ids) - set(prompts_to_load)
                if missing:
                    logger.warning(f"Requested prompt IDs not found in embeddings file: {missing}")
            else:
                logger.info("Processing all prompts found in the embeddings file.")
                prompts_to_load = all_prompt_ids_in_file

            if not prompts_to_load:
                logger.error("No prompts selected or found to plot.")
                return

            logger.info(f"Loading data for prompts: {prompts_to_load}")
            max_layer_idx = -1

            for prompt_id in prompts_to_load:
                if prompt_id not in f: # Should not happen if logic above is correct, but double-check
                    logger.warning(f"Prompt ID {prompt_id} not found. Skipping.")
                    continue
                prompt_group = f[prompt_id]

                # Check if the required activation type exists
                if activation_type not in prompt_group:
                    logger.error(f"Activation type '{activation_type}' not found for prompt '{prompt_id}' in this embeddings file. Skipping prompt.")
                    continue

                act_group = prompt_group[activation_type]

                # Load data
                embeddings = act_group['embeddings'][:]
                layer_indices = act_group['layer_indices'][:]
                token_indices = act_group['token_indices'][:]

                # Basic consistency check
                if not (len(embeddings) == len(layer_indices) == len(token_indices)):
                    logger.error(f"Data length mismatch for prompt '{prompt_id}', activation '{activation_type}'. Skipping.")
                    continue
                if embeddings.shape[1] != final_components:
                     logger.error(f"Embedding dimension mismatch for prompt '{prompt_id}', activation '{activation_type}'. Expected {final_components}, got {embeddings.shape[1]}. Skipping.")
                     continue

                all_embeddings.append(embeddings)
                max_layer_idx = max(max_layer_idx, np.max(layer_indices))

                # Load prompt metadata attributes using safe getter
                prompt_attrs = {key: get_hdf5_attr(prompt_group, key) for key in prompt_group.attrs.keys()}
                prompt_metadata_store[prompt_id] = prompt_attrs
                shift_idx_val = prompt_attrs.get('shift_token_idx', -1)
                # Ensure shift_idx is an integer
                try:
                    shift_idx = int(shift_idx_val) if shift_idx_val is not None else -1
                except (ValueError, TypeError):
                    shift_idx = -1
                prompt_label = prompt_attrs.get('prompt_label', prompt_id) # Use ID if no label

                # Create metadata rows for this prompt
                num_points = len(embeddings)
                for i in range(num_points):
                    metadata_list.append({
                        "prompt_id": prompt_id,
                        "prompt_label": prompt_label, # Use label or ID
                        "layer": layer_indices[i],
                        "token_idx": token_indices[i],
                        "shift_token_idx": shift_idx # Store shift index associated with this prompt
                    })

            if not all_embeddings:
                 logger.error("Failed to load any valid embedding data.")
                 return

            # Combine embeddings and create DataFrame
            combined_embeddings = np.vstack(all_embeddings)
            plot_df = pd.DataFrame(metadata_list)
            for dim in range(final_components):
                plot_df[f'dim_{dim}'] = combined_embeddings[:, dim]

            logger.info(f"Loaded data shape: {combined_embeddings.shape}")
            logger.info(f"DataFrame shape: {plot_df.shape}")

    except FileNotFoundError:
        logger.error(f"Embeddings HDF5 file not found: {embeddings_hdf5_path}")
        return
    except Exception as e:
        logger.error(f"Error loading data from HDF5 file {embeddings_hdf5_path}: {e}", exc_info=True)
        return

    # --- Plotting ---
    logger.info("Generating plot...")
    fig = plt.figure(figsize=(16, 12)) # Larger figure size

    if final_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_proj_type('persp', focal_length=0.3)
        x_col, y_col, z_col = 'dim_0', 'dim_1', 'dim_2'
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
    else: # 2D
        ax = fig.add_subplot(111)
        x_col, y_col, z_col = 'dim_0', 'dim_1', None

    # --- Define Plot Styles ---
    # Use prompt IDs directly for markers if plotting all, or use labels if few
    unique_prompts = prompts_to_load
    markers = ['^', 'o', 's', 'd', 'p', '*', 'X', 'P', 'v', '<', '>'] # Cycle through markers
    prompt_styles = {pid: {'marker': markers[i % len(markers)],
                           'label': pid} # Use prompt ID as label
                     for i, pid in enumerate(unique_prompts)}

    # --- Create Scatter Plot ---
    layer_cmaps = {
      unique_prompts[0]: plt.cm.viridis,
      unique_prompts[1]: plt.cm.inferno
    }
    norm = mcolors.Normalize(vmin=0, vmax=max_layer_idx)
    legend_elements = []
    clist = ['#ff0000','#0000ff']

    for pid, style in prompt_styles.items():
        if pid not in prompts_to_load: continue # Skip if not loaded
        subset_df = plot_df[plot_df['prompt_id'] == pid]
        shift_idx = prompt_metadata_store.get(pid, {}).get('shift_token_idx', -1)
        if shift_idx is not None:
          subset_df = subset_df[subset_df['token_idx'] != shift_idx]
        if subset_df.empty: continue

        # Separate positional and keyword arguments
        x_data = subset_df[x_col]
        y_data = subset_df[y_col]
        scatter_kwargs = {
            'c': subset_df['layer'],
            'cmap': layer_cmaps[pid],
            'norm': norm,
            'marker': style['marker'],
            's': 25,
            'alpha': 0.3,
            'label': style['label'] # Label for internal use, legend handled separately
        }

        if final_components == 3:
            z_data = subset_df[z_col]
            scatter = ax.scatter(x_data, y_data, z_data, **scatter_kwargs) # Pass x, y, z positionally
        else:
            scatter = ax.scatter(x_data, y_data, **scatter_kwargs) # Pass x, y positionally

    # --- Plot Trajectories ---
    if plot_token_indices:
        logger.info(f"Plotting trajectories for tokens: {plot_token_indices}")
        # Use distinct linestyles or slightly different colors for prompt types
        line_styles = ['-', '--', '-.', ':']
        # Use colors based on marker color or a separate palette
        traj_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_prompts)))

        for i, (pid, style) in enumerate(prompt_styles.items()):
            if pid not in prompts_to_load: continue
            prompt_subset_df = plot_df[plot_df['prompt_id'] == pid]
            prompt_seq_len = prompt_metadata_store[pid].get('sequence_length', 0)

            for token_idx in plot_token_indices:
                if token_idx >= prompt_seq_len:
                    logger.warning(f"Token index {token_idx} is out of bounds for prompt {pid} (length {prompt_seq_len}). Skipping trajectory.")
                    continue

                token_df = prompt_subset_df[prompt_subset_df['token_idx'] == token_idx].sort_values('layer')
                if not token_df.empty:
                    line_color = traj_colors[i % len(traj_colors)]
                    plot_args = {
                        'xs': token_df[x_col],
                        'ys': token_df[y_col],
                        'linestyle': line_styles[i % len(line_styles)],
                        'color': line_color,
                        'linewidth': 1.0,
                        'alpha': 0.4,
                    }
                    if final_components == 3:
                        plot_args['zs'] = token_df[z_col]
                        ax.plot(**plot_args)
                        # 3D Start/End markers
                        ax.scatter(token_df[x_col].iloc[0], token_df[y_col].iloc[0], token_df[z_col].iloc[0], marker='>', color=line_color, s=50, alpha=0.4, zorder=3)
                        ax.scatter(token_df[x_col].iloc[-1], token_df[y_col].iloc[-1], token_df[z_col].iloc[-1], marker='X', color=line_color, s=50, alpha=0.4, zorder=3)
                    else:
                        ax.plot(**plot_args)
                        # 2D Start/End markers
                        ax.scatter(token_df[x_col].iloc[0], token_df[y_col].iloc[0], marker='>', color=line_color, s=50, alpha=0.8, zorder=3)
                        ax.scatter(token_df[x_col].iloc[-1], token_df[y_col].iloc[-1], marker='X', color=line_color, s=50, alpha=0.8, zorder=3)

    # --- Highlight Shift Points ---
    shift_points_plotted = False
    shift_legend_label_added = False
    for pid in prompts_to_load:
        shift_idx = prompt_metadata_store[pid].get('shift_token_idx', -1)
        # Ensure shift_idx is a valid integer >= 0
        try:
            shift_idx = int(shift_idx) if shift_idx is not None else -1
        except (ValueError, TypeError):
            shift_idx = -1

        if shift_idx >= 0:
            logger.info(f"Highlighting shift token index {shift_idx} for prompt {pid}")
            shift_points_df = plot_df[
                (plot_df['prompt_id'] == pid) &
                (plot_df['token_idx'] == shift_idx)
            ]
            if not shift_points_df.empty:
                current_label = f'Shift Token ({shift_idx})' if not shift_legend_label_added else None
                # Separate positional and keyword arguments for shift points
                x_shift_data = shift_points_df[x_col]
                y_shift_data = shift_points_df[y_col]
                scatter_shift_kwargs = {
                    'marker': SHIFT_POINT_MARKER,
                    'color': SHIFT_POINT_COLOR,
                    's': SHIFT_POINT_SIZE,
                    'alpha': 0.5,
                    'label': current_label, # Use label for legend
                    'zorder': 5
                }

                if final_components == 3:
                    z_shift_data = shift_points_df[z_col]
                    # Pass x, y, z positionally
                    ax.scatter(x_shift_data, y_shift_data, z_shift_data, **scatter_shift_kwargs)
                else:
                    # Pass x, y positionally
                    ax.scatter(x_shift_data, y_shift_data, **scatter_shift_kwargs)

                if not shift_legend_label_added:
                     legend_elements.append(plt.Line2D([0], [0], marker=SHIFT_POINT_MARKER, color='w', label=f'Shift Token',
                                                  markerfacecolor=SHIFT_POINT_COLOR, markersize=10))
                     shift_legend_label_added = True # Ensure legend entry added only once

            else:
                logger.warning(f"Could not find data points for shift token index {shift_idx} in prompt {pid}.")


    # --- Final Plot Setup ---
    # Construct title from reduction config
    pca_str = f"PCA({pca_components})" if use_pca else "No PCA"
    norm_str = f"Norm({normalize_method})" if normalize_method else "No Norm"
    if reduction_method == 'UMAP':
        red_details = f"UMAP(n={umap_neighbors}, d={umap_min_dist})"
    else: # Assumes TSNE
        red_details = f"tSNE(p={tsne_perplexity})"
    title = (f"{final_components}D {reduction_method} Projection of '{activation_type}' Activations\n"
             f"Method: {norm_str} -> {pca_str} -> {red_details}\n"
             f"Prompts: {', '.join(prompts_to_load)}")
    ax.set_title(title, fontsize=12)

    ax.set_xlabel(f'{reduction_method} Dimension 1')
    ax.set_ylabel(f'{reduction_method} Dimension 2')
    if final_components == 3:
        ax.set_zlabel(f'{reduction_method} Dimension 3')

    ax.grid(True)

    # --- Output Plot ---
    if output_plot_file:
        # Ensure output directory exists for plot file
        plot_output_dir = os.path.dirname(output_plot_file)
        if plot_output_dir and not os.path.exists(plot_output_dir):
            try:
                os.makedirs(plot_output_dir)
                logger.info(f"Created plot output directory: {plot_output_dir}")
            except OSError as e:
                logger.error(f"Failed to create plot output directory {plot_output_dir}: {e}")
                # Continue to try saving in current dir or fail there
        logger.info(f"Saving plot to: {output_plot_file}")
        plt.savefig(output_plot_file, dpi=300, bbox_inches='tight')
    else:
        logger.info("Displaying plot...")
        plt.show()

    plt.close(fig)
    logger.info("Analysis plot script finished.")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze and plot reduced LLM activation embeddings."
    )
    parser.add_argument(
        "reduction_config_file",
        type=str,
        help="Path to the JSON reduction configuration file used to generate the embeddings."
    )
    parser.add_argument(
        "--prompt_ids",
        type=str,
        default=None,
        help="Optional comma-separated list of specific prompt IDs to plot (e.g., 'id1,id2'). If None, plots all prompts."
    )
    parser.add_argument(
        "--plot_token_indices",
        type=str,
        default=None,
        help="Optional comma-separated list of token indices to plot as trajectories (e.g., '0,10,20')."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional path to save the output plot PNG file. If not provided, the plot is displayed."
    )

    args = parser.parse_args()

    # Parse optional arguments
    prompt_ids_to_plot = parse_prompt_ids(args.prompt_ids)
    token_indices_to_plot = parse_token_indices(args.plot_token_indices)
    if args.plot_token_indices is not None and token_indices_to_plot is None:
         sys.exit(1) # Exit if parsing failed

    run_analysis_plot(
        reduction_config_path=args.reduction_config_file,
        specific_prompt_ids=prompt_ids_to_plot,
        plot_token_indices=token_indices_to_plot,
        output_plot_file=args.output_file
    )

