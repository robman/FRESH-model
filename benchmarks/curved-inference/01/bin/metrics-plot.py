import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path # Use pathlib for path operations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Configuration ---
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Define expected metric filenames based on base name
METRIC_FILENAMES = {
    "layerwise_deviation": "{base}-layerwise-deviation.csv",
    "direction_deviation": "{base}-direction-deviation.csv",
    "cosine_similarity": "{base}-cosine-similarity.csv",
    # Add others later if needed
    # "trajectory_length": "{base}-trajectory-length.csv",
    # "local_curvature": "{base}-local-curvature.csv",
}

# --- Helper Functions ---

def load_metric_csv(filepath):
    """Loads a metric CSV file into a pandas DataFrame."""
    if not Path(filepath).is_file():
        logger.warning(f"Metric file not found: {filepath}")
        return None
    try:
        # Explicitly handle potential type issues during loading if needed
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        # Clean up potential NaN strings if necessary
        df.replace('None', np.nan, inplace=True)
        df.replace('N/A', np.nan, inplace=True)
        return df
    except pd.errors.EmptyDataError:
        logger.warning(f"Metric file is empty: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Failed to load or parse CSV file {filepath}: {e}")
        return None

def get_plot_title_info(df, metric_files_base):
    """Extracts representative config info from the first row of the DataFrame or basename."""
    # Default info structure
    info = {'method': 'UNKNOWN', 'activation': 'UNKNOWN', 'dim': 'UNKNOWN', 'config_basename': 'UNKNOWN', 'pca': 'N/A', 'norm': 'None'}
    
    # Try getting info from the DataFrame first
    if df is not None and not df.empty:
        first_row = df.iloc[0]
        info['config_basename'] = first_row.get('config_basename', 'UNKNOWN')
        info['method'] = first_row.get('reduction_method', 'UNKNOWN')
        info['activation'] = first_row.get('activation_type_reduced', first_row.get('activation_type_capture', 'UNKNOWN')) # Use reduced, fallback to capture
        info['dim'] = f"{first_row.get('final_components', 'N/A')}D"
        info['pca'] = first_row.get('pca_components', 'N/A')
        info['norm'] = first_row.get('norm_method', 'None')
        if info['norm'] is None or pd.isna(info['norm']): info['norm'] = 'None'
        return info # Return info derived from data

    # Fallback: Try parsing the metric_files_base if DataFrame is empty/None
    config_basename = os.path.basename(metric_files_base)
    info['config_basename'] = config_basename # Store the original base name
    parts = config_basename.split('-') # Split by hyphen, assuming structure like 'config_cs_reduce_umap_residual_3d-emotional'
    if len(parts) > 1:
         config_part = parts[0] # e.g., 'config_cs_reduce_umap_residual_3d'
         config_parts = config_part.split('_')
         if len(config_parts) >= 5 and config_parts[0] == 'config' and config_parts[2] == 'reduce':
             info['method'] = config_parts[3].upper()
             info['activation'] = config_parts[4]
             info['dim'] = config_parts[5]
    # PCA and Norm info cannot be reliably parsed from filename, leave as default
    logger.warning("Could not load metric data to extract full config info. Plot titles may be partial.")
    return info



def plot_layerwise_metric(df, metric_col, y_label, title_suffix, output_dir, metric_files_base, plot_config_info):
    """Generates line plots for layer-wise metrics (_cs vs _ctrl pairs)."""
    # Added metric_files_base argument
    if df is None or df.empty:
        logger.warning(f"No data provided for {y_label} plot. Skipping.")
        return

    required_cols = ['prompt_id_cs', 'prompt_id_ctrl', metric_col]
    if 'layer' in df.columns:
        sort_col = 'layer'
        required_cols.append('layer')
        x_label = 'Layer Index'
    elif 'layer_transition' in df.columns:
        sort_col = 'layer_transition'
        required_cols.append('layer_transition')
        x_label = 'Layer Transition'
    else:
        logger.error(f"Cannot determine layer column for {y_label} plot. Skipping.")
        return

    if not all(col in df.columns for col in required_cols):
        logger.error(f"Metric data for {y_label} missing required columns: {required_cols}. Skipping plot.")
        return

    # Convert numeric columns, coercing errors
    if sort_col == 'layer':
        df[sort_col] = pd.to_numeric(df[sort_col], errors='coerce')
    df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce')

    # Create a unique pair identifier for grouping/plotting
    df['pair_id'] = df['prompt_id_cs'] # Use cs_id as the representative ID for the pair

    # Drop rows where essential data is missing
    df = df.dropna(subset=[sort_col, metric_col, 'pair_id'])
    if df.empty:
        logger.warning(f"No valid data points after dropping NaNs for {metric_col}. Skipping plot.")
        return

    # Sort data for plotting lines correctly
    if sort_col == 'layer_transition':
        df['sort_key'] = df[sort_col].str.extract(r'L(\d+)_').astype(int)
        df = df.sort_values(by=['pair_id', 'sort_key'])
    else: # sort by layer
        df = df.sort_values(by=['pair_id', sort_col])

    unique_pairs = df['pair_id'].unique()
    num_plots = len(unique_pairs)
    if num_plots == 0:
        logger.warning(f"No CS/CTRL pairs found in the data for {y_label} plot.")
        return

    # Determine grid size for subplots
    ncols = min(3, num_plots) # Max 3 columns
    nrows = (num_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5), squeeze=False, sharey=True) # Share Y axis
    axes = axes.flatten() # Flatten grid for easy iteration

    plot_idx = 0
    for pair_id in unique_pairs:
        pair_df = df[df['pair_id'] == pair_id]
        if pair_df.empty: continue

        ax = axes[plot_idx]

        # Handle different x-axis types (layer vs transition)
        if sort_col == 'layer':
            x_data = pair_df['layer']
        else: # layer_transition
            x_data = pair_df['layer_transition']
            ax.tick_params(axis='x', rotation=45, labelsize=8) # Rotate labels if transitions

        ax.plot(x_data, pair_df[metric_col], marker='.', linestyle='-', label=metric_col)

        # Extract short name for title
        short_pair_name = pair_id.replace('_cs', '').replace('_ctrl', '')
        ax.set_title(f"{short_pair_name}", fontsize=10)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True)
        # ax.legend() # Legend might be redundant if only one line per plot

        plot_idx += 1

    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        fig.delaxes(axes[i])

    # Construct title using info derived from DataFrame
    title = (f"{y_label} per Layer\n"
             f"Config: {plot_config_info['config_basename']} " # Use config_basename from info dict
             f"({plot_config_info['method']} {plot_config_info['dim']}, {plot_config_info['activation']}, PCA={plot_config_info['pca']}, Norm={plot_config_info['norm']})")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

    # Save plot
    # CORRECTED: Use the original metric_files_base for the output filename prefix
    output_filename_base = os.path.basename(metric_files_base)
    plot_filename = Path(output_dir) / f"{output_filename_base}-{metric_col}.png"
    try:
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot: {plot_filename}")
    except Exception as e:
        logger.error(f"Failed to save plot {plot_filename}: {e}")
    plt.close(fig)


def plot_cosine_similarity(df, output_dir, metric_files_base, plot_config_info):
    """Generates line plots for layer-wise cosine similarity, faceted by capture type."""
    # Added metric_files_base argument
    if df is None or df.empty:
        logger.warning("No data provided for Cosine Similarity plot. Skipping.")
        return

    required_cols = ['layer', 'avg_cosine_similarity_layer', 'activation_type_capture', 'prompt_id_cs']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Cosine similarity data missing required columns: {required_cols}. Skipping plot.")
        return

    # Convert numeric columns, coercing errors
    df['layer'] = pd.to_numeric(df['layer'], errors='coerce')
    df['avg_cosine_similarity_layer'] = pd.to_numeric(df['avg_cosine_similarity_layer'], errors='coerce')

    df = df.dropna(subset=['layer', 'avg_cosine_similarity_layer', 'activation_type_capture'])
    if df.empty:
        logger.warning("No valid cosine similarity data points after dropping NaNs. Skipping plot.")
        return

    # Create a short pair name for faceting titles
    df['pair_name'] = df['prompt_id_cs'].str.replace('_cs', '', regex=False)

    # Use seaborn's FacetGrid for plotting each pair and activation type separately
    # Facet by activation type first (row), then by pair (col)
    g = sns.FacetGrid(df, row="activation_type_capture", col="pair_name", hue="pair_name",
                      height=4, aspect=1.2, margin_titles=True, sharey=True)
    g.map(sns.lineplot, "layer", "avg_cosine_similarity_layer", marker='.')

    # Set titles and labels
    g.set_axis_labels("Layer Index", "Avg Cosine Similarity")
    # Extract info for titles
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    # Add overall title
    title = (f"Layer-wise Cosine Similarity (Raw Activations)\n"
             f"Config: {plot_config_info['config_basename']} " # Use config_basename from info dict
             f"({plot_config_info['method']} {plot_config_info['dim']}, {plot_config_info['activation']}, PCA={plot_config_info['pca']}, Norm={plot_config_info['norm']})")
    g.figure.suptitle(title, y=1.03, fontsize=14)

    # Adjust layout
    g.tight_layout()

    # Save plot
    # CORRECTED: Use the original metric_files_base for the output filename prefix
    output_filename_base = os.path.basename(metric_files_base)
    plot_filename = Path(output_dir) / f"{output_filename_base}-cosine_similarity.png"
    try:
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot: {plot_filename}")
    except Exception as e:
        logger.error(f"Failed to save plot {plot_filename}: {e}")
    plt.close(g.figure)


# --- Main Plotting Function ---

def run_metrics_plot(metric_files_base, output_dir):
    """
    Loads metric CSV files and generates plots.

    Args:
        metric_files_base (str): Base path/name for the metric CSV files.
        output_dir (str): Directory to save the generated plots.
    """
    logger.info(f"Generating plots for metrics based on: {metric_files_base}")
    logger.info(f"Saving plots to directory: {output_dir}")

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- Load Data ---
    deviation_df = load_metric_csv(METRIC_FILENAMES["layerwise_deviation"].format(base=metric_files_base))
    direction_df = load_metric_csv(METRIC_FILENAMES["direction_deviation"].format(base=metric_files_base))
    cosine_df = load_metric_csv(METRIC_FILENAMES["cosine_similarity"].format(base=metric_files_base))
    # Load curvature/length later if needed

    # --- Extract Config Info from Basename/Data ---
    # Prioritise reading from a loaded df, fallback to parsing basename
    plot_config_info = None
    # Try loading from deviation_df first as it's likely most representative
    if deviation_df is not None: plot_config_info = get_plot_title_info(deviation_df, metric_files_base)
    elif direction_df is not None: plot_config_info = get_plot_title_info(direction_df, metric_files_base)
    elif cosine_df is not None: plot_config_info = get_plot_title_info(cosine_df, metric_files_base)
    else:
        logger.warning("Could not load any metric data to extract config info. Plot titles may be generic.")
        plot_config_info = get_plot_title_info(None, metric_files_base) # Pass base name for parsing


    # --- Generate Plots ---

    # Plot Layer-wise Euclidean Deviation
    plot_layerwise_metric(
        df=deviation_df,
        metric_col='avg_euclidean_distance_layer',
        y_label='Avg Euclidean Distance',
        title_suffix='Layer-wise Euclidean Deviation',
        output_dir=output_dir,
        metric_files_base=metric_files_base, # Pass original base name
        plot_config_info=plot_config_info
    )

    # Plot Layer-wise Direction Deviation Angle
    plot_layerwise_metric(
        df=direction_df,
        metric_col='avg_angle_degrees_layer',
        y_label='Avg Angle Between Steps (degrees)',
        title_suffix='Layer-wise Direction Deviation',
        output_dir=output_dir,
        metric_files_base=metric_files_base, # Pass original base name
        plot_config_info=plot_config_info
    )

    # Plot Layer-wise Cosine Similarity
    plot_cosine_similarity(
        df=cosine_df,
        output_dir=output_dir,
        metric_files_base=metric_files_base, # Pass original base name
        plot_config_info=plot_config_info
    )

    # Add calls to plot other metrics (curvature, length distributions) here later

    logger.info("Metrics plotting finished.")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plots from calculated activation metrics."
    )
    parser.add_argument(
        "metric_files_base",
        type=str,
        help="Base name/path of the metric CSV files generated by analyse_metrics.py (e.g., 'analysis/full/metrics_umap_residual')."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the generated plot image files."
    )
    # Add arguments later for filtering prompts or selecting plot types if needed

    args = parser.parse_args()

    run_metrics_plot(
        metric_files_base=args.metric_files_base,
        output_dir=args.output_dir
    )

