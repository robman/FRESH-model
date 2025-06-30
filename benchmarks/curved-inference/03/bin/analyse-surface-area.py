"""
Enhanced Semantic Surface Area Analysis with Length Normalization

This script calculates semantic surface area metrics including length-normalized versions
to control for response length confounds in geometric deception analysis.
"""

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
import re

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_and_validate_data(curvature_file, salience_file, prompt_ids):
    """Load and validate curvature and salience data"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading curvature data from: {curvature_file}")
    curvature_df = pd.read_csv(curvature_file)
    
    logger.info(f"Loading salience data from: {salience_file}")
    salience_df = pd.read_csv(salience_file)
    
    logger.info(f"Loaded {len(curvature_df)} curvature records and {len(salience_df)} salience records")
    
    # Filter by prompt IDs if specified
    if prompt_ids:
        prompt_list = [p.strip() for p in prompt_ids.split(',')]
        curvature_df = curvature_df[curvature_df['prompt_id'].isin(prompt_list)]
        salience_df = salience_df[salience_df['prompt_id'].isin(prompt_list)]
        logger.info(f"Filtered to {len(curvature_df)} curvature and {len(salience_df)} salience records for specified prompt IDs")
    
    return curvature_df, salience_df

def merge_datasets(curvature_df, salience_df):
    """Merge curvature and salience datasets on key columns"""
    logger = logging.getLogger(__name__)
    
    # Debug: Print column names to identify salience column
    logger.info(f"Curvature columns: {list(curvature_df.columns)}")
    logger.info(f"Salience columns: {list(salience_df.columns)}")
    
    # Auto-detect salience column name
    salience_cols = [col for col in salience_df.columns if 'salience' in col.lower()]
    if not salience_cols:
        raise ValueError("No salience column found in salience dataframe")
    
    salience_col = salience_cols[0]  # Use first salience column found
    logger.info(f"Using salience column: {salience_col}")
    
    # Define merge keys
    merge_keys = ['prompt_id', 'run_id', 'generation_step_idx']
    
    # Merge datasets
    merged_df = pd.merge(
        curvature_df, 
        salience_df, 
        on=merge_keys, 
        how='inner',
        suffixes=('_curvature', '_salience')
    )
    
    logger.info(f"Successfully joined {len(merged_df)} records on {merge_keys}")
    
    # Remove records with NaN values in critical columns
    critical_cols = ['mean_3point_curvature', salience_col]
    initial_count = len(merged_df)
    merged_df = merged_df.dropna(subset=critical_cols)
    final_count = len(merged_df)
    
    logger.info(f"After removing NaN values: {final_count} records remain")
    if initial_count != final_count:
        logger.warning(f"Removed {initial_count - final_count} records due to NaN values")
    
    # Store the salience column name for later use
    merged_df._salience_col = salience_col
    
    return merged_df

def calculate_surface_area_metrics(merged_df, gamma_values):
    """Calculate semantic surface area with various normalizations"""
    logger = logging.getLogger(__name__)
    
    # Parse gamma values
    gamma_list = [float(g.strip()) for g in gamma_values.split(',')]
    logger.info(f"Calculating surface area for gamma values: {gamma_list}")
    
    results = []
    unique_runs = merged_df.groupby(['prompt_id', 'run_id']).size().reset_index(name='step_count')
    logger.info(f"Processing {len(unique_runs)} unique runs for surface area calculation")
    
    for _, run_info in unique_runs.iterrows():
        prompt_id = run_info['prompt_id']
        run_id = run_info['run_id']
        
        # Get data for this specific run
        run_data = merged_df[
            (merged_df['prompt_id'] == prompt_id) & 
            (merged_df['run_id'] == run_id)
        ].copy()
        
        if len(run_data) == 0:
            continue
            
        # Sort by generation step to ensure proper ordering
        run_data = run_data.sort_values('generation_step_idx')
        
        # Extract key metrics for this run
        curvature_values = run_data['mean_3point_curvature'].values
        # Use the auto-detected salience column
        salience_col = getattr(merged_df, '_salience_col', 'residual_path_salience')
        salience_values = run_data[salience_col].values
        num_steps = len(run_data)
        
        # Calculate aggregate statistics
        total_salience = np.sum(salience_values)
        total_curvature = np.sum(curvature_values)
        mean_salience = np.mean(salience_values)
        mean_curvature = np.mean(curvature_values)
        max_salience = np.max(salience_values)
        max_curvature = np.max(curvature_values)
        
        # Calculate surface area for each gamma value
        for gamma in gamma_list:
            # Core surface area calculation: A' = Σ(salience + γ × curvature)
            surface_area_components = salience_values + gamma * curvature_values
            total_surface_area = np.sum(surface_area_components)
            
            # Component contributions
            total_salience_contribution = total_salience
            total_curvature_contribution = gamma * total_curvature
            
            # LENGTH NORMALIZATION METRICS
            normalized_surface_area = total_surface_area / num_steps
            normalized_salience_contribution = total_salience_contribution / num_steps
            normalized_curvature_contribution = total_curvature_contribution / num_steps
            
            # Per-step surface area variance (geometric complexity measure)
            surface_area_variance = np.var(surface_area_components)
            surface_area_std = np.std(surface_area_components)
            
            # Curvature-to-salience ratio (geometric shape measure)
            curvature_salience_ratio = total_curvature / total_salience if total_salience > 0 else 0
            
            config_basename = run_data.iloc[0]['config_basename_curvature']

            result = {
                # Identifiers
                'config_basename': config_basename, 
                'prompt_id': prompt_id,
                'run_id': run_id,
                #'kappa_parameter': float(config_basename.split('-k')[1].split('_')[0]) if '-k' in config_basename else 0.0,
                'kappa_parameter': float(re.search(r'-k([\d.]+)', config_basename).group(1)) if re.search(r'-k([\d.]+)', config_basename) else 0.0, 
                'gamma_weighting_parameter': gamma,
                
                # Raw surface area metrics
                'total_surface_area_A_prime': total_surface_area,
                'total_salience_contribution': total_salience_contribution,
                'total_curvature_contribution': total_curvature_contribution,
                
                # LENGTH-NORMALIZED METRICS (NEW)
                'normalized_surface_area': normalized_surface_area,
                'normalized_salience_contribution': normalized_salience_contribution,
                'normalized_curvature_contribution': normalized_curvature_contribution,
                
                # Step count and averages
                'num_generation_steps_included': num_steps,
                'mean_step_salience': mean_salience,
                'mean_step_curvature': mean_curvature,
                'max_step_salience': max_salience,
                'max_step_curvature': max_curvature,
                
                # Geometric complexity metrics (NEW)
                'surface_area_variance': surface_area_variance,
                'surface_area_std': surface_area_std,
                'curvature_salience_ratio': curvature_salience_ratio,
                
                # Additional geometric measures
                'surface_area_range': np.max(surface_area_components) - np.min(surface_area_components),
                'surface_area_coefficient_of_variation': surface_area_std / normalized_surface_area if normalized_surface_area > 0 else 0
            }
            
            results.append(result)
    
    return pd.DataFrame(results)

def save_results(results_df, output_file):
    """Save results to CSV with summary statistics"""
    logger = logging.getLogger(__name__)
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("Writing final {} records to CSV...".format(len(results_df)))
    results_df.to_csv(output_file, index=False)
    logger.info(f"Semantic surface area analysis complete. Results saved to {output_file}")
    
    # Print summary statistics
    unique_prompts = results_df['prompt_id'].nunique()
    unique_gamma = results_df['gamma_weighting_parameter'].nunique()
    
    logger.info(f"Summary: Processed {len(results_df)} total records for {unique_prompts} prompts and {unique_gamma} gamma values")
    
    # Print gamma-specific summaries for both raw and normalized metrics
    for gamma in sorted(results_df['gamma_weighting_parameter'].unique()):
        gamma_data = results_df[results_df['gamma_weighting_parameter'] == gamma]
        raw_mean = gamma_data['total_surface_area_A_prime'].mean()
        norm_mean = gamma_data['normalized_surface_area'].mean()
        logger.info(f"  γ={gamma}: Raw surface area = {raw_mean:.4f}, Normalized = {norm_mean:.4f}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Calculate semantic surface area with length normalization controls'
    )
    parser.add_argument('curvature_file', help='Path to curvature CSV file')
    parser.add_argument('salience_file', help='Path to salience CSV file')
    parser.add_argument('--output_base', required=True, help='Base path for output file')
    parser.add_argument('--prompt_ids', help='Comma-separated list of prompt IDs to include')
    parser.add_argument('--gamma_values', default='0.0,0.1,0.5,1.0', 
                       help='Comma-separated gamma values for weighting curvature')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Load and validate data
        curvature_df, salience_df = load_and_validate_data(
            args.curvature_file, args.salience_file, args.prompt_ids
        )
        
        # Merge datasets
        merged_df = merge_datasets(curvature_df, salience_df)
        
        # Calculate surface area metrics
        results_df = calculate_surface_area_metrics(merged_df, args.gamma_values)
        
        # Generate output filename
        output_file = f"{args.output_base}-semantic-surface-area.csv"
        
        # Save results
        save_results(results_df, output_file)
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
