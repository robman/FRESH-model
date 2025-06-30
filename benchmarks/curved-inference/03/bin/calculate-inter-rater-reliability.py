import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict

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
# Inter-rater reliability calculations
# -----------------------------------------------------------------------------

def fleiss_kappa(ratings_matrix):
    """
    Calculate Fleiss' Kappa for multiple raters and multiple categories.
    
    Args:
        ratings_matrix: numpy array where rows = subjects, columns = categories
                       Each cell contains the number of raters who assigned that category
    
    Returns:
        float: Fleiss' Kappa coefficient
    """
    n_subjects, n_categories = ratings_matrix.shape
    n_raters = ratings_matrix.sum(axis=1)[0]  # Assuming same number of raters for all subjects
    
    # Calculate proportion of raters assigning each category
    p_j = ratings_matrix.sum(axis=0) / (n_subjects * n_raters)
    
    # Calculate P_e (expected agreement)
    P_e = np.sum(p_j ** 2)
    
    # Calculate P_i for each subject (observed agreement)
    P_i_values = []
    for i in range(n_subjects):
        r_ij = ratings_matrix[i, :]
        P_i = (np.sum(r_ij ** 2) - n_raters) / (n_raters * (n_raters - 1))
        P_i_values.append(P_i)
    
    # Calculate P_bar (mean observed agreement)
    P_bar = np.mean(P_i_values)
    
    # Calculate Fleiss' Kappa
    if P_e == 1.0:
        return 1.0 if P_bar == 1.0 else 0.0
    
    kappa = (P_bar - P_e) / (1 - P_e)
    return kappa

def pairwise_agreement(rater1_labels, rater2_labels):
    """Calculate simple pairwise agreement percentage."""
    if len(rater1_labels) != len(rater2_labels):
        return 0.0
    
    agreements = sum(1 for a, b in zip(rater1_labels, rater2_labels) if a == b)
    return agreements / len(rater1_labels)

def krippendorff_alpha_nominal(ratings_df):
    """
    Calculate Krippendorff's Alpha for nominal data.
    
    Args:
        ratings_df: DataFrame with columns=['subject_id', 'rater', 'rating']
    
    Returns:
        float: Krippendorff's Alpha coefficient
    """
    # Pivot to get raters as columns
    pivot_df = ratings_df.pivot(index='subject_id', columns='rater', values='rating')
    
    # Count observed disagreements
    observed_disagreements = 0
    total_pairs = 0
    
    for subject in pivot_df.index:
        subject_ratings = pivot_df.loc[subject].dropna()
        if len(subject_ratings) > 1:
            for i, rating1 in enumerate(subject_ratings):
                for j, rating2 in enumerate(subject_ratings):
                    if i < j:  # Avoid double counting
                        total_pairs += 1
                        if rating1 != rating2:
                            observed_disagreements += 1
    
    if total_pairs == 0:
        return np.nan
    
    observed_disagreement_rate = observed_disagreements / total_pairs
    
    # Count expected disagreements (based on marginal distributions)
    all_ratings = []
    for subject in pivot_df.index:
        subject_ratings = pivot_df.loc[subject].dropna().tolist()
        all_ratings.extend(subject_ratings)
    
    if len(all_ratings) == 0:
        return np.nan
    
    # Calculate expected disagreement rate
    unique_ratings = list(set(all_ratings))
    rating_probs = {rating: all_ratings.count(rating) / len(all_ratings) for rating in unique_ratings}
    
    expected_disagreement_rate = 1 - sum(prob ** 2 for prob in rating_probs.values())
    
    if expected_disagreement_rate == 0:
        return 1.0 if observed_disagreement_rate == 0 else 0.0
    
    alpha = 1 - (observed_disagreement_rate / expected_disagreement_rate)
    return alpha

# -----------------------------------------------------------------------------
# Unanimous export function for MOLES
# -----------------------------------------------------------------------------

def export_unanimous_responses_by_prompt(consensus_df: pd.DataFrame, output_path: Path) -> None:
    """
    Export unanimous responses grouped by prompt_id for MOLES framework.
    This creates files that match the surface area file structure.
    
    Args:
        consensus_df: Dataframe with consensus classifications
        output_path: Path to the main output file (used to determine output directory)
    """
    output_dir = output_path.parent
    
    # Extract the base filename (without _classifications.csv suffix)
    base_name = output_path.stem.replace('_classifications', '')
    
    binary_dimensions = [
        'contains_self_experience', 'contains_self_model', 'contains_self_delusion',
        'contains_self_uncertainty', 'contains_factual_response', 'contains_hallucination',
        'contains_theory_of_mind', 'contains_imaginative_construction', 
        'contains_interpretive_inference', 'contains_semantic_overfitting',
        'shows_computational_work'
    ]
    
    categorical_dimensions = ['primary_stance', 'confidence_level']
    
    # Get unique prompt_ids
    unique_prompts = consensus_df['prompt_id'].unique()
    logger.info(f"Exporting consensus data for {len(unique_prompts)} unique prompts")
    
    for prompt_id in unique_prompts:
        prompt_data = consensus_df[consensus_df['prompt_id'] == prompt_id].copy()
        logger.info(f"Processing {prompt_id}: {len(prompt_data)} responses")
        
        # 1. All binary unanimous for this prompt
        all_unanimous_cols = [f"{dim}_unanimous" for dim in binary_dimensions]
        unanimous_all = prompt_data[
            prompt_data[all_unanimous_cols].all(axis=1)
        ].copy()
        
        unanimous_all_file = output_dir / f"{base_name}-{prompt_id}-consensus_unanimous_all_binary.csv"
        unanimous_all.to_csv(unanimous_all_file, index=False)
        logger.info(f"  Saved {len(unanimous_all)} unanimous (all binary) responses to {unanimous_all_file.name}")
        
        # 2. All dimensions unanimous for this prompt
        all_dimensions_cols = [f"{dim}_unanimous" for dim in binary_dimensions + categorical_dimensions]
        unanimous_everything = prompt_data[
            prompt_data[all_dimensions_cols].all(axis=1)
        ].copy()
        
        unanimous_everything_file = output_dir / f"{base_name}-{prompt_id}-consensus_unanimous_everything.csv"
        unanimous_everything.to_csv(unanimous_everything_file, index=False)
        logger.info(f"  Saved {len(unanimous_everything)} unanimous (all dimensions) responses to {unanimous_everything_file.name}")
        
        # 3. Individual dimension exports for this prompt (optional)
        for dimension in binary_dimensions + categorical_dimensions:
            unanimous_dim = prompt_data[
                prompt_data[f'{dimension}_unanimous'] == True
            ].copy()
            
            unanimous_dim_file = output_dir / f"{base_name}-{prompt_id}-consensus_unanimous_{dimension}.csv"
            unanimous_dim.to_csv(unanimous_dim_file, index=False)
            logger.info(f"  Saved {len(unanimous_dim)} responses unanimous on {dimension}")
        
        # 4. All consensus data for this prompt (not just unanimous)
        all_consensus_file = output_dir / f"{base_name}-{prompt_id}-consensus_all.csv"
        prompt_data.to_csv(all_consensus_file, index=False)
        logger.info(f"  Saved {len(prompt_data)} total consensus responses to {all_consensus_file.name}")

# -----------------------------------------------------------------------------
# Main analysis function for MOLES
# -----------------------------------------------------------------------------

def calculate_inter_rater_reliability(csv_files: list, output_file: str):
    """
    Calculate inter-rater reliability metrics for multiple MOLES classifications.
    
    Args:
        csv_files: List of CSV file paths containing MOLES classifications
        output_file: Path to save reliability analysis results
    """
    
    logger.info(f"Loading {len(csv_files)} MOLES classification files...")
    
    # Load all CSV files
    llm_data = {}
    for i, csv_file in enumerate(csv_files):
        try:
            df = pd.read_csv(csv_file)
            
            # Extract LLM name from filename
            csv_path = Path(csv_file)
            # Expecting filenames like: "output-claude_claude_3_5_sonnet_20241022.csv"
            if '-' in csv_path.stem:
                rater_name = csv_path.stem.split('-', 1)[1]
            else:
                rater_name = f"LLM_{i+1}_{csv_path.stem}"
            
            # Normalize column names (handle different capitalization)
            df.columns = df.columns.str.lower()
            
            # MOLES framework columns
            binary_columns = [
                'contains_self_experience', 'contains_self_model', 'contains_self_delusion',
                'contains_self_uncertainty', 'contains_factual_response', 'contains_hallucination',
                'contains_theory_of_mind', 'contains_imaginative_construction', 
                'contains_interpretive_inference', 'contains_semantic_overfitting',
                'shows_computational_work'
            ]
            
            categorical_columns = ['primary_stance', 'confidence_level']
            
            required_columns = binary_columns + categorical_columns
            
            # Check for required columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing columns in {csv_file}: {missing_columns}")
                continue
            
            # Create unique identifier for each response
            df['response_id'] = df['prompt_id'].astype(str) + '_' + df['run_id'].astype(str)
            
            # Clean and validate data
            df = df.dropna(subset=['response_id'] + required_columns)
            
            # Normalize Y/N values for binary columns
            for col in binary_columns:
                df[col] = df[col].astype(str).str.upper().str.strip()
                # Filter out rows with invalid Y/N values
                df = df[df[col].isin(['Y', 'N'])]
            
            # Normalize categorical columns
            for col in categorical_columns:
                df[col] = df[col].astype(str).str.lower().str.strip()
            
            # Remove duplicates (keep first occurrence)
            df = df.drop_duplicates(subset=['response_id'], keep='first')
            
            # Filter out ERROR rows
            error_mask = df[required_columns].eq('ERROR').any(axis=1)
            if error_mask.sum() > 0:
                logger.info(f"Filtering out {error_mask.sum()} ERROR rows from {rater_name}")
                df = df[~error_mask]
            
            llm_data[rater_name] = df
            logger.info(f"Loaded {len(df)} valid ratings from {rater_name}")
            
        except Exception as e:
            logger.error(f"Error loading {csv_file}: {e}")
            continue
    
    if len(llm_data) < 2:
        logger.error("Need at least 2 LLM classification files for reliability analysis")
        return
    
    # Find common response IDs across all raters
    common_responses = None
    for rater_name, df in llm_data.items():
        response_ids = set(df['response_id'].unique())
        if common_responses is None:
            common_responses = response_ids
        else:
            common_responses = common_responses.intersection(response_ids)
    
    logger.info(f"Found {len(common_responses)} responses rated by all {len(llm_data)} LLMs")
    
    if len(common_responses) == 0:
        logger.error("No common responses found across all raters")
        return
    
    # MOLES framework dimensions
    binary_dimensions = [
        'contains_self_experience', 'contains_self_model', 'contains_self_delusion',
        'contains_self_uncertainty', 'contains_factual_response', 'contains_hallucination',
        'contains_theory_of_mind', 'contains_imaginative_construction', 
        'contains_interpretive_inference', 'contains_semantic_overfitting',
        'shows_computational_work'
    ]
    
    categorical_dimensions = ['primary_stance', 'confidence_level']
    all_dimensions = binary_dimensions + categorical_dimensions
    
    reliability_results = {
        'summary': {
            'n_raters': len(llm_data),
            'n_common_responses': len(common_responses),
            'rater_names': list(llm_data.keys())
        }
    }
    
    # Initialize results for each dimension
    for dimension in all_dimensions:
        reliability_results[dimension] = {}
    
    # Analyze each classification dimension
    for dimension in all_dimensions:
        logger.info(f"Analyzing inter-rater reliability for {dimension}...")
        
        analysis_responses = common_responses
        
        # Create ratings matrix for this dimension
        ratings_long = []
        for rater_name, df in llm_data.items():
            analysis_df = df[df['response_id'].isin(analysis_responses)]
            for _, row in analysis_df.iterrows():
                ratings_long.append({
                    'subject_id': row['response_id'],
                    'rater': rater_name,
                    'rating': row[dimension]
                })
        
        ratings_df = pd.DataFrame(ratings_long)
        
        # Skip analysis if no data for this dimension
        if len(ratings_df) == 0:
            logger.warning(f"No data found for {dimension}, skipping...")
            reliability_results[dimension] = {
                'categories': [],
                'pairwise_agreements': {},
                'mean_pairwise_agreement': np.nan,
                'krippendorff_alpha': np.nan,
                'fleiss_kappa': np.nan,
                'category_counts': {},
                'total_ratings': 0
            }
            continue
        
        # Get unique categories and counts
        unique_categories = sorted(ratings_df['rating'].unique())
        category_counts = ratings_df['rating'].value_counts().to_dict()
        total_ratings = len(ratings_df)
        
        logger.info(f"Categories for {dimension}: {unique_categories}")
        logger.info(f"Category distribution: {category_counts}")
        
        # Calculate pairwise agreements
        pairwise_agreements = {}
        rater_names = list(llm_data.keys())
        for rater1, rater2 in combinations(rater_names, 2):
            rater1_ratings = ratings_df[ratings_df['rater'] == rater1].set_index('subject_id')['rating']
            rater2_ratings = ratings_df[ratings_df['rater'] == rater2].set_index('subject_id')['rating']
            
            # Align ratings by subject_id
            common_subjects = rater1_ratings.index.intersection(rater2_ratings.index)
            if len(common_subjects) == 0:
                pairwise_agreements[f"{rater1}_vs_{rater2}"] = np.nan
                continue
                
            rater1_aligned = rater1_ratings.loc[common_subjects]
            rater2_aligned = rater2_ratings.loc[common_subjects]
            
            agreement = pairwise_agreement(rater1_aligned.tolist(), rater2_aligned.tolist())
            pairwise_agreements[f"{rater1}_vs_{rater2}"] = agreement
        
        # Calculate Krippendorff's Alpha
        try:
            alpha = krippendorff_alpha_nominal(ratings_df)
        except Exception as e:
            logger.warning(f"Could not calculate Krippendorff's Alpha for {dimension}: {e}")
            alpha = np.nan
        
        # Calculate Fleiss' Kappa (if 3+ raters)
        fleiss_kappa_val = np.nan
        if len(llm_data) >= 3:
            try:
                # Create ratings matrix for Fleiss' Kappa
                pivot_df = ratings_df.pivot_table(
                    index='subject_id', 
                    columns='rating', 
                    values='rater',
                    aggfunc='count',
                    fill_value=0
                )
                
                fleiss_kappa_val = fleiss_kappa(pivot_df.values)
            except Exception as e:
                logger.warning(f"Could not calculate Fleiss' Kappa for {dimension}: {e}")
        
        # Store results
        reliability_results[dimension] = {
            'categories': unique_categories,
            'pairwise_agreements': pairwise_agreements,
            'mean_pairwise_agreement': np.mean(list(pairwise_agreements.values())),
            'krippendorff_alpha': alpha,
            'fleiss_kappa': fleiss_kappa_val,
            'category_counts': category_counts,
            'total_ratings': total_ratings
        }
        
        if not np.isnan(reliability_results[dimension]['mean_pairwise_agreement']):
            logger.info(f"{dimension} - Mean pairwise agreement: {reliability_results[dimension]['mean_pairwise_agreement']:.3f}")
        if not np.isnan(alpha):
            logger.info(f"{dimension} - Krippendorff's Alpha: {alpha:.3f}")
        if not np.isnan(fleiss_kappa_val):
            logger.info(f"{dimension} - Fleiss' Kappa: {fleiss_kappa_val:.3f}")
    
    # Create consensus classifications
    logger.info("Creating consensus classifications...")
    consensus_data = []
    
    for response_id in common_responses:
        consensus_row = {'response_id': response_id}
        
        # Extract response metadata from first rater
        first_rater_df = list(llm_data.values())[0]
        response_row = first_rater_df[first_rater_df['response_id'] == response_id].iloc[0]
        consensus_row.update({
            'prompt_id': response_row['prompt_id'],
            'run_id': response_row['run_id'],
            'response_text': response_row['response_text']
        })
        
        # Calculate consensus for each dimension
        for dimension in all_dimensions:
            votes = []
            for rater_name, df in llm_data.items():
                rating = df[df['response_id'] == response_id][dimension].iloc[0]
                votes.append(rating)
                consensus_row[f"{dimension}_{rater_name}"] = rating
            
            # Majority vote consensus
            vote_counts = pd.Series(votes).value_counts()
            consensus_row[f"{dimension}_consensus"] = vote_counts.index[0]
            consensus_row[f"{dimension}_agreement_count"] = vote_counts.iloc[0]
            consensus_row[f"{dimension}_total_raters"] = len(votes)
            consensus_row[f"{dimension}_unanimous"] = vote_counts.iloc[0] == len(votes)
        
        consensus_data.append(consensus_row)
    
    # Save results
    consensus_df = pd.DataFrame(consensus_data)
    consensus_df.to_csv(output_file, index=False)
    logger.info(f"Saved consensus classifications and reliability analysis to {output_file}")
    
    # Export unanimous responses by prompt (for matching surface area files)
    export_unanimous_responses_by_prompt(consensus_df, Path(output_file))
    
    # Print summary
    print("\n" + "="*80)
    print("MOLES INTER-RATER RELIABILITY ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nDataset Overview:")
    print(f"  Number of raters: {reliability_results['summary']['n_raters']}")
    print(f"  Common responses: {reliability_results['summary']['n_common_responses']}")
    print(f"  Raters: {', '.join(reliability_results['summary']['rater_names'])}")
    
    # Binary dimensions analysis
    print(f"\n{'='*50}")
    print("BINARY DIMENSIONS (Y/N)")
    print(f"{'='*50}")
    
    for dimension in binary_dimensions:
        results = reliability_results[dimension]
        print(f"\n{dimension.replace('_', ' ').title()}:")
        
        if len(results['categories']) == 0:
            print(f"  No data available for analysis")
            continue
            
        print(f"  Categories: {', '.join(results['categories'])}")
        
        # Display category counts
        if results['category_counts']:
            print(f"  Rating distribution (total: {results['total_ratings']} ratings):")
            for category in sorted(results['category_counts'].keys()):
                count = results['category_counts'][category]
                percentage = (count / results['total_ratings']) * 100
                print(f"    {category}: {count} ({percentage:.1f}%)")
        
        if not np.isnan(results['mean_pairwise_agreement']):
            print(f"  Mean pairwise agreement: {results['mean_pairwise_agreement']:.3f}")
        else:
            print(f"  Mean pairwise agreement: N/A")
        
        if not np.isnan(results['krippendorff_alpha']):
            alpha_interp = "Excellent" if results['krippendorff_alpha'] > 0.8 else \
                          "Good" if results['krippendorff_alpha'] > 0.67 else \
                          "Tentative" if results['krippendorff_alpha'] > 0.33 else "Poor"
            print(f"  Krippendorff's Alpha: {results['krippendorff_alpha']:.3f} ({alpha_interp})")
        
        if not np.isnan(results['fleiss_kappa']):
            kappa_interp = "Excellent" if results['fleiss_kappa'] > 0.8 else \
                          "Good" if results['fleiss_kappa'] > 0.6 else \
                          "Moderate" if results['fleiss_kappa'] > 0.4 else \
                          "Fair" if results['fleiss_kappa'] > 0.2 else "Poor"
            print(f"  Fleiss' Kappa: {results['fleiss_kappa']:.3f} ({kappa_interp})")
        
        print("  Pairwise agreements:")
        for pair, agreement in results['pairwise_agreements'].items():
            if not np.isnan(agreement):
                print(f"    {pair}: {agreement:.3f}")
            else:
                print(f"    {pair}: N/A")
    
    # Categorical dimensions analysis
    print(f"\n{'='*50}")
    print("CATEGORICAL DIMENSIONS")
    print(f"{'='*50}")
    
    for dimension in categorical_dimensions:
        results = reliability_results[dimension]
        print(f"\n{dimension.replace('_', ' ').title()}:")
        
        if len(results['categories']) == 0:
            print(f"  No data available for analysis")
            continue
            
        print(f"  Categories: {', '.join(results['categories'])}")
        
        # Display category counts
        if results['category_counts']:
            print(f"  Rating distribution (total: {results['total_ratings']} ratings):")
            for category in sorted(results['category_counts'].keys()):
                count = results['category_counts'][category]
                percentage = (count / results['total_ratings']) * 100
                print(f"    {category}: {count} ({percentage:.1f}%)")
        
        if not np.isnan(results['mean_pairwise_agreement']):
            print(f"  Mean pairwise agreement: {results['mean_pairwise_agreement']:.3f}")
        
        if not np.isnan(results['krippendorff_alpha']):
            alpha_interp = "Excellent" if results['krippendorff_alpha'] > 0.8 else \
                          "Good" if results['krippendorff_alpha'] > 0.67 else \
                          "Tentative" if results['krippendorff_alpha'] > 0.33 else "Poor"
            print(f"  Krippendorff's Alpha: {results['krippendorff_alpha']:.3f} ({alpha_interp})")
        
        if not np.isnan(results['fleiss_kappa']):
            kappa_interp = "Excellent" if results['fleiss_kappa'] > 0.8 else \
                          "Good" if results['fleiss_kappa'] > 0.6 else \
                          "Moderate" if results['fleiss_kappa'] > 0.4 else \
                          "Fair" if results['fleiss_kappa'] > 0.2 else "Poor"
            print(f"  Fleiss' Kappa: {results['fleiss_kappa']:.3f} ({kappa_interp})")
        
        print("  Pairwise agreements:")
        for pair, agreement in results['pairwise_agreements'].items():
            if not np.isnan(agreement):
                print(f"    {pair}: {agreement:.3f}")
            else:
                print(f"    {pair}: N/A")
    
    # Unanimous agreement statistics
    print(f"\n{'='*50}")
    print("CONSENSUS STATISTICS")
    print(f"{'='*50}")
    
    for dimension in all_dimensions:
        unanimous_count = consensus_df[f'{dimension}_unanimous'].sum()
        print(f"  Unanimous {dimension}: {unanimous_count}/{len(consensus_df)} ({unanimous_count/len(consensus_df)*100:.1f}%)")

    # Overall unanimous agreement
    binary_unanimous = consensus_df[[f"{dim}_unanimous" for dim in binary_dimensions]].all(axis=1).sum()
    print(f"  Unanimous on ALL BINARY dimensions: {binary_unanimous}/{len(consensus_df)} ({binary_unanimous/len(consensus_df)*100:.1f}%)")
    
    all_unanimous = consensus_df[[f"{dim}_unanimous" for dim in all_dimensions]].all(axis=1).sum()
    print(f"  Unanimous on ALL dimensions: {all_unanimous}/{len(consensus_df)} ({all_unanimous/len(consensus_df)*100:.1f}%)")

# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate inter-rater reliability for MOLES response classifications.")
    parser.add_argument("csv_files", nargs='+', help="List of CSV files containing MOLES classifications")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file for consensus classifications and reliability analysis")
    
    args = parser.parse_args()
    
    # Validate input files
    valid_files = []
    for csv_file in args.csv_files:
        if Path(csv_file).exists():
            valid_files.append(csv_file)
        else:
            logger.warning(f"File not found: {csv_file}")
    
    if len(valid_files) < 2:
        logger.error("Need at least 2 valid CSV files for reliability analysis")
        sys.exit(1)
    
    calculate_inter_rater_reliability(valid_files, args.output)
