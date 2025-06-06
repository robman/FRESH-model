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
# Unanimous export function (NEW)
# -----------------------------------------------------------------------------

def export_unanimous_responses(consensus_df: pd.DataFrame, output_path: Path) -> None:
    """
    Export unanimous responses in three different configurations.
    
    Args:
        consensus_df: Dataframe with consensus classifications
        output_path: Path to the main output file (used to determine output directory)
    """
    output_dir = output_path.parent
    
    # Extract the base filename (without _classifications.csv suffix)
    base_name = output_path.stem.replace('_classifications', '')
    
    # 1. Unanimous on both dimensions
    unanimous_both = consensus_df[
        (consensus_df['transparency_level_unanimous'] == True) & 
        (consensus_df['response_type_unanimous'] == True)
    ].copy()
    
    unanimous_both_file = output_dir / f"{base_name}_unanimous.csv"
    unanimous_both.to_csv(unanimous_both_file, index=False)
    logger.info(f"Saved {len(unanimous_both)} responses unanimous on both dimensions to {unanimous_both_file}")
    
    # 2. Unanimous on transparency level only
    unanimous_transparency = consensus_df[
        consensus_df['transparency_level_unanimous'] == True
    ].copy()
    
    unanimous_transparency_file = output_dir / f"{base_name}_unanimous-transparency_level.csv"
    unanimous_transparency.to_csv(unanimous_transparency_file, index=False)
    logger.info(f"Saved {len(unanimous_transparency)} responses unanimous on transparency level to {unanimous_transparency_file}")
    
    # 3. Unanimous on response type only
    unanimous_response_type = consensus_df[
        consensus_df['response_type_unanimous'] == True
    ].copy()
    
    unanimous_response_type_file = output_dir / f"{base_name}_unanimous-response_type.csv"
    unanimous_response_type.to_csv(unanimous_response_type_file, index=False)
    logger.info(f"Saved {len(unanimous_response_type)} responses unanimous on response type to {unanimous_response_type_file}")

# -----------------------------------------------------------------------------
# Main analysis function
# -----------------------------------------------------------------------------

def calculate_inter_rater_reliability(csv_files: list, output_file: str):
    """
    Calculate inter-rater reliability metrics for multiple LLM classifications.
    
    Args:
        csv_files: List of CSV file paths containing LLM classifications
        output_file: Path to save reliability analysis results
    """
    
    logger.info(f"Loading {len(csv_files)} classification files...")
    
    # Load all CSV files
    llm_data = {}
    for i, csv_file in enumerate(csv_files):
        try:
            df = pd.read_csv(csv_file)
            rater_name = f"LLM_{i+1}_{Path(csv_file).stem.split('response')[-1]}"
            
            # Normalize column names (handle different capitalization)
            df.columns = df.columns.str.lower()
            if 'transparency_level' not in df.columns:
                df['transparency_level'] = df.get('transparency level', df.get('transparencylevel'))
            if 'response_type' not in df.columns:
                df['response_type'] = df.get('response type', df.get('responsetype'))
            
            # Create unique identifier for each response
            df['response_id'] = df['prompt_id'].astype(str) + '_' + df['run_id'].astype(str)
            
            # Clean and validate data
            df = df.dropna(subset=['response_id', 'transparency_level', 'response_type'])
            df['transparency_level'] = df['transparency_level'].str.lower().str.strip()
            df['response_type'] = df['response_type'].str.lower().str.strip()
            
            # Remove duplicates (keep first occurrence)
            df = df.drop_duplicates(subset=['response_id'], keep='first')
            
            llm_data[rater_name] = df
            logger.info(f"Loaded {len(df)} ratings from {rater_name}")
            
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
    
    # Prepare data for reliability analysis
    reliability_results = {
        'summary': {
            'n_raters': len(llm_data),
            'n_common_responses': len(common_responses),
            'rater_names': list(llm_data.keys())
        },
        'transparency_level': {},
        'response_type': {}
    }
    
    # Analyze each classification dimension
    for dimension in ['transparency_level', 'response_type']:
        logger.info(f"Analyzing inter-rater reliability for {dimension}...")
        
        # Create ratings matrix for this dimension
        ratings_long = []
        for rater_name, df in llm_data.items():
            common_df = df[df['response_id'].isin(common_responses)]
            for _, row in common_df.iterrows():
                ratings_long.append({
                    'subject_id': row['response_id'],
                    'rater': rater_name,
                    'rating': row[dimension]
                })
        
        ratings_df = pd.DataFrame(ratings_long)
        
        # Get unique categories
        unique_categories = sorted(ratings_df['rating'].unique())
        logger.info(f"Categories for {dimension}: {unique_categories}")
        
        # Calculate pairwise agreements
        pairwise_agreements = {}
        rater_names = list(llm_data.keys())
        for rater1, rater2 in combinations(rater_names, 2):
            rater1_ratings = ratings_df[ratings_df['rater'] == rater1].set_index('subject_id')['rating']
            rater2_ratings = ratings_df[ratings_df['rater'] == rater2].set_index('subject_id')['rating']
            
            # Align ratings by subject_id
            common_subjects = rater1_ratings.index.intersection(rater2_ratings.index)
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
            'fleiss_kappa': fleiss_kappa_val
        }
        
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
        for dimension in ['transparency_level', 'response_type']:
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
    
    # Export unanimous responses (NEW)
    export_unanimous_responses(consensus_df, Path(output_file))
    
    # Print summary
    print("\n" + "="*60)
    print("INTER-RATER RELIABILITY ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\nDataset Overview:")
    print(f"  Number of raters: {reliability_results['summary']['n_raters']}")
    print(f"  Common responses: {reliability_results['summary']['n_common_responses']}")
    print(f"  Raters: {', '.join(reliability_results['summary']['rater_names'])}")
    
    for dimension in ['transparency_level', 'response_type']:
        results = reliability_results[dimension]
        print(f"\n{dimension.replace('_', ' ').title()} Reliability:")
        print(f"  Categories: {', '.join(results['categories'])}")
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
            print(f"    {pair}: {agreement:.3f}")
    
    # Unanimous agreement statistics
    unanimous_transparency = consensus_df['transparency_level_unanimous'].sum()
    unanimous_response_type = consensus_df['response_type_unanimous'].sum()
    
    print(f"\nConsensus Statistics:")
    print(f"  Unanimous transparency agreement: {unanimous_transparency}/{len(consensus_df)} ({unanimous_transparency/len(consensus_df)*100:.1f}%)")
    print(f"  Unanimous response type agreement: {unanimous_response_type}/{len(consensus_df)} ({unanimous_response_type/len(consensus_df)*100:.1f}%)")
    print(f"  Unanimous on both dimensions: {(consensus_df['transparency_level_unanimous'] & consensus_df['response_type_unanimous']).sum()}/{len(consensus_df)}")

# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate inter-rater reliability for LLM response classifications.")
    parser.add_argument("csv_files", nargs='+', help="List of CSV files containing LLM classifications")
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
