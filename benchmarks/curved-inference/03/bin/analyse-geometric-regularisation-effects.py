import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def analyze_surface_area_decomposition(df, group_col, output_dir):
    """Analyze how salience and curvature contributions change with regularization"""
    
    print(f"\n{'='*60}")
    print(f"SURFACE AREA DECOMPOSITION ANALYSIS")
    print(f"{'='*60}")
    
    # Required columns
    required_cols = ['total_salience_contribution', 'total_curvature_contribution', 
                     'total_surface_area_A_prime', 'mean_step_salience', 'mean_step_curvature']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing columns for decomposition analysis: {missing_cols}")
        return None
    
    results = {}
    groups = sorted(df[group_col].unique())
    
    # 1. Component analysis by group
    print(f"\nComponent Statistics by Group:")
    print("-" * 40)
    
    component_stats = []
    for group in groups:
        group_data = df[df[group_col] == group]
        
        stats_row = {
            'group': group,
            'n': len(group_data),
            'total_surface_area': group_data['total_surface_area_A_prime'].mean(),
            'salience_contribution': group_data['total_salience_contribution'].mean(),
            'curvature_contribution': group_data['total_curvature_contribution'].mean(),
            'salience_proportion': group_data['total_salience_contribution'].mean() / group_data['total_surface_area_A_prime'].mean(),
            'curvature_proportion': group_data['total_curvature_contribution'].mean() / group_data['total_surface_area_A_prime'].mean(),
            'mean_salience': group_data['mean_step_salience'].mean(),
            'mean_curvature': group_data['mean_step_curvature'].mean()
        }
        component_stats.append(stats_row)
    
    component_df = pd.DataFrame(component_stats)
    print(component_df.round(3))
    
    # 2. Test individual components for group differences
    print(f"\nStatistical Tests for Components:")
    print("-" * 40)
    
    component_tests = {}
    for component in ['total_salience_contribution', 'total_curvature_contribution', 
                     'mean_step_salience', 'mean_step_curvature']:
        
        if len(groups) == 2:
            group1_data = df[df[group_col] == groups[0]][component].dropna()
            group2_data = df[df[group_col] == groups[1]][component].dropna()
            
            if len(group1_data) >= 2 and len(group2_data) >= 2:
                stat, p_val = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                
                # Effect size (Cliff's delta)
                n1, n2 = len(group1_data), len(group2_data)
                dominance = 0
                for x in group1_data:
                    for y in group2_data:
                        if x > y:
                            dominance += 1
                        elif x < y:
                            dominance -= 1
                cliffs_d = dominance / (n1 * n2) if (n1 * n2) > 0 else 0
                
                mean_diff = group2_data.mean() - group1_data.mean()
                rel_diff = (mean_diff / group1_data.mean()) * 100 if group1_data.mean() != 0 else 0
                
                component_tests[component] = {
                    'statistic': stat,
                    'p_value': p_val,
                    'cliffs_delta': cliffs_d,
                    'mean_difference': mean_diff,
                    'relative_difference_percent': rel_diff,
                    'direction': 'increase' if mean_diff > 0 else 'decrease'
                }
                
                direction_symbol = "↑" if mean_diff > 0 else "↓"
                significance = "*" if p_val < 0.05 else ""
                
                print(f"{component}:")
                print(f"  {groups[0]} → {groups[1]}: {direction_symbol} {rel_diff:+.1f}% (p={p_val:.3f}){significance}")
                print(f"  Cliff's δ: {cliffs_d:.3f}")
    
    # 3. Compensation analysis
    print(f"\nCompensation Analysis:")
    print("-" * 40)
    
    if len(groups) == 2 and 'total_salience_contribution' in component_tests and 'total_curvature_contribution' in component_tests:
        salience_change = component_tests['total_salience_contribution']['relative_difference_percent']
        curvature_change = component_tests['total_curvature_contribution']['relative_difference_percent']
        
        print(f"Salience change: {salience_change:+.1f}%")
        print(f"Curvature change: {curvature_change:+.1f}%")
        
        # Check for compensation pattern
        if salience_change > 0 and curvature_change < 0:
            compensation_ratio = abs(salience_change) / abs(curvature_change)
            print(f"\n✓ COMPENSATION DETECTED:")
            print(f"  Pattern: Curvature ↓, Salience ↑")
            print(f"  Compensation ratio: {compensation_ratio:.2f}")
            
            if compensation_ratio > 1.0:
                print(f"  → Salience increase DOMINATES curvature decrease")
            else:
                print(f"  → Curvature decrease DOMINATES salience increase")
                
        elif salience_change < 0 and curvature_change < 0:
            print(f"\n→ Both components decrease (no compensation)")
        elif salience_change > 0 and curvature_change > 0:
            print(f"\n→ Both components increase (amplification)")
        else:
            print(f"\n→ Mixed pattern (salience ↓, curvature ↑)")
    
    # 4. Correlation analysis
    print(f"\nComponent Correlations:")
    print("-" * 40)
    
    correlations = {}
    for comp1, comp2 in [('total_salience_contribution', 'total_curvature_contribution'),
                        ('mean_step_salience', 'mean_step_curvature'),
                        ('total_salience_contribution', 'total_surface_area_A_prime'),
                        ('total_curvature_contribution', 'total_surface_area_A_prime')]:
        
        corr = df[comp1].corr(df[comp2])
        correlations[f"{comp1}_vs_{comp2}"] = corr
        print(f"{comp1} vs {comp2}: r = {corr:.3f}")
    
    # 5. Save detailed results
    results = {
        'component_statistics': component_df,
        'component_tests': component_tests,
        'correlations': correlations
    }
    
    # Save component statistics
    comp_path = Path(output_dir) / 'component_decomposition.csv'
    component_df.to_csv(comp_path, index=False)
    logger.info(f"Saved component decomposition to {comp_path}")
    
    return results

def create_decomposition_visualization(df, group_col, output_dir, prompt_name=""):
    """Create visualizations showing salience vs curvature decomposition"""
    
    logger.info("Creating decomposition visualizations...")
    
    groups = sorted(df[group_col].unique())
    n_groups = len(groups)
    
    if n_groups < 2:
        logger.warning("Need at least 2 groups for decomposition visualization")
        return
    
    # Set up colors
    colors = ['#E74C3C', '#3498DB', '#F39C12'][:n_groups]
    palette = dict(zip(groups, colors))
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Surface Area Decomposition: {prompt_name}', fontsize=14, fontweight='bold')
    
    # 1. Stacked bar chart of components
    component_means = []
    salience_means = []
    curvature_means = []
    
    for group in groups:
        group_data = df[df[group_col] == group]
        salience_mean = group_data['total_salience_contribution'].mean()
        curvature_mean = group_data['total_curvature_contribution'].mean()
        
        salience_means.append(salience_mean)
        curvature_means.append(curvature_mean)
        component_means.append(salience_mean + curvature_mean)
    
    x_pos = range(len(groups))
    
    # Stacked bars
    bars1 = ax1.bar(x_pos, salience_means, color='lightcoral', alpha=0.8, 
                    label='Salience Contribution', edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x_pos, curvature_means, bottom=salience_means, color='lightblue', alpha=0.8,
                    label='Curvature Contribution', edgecolor='black', linewidth=1)
    
    # Add total labels
    for i, (sal, curv) in enumerate(zip(salience_means, curvature_means)):
        total = sal + curv
        ax1.text(i, total + max(component_means) * 0.02, f'{total:.0f}', 
                ha='center', va='bottom', fontweight='bold')
    
    ax1.set_title('Total Surface Area Components')
    ax1.set_ylabel('Contribution to Surface Area')
    ax1.set_xlabel('Group')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(groups)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Component proportions
    proportions_salience = [s/(s+c) for s, c in zip(salience_means, curvature_means)]
    proportions_curvature = [c/(s+c) for s, c in zip(salience_means, curvature_means)]
    
    bars1 = ax2.bar(x_pos, proportions_salience, color='lightcoral', alpha=0.8, 
                    label='Salience %', edgecolor='black', linewidth=1)
    bars2 = ax2.bar(x_pos, proportions_curvature, bottom=proportions_salience, 
                    color='lightblue', alpha=0.8, label='Curvature %', edgecolor='black', linewidth=1)
    
    # Add percentage labels
    for i, (prop_sal, prop_curv) in enumerate(zip(proportions_salience, proportions_curvature)):
        ax2.text(i, prop_sal/2, f'{prop_sal*100:.1f}%', ha='center', va='center', fontweight='bold')
        ax2.text(i, prop_sal + prop_curv/2, f'{prop_curv*100:.1f}%', ha='center', va='center', fontweight='bold')
    
    ax2.set_title('Component Proportions')
    ax2.set_ylabel('Proportion of Total Surface Area')
    ax2.set_xlabel('Group')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(groups)
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Scatter plot: Salience vs Curvature
    for i, group in enumerate(groups):
        group_data = df[df[group_col] == group]
        ax3.scatter(group_data['total_curvature_contribution'], 
                   group_data['total_salience_contribution'],
                   color=palette[group], alpha=0.7, s=60, 
                   label=group, edgecolors='black', linewidth=0.5)
    
    # Add correlation line
    all_curv = df['total_curvature_contribution']
    all_sal = df['total_salience_contribution']
    
    if len(all_curv.dropna()) > 1 and len(all_sal.dropna()) > 1:
        corr_coef = all_curv.corr(all_sal)
        z = np.polyfit(all_curv.dropna(), all_sal.dropna(), 1)
        p = np.poly1d(z)
        x_trend = np.linspace(all_curv.min(), all_curv.max(), 100)
        ax3.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
        ax3.text(0.05, 0.95, f'r = {corr_coef:.3f}', transform=ax3.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax3.set_title('Salience vs Curvature Contributions')
    ax3.set_xlabel('Curvature Contribution')
    ax3.set_ylabel('Salience Contribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Change analysis (if 2 groups)
    if len(groups) == 2:
        components = ['Salience', 'Curvature']
        changes = [
            ((salience_means[1] - salience_means[0]) / salience_means[0]) * 100,
            ((curvature_means[1] - curvature_means[0]) / curvature_means[0]) * 100
        ]
        
        colors_change = ['lightcoral' if c >= 0 else 'lightblue' for c in changes]
        bars = ax4.bar(components, changes, color=colors_change, alpha=0.8, 
                      edgecolor='black', linewidth=1)
        
        # Add value labels
        for bar, change in zip(bars, changes):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                    f'{change:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                    fontweight='bold')
        
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_title(f'Component Changes: {groups[0]} → {groups[1]}')
        ax4.set_ylabel('Relative Change (%)')
        ax4.grid(True, alpha=0.3)
        
        # Add compensation analysis text
        if changes[0] > 0 and changes[1] < 0:
            compensation_text = "COMPENSATION:\nSalience ↑, Curvature ↓"
            color = 'orange'
        elif changes[0] < 0 and changes[1] < 0:
            compensation_text = "BOTH DECREASE:\nNo compensation"
            color = 'red'
        elif changes[0] > 0 and changes[1] > 0:
            compensation_text = "BOTH INCREASE:\nAmplification"
            color = 'green'
        else:
            compensation_text = "MIXED PATTERN"
            color = 'gray'
        
        ax4.text(0.02, 0.98, compensation_text, transform=ax4.transAxes, 
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
                verticalalignment='top', fontsize=9, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Change analysis\nrequires exactly\n2 groups', 
                ha='center', va='center', transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax4.set_title('Component Changes')
    
    plt.tight_layout()
    
    # Save visualization
    output_path = Path(output_dir) / f'decomposition_analysis_{prompt_name.replace(" ", "_")}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved decomposition visualization to {output_path}")
    plt.close()

def load_consensus_intersection(consensus_files, labels):
    """Find intersection of consensus (prompt_id, run_id) pairs across all kappa values"""
    consensus_data = {}
    
    for consensus_file, label in zip(consensus_files, labels):
        if not Path(consensus_file).exists():
            logger.warning(f"Consensus file not found: {consensus_file}")
            consensus_data[label] = set()
            continue
            
        try:
            df = pd.read_csv(consensus_file)
            if len(df) == 0:
                logger.warning(f"Empty consensus file: {consensus_file}")
                consensus_data[label] = set()
            else:
                consensus_pairs = set(zip(df['prompt_id'], df['run_id']))
                consensus_data[label] = consensus_pairs
                logger.info(f"{label}: {len(consensus_pairs)} consensus pairs")
        except Exception as e:
            logger.error(f"Error loading {consensus_file}: {e}")
            consensus_data[label] = set()
    
    # Find intersection across all kappa values that have data
    valid_sets = [pairs for pairs in consensus_data.values() if len(pairs) > 0]
    
    if len(valid_sets) == 0:
        logger.warning("No valid consensus data found")
        return set(), consensus_data
    
    intersection = valid_sets[0]
    for consensus_set in valid_sets[1:]:
        intersection = intersection.intersection(consensus_set)
    
    logger.info(f"Consensus intersection: {len(intersection)} (prompt_id, run_id) pairs")
    return intersection, consensus_data

def load_and_filter_surface_data(surface_files, consensus_intersection, labels):
    """Load surface area data and filter to consensus intersection with robust error handling"""
    datasets = []
    
    for i, (surface_file, label) in enumerate(zip(surface_files, labels)):
        if not Path(surface_file).exists():
            logger.warning(f"Surface file not found: {surface_file}")
            continue
            
        try:
            df = pd.read_csv(surface_file)
            
            if len(df) == 0:
                logger.warning(f"Empty surface file: {surface_file}")
                continue
            
            # Filter to consensus intersection if it exists
            if len(consensus_intersection) > 0:
                mask = df.apply(lambda row: (row['prompt_id'], row['run_id']) in consensus_intersection, axis=1)
                df_filtered = df[mask].copy()
            else:
                logger.warning(f"No consensus intersection - using all data from {surface_file}")
                df_filtered = df.copy()
            
            df_filtered['kappa_condition'] = label
            logger.info(f"{label}: {len(df_filtered)} records after filtering")
            
            if len(df_filtered) > 0:
                datasets.append(df_filtered)
            
        except Exception as e:
            logger.error(f"Error loading {surface_file}: {e}")
            continue
    
    if len(datasets) == 0:
        logger.error("No valid surface data loaded")
        return pd.DataFrame()
    
    combined_df = pd.concat(datasets, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined_df)} total records")
    return combined_df

def analyze_regularization_effects_robust(combined_df, gamma_value, labels):
    """Robust regularization effects analysis that handles missing conditions"""
    results = []
    
    if gamma_value is None:
        gamma_values = sorted(combined_df['gamma_weighting_parameter'].unique())
        logger.info(f"Analyzing all gamma values: {gamma_values}")
    else:
        gamma_values = [gamma_value]
        logger.info(f"Analyzing specific gamma value: {gamma_value}")
    
    for gamma in gamma_values:
        gamma_data = combined_df[combined_df['gamma_weighting_parameter'] == gamma]
        
        # Check what kappa conditions we actually have in the data
        available_conditions = gamma_data['kappa_condition'].unique()
        logger.info(f"Gamma {gamma}: Available conditions: {available_conditions}")
        
        # Skip if we don't have data for analysis
        if len(available_conditions) < 1:
            logger.warning(f"Gamma {gamma}: No conditions available for analysis")
            continue
            
        # Try to create pivot table with available conditions
        try:
            pivot_df = gamma_data.pivot_table(
                index=['prompt_id', 'run_id'], 
                columns='kappa_condition',
                values=['total_surface_area_A_prime', 'normalized_surface_area', 
                        'total_salience_contribution', 'total_curvature_contribution'],
                aggfunc='first'
            )
            
            logger.info(f"Gamma {gamma}: Pivot table shape: {pivot_df.shape}")
            logger.info(f"Gamma {gamma}: Pivot columns: {pivot_df.columns.tolist()}")
            
            # Analyze all available pairs of conditions
            for i, cond1 in enumerate(available_conditions):
                for j, cond2 in enumerate(available_conditions):
                    if i >= j:  # Only analyze each pair once
                        continue
                        
                    for metric in ['total_surface_area_A_prime', 'normalized_surface_area']:
                        try:
                            before_values = pivot_df[(metric, cond1)].dropna()
                            after_values = pivot_df[(metric, cond2)].dropna()
                            
                            # Find common indices (paired observations)
                            common_index = before_values.index.intersection(after_values.index)
                            
                            if len(common_index) < 2:
                                logger.info(f"Gamma {gamma}, {cond1} vs {cond2}, {metric}: Insufficient paired data ({len(common_index)} pairs)")
                                continue
                                
                            before_paired = before_values[common_index]
                            after_paired = after_values[common_index]
                            
                            # Perform paired t-test
                            stat, p_value = stats.ttest_rel(before_paired, after_paired)
                            effect_size = (after_paired.mean() - before_paired.mean()) / before_paired.std() if before_paired.std() > 0 else 0
                            
                            results.append({
                                'gamma_value': gamma,
                                'comparison': f"{cond1} vs {cond2}",
                                'metric': metric,
                                'n_pairs': len(common_index),
                                'before_mean': before_paired.mean(),
                                'after_mean': after_paired.mean(),
                                'mean_difference': after_paired.mean() - before_paired.mean(),
                                'effect_size_cohens_d': effect_size,
                                't_statistic': stat,
                                'p_value': p_value
                            })
                            
                        except Exception as e:
                            logger.warning(f"Error analyzing {cond1} vs {cond2} for {metric}: {e}")
                            continue
                            
        except Exception as e:
            logger.error(f"Error creating pivot table for gamma {gamma}: {e}")
            continue
    
    if len(results) == 0:
        logger.warning("No valid regularization effects could be computed")
        return pd.DataFrame()
    
    return pd.DataFrame(results)

def enhanced_descriptive_stats(df, measure_col, group_col):
    """Enhanced descriptive statistics with confidence intervals."""
    results = []
    
    for group in df[group_col].unique():
        group_data = df[df[group_col] == group][measure_col].dropna()
        
        if len(group_data) > 0:
            stats_dict = {
                'group': group,
                'n': len(group_data),
                'mean': group_data.mean(),
                'std': group_data.std(),
                'median': group_data.median(),
                'min': group_data.min(),
                'max': group_data.max(),
                'sem': group_data.sem()
            }
            
            # Bootstrap CI for mean (only if n >= 3)
            if len(group_data) >= 3:
                n_bootstrap = min(1000, len(group_data) * 100)  # Adjust for small samples
                bootstrap_means = []
                for _ in range(n_bootstrap):
                    bootstrap_sample = np.random.choice(group_data, size=len(group_data), replace=True)
                    bootstrap_means.append(np.mean(bootstrap_sample))
                
                ci_lower = np.percentile(bootstrap_means, 2.5)
                ci_upper = np.percentile(bootstrap_means, 97.5)
                stats_dict['ci_lower'] = ci_lower
                stats_dict['ci_upper'] = ci_upper
                stats_dict['ci_width'] = ci_upper - ci_lower
            else:
                stats_dict['ci_lower'] = np.nan
                stats_dict['ci_upper'] = np.nan
                stats_dict['ci_width'] = np.nan
            
            results.append(stats_dict)
    
    return pd.DataFrame(results)

def perform_robust_group_comparison(df, measure_col, group_col, measure_name):
    """Robust group comparison that handles small samples and missing data"""
    
    print(f"\n{'='*60}")
    print(f"ROBUST GROUP ANALYSIS: {measure_name} by {group_col}")
    print(f"{'='*60}")
    
    # Get groups and check data availability
    groups = df[group_col].unique()
    group_data = [df[df[group_col] == group][measure_col].dropna() for group in groups]
    
    # Filter out empty groups
    valid_groups = []
    valid_data = []
    for i, (group, data) in enumerate(zip(groups, group_data)):
        if len(data) > 0:
            valid_groups.append(group)
            valid_data.append(data)
        else:
            logger.warning(f"Group '{group}' has no valid data for {measure_col}")
    
    if len(valid_groups) < 1:
        print(f"No groups with valid data found for {measure_col}")
        return {"error": "no_valid_data", "message": "No groups with valid data"}
    
    if len(valid_groups) == 1:
        print(f"Only one group with data: {valid_groups[0]} (n={len(valid_data[0])})")
        desc_stats = enhanced_descriptive_stats(df[df[group_col] == valid_groups[0]], measure_col, group_col)
        print(desc_stats.round(3))
        return {
            "error": "single_group",
            "group": valid_groups[0],
            "n": len(valid_data[0]),
            "descriptive_stats": desc_stats
        }
    
    # Enhanced descriptive statistics
    print(f"Enhanced Descriptive Statistics:")
    desc_stats = enhanced_descriptive_stats(df, measure_col, group_col)
    print(desc_stats.round(3))
    
    # Check sample sizes and warn about small samples
    min_n = min(len(data) for data in valid_data)
    total_n = sum(len(data) for data in valid_data)
    
    print(f"\nSample Size Summary:")
    for group, data in zip(valid_groups, valid_data):
        asterisk = "*" if len(data) < 5 else ""
        print(f"  {group}: n = {len(data)}{asterisk}")
    print(f"  Total: N = {total_n}")
    if min_n < 5:
        print(f"  Warning: Minimum group size is {min_n} - interpret results with caution")
    
    # Choose appropriate statistical test
    if len(valid_groups) == 2 and min_n >= 2:
        # Two group comparison
        try:
            # Use Mann-Whitney U for small samples or non-normal data
            stat, p_value = mannwhitneyu(valid_data[0], valid_data[1], alternative='two-sided')
            test_name = "Mann-Whitney U test"
            
            # Calculate effect size (Cliff's delta)
            n1, n2 = len(valid_data[0]), len(valid_data[1])
            dominance = 0
            for x in valid_data[0]:
                for y in valid_data[1]:
                    if x > y:
                        dominance += 1
                    elif x < y:
                        dominance -= 1
            
            cliffs_d = dominance / (n1 * n2) if (n1 * n2) > 0 else 0
            
            # Effect size interpretation
            abs_d = abs(cliffs_d)
            if abs_d < 0.147:
                effect_interp = "Negligible"
            elif abs_d < 0.33:
                effect_interp = "Small"
            elif abs_d < 0.474:
                effect_interp = "Medium"
            else:
                effect_interp = "Large"
            
            print(f"\n{test_name} Results:")
            print(f"  Test statistic: {stat:.4f}")
            print(f"  p-value: {p_value:.6f}")
            print(f"  Cliff's delta: {cliffs_d:.4f} ({effect_interp})")
            
            # Practical significance
            mean_diff = valid_data[1].mean() - valid_data[0].mean()
            rel_diff = (mean_diff / valid_data[0].mean()) * 100 if valid_data[0].mean() != 0 else 0
            print(f"  Mean difference: {mean_diff:.2f} ({rel_diff:.1f}%)")
            
            return {
                'test_name': test_name,
                'statistic': stat,
                'p_value': p_value,
                'groups': valid_groups,
                'group_means': [data.mean() for data in valid_data],
                'cliffs_delta': cliffs_d,
                'effect_interpretation': effect_interp,
                'mean_difference': mean_diff,
                'relative_difference_percent': rel_diff,
                'descriptive_stats': desc_stats,
                'sample_sizes': [len(data) for data in valid_data]
            }
            
        except Exception as e:
            print(f"Statistical test failed: {e}")
            return {"error": "test_failed", "exception": str(e), "descriptive_stats": desc_stats}
    
    elif len(valid_groups) > 2 and min_n >= 2:
        # Multiple group comparison
        try:
            stat, p_value = kruskal(*valid_data)
            test_name = "Kruskal-Wallis test"
            
            print(f"\n{test_name} Results:")
            print(f"  Test statistic: {stat:.4f}")
            print(f"  p-value: {p_value:.6f}")
            
            return {
                'test_name': test_name,
                'statistic': stat,
                'p_value': p_value,
                'groups': valid_groups,
                'group_means': [data.mean() for data in valid_data],
                'descriptive_stats': desc_stats,
                'sample_sizes': [len(data) for data in valid_data]
            }
            
        except Exception as e:
            print(f"Statistical test failed: {e}")
            return {"error": "test_failed", "exception": str(e), "descriptive_stats": desc_stats}
    
    else:
        print(f"Insufficient data for statistical testing (min_n={min_n}, groups={len(valid_groups)})")
        return {
            "error": "insufficient_data",
            "min_n": min_n,
            "n_groups": len(valid_groups),
            "descriptive_stats": desc_stats
        }

def create_robust_visualization(df, surface_area_col, group_col, output_dir, prompt_name=""):
    """Create visualizations that work with small samples and missing data"""
    
    if len(df) == 0:
        logger.warning("No data available for visualization")
        return
    
    logger.info("Creating robust visualizations...")
    
    # Set style
    plt.style.use('default')
    sns.set_style("whitegrid", {"grid.alpha": 0.3})
    plt.rcParams.update({'font.size': 10, 'figure.titlesize': 12})
    
    # Get groups and data
    groups = sorted(df[group_col].unique())
    n_groups = len(groups)
    
    if n_groups == 0:
        logger.warning("No groups found for visualization")
        return
    
    # Color palette
    if n_groups <= 3:
        colors = ['#E74C3C', '#3498DB', '#F39C12'][:n_groups]
        palette = dict(zip(groups, colors))
    else:
        palette = dict(zip(groups, sns.color_palette("Set1", n_groups)))
    
    # Create figure layout
    if n_groups == 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    fig.suptitle(f'Geometric Analysis: {prompt_name}', fontsize=14, fontweight='bold')
    
    # 1. Bar plot with error bars
    desc_stats = enhanced_descriptive_stats(df, surface_area_col, group_col)
    desc_stats = desc_stats.set_index('group').reindex(groups).reset_index()
    
    bars = ax1.bar(range(len(groups)), desc_stats['mean'], 
                   color=[palette[g] for g in desc_stats['group']], alpha=0.8, 
                   edgecolor='black', linewidth=1)
    
    # Add error bars if we have CI data
    if not desc_stats['ci_lower'].isna().all():
        errors_lower = desc_stats['mean'] - desc_stats['ci_lower']
        errors_upper = desc_stats['ci_upper'] - desc_stats['mean']
        ax1.errorbar(range(len(groups)), desc_stats['mean'], 
                     yerr=[errors_lower, errors_upper], 
                     fmt='none', color='black', capsize=5, linewidth=2)
    
    # Add value labels and sample size warnings
    for i, (bar, mean_val, n_val) in enumerate(zip(bars, desc_stats['mean'], desc_stats['n'])):
        label_text = f'{mean_val:.0f}'
        if n_val < 5:
            label_text += '*'
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05, 
                 label_text, ha='center', va='bottom', fontweight='bold')
    
    ax1.set_title('Mean Surface Area by Group')
    ax1.set_ylabel('Surface Area (A′)')
    ax1.set_xlabel('Group')
    ax1.set_xticks(range(len(groups)))
    ax1.set_xticklabels(groups, rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribution plot
    if n_groups == 1:
        # Single group histogram
        data = df[surface_area_col].dropna()
        ax2.hist(data, bins=min(10, len(data)//2 + 1), alpha=0.7, 
                 color=list(palette.values())[0], edgecolor='black')
        ax2.axvline(data.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {data.mean():.0f}')
        ax2.legend()
    else:
        # Multiple groups - use violin or box plot depending on sample sizes
        min_group_size = min(len(df[df[group_col] == g][surface_area_col].dropna()) for g in groups)
        
        if min_group_size >= 5:
            # Use violin plot for larger samples
            violin_data = [df[df[group_col] == group][surface_area_col].dropna() for group in groups]
            parts = ax2.violinplot(violin_data, positions=range(len(groups)), widths=0.6)
            
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(list(palette.values())[i])
                pc.set_alpha(0.7)
        else:
            # Use box plot with individual points for small samples
            box_data = [df[df[group_col] == group][surface_area_col].dropna() for group in groups]
            bp = ax2.boxplot(box_data, positions=range(len(groups)), patch_artist=True)
            
            for i, (patch, group) in enumerate(zip(bp['boxes'], groups)):
                patch.set_facecolor(palette[group])
                patch.set_alpha(0.7)
            
            # Add individual points for very small samples
            for i, group in enumerate(groups):
                group_data = df[df[group_col] == group][surface_area_col].dropna()
                if len(group_data) <= 10:
                    y_data = group_data.values
                    x_data = np.random.normal(i, 0.04, size=len(y_data))
                    ax2.scatter(x_data, y_data, alpha=0.8, s=30, color='darkred', 
                               edgecolors='black', linewidth=0.5, zorder=10)
    
    ax2.set_title('Distribution by Group')
    ax2.set_ylabel('Surface Area (A′)')
    ax2.set_xlabel('Group')
    if n_groups > 1:
        ax2.set_xticks(range(len(groups)))
        ax2.set_xticklabels(groups, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Sample composition (if multiple groups)
    if n_groups > 1:
        group_sizes = desc_stats.set_index('group')['n']
        
        # Pie chart
        wedges, texts, autotexts = ax3.pie(group_sizes.values, labels=group_sizes.index, 
                                          colors=[palette[g] for g in group_sizes.index],
                                          autopct='%1.1f%%', startangle=90)
        
        # Highlight small groups
        small_groups = group_sizes[group_sizes < 5].index.tolist()
        if small_groups:
            ax3.text(0, -1.3, f"*n < 5: {', '.join(small_groups)}", ha='center', va='center', 
                    fontsize=9, style='italic', transform=ax3.transData)
        
        ax3.set_title('Sample Sizes by Group')
        
        # 4. Summary statistics table
        ax4.axis('off')
        
        # Create summary text
        summary_text = f"Analysis Summary\n{'='*20}\n\n"
        summary_text += f"Total N: {len(df)}\n"
        summary_text += f"Groups: {n_groups}\n\n"
        
        for _, row in desc_stats.iterrows():
            asterisk = '*' if row['n'] < 5 else ''
            summary_text += f"{row['group']}: {row['mean']:.0f} ± {row['std']:.0f} (n={row['n']}){asterisk}\n"
        
        if any(desc_stats['n'] < 5):
            summary_text += f"\n*Small sample warning\n"
        
        # Check for gamma information
        if 'gamma_weighting_parameter' in df.columns:
            gamma_vals = df['gamma_weighting_parameter'].unique()
            summary_text += f"\nGamma: {gamma_vals[0] if len(gamma_vals) == 1 else 'Multiple'}\n"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / f'robust_analysis_{prompt_name.replace(" ", "_")}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved visualization to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Robust analysis of geometric surface area across conditions.")
    parser.add_argument('surface_files', nargs='+', help='Surface area CSV files for different conditions')
    parser.add_argument('--consensus_files', nargs='+', help='Consensus CSV files (optional)')
    parser.add_argument('--labels', required=True, help='Comma-separated labels for conditions')
    parser.add_argument("--output_dir", default="robust_analysis_results", help="Output directory")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma value to analyze")
    parser.add_argument("--prompt_name", default="", help="Name for this prompt type")

    args = parser.parse_args()
    
    labels = [label.strip() for label in args.labels.split(',')]
    
    if len(args.surface_files) != len(labels):
        logger.error("Number of surface files must match number of labels")
        return
    
    # Handle consensus files
    consensus_files = args.consensus_files if args.consensus_files else []
    if consensus_files and len(consensus_files) != len(labels):
        logger.warning("Number of consensus files doesn't match labels - proceeding without consensus filtering")
        consensus_files = []
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data with robust error handling
    if consensus_files:
        logger.info("Finding consensus intersection...")
        consensus_intersection, consensus_data = load_consensus_intersection(consensus_files, labels)
    else:
        logger.info("No consensus files provided - using all surface area data")
        consensus_intersection = set()
        consensus_data = {}
    
    logger.info("Loading surface area data...")
    merged_df = load_and_filter_surface_data(args.surface_files, consensus_intersection, labels)
    
    if len(merged_df) == 0:
        logger.error("No data available for analysis")
        return
    
    # Filter to specific gamma
    if 'gamma_weighting_parameter' in merged_df.columns:
        gamma_data = merged_df[merged_df['gamma_weighting_parameter'] == args.gamma]
        if len(gamma_data) == 0:
            logger.warning(f"No data found for gamma={args.gamma}, using all gamma values")
            gamma_data = merged_df
        else:
            merged_df = gamma_data
            logger.info(f"Filtered to gamma={args.gamma}: {len(merged_df)} records")
    
    # Print data overview
    print(f"\n{'='*60}")
    print(f"ROBUST ANALYSIS: {args.prompt_name}")
    print(f"{'='*60}")
    print(f"Total records: {len(merged_df)}")
    print(f"Conditions: {labels}")
    print(f"Gamma: {args.gamma}")
    
    if 'kappa_condition' in merged_df.columns:
        condition_counts = merged_df['kappa_condition'].value_counts()
        print(f"Condition distribution: {condition_counts.to_dict()}")
    
    # Main analysis
    surface_area_col = 'total_surface_area_A_prime'
    
    if 'kappa_condition' in merged_df.columns and len(merged_df['kappa_condition'].unique()) > 0:
        # Analysis by kappa condition
        results = perform_robust_group_comparison(
            merged_df, surface_area_col, 'kappa_condition', 
            f"Surface Area Analysis: {args.prompt_name}"
        )
        
        # Regularization effects analysis
        if len(labels) >= 2:
            logger.info("Analyzing regularization effects...")
            reg_results = analyze_regularization_effects_robust(merged_df, args.gamma, labels)
            
            if not reg_results.empty:
                print(f"\n{'='*60}")
                print(f"REGULARIZATION EFFECTS")
                print(f"{'='*60}")
                print(reg_results.round(4))
                
                # Save results
                reg_path = output_dir / 'regularization_effects.csv'
                reg_results.to_csv(reg_path, index=False)
                logger.info(f"Saved regularization effects to {reg_path}")
            else:
                logger.warning("No regularization effects could be computed")
    
    # Create visualizations
    create_robust_visualization(merged_df, surface_area_col, 'kappa_condition', 
                               output_dir, args.prompt_name)

    # Decomposition analysis
    if len(merged_df) > 0 and 'kappa_condition' in merged_df.columns:
        decomp_results = analyze_surface_area_decomposition(merged_df, 'kappa_condition', output_dir)
        
        # Create decomposition visualizations
        create_decomposition_visualization(merged_df, 'kappa_condition', output_dir, args.prompt_name)
        
        # Update summary file with decomposition results
        if decomp_results:
            summary_path = output_dir / 'analysis_summary.txt'
            with open(summary_path, 'a') as f:  # Append mode
                f.write(f"\n\nDECOMPOSITION ANALYSIS:\n")
                f.write(f"{'='*30}\n")
                
                if 'component_tests' in decomp_results:
                    for component, test_result in decomp_results['component_tests'].items():
                        f.write(f"{component}:\n")
                        f.write(f"  Change: {test_result['relative_difference_percent']:+.1f}%\n")
                        f.write(f"  P-value: {test_result['p_value']:.4f}\n")
                        f.write(f"  Direction: {test_result['direction']}\n\n")
                
                # Add compensation summary
                if len(merged_df['kappa_condition'].unique()) == 2:
                    sal_test = decomp_results['component_tests'].get('total_salience_contribution')
                    curv_test = decomp_results['component_tests'].get('total_curvature_contribution')
                    
                    if sal_test and curv_test:
                        sal_change = sal_test['relative_difference_percent']
                        curv_change = curv_test['relative_difference_percent']
                        
                        f.write(f"Compensation Analysis:\n")
                        if sal_change > 0 and curv_change < 0:
                            f.write(f"  ✓ COMPENSATION DETECTED\n")
                            f.write(f"  Salience: +{sal_change:.1f}%, Curvature: {curv_change:.1f}%\n")
                        else:
                            f.write(f"  No compensation pattern\n")
    
    # Save processed data
    output_path = output_dir / 'processed_analysis_data.csv'
    merged_df.to_csv(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")
    
    # Save summary
    summary_path = output_dir / 'analysis_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"ROBUST GEOMETRIC ANALYSIS SUMMARY\n")
        f.write(f"{'='*40}\n\n")
        f.write(f"Prompt: {args.prompt_name}\n")
        f.write(f"Analysis Date: {pd.Timestamp.now()}\n")
        f.write(f"Gamma Parameter: {args.gamma}\n")
        f.write(f"Conditions: {', '.join(labels)}\n")
        f.write(f"Total Records: {len(merged_df)}\n\n")
        
        if 'results' in locals() and isinstance(results, dict) and 'error' not in results:
            f.write(f"Statistical Test: {results['test_name']}\n")
            f.write(f"Test Statistic: {results['statistic']:.4f}\n")
            f.write(f"P-value: {results['p_value']:.6f}\n")
            if 'cliffs_delta' in results:
                f.write(f"Effect Size (Cliff's δ): {results['cliffs_delta']:.4f}\n")
                f.write(f"Effect Interpretation: {results['effect_interpretation']}\n")
            if 'mean_difference' in results:
                f.write(f"Mean Difference: {results['mean_difference']:.2f}\n")
                f.write(f"Relative Difference: {results['relative_difference_percent']:.1f}%\n")
        elif 'results' in locals() and isinstance(results, dict) and 'error' in results:
            f.write(f"Analysis Status: {results['error']}\n")
            f.write(f"Message: {results.get('message', 'No additional details')}\n")
        
        # Add consensus information if available
        if consensus_data:
            f.write(f"\nConsensus Information:\n")
            for label, pairs in consensus_data.items():
                f.write(f"  {label}: {len(pairs)} consensus pairs\n")
            f.write(f"  Intersection: {len(consensus_intersection)} pairs\n")
        
        # Add data quality warnings
        f.write(f"\nData Quality Notes:\n")
        if 'kappa_condition' in merged_df.columns:
            for condition in merged_df['kappa_condition'].unique():
                n = len(merged_df[merged_df['kappa_condition'] == condition])
                warning = " (SMALL SAMPLE - interpret with caution)" if n < 5 else ""
                f.write(f"  {condition}: n={n}{warning}\n")
    
    logger.info(f"Saved analysis summary to {summary_path}")
    
    # Final summary to console
    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE: {args.prompt_name}")
    print(f"{'='*60}")
    
    if 'results' in locals() and isinstance(results, dict):
        if 'error' not in results:
            print(f"✓ Statistical analysis completed successfully")
            print(f"  Test: {results['test_name']}")
            print(f"  P-value: {results['p_value']:.6f}")
            if 'cliffs_delta' in results:
                print(f"  Effect size: {results['cliffs_delta']:.4f} ({results['effect_interpretation']})")
        else:
            print(f"⚠ Analysis limitation: {results['error']}")
            if 'message' in results:
                print(f"  Details: {results['message']}")
    
    print(f"\nFiles generated:")
    print(f"  - processed_analysis_data.csv: Cleaned dataset")
    print(f"  - robust_analysis_{args.prompt_name.replace(' ', '_')}.png: Visualization")
    print(f"  - analysis_summary.txt: Complete summary")
    if 'reg_results' in locals() and not reg_results.empty:
        print(f"  - regularization_effects.csv: Before/after comparison")
    
    print(f"\nOutput directory: {output_dir}")

if __name__ == "__main__":
    main()
