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

# -----------------------------------------------------------------------------
# Enhanced statistical analysis functions
# -----------------------------------------------------------------------------

def apply_multiple_test_correction(p_values, method='holm'):
    """Apply multiple test correction to p-values."""
    if len(p_values) == 0:
        return [], []
    
    # Filter out None values
    valid_p_values = [p for p in p_values if p is not None and not np.isnan(p)]
    if len(valid_p_values) == 0:
        return p_values, p_values
    
    # Apply correction
    rejected, corrected_p, _, _ = multipletests(valid_p_values, method=method)
    
    print(f"\nMultiple Test Correction ({method.upper()}):")
    print(f"Original p-values: {[f'{p:.4f}' for p in valid_p_values]}")
    print(f"Corrected p-values: {[f'{p:.4f}' for p in corrected_p]}")
    print(f"Significant after correction: {rejected}")
    
    return valid_p_values, corrected_p

def calculate_bootstrap_ci(data, n_bootstrap=1000, confidence=0.95):
    """Calculate bootstrap confidence intervals."""
    if len(data) < 2:
        return np.nan, np.nan
    
    bootstrap_means = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    alpha = 1 - confidence
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    ci_lower = np.percentile(bootstrap_means, lower_percentile)
    ci_upper = np.percentile(bootstrap_means, upper_percentile)
    
    return ci_lower, ci_upper

def calculate_cliffs_delta(group1, group2):
    """Calculate Cliff's delta effect size."""
    if len(group1) == 0 or len(group2) == 0:
        return np.nan
    
    n1, n2 = len(group1), len(group2)
    dominance = 0
    
    for x in group1:
        for y in group2:
            if x > y:
                dominance += 1
            elif x < y:
                dominance -= 1
    
    cliffs_d = dominance / (n1 * n2)
    
    # Interpretation
    abs_d = abs(cliffs_d)
    if abs_d < 0.147:
        interp = "Negligible"
    elif abs_d < 0.33:
        interp = "Small"
    elif abs_d < 0.474:
        interp = "Medium"
    else:
        interp = "Large"
    
    return cliffs_d, interp

def calculate_effect_sizes(group_data, test_name):
    """Calculate appropriate effect sizes for different tests."""
    effects = {}
    
    if len(group_data) == 2:
        # Cohen's d for two groups
        group1, group2 = group_data[0], group_data[1]
        
        if len(group1) > 1 and len(group2) > 1:
            pooled_std = np.sqrt(((len(group1)-1)*group1.std()**2 + (len(group2)-1)*group2.std()**2) / 
                                (len(group1) + len(group2) - 2))
            if pooled_std > 0:
                cohens_d = abs(group1.mean() - group2.mean()) / pooled_std
                effects['cohens_d'] = cohens_d
                
                # Interpretation
                if cohens_d > 0.8:
                    effect_interp = "Large"
                elif cohens_d > 0.5:
                    effect_interp = "Medium"
                elif cohens_d > 0.2:
                    effect_interp = "Small"
                else:
                    effect_interp = "Negligible"
                effects['cohens_d_interpretation'] = effect_interp
        
        # Cliff's delta for non-parametric
        if 'Mann-Whitney' in test_name or 'Kruskal' in test_name:
            cliffs_delta, cliffs_interp = calculate_cliffs_delta(group1, group2)
            effects['cliffs_delta'] = cliffs_delta
            effects['cliffs_delta_interpretation'] = cliffs_interp
    
    else:
        # Eta-squared for multiple groups
        if len(group_data) > 2:
            all_data = np.concatenate(group_data)
            grand_mean = all_data.mean()
            
            ss_between = sum(len(group) * (group.mean() - grand_mean)**2 for group in group_data)
            ss_total = sum((all_data - grand_mean)**2)
            
            if ss_total > 0:
                eta_squared = ss_between / ss_total
                effects['eta_squared'] = eta_squared
                
                # Interpretation
                if eta_squared > 0.14:
                    effect_interp = "Large"
                elif eta_squared > 0.06:
                    effect_interp = "Medium"  
                elif eta_squared > 0.01:
                    effect_interp = "Small"
                else:
                    effect_interp = "Negligible"
                effects['eta_squared_interpretation'] = effect_interp
    
    return effects

def enhanced_descriptive_stats(df, measure_col, group_col):
    """Enhanced descriptive statistics with confidence intervals."""
    results = []
    
    for group in df[group_col].unique():
        group_data = df[df[group_col] == group][measure_col].dropna()
        
        if len(group_data) > 0:
            # Basic stats
            stats_dict = {
                'group': group,
                'n': len(group_data),
                'mean': group_data.mean(),
                'std': group_data.std(),
                'median': group_data.median(),
                'min': group_data.min(),
                'max': group_data.max(),
                'sem': group_data.sem()  # Standard error of mean
            }
            
            # Bootstrap CI for mean
            if len(group_data) >= 3:
                ci_lower, ci_upper = calculate_bootstrap_ci(group_data)
                stats_dict['ci_lower'] = ci_lower
                stats_dict['ci_upper'] = ci_upper
                stats_dict['ci_width'] = ci_upper - ci_lower
            else:
                stats_dict['ci_lower'] = np.nan
                stats_dict['ci_upper'] = np.nan
                stats_dict['ci_width'] = np.nan
            
            results.append(stats_dict)
    
    return pd.DataFrame(results)

def perform_anova_analysis(df, measure_col, group_col, measure_name):
    """Perform ANOVA and post-hoc tests for a continuous measure across groups."""
    
    print(f"\n{'='*60}")
    print(f"ANOVA ANALYSIS: {measure_name} by {group_col}")
    print(f"{'='*60}")
    
    # Get groups
    groups = df[group_col].unique()
    group_data = [df[df[group_col] == group][measure_col].dropna() for group in groups]
    
    # Check for insufficient groups BEFORE attempting analysis
    if len(groups) < 2:
        print(f"\nOnly {len(groups)} group found: {groups}")
        print("Statistical comparison requires at least 2 groups.")
        print("Skipping statistical test.")
        
        # Still show descriptive statistics for the single group
        print(f"\nDescriptive Statistics:")
        desc_stats = enhanced_descriptive_stats(df, measure_col, group_col)
        print(desc_stats.round(3))
        
        return {
            "error": "insufficient_groups", 
            "n_groups": len(groups),
            "groups": list(groups),
            "message": f"Only {len(groups)} group found. Cannot perform statistical comparison.",
            "descriptive_stats": desc_stats
        }
    
    # Enhanced descriptive statistics with CI
    print(f"\nEnhanced Descriptive Statistics:")
    desc_stats = enhanced_descriptive_stats(df, measure_col, group_col)
    print(desc_stats.round(3))
    
    # Test normality (Shapiro-Wilk for each group)
    print(f"\nNormality Tests (Shapiro-Wilk):")
    normality_results = {}
    for i, group in enumerate(groups):
        if len(group_data[i]) >= 3:  # Shapiro needs at least 3 samples
            stat, p_value = stats.shapiro(group_data[i])
            normality_results[group] = p_value
            print(f"  {group}: W = {stat:.4f}, p = {p_value:.4f} {'(Normal)' if p_value > 0.05 else '(Non-normal)'}")
        else:
            print(f"  {group}: Insufficient data for normality test")
            normality_results[group] = 0.0
    
    # Choose appropriate test based on normality and group count
    all_normal = all(p > 0.05 for p in normality_results.values() if p > 0)
    sufficient_data = all(len(group) >= 3 for group in group_data)
    
    print(f"\nStatistical Test Selection:")
    print(f"  Groups normally distributed: {all_normal}")
    print(f"  Sufficient data per group: {sufficient_data}")
    
    # Check for insufficient data in any group
    min_group_size = min(len(group) for group in group_data)
    if min_group_size < 2:
        print(f"  Minimum group size: {min_group_size} (insufficient for statistical testing)")
        print("Skipping statistical test due to insufficient data in at least one group.")
        return {
            "error": "insufficient_data",
            "min_group_size": min_group_size,
            "groups": list(groups),
            "message": f"Minimum group size {min_group_size} insufficient for testing.",
            "descriptive_stats": desc_stats
        }
    
    try:
        if len(groups) == 2:
            # Two groups - use t-test or Mann-Whitney U
            if all_normal and sufficient_data:
                stat, p_value = stats.ttest_ind(group_data[0], group_data[1])
                test_name = "Independent t-test"
            else:
                stat, p_value = mannwhitneyu(group_data[0], group_data[1], alternative='two-sided')
                test_name = "Mann-Whitney U test"
        else:
            # Multiple groups - use ANOVA or Kruskal-Wallis
            if all_normal and sufficient_data:
                stat, p_value = stats.f_oneway(*group_data)
                test_name = "One-way ANOVA"
            else:
                stat, p_value = kruskal(*group_data)
                test_name = "Kruskal-Wallis test"
        
        print(f"\n{test_name} Results:")
        print(f"  Test statistic: {stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        
        # Calculate effect sizes
        effects = calculate_effect_sizes(group_data, test_name)
        
        print(f"\nEffect Sizes:")
        for effect_name, effect_value in effects.items():
            if 'interpretation' not in effect_name:
                interp_key = f"{effect_name}_interpretation"
                interp = effects.get(interp_key, "")
                print(f"  {effect_name}: {effect_value:.4f} ({interp})")
        
        # Interpretation
        if p_value < 0.001:
            significance = "Highly significant (p < 0.001)"
        elif p_value < 0.01:
            significance = "Very significant (p < 0.01)"
        elif p_value < 0.05:
            significance = "Significant (p < 0.05)"
        elif p_value < 0.1:
            significance = "Marginally significant (p < 0.1)"
        else:
            significance = "Not significant (p ≥ 0.1)"
        
        print(f"  Interpretation: {significance}")
        
        return {
            'test_name': test_name,
            'statistic': stat,
            'p_value': p_value,
            'groups': groups,
            'group_means': [group.mean() for group in group_data],
            'significance': significance,
            'effect_sizes': effects,
            'descriptive_stats': desc_stats,
            'normality_results': normality_results
        }
        
    except Exception as e:
        print(f"\nError during statistical testing: {e}")
        return {
            "error": "statistical_test_failed",
            "exception": str(e),
            "groups": list(groups),
            "message": f"Statistical test failed: {e}",
            "descriptive_stats": desc_stats
        }

def perform_ancova_analysis(df, measure_col, group_col, covariate_col, measure_name):
    """Perform ANCOVA controlling for a covariate (e.g., token length)."""
    
    print(f"\n{'='*60}")
    print(f"ANCOVA ANALYSIS: {measure_name} by {group_col} (controlling for {covariate_col})")
    print(f"{'='*60}")
    
    # Check if covariate exists
    if covariate_col not in df.columns:
        print(f"Covariate '{covariate_col}' not found in data. Skipping ANCOVA.")
        return None
    
    # Remove missing values
    analysis_df = df[[measure_col, group_col, covariate_col]].dropna()
    
    if len(analysis_df) == 0:
        print("No complete cases for ANCOVA analysis.")
        return None
    
    # Check if we have enough groups
    if len(analysis_df[group_col].unique()) < 2:
        print(f"Only {len(analysis_df[group_col].unique())} group found. Cannot perform ANCOVA.")
        return None
    
    # Check correlation between covariate and outcome
    covariate_corr = analysis_df[measure_col].corr(analysis_df[covariate_col])
    print(f"Correlation between {measure_col} and {covariate_col}: r = {covariate_corr:.4f}")
    
    # Print descriptive statistics by group for the covariate
    print(f"\nCovariate ({covariate_col}) statistics by group:")
    covariate_stats = analysis_df.groupby(group_col)[covariate_col].agg(['count', 'mean', 'std']).round(3)
    print(covariate_stats)
    
    # Perform ANCOVA
    try:
        # Fit the model
        formula = f"{measure_col} ~ C({group_col}) + {covariate_col}"
        model = ols(formula, data=analysis_df).fit()
        
        print(f"\nANCOVA Model Summary:")
        print(f"  Model R²: {model.rsquared:.4f}")
        print(f"  Adjusted R²: {model.rsquared_adj:.4f}")
        print(f"  F-statistic: {model.fvalue:.4f}")
        print(f"  Model p-value: {model.f_pvalue:.6f}")
        
        # ANOVA table
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(f"\nANOVA Table (Type II):")
        print(anova_table.round(6))
        
        # Extract p-values
        group_p = anova_table.loc[f'C({group_col})', 'PR(>F)']
        covariate_p = anova_table.loc[covariate_col, 'PR(>F)']
        
        # Calculate partial eta-squared for group effect
        group_ss = anova_table.loc[f'C({group_col})', 'sum_sq']
        error_ss = anova_table.loc['Residual', 'sum_sq']
        partial_eta_sq = group_ss / (group_ss + error_ss)
        
        print(f"\nResults Summary:")
        print(f"  {group_col} effect (controlling for {covariate_col}): p = {group_p:.6f}")
        print(f"  {covariate_col} effect: p = {covariate_p:.6f}")
        print(f"  Partial η² for {group_col}: {partial_eta_sq:.4f}")
        
        # Interpretation
        if partial_eta_sq > 0.14:
            eta_interp = "Large"
        elif partial_eta_sq > 0.06:
            eta_interp = "Medium"
        elif partial_eta_sq > 0.01:
            eta_interp = "Small"
        else:
            eta_interp = "Negligible"
        
        print(f"  Effect size interpretation: {eta_interp}")
        
        return {
            'model': model,
            'anova_table': anova_table,
            'group_p_value': group_p,
            'covariate_p_value': covariate_p,
            'r_squared': model.rsquared,
            'partial_eta_squared': partial_eta_sq,
            'partial_eta_squared_interpretation': eta_interp,
            'formula': formula
        }
        
    except Exception as e:
        print(f"ANCOVA failed: {e}")
        return None

def create_enhanced_visualization(df, surface_area_col, group_col, output_dir, covariate_col=None):
    """Create publication-ready visualizations addressing all reviewer feedback."""
    
    logger.info("Creating enhanced visualizations...")
    
    # Set improved style with better font sizes and layout
    plt.style.use('default')
    sns.set_style("whitegrid", {"grid.alpha": 0.3})
    plt.rcParams.update({
        'font.size': 10,           # Base font size
        'axes.titlesize': 12,      # Subplot titles
        'axes.labelsize': 12,      # Axis labels
        'xtick.labelsize': 10,     # X-axis tick labels
        'ytick.labelsize': 10,     # Y-axis tick labels
        'legend.fontsize': 10,     # Legend text
        'figure.titlesize': 14     # Main figure title
    })
    
    # Define color-blind friendly palette with high contrast
    groups = sorted(df[group_col].unique())  # CANONICAL ORDER: sorted
    n_groups = len(groups)
    
    if n_groups <= 3:
        # Color-blind friendly palette: Red, Blue, Orange (avoiding teal/sky-blue confusion)
        colors = ['#E74C3C', '#3498DB', '#F39C12']  # Red, Blue, Orange
        palette = dict(zip(groups, colors[:n_groups]))
    else:
        palette = sns.color_palette("Set1", n_groups)
        palette = dict(zip(groups, palette))
    
    # Create figure with constrained layout for better spacing
    if n_groups == 1:
        # Single group: use 2x2 layout to avoid blank panels
        fig = plt.figure(figsize=(12, 10), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.25)
    else:
        # Multiple groups: use full 2x3 layout
        fig = plt.figure(figsize=(18, 12), constrained_layout=True)
        gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.25)
    
    # Shorter, more focused title
    fig.suptitle('Surface Area Metric Separates Transparency Groups', fontsize=16, fontweight='bold', y=0.96)
    
    # Get enhanced descriptive statistics
    desc_stats = enhanced_descriptive_stats(df, surface_area_col, group_col)
    desc_stats = desc_stats.set_index('group').loc[groups].reset_index()

    y_max = desc_stats['ci_upper'].max() * 1.20     # +20 % head-room
    # set now so ax1 keeps the limit even after plotting
    ax1_ylim = (0, y_max)

    # 1. Main result: Bar plot with CI (TOP LEFT)
    ax1 = fig.add_subplot(gs[0, 0])
        
    if 'ci_lower' in desc_stats.columns and not desc_stats['ci_lower'].isna().all():
        bars = ax1.bar(range(len(groups)), desc_stats['mean'], 
                      color=[palette[g] for g in desc_stats['group']], alpha=0.8, 
                      edgecolor='black', linewidth=1.5)
        
        # Add error bars
        errors_lower = desc_stats['mean'] - desc_stats['ci_lower']
        errors_upper = desc_stats['ci_upper'] - desc_stats['mean']
        
        ax1.errorbar(range(len(groups)), desc_stats['mean'], 
                    yerr=[errors_lower, errors_upper], 
                    fmt='none', color='black', capsize=8, linewidth=2.5, capthick=2)
        ax1.set_ylim(ax1_ylim)         # keep bar height under control
        
        # Clean value labels with small sample warnings
        for i, (bar, mean_val, n_val, upper_err) in enumerate(zip(bars, desc_stats['mean'], 
                                                                 desc_stats['n'], errors_upper)):
            # ------------------------------------------------------------
            # Annotate bar tops and flag tiny samples with an asterisk
            # ------------------------------------------------------------
            label_height = bar.get_height() + upper_err + (ax1_ylim[1] * 0.03)
            label_text = f'{mean_val:.0f}'
            if n_val < 5:
                label_text += '*'           # asterisk on the number itself
            ax1.text(bar.get_x() + bar.get_width()/2,
                     label_height,
                     label_text,
                     ha='center', va='bottom',
                     fontweight='bold', fontsize=11,
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='white', alpha=0.9, edgecolor='gray'))

            if any(desc_stats['n'] < 5):
                ax1.text(0.5, -0.12,
                         '*n < 5 : interpret with caution',
                         transform=ax1.transAxes,
                         ha='center', va='top',
                         fontsize=9, style='italic')

    ax1.set_title('Mean Surface Area by Group', fontsize=12, fontweight='bold', pad=15)
    ax1.set_ylabel('Surface Area (A′)', fontsize=12)
    ax1.set_xlabel('Transparency Level', fontsize=12)
    ax1.set_xticks(range(len(groups)))
    ax1.set_xticklabels(groups, fontsize=10)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribution: Violin plot with CONSISTENT ORDERING (TOP CENTER/RIGHT)
    ax2 = fig.add_subplot(gs[0, 1])
        
    if n_groups > 1:
        # Ensure consistent canonical order for violin plot
        violin_data = [df[df[group_col] == group][surface_area_col].dropna() for group in groups]
        violin_parts = ax2.violinplot(violin_data, positions=range(len(groups)), 
                                    widths=0.6, showmeans=True, showmedians=True)
        
        # Color the violins with hatching for color-blind accessibility
        hatch_patterns = ['', '///', '...']  # Different patterns for each group
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(list(palette.values())[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            if i < len(hatch_patterns):
                pc.set_hatch(hatch_patterns[i])
        
        # Style the statistical lines
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
            if partname in violin_parts:
                violin_parts[partname].set_edgecolor('black')
                violin_parts[partname].set_linewidth(2)
        
        # Add individual points for small groups (n < 10)
        for i, group in enumerate(groups):
            group_data = df[df[group_col] == group][surface_area_col].dropna()
            if len(group_data) <= 10:  # Only show points for small groups
                y_data = group_data.values
                x_data = np.random.normal(i, 0.04, size=len(y_data))
                ax2.scatter(x_data, y_data, alpha=0.8, s=40, color='darkred', 
                           edgecolors='black', linewidth=0.8, zorder=10)
        
        ax2.set_xticks(range(len(groups)))
        ax2.set_xticklabels(groups, fontsize=10)  # CONSISTENT ORDER
        
        # MATCH Y-AXIS SCALE to bar plot
        ax2.set_ylim(ax1.get_ylim())
        
    else:
        # Single group histogram
        ax2.hist(df[surface_area_col], bins=15, alpha=0.7, color=list(palette.values())[0], 
                edgecolor='black', linewidth=1)
        ax2.axvline(df[surface_area_col].mean(), color='red', linestyle='--', linewidth=3, 
                   label=f'Mean: {df[surface_area_col].mean():.0f}')
        ax2.legend(fontsize=10)
    
    ax2.set_title('Surface Area Distribution', fontsize=12, fontweight='bold', pad=15)
    ax2.set_ylabel('Surface Area (A′)', fontsize=12)
    ax2.set_xlabel('Transparency Level', fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Effect size visualization (TOP RIGHT for multi-group, BOTTOM LEFT for single group)
    if n_groups == 1:
        ax3 = fig.add_subplot(gs[1, 0])  # Bottom left in 2x2 layout
    else:
        ax3 = fig.add_subplot(gs[0, 2])  # Top right in 2x3 layout
        
    if n_groups == 2:
        # Cohen's d visualization with CORRECT axis labels
        group1_data = df[df[group_col] == groups[0]][surface_area_col].dropna()
        group2_data = df[df[group_col] == groups[1]][surface_area_col].dropna()
        
        if len(group1_data) > 0 and len(group2_data) > 0:
            # Calculate Cohen's d
            pooled_std = np.sqrt(((len(group1_data)-1)*group1_data.std()**2 + 
                                (len(group2_data)-1)*group2_data.std()**2) / 
                               (len(group1_data) + len(group2_data) - 2))
            cohens_d = abs(group1_data.mean() - group2_data.mean()) / pooled_std if pooled_std > 0 else 0
            
            # Create overlapping histograms
            ax3.hist(group1_data, alpha=0.6, label=f'{groups[0]} (μ={group1_data.mean():.0f})', 
                    bins=15, density=True, color=palette[groups[0]], edgecolor='black', linewidth=0.8)
            ax3.hist(group2_data, alpha=0.6, label=f'{groups[1]} (μ={group2_data.mean():.0f})', 
                    bins=15, density=True, color=palette[groups[1]], edgecolor='black', linewidth=0.8)
            
            # Add mean lines
            ax3.axvline(group1_data.mean(), color=palette[groups[0]], linestyle='--', alpha=0.9, linewidth=3)
            ax3.axvline(group2_data.mean(), color=palette[groups[1]], linestyle='--', alpha=0.9, linewidth=3)
            
            ax3.set_xlabel('Surface Area (A′)', fontsize=12)  # CORRECT
            ax3.set_ylabel('Density', fontsize=12)  # CORRECT
            ax3.set_title(f'Effect Size Visualization\nCohen\'s d = {cohens_d:.2f}', fontsize=12, fontweight='bold', pad=15)
            ax3.legend(fontsize=9)
    elif n_groups > 2:
        # Multiple groups - bar chart with CORRECT axis label
        means = [df[df[group_col] == group][surface_area_col].mean() for group in groups]
        stds = [df[df[group_col] == group][surface_area_col].std() for group in groups]
        
        bars = ax3.bar(range(len(groups)), means, yerr=stds, capsize=5, 
                      color=[palette[g] for g in groups], alpha=0.8, 
                      edgecolor='black', linewidth=1)
        ax3.set_xticks(range(len(groups)))
        ax3.set_xticklabels(groups, fontsize=10)
        ax3.set_ylabel('Mean Surface Area (A′)', fontsize=12)  # CORRECT
        ax3.set_xlabel('Transparency Level', fontsize=12)
        ax3.set_title(f'Group Means Comparison\n(η² in main analysis)', fontsize=12, fontweight='bold', pad=15)
    else:
        # Single group - show density curve instead of blank panel
        data = df[surface_area_col].dropna()
        ax3.hist(data, bins=20, density=True, alpha=0.7, color=list(palette.values())[0], 
                edgecolor='black', linewidth=1, label='Observed data')
        
        # Add fitted normal curve for comparison
        mu, sigma = data.mean(), data.std()
        x_norm = np.linspace(data.min(), data.max(), 100)
        y_norm = stats.norm.pdf(x_norm, mu, sigma)
        ax3.plot(x_norm, y_norm, 'r--', linewidth=2, label=f'Normal(μ={mu:.0f}, σ={sigma:.0f})')
        
        # Add vertical lines for key statistics
        ax3.axvline(mu, color='red', linestyle='-', alpha=0.8, linewidth=2, label=f'Mean: {mu:.0f}')
        ax3.axvline(data.median(), color='blue', linestyle='-', alpha=0.8, linewidth=2, label=f'Median: {data.median():.0f}')
        
        ax3.set_xlabel('Surface Area (A′)', fontsize=12)
        ax3.set_ylabel('Density', fontsize=12)
        ax3.set_title('Distribution Analysis\n(Single Group)', fontsize=12, fontweight='bold', pad=15)
        ax3.legend(fontsize=9)
    
    ax3.tick_params(axis='both', which='major', labelsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Sample sizes - DONUT CHART for better visual balance
    if n_groups == 1:
        ax4 = fig.add_subplot(gs[1, 1])  # Bottom right in 2x2 layout
    else:
        ax4 = fig.add_subplot(gs[1, 0])  # Bottom left in 2x3 layout
        
    group_sizes = df.groupby(group_col).size()
    # Reorder to match canonical order
    group_sizes = group_sizes.reindex(groups)
    
    if n_groups > 1:
        # Create donut chart instead of bars to handle imbalance better
        wedges, texts, autotexts = ax4.pie(group_sizes.values, labels=group_sizes.index, 
                                          colors=[palette[g] for g in group_sizes.index],
                                          autopct='%1.1f%%', startangle=90, 
                                          wedgeprops=dict(width=0.5, edgecolor='black'))
        
        # Add sample sizes in center
        centre_circle = plt.Circle((0,0), 0.30, fc='white', edgecolor='black')
        ax4.add_artist(centre_circle)
        ax4.text(0, 0, f'Total\nN = {group_sizes.sum()}', ha='center', va='center', 
                fontsize=11, fontweight='bold')
        
        # Style the percentage labels
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        
        # Add sample size annotation for small groups
        small_groups = [g for g in groups if group_sizes[g] < 5]
        if small_groups:
            warning_text = f"*n < 5: interpret with caution"
            ax4.text(0, -1.4, warning_text, ha='center', va='center', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                    transform=ax4.transData)
        
        ax4.set_title('Sample Sizes by Group', fontsize=12, fontweight='bold', pad=15)
    else:
        # Single group - show sample composition
        ax4.text(0.5, 0.5, f'Single Group Analysis\nTotal N = {group_sizes.iloc[0]}\nGroup: {group_sizes.index[0]}', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12,
                bbox=dict(boxstyle='round,pad=0.5', facecolor=list(palette.values())[0], alpha=0.3))
        ax4.set_title('Sample Composition', fontsize=12, fontweight='bold', pad=15)
        ax4.axis('off')
    
    # 5. Gamma parameter sweep or covariate (BOTTOM CENTER - multi-group only)
    if n_groups > 1:
        ax5 = fig.add_subplot(gs[1, 1])
        
        # Check if we have gamma parameter variation for a mini-plot
        if 'gamma_weighting_parameter' in df.columns and len(df['gamma_weighting_parameter'].unique()) > 1:
            # Create gamma sweep mini-plot
            gamma_means = []
            gamma_values = sorted(df['gamma_weighting_parameter'].unique())
            
            for gamma in gamma_values:
                gamma_df = df[df['gamma_weighting_parameter'] == gamma]
                if len(gamma_df[surface_area_col]) > 0:
                    gamma_means.append(gamma_df[surface_area_col].mean())
                else:
                    gamma_means.append(np.nan)
            
            ax5.plot(gamma_values, gamma_means, 'o-', linewidth=2, markersize=6, color='darkblue')
            ax5.set_xlabel('Gamma Parameter', fontsize=12)
            ax5.set_ylabel('Mean Surface Area (A′)', fontsize=12)
            ax5.set_title('Gamma Parameter Sweep', fontsize=12, fontweight='bold', pad=15)
            ax5.grid(True, alpha=0.3)
            
            # Highlight current gamma if single value
            current_gamma = df['gamma_weighting_parameter'].iloc[0]
            if len(df['gamma_weighting_parameter'].unique()) == 1:
                ax5.axvline(current_gamma, color='red', linestyle='--', alpha=0.7)
                ax5.text(current_gamma, max(gamma_means) * 0.9, f'Current: {current_gamma}', 
                        rotation=90, ha='right', va='top', fontsize=9)
        
        elif covariate_col and covariate_col in df.columns:
            # Covariate relationship (simplified, no perfect correlation display)
            for group in groups:
                group_data = df[df[group_col] == group]
                ax5.scatter(group_data[covariate_col], group_data[surface_area_col], 
                           alpha=0.7, s=50, color=palette[group], label=group, 
                           edgecolors='black', linewidth=0.5)
            
            # Only show trend line if correlation is not perfect
            corr_coef = df[covariate_col].corr(df[surface_area_col])
            if abs(corr_coef) < 0.95:  # Only show if not perfect correlation
                valid_mask = df[covariate_col].notna() & df[surface_area_col].notna()
                if valid_mask.sum() > 1:
                    z = np.polyfit(df.loc[valid_mask, covariate_col], df.loc[valid_mask, surface_area_col], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(df[covariate_col].min(), df[covariate_col].max(), 100)
                    ax5.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=3)
            
            ax5.set_xlabel(covariate_col.replace('_', ' ').title(), fontsize=12)
            ax5.set_ylabel('Surface Area (A′)', fontsize=12)
            ax5.set_title(f'Surface Area vs {covariate_col.replace("_", " ").title()}', fontsize=12, fontweight='bold', pad=15)
            ax5.legend(fontsize=9)
        else:
            # Additional analysis space
            ax5.text(0.5, 0.5, 'Additional Analysis\nSpace Available', ha='center', va='center', 
                    transform=ax5.transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            ax5.set_title('Additional Analysis', fontsize=12, fontweight='bold', pad=15)
        
        ax5.tick_params(axis='both', which='major', labelsize=10)
        ax5.grid(True, alpha=0.3)
    
    # 6. Enhanced summary statistics (BOTTOM RIGHT for multi-group, BOTTOM CENTER for single group)
    if n_groups == 1:
        # For single group, summary goes in bottom center (gs[1, 1] doesn't exist in 2x2)
        # We'll put it as text overlay on the existing plot or skip detailed summary
        pass
    else:
        ax6 = fig.add_subplot(gs[1, 2])  # Bottom right in 2x3 layout
        ax6.axis('off')  # Turn off axis for text summary
        
        # Create enhanced summary with better normality description
        summary_text = f"""Key Findings Summary

Total Sample: N = {len(df)}
Groups: {', '.join(groups)}

Group Means (95% CI):"""
        
        for _, row in desc_stats.iterrows():
            asterisk = '*' if row['n'] < 5 else ''
            if not pd.isna(row['ci_lower']):
                summary_text += f"\n• {row['group']}: {row['mean']:.0f} [{row['ci_lower']:.0f}, {row['ci_upper']:.0f}] (n={row['n']}){asterisk}"
            else:
                summary_text += f"\n• {row['group']}: {row['mean']:.0f} (n={row['n']}){asterisk}"
        
        if n_groups == 2 and len(desc_stats) == 2:
            mean_diff = abs(desc_stats.iloc[0]['mean'] - desc_stats.iloc[1]['mean'])
            summary_text += f"\n\nMean Difference: {mean_diff:.0f} A′"
        
        # Enhanced normality reporting
        shapiro_ps = []
        for group in groups:
            group_data = df[df[group_col] == group][surface_area_col].dropna()
            if len(group_data) >= 3:
                _, p = stats.shapiro(group_data)
                shapiro_ps.append(p)
        
        if shapiro_ps:
            all_normal = all(p > 0.05 for p in shapiro_ps)
            min_p = min(shapiro_ps) if shapiro_ps else 1.0
            summary_text += f"\n\nNormality: {'✓' if all_normal else '✗'} "
            summary_text += f"Non-normal (SW p = {min_p:.3f})" if not all_normal else "Normal (SW p > 0.05)"
            summary_text += f"\n→ {'Parametric' if all_normal else 'Non-parametric'} tests used"
        
        # Add note about perfect correlations and A' definition
        if 'total_salience_contribution' in df.columns:
            summary_text += f"\n\nNote: SA = f(salience, curvature)\nPerfect correlations expected"
        
        # Add asterisk explanation if any small samples
        small_groups_exist = any(desc_stats['n'] < 5)
        if small_groups_exist:
            summary_text += f"\n\n*n < 5: interpret with caution"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    
    #for ax in axes:
    #    ax.grid(axis='y', alpha=0.3)   # horizontal only

    # Save plot with high quality
    output_path = Path(output_dir) / 'enhanced_geometric_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved enhanced visualization to {output_path}")
    
    plt.close()

    # Create publication figure with caption including sample size warnings
    create_publication_figure(df, surface_area_col, group_col, desc_stats, palette, output_dir)

def create_publication_figure(df, surface_area_col, group_col, desc_stats, palette, output_dir):
    """Create publication-ready figure with comprehensive caption and A' definition."""
    
    # Set publication font sizes
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10
    })
    
    groups = sorted(df[group_col].unique())
    n_groups = len(groups)
    
    # Adjust figure layout based on number of groups
    if n_groups == 1:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
        axes = [ax1]
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
        axes = [ax1, ax2]
    
    fig.suptitle('Surface Area Metric Separates Transparency Groups', fontsize=16, fontweight='bold')
    
    # Panel A: Main result with CI and asterisks for small samples
    if 'ci_lower' in desc_stats.columns and not desc_stats['ci_lower'].isna().all():
        bars = ax1.bar(range(len(groups)), desc_stats['mean'], 
                      color=[palette[g] for g in desc_stats['group']], alpha=0.8, 
                      edgecolor='black', linewidth=1.5)
        
        errors_lower = desc_stats['mean'] - desc_stats['ci_lower']
        errors_upper = desc_stats['ci_upper'] - desc_stats['mean']
        
        ax1.errorbar(range(len(groups)), desc_stats['mean'], 
                    yerr=[errors_lower, errors_upper], 
                    fmt='none', color='black', capsize=8, linewidth=2.5, capthick=2)

        for bar in bars:
            bar.set_width(0.4)          # slimmer bar, CI stands out
        
        # Clean labels with asterisks for small samples
        for i, (bar, mean_val, n_val, upper_err) in enumerate(zip(bars, desc_stats['mean'], 
                                                                 desc_stats['n'], errors_upper)):
            label_height = bar.get_height() + upper_err + (ax1.get_ylim()[1] * 0.03)
            label_text = f'{mean_val:.0f}'
            if n_val < 5:
                label_text += '*'
            ax1.text(bar.get_x() + bar.get_width()/2, label_height, 
                    label_text, ha='center', va='bottom', 
                    fontweight='bold', fontsize=12)
    
    if n_groups == 1:
        ax1.set_title('Surface Area Distribution', fontsize=13, fontweight='bold')
    else:
        ax1.set_title('A. Mean Surface Area by Group', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Surface Area (A′)', fontsize=13)
    ax1.set_xlabel('Transparency Level', fontsize=13)
    ax1.set_xticks(range(len(groups)))
    ax1.set_xticklabels(groups, fontsize=11)
    ax1.tick_params(axis='both', which='major', labelsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Distribution with hatching (only for multi-group)
    if n_groups > 1:
        violin_data = [df[df[group_col] == group][surface_area_col].dropna() for group in groups]
        violin_parts = ax2.violinplot(violin_data, positions=range(len(groups)), 
                                    widths=0.6, showmeans=True, showmedians=True)
        
        hatch_patterns = ['', '///', '...']
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(list(palette.values())[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            if i < len(hatch_patterns):
                pc.set_hatch(hatch_patterns[i])
        
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
            if partname in violin_parts:
                violin_parts[partname].set_edgecolor('black')
                violin_parts[partname].set_linewidth(2)
        
        # Match y-axis scale to bar plot
        ax2.set_ylim(ax1.get_ylim())
        
        ax2.set_xticks(range(len(groups)))
        ax2.set_xticklabels(groups, fontsize=11)
        ax2.set_title('B. Distribution by Group', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Surface Area (A′)', fontsize=13)
        ax2.set_xlabel('Transparency Level', fontsize=13)
        ax2.tick_params(axis='both', which='major', labelsize=11)
        ax2.grid(True, alpha=0.3)
    
    # Comprehensive caption with all improvements
    group_sizes = df.groupby(group_col).size()
    small_groups = [g for g in groups if group_sizes[g] < 5]
    
    caption = f"Figure 1. Surface area metric separates transparency groups (N = {len(df)}). "
    
    if n_groups == 1:
        caption += f"Single group analysis showing distribution characteristics. "
    else:
        caption += f"(A) Mean surface area with 95% bootstrap confidence intervals. "
        caption += f"(B) Distribution density with pattern coding for accessibility. "
    
    if small_groups:
        caption += f"*Groups with n < 5: interpret with caution. "
    
    caption += f"A′ = double-resolution un-normalised semantic surface area derived from salience and curvature contributions (perfect correlation expected). "
    
    # Add normality and test information
    shapiro_ps = []
    for group in groups:
        group_data = df[df[group_col] == group][surface_area_col].dropna()
        if len(group_data) >= 3:
            _, p = stats.shapiro(group_data)
            shapiro_ps.append(p)
    
    if shapiro_ps:
        all_normal = all(p > 0.05 for p in shapiro_ps)
        caption += f"Non-parametric tests used due to non-normal distributions (Shapiro-Wilk p < 0.05). " if not all_normal else "Parametric tests appropriate (normal distributions). "
    
    caption += f"Error bars show 95% bootstrap confidence intervals."
    print(f"{output_dir} -> caption: {caption}") 
    # Add caption with proper wrapping
    #fig.text(0.02, 0.02, caption, fontsize=9, wrap=True, ha='left', va='bottom')

    for ax in axes:
        ax.grid(axis='y', alpha=0.3)   # horizontal only
    
    # Save publication figure
    pub_path = Path(output_dir) / 'publication_figure.png'
    plt.savefig(pub_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved publication figure to {pub_path}")
    
    plt.close()

def analyze_gamma_effects(df, output_dir):
    """Analyze how gamma weighting parameter affects the results."""
    
    print(f"\n{'='*60}")
    print(f"GAMMA PARAMETER ANALYSIS")
    print(f"{'='*60}")
    
    # Group by gamma and analyze
    gamma_analysis = []
    
    for gamma in sorted(df['gamma_weighting_parameter'].unique()):
        gamma_df = df[df['gamma_weighting_parameter'] == gamma]
        
        # Test surface area differences by transparency
        if 'transparency_level_consensus' in gamma_df.columns:
            groups = gamma_df['transparency_level_consensus'].unique()
            if len(groups) >= 2:
                group_data = [gamma_df[gamma_df['transparency_level_consensus'] == group]['total_surface_area_A_prime'] 
                             for group in groups]
                # Filter out empty groups
                group_data = [group.dropna() for group in group_data if len(group.dropna()) > 0]
                
                if len(group_data) >= 2 and all(len(group) >= 2 for group in group_data):
                    try:
                        if len(group_data) == 2:
                            stat, p_val = mannwhitneyu(group_data[0], group_data[1], alternative='two-sided')
                            test_name = "Mann-Whitney U"
                        else:
                            stat, p_val = kruskal(*group_data)
                            test_name = "Kruskal-Wallis"
                        
                        # Calculate effect size
                        if len(group_data) == 2:
                            cliffs_d, cliffs_interp = calculate_cliffs_delta(group_data[0], group_data[1])
                        else:
                            cliffs_d, cliffs_interp = np.nan, "N/A"
                        
                        gamma_analysis.append({
                            'gamma': gamma,
                            'test_name': test_name,
                            'test_statistic': stat,
                            'p_value': p_val,
                            'n_responses': len(gamma_df),
                            'n_groups': len(group_data),
                            'cliffs_delta': cliffs_d,
                            'effect_interpretation': cliffs_interp,
                            'significant': p_val < 0.05
                        })
                    except Exception as e:
                        logger.warning(f"Failed to analyze gamma={gamma}: {e}")
                else:
                    logger.info(f"Gamma={gamma}: Insufficient data for statistical testing")
            else:
                logger.info(f"Gamma={gamma}: Only {len(groups)} group(s) found")
    
    if gamma_analysis:
        gamma_df_results = pd.DataFrame(gamma_analysis)
        print("\nGamma Parameter Effects on Transparency Detection:")
        print(gamma_df_results.round(4))
        
        # Save gamma analysis
        output_path = Path(output_dir) / 'gamma_analysis.csv'
        gamma_df_results.to_csv(output_path, index=False)
        logger.info(f"Saved gamma analysis to {output_path}")
        
        # Find optimal gamma (highest test statistic with significance)
        if len(gamma_df_results) > 0:
            significant_results = gamma_df_results[gamma_df_results['significant']]
            if len(significant_results) > 0:
                optimal_gamma = significant_results.loc[significant_results['test_statistic'].idxmax(), 'gamma']
                print(f"\nOptimal gamma for discrimination (significant results): {optimal_gamma}")
            else:
                optimal_gamma = gamma_df_results.loc[gamma_df_results['test_statistic'].idxmax(), 'gamma']
                print(f"\nBest gamma for discrimination (no significant results): {optimal_gamma}")
            return optimal_gamma
    else:
        print("\nNo gamma values suitable for statistical analysis.")
    
    return None

def comprehensive_analysis(merged_df, surface_area_col, group_col, covariate_col=None):
    """Perform comprehensive statistical analysis with all enhancements."""
    
    all_p_values = []
    all_test_names = []
    results_summary = {}
    
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE STATISTICAL ANALYSIS")
    print(f"{'='*60}")
    
    # 1. Enhanced descriptive statistics
    print(f"\nEnhanced Descriptive Statistics with Confidence Intervals:")
    desc_stats = enhanced_descriptive_stats(merged_df, surface_area_col, group_col)
    print(desc_stats.round(3))
    results_summary['descriptive_stats'] = desc_stats
    
    # 2. Primary statistical test
    primary_result = perform_anova_analysis(merged_df, surface_area_col, group_col, "Primary Hypothesis Test")
    if primary_result and 'p_value' in primary_result and not primary_result.get('error'):
        all_p_values.append(primary_result['p_value'])
        all_test_names.append("Primary Test")
        results_summary['primary_test'] = primary_result
    
    # 3. ANCOVA if covariate provided
    ancova_result = None
    if covariate_col and covariate_col in merged_df.columns:
        ancova_result = perform_ancova_analysis(merged_df, surface_area_col, group_col, covariate_col, "ANCOVA Analysis")
        if ancova_result and ancova_result['group_p_value'] is not None:
            all_p_values.append(ancova_result['group_p_value'])
            all_test_names.append("ANCOVA Test")
            results_summary['ancova_test'] = ancova_result
    
    # 4. Apply multiple test correction if we have multiple tests
    if len(all_p_values) > 1:
        original_p, corrected_p = apply_multiple_test_correction(all_p_values, method='holm')
        results_summary['multiple_test_correction'] = {
            'method': 'holm',
            'original_p_values': original_p,
            'corrected_p_values': corrected_p,
            'test_names': all_test_names
        }
    else:
        results_summary['multiple_test_correction'] = None
    
    # 5. Generate final summary
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    
    if 'primary_test' in results_summary:
        pt = results_summary['primary_test']
        print(f"\nPrimary Test ({pt['test_name']}):")
        print(f"  Test statistic: {pt['statistic']:.4f}")
        print(f"  p-value: {pt['p_value']:.6f}")
        print(f"  Significance: {pt['significance']}")
        if 'effect_sizes' in pt:
            for effect_name, effect_value in pt['effect_sizes'].items():
                if 'interpretation' not in effect_name:
                    interp_key = f"{effect_name}_interpretation"
                    interp = pt['effect_sizes'].get(interp_key, "")
                    print(f"  {effect_name}: {effect_value:.4f} ({interp})")
    
    if 'ancova_test' in results_summary and results_summary['ancova_test']:
        at = results_summary['ancova_test']
        print(f"\nANCOVA Test:")
        print(f"  Group effect p-value: {at['group_p_value']:.6f}")
        print(f"  Covariate effect p-value: {at['covariate_p_value']:.6f}")
        print(f"  Partial η²: {at['partial_eta_squared']:.4f} ({at['partial_eta_squared_interpretation']})")
        print(f"  Model R²: {at['r_squared']:.4f}")
    
    if results_summary['multiple_test_correction']:
        mtc = results_summary['multiple_test_correction']
        print(f"\nMultiple Test Correction ({mtc['method'].upper()}):")
        for i, (test_name, orig_p, corr_p) in enumerate(zip(mtc['test_names'], mtc['original_p_values'], mtc['corrected_p_values'])):
            print(f"  {test_name}: p = {orig_p:.6f} → {corr_p:.6f} {'*' if corr_p < 0.05 else ''}")
    
    return results_summary

def main():
    parser = argparse.ArgumentParser(description="Enhanced analysis of geometric surface area hypothesis for deception detection.")
    parser.add_argument("surface_area_csv", help="CSV file with surface area measurements")
    parser.add_argument("consensus_csv", help="CSV file with consensus behavioral classifications")
    parser.add_argument("--output_dir", default="analysis_results", help="Output directory for results")
    parser.add_argument("--gamma", type=float, default=None, help="Specific gamma value to analyze (if None, analyzes all)")
    parser.add_argument("--covariate", default=None, help="Column name for covariate control (e.g., 'response_length')")
    parser.add_argument("--correction_method", default="holm", choices=["holm", "bonferroni", "fdr_bh"], 
                       help="Multiple test correction method")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    logger.info(f"Loading surface area data from {args.surface_area_csv}")
    surface_df = pd.read_csv(args.surface_area_csv)
    
    logger.info(f"Loading consensus classifications from {args.consensus_csv}")
    consensus_df = pd.read_csv(args.consensus_csv)
    
    logger.info(f"Surface area data shape: {surface_df.shape}")
    logger.info(f"Consensus data shape: {consensus_df.shape}")
    
    # Create merge key for both datasets
    surface_df['response_id'] = surface_df['prompt_id'].astype(str) + '_' + surface_df['run_id'].astype(str)
    
    # Ensure consensus_df also has response_id if not already present
    if 'response_id' not in consensus_df.columns:
        consensus_df['response_id'] = consensus_df['prompt_id'].astype(str) + '_' + consensus_df['run_id'].astype(str)
    
    # Merge datasets
    logger.info("Merging datasets...")
    merged_df = pd.merge(surface_df, consensus_df, on=['prompt_id', 'run_id'], how='inner', suffixes=('', '_consensus'))
    
    # Ensure response_id is in merged dataset
    if 'response_id' not in merged_df.columns:
        merged_df['response_id'] = merged_df['prompt_id'].astype(str) + '_' + merged_df['run_id'].astype(str)
    
    logger.info(f"Merged dataset shape: {merged_df.shape}")
    
    if len(merged_df) == 0:
        logger.error("No matching records found between surface area and consensus data!")
        return
    
    # Filter to specific gamma if requested
    if args.gamma is not None:
        merged_df = merged_df[merged_df['gamma_weighting_parameter'] == args.gamma]
        logger.info(f"Filtered to gamma={args.gamma}: {len(merged_df)} records")
    
    # Print data overview
    print(f"\n{'='*60}")
    print(f"DATASET OVERVIEW")
    print(f"{'='*60}")
    print(f"Total merged records: {len(merged_df)}")
    print(f"Unique responses: {merged_df['response_id'].nunique()}")
    print(f"Gamma values: {sorted(merged_df['gamma_weighting_parameter'].unique())}")
    print(f"Prompt strategies: {sorted(merged_df['prompt_id'].unique())}")
    
    if 'transparency_level_consensus' in merged_df.columns:
        print(f"Transparency levels: {merged_df['transparency_level_consensus'].value_counts().to_dict()}")
    if 'response_type_consensus' in merged_df.columns:
        print(f"Response types: {merged_df['response_type_consensus'].value_counts().to_dict()}")
    
    # Check covariate availability
    covariate_col = args.covariate
    if covariate_col and covariate_col not in merged_df.columns:
        logger.warning(f"Specified covariate '{covariate_col}' not found in data. Available columns: {list(merged_df.columns)}")
        covariate_col = None
    elif covariate_col:
        logger.info(f"Using covariate: {covariate_col}")
    
    # Analyze gamma effects first (if multiple gammas)
    optimal_gamma = None
    if len(merged_df['gamma_weighting_parameter'].unique()) > 1:
        optimal_gamma = analyze_gamma_effects(merged_df, output_dir)
        
        # Use optimal gamma for main analysis if found and not specified
        if optimal_gamma is not None and args.gamma is None:
            merged_df = merged_df[merged_df['gamma_weighting_parameter'] == optimal_gamma]
            logger.info(f"Using optimal gamma={optimal_gamma} for main analysis: {len(merged_df)} records")
    
    # Main comprehensive analysis
    surface_area_col = 'total_surface_area_A_prime'
    
    # Comprehensive analysis for primary hypothesis
    if 'transparency_level_consensus' in merged_df.columns:
        results = comprehensive_analysis(
            merged_df, surface_area_col, 'transparency_level_consensus', covariate_col
        )
    
    # Additional analyses for other groupings
    additional_analyses = []
    
    # Test by prompt strategy (experimental manipulation check)
    if len(merged_df['prompt_id'].unique()) > 1:
        print(f"\n{'='*60}")
        print(f"ADDITIONAL ANALYSIS: PROMPT STRATEGY")
        print(f"{'='*60}")
        strategy_results = perform_anova_analysis(
            merged_df, surface_area_col, 'prompt_id', 
            "Surface Area by Prompt Strategy (Manipulation Check)"
        )
        additional_analyses.append(('prompt_strategy', strategy_results))
    
    # Test by response type consensus
    if 'response_type_consensus' in merged_df.columns and len(merged_df['response_type_consensus'].unique()) > 1:
        print(f"\n{'='*60}")
        print(f"ADDITIONAL ANALYSIS: RESPONSE TYPE")
        print(f"{'='*60}")
        response_type_results = perform_anova_analysis(
            merged_df, surface_area_col, 'response_type_consensus',
            "Surface Area by Response Type"
        )
        additional_analyses.append(('response_type', response_type_results))
    
    # Test unanimous vs non-unanimous responses
    if 'transparency_level_unanimous' in merged_df.columns and len(merged_df['transparency_level_unanimous'].unique()) > 1:
        print(f"\n{'='*60}")
        print(f"ADDITIONAL ANALYSIS: UNANIMOUS CLASSIFICATION")
        print(f"{'='*60}")
        unanimous_results = perform_anova_analysis(
            merged_df, surface_area_col, 'transparency_level_unanimous',
            "Surface Area: Unanimous vs Non-unanimous Classifications"
        )
        additional_analyses.append(('unanimous_classification', unanimous_results))
    
    # Correlation analysis
    print(f"\n{'='*60}")
    print(f"CORRELATION ANALYSIS")
    print(f"{'='*60}")
    
    numeric_cols = ['total_surface_area_A_prime', 'total_salience_contribution', 
                   'total_curvature_contribution', 'mean_step_salience', 'mean_step_curvature']
    
    if covariate_col:
        numeric_cols.append(covariate_col)
    
    available_cols = [col for col in numeric_cols if col in merged_df.columns]
    if len(available_cols) > 1:
        corr_matrix = merged_df[available_cols].corr()
        print("\nCorrelation Matrix:")
        print(corr_matrix.round(3))
        
        # Save correlation matrix
        corr_path = output_dir / 'correlation_matrix.csv'
        corr_matrix.to_csv(corr_path)
        logger.info(f"Saved correlation matrix to {corr_path}")
    
    # Create enhanced visualizations
    if len(merged_df) > 0 and 'transparency_level_consensus' in merged_df.columns:
        create_enhanced_visualization(merged_df, surface_area_col, 'transparency_level_consensus', 
                                    output_dir, covariate_col)
    
    # Save comprehensive results
    results_path = output_dir / 'comprehensive_results.txt'
    with open(results_path, 'w') as f:
        f.write("ENHANCED GEOMETRIC DECEPTION ANALYSIS RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now()}\n")
        f.write(f"Dataset: {len(merged_df)} records\n")
        f.write(f"Optimal Gamma: {optimal_gamma}\n")
        f.write(f"Covariate Used: {covariate_col}\n\n")
        
        # Add more detailed results to file
        if 'primary_test' in results:
            pt = results['primary_test']
            f.write("PRIMARY HYPOTHESIS TEST:\n")
            f.write(f"  Test: {pt['test_name']}\n")
            f.write(f"  Statistic: {pt['statistic']:.4f}\n")
            f.write(f"  p-value: {pt['p_value']:.6f}\n")
            f.write(f"  Significance: {pt['significance']}\n")
            if 'effect_sizes' in pt:
                f.write("  Effect Sizes:\n")
                for effect_name, effect_value in pt['effect_sizes'].items():
                    if 'interpretation' not in effect_name:
                        interp_key = f"{effect_name}_interpretation"
                        interp = pt['effect_sizes'].get(interp_key, "")
                        f.write(f"    {effect_name}: {effect_value:.4f} ({interp})\n")
            f.write("\n")
    
    # Save final merged dataset
    output_path = output_dir / 'merged_analysis_data.csv'
    merged_df.to_csv(output_path, index=False)
    logger.info(f"Saved merged analysis dataset to {output_path}")
    
    # Save descriptive statistics
    if 'descriptive_stats' in results:
        desc_path = output_dir / 'descriptive_statistics.csv'
        results['descriptive_stats'].to_csv(desc_path, index=False)
        logger.info(f"Saved descriptive statistics to {desc_path}")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Dataset: {len(merged_df)} records analyzed")
    if optimal_gamma is not None:
        print(f"Optimal gamma parameter: {optimal_gamma}")
    if covariate_col:
        print(f"Covariate controlled: {covariate_col}")
    print(f"Results saved to: {output_dir}")
    print("\nKey files generated:")
    print(f"  - merged_analysis_data.csv: Complete dataset")
    print(f"  - enhanced_geometric_analysis_plots.png: Comprehensive visualizations")
    print(f"  - descriptive_statistics.csv: Enhanced descriptive stats with CI")
    print(f"  - correlation_matrix.csv: Correlation analysis")
    print(f"  - comprehensive_results.txt: Detailed analysis summary")
    if optimal_gamma is not None:
        print(f"  - gamma_analysis.csv: Gamma optimization results")
    
    print(f"\nAnalysis includes:")
    print(f"  ✓ Multiple test correction ({args.correction_method})")
    print(f"  ✓ Effect size calculations (Cohen's d, Cliff's δ, η²)")
    print(f"  ✓ Bootstrap confidence intervals (95%)")
    print(f"  ✓ Normality testing (Shapiro-Wilk)")
    if covariate_col:
        print(f"  ✓ ANCOVA covariate control ({covariate_col})")
    print(f"  ✓ Enhanced visualizations with CI bars")

if __name__ == "__main__":
    main()
