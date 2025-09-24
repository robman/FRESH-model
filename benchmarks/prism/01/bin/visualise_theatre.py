#!/usr/bin/env python3
"""
Theatre Detection Focused Visualization Script

Creates three focused plots that directly support the paper's narrative:
1. Dumbbell chart: External vs Internal arbitration by scenario (sorted by gap)
2. Stacked bars: Theatre Exposure Index distribution by model family
3. EFE waterfall grid: Decomposition of G^ for top hidden theatre cases

Usage:
    python bin/visualise_theatre2.py \
    --results_dir results_analysis \
    --output_dir results_analysis/plots

Input Requirements:
    - raw_data.csv (from analyse_theatre.py)

Output:
    - dumbbell_arbitration.png
    - tei_mix_by_model.png
    - efe_waterfalls.png
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class FocusedTheatreVisualizer:
    def __init__(self, results_dir: str, output_dir: Optional[str] = None):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir) if output_dir else self.results_dir / 'plots'

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"📈 Focused Theatre Visualizer initialized")
        print(f"🔍 Results directory: {self.results_dir}")
        print(f"📊 Output directory: {self.output_dir}")

    def load_data(self) -> pd.DataFrame:
        """Load raw trial data from CSV"""
        print("📊 Loading raw trial data...")

        raw_data_file = self.results_dir / 'raw_data.csv'
        if not raw_data_file.exists():
            raise FileNotFoundError(f"Raw data file not found: {raw_data_file}")

        df = pd.read_csv(raw_data_file)
        print(f"✅ Loaded {len(df)} trials")

        # Verify required columns exist
        required_cols = ['scenario', 'model', 'x_arbitration_rate', 'i_arbitration_rate',
                        'theatre_exposure_index', 'efe_R', 'efe_E', 'efe_Ghat']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return df

    def extract_model_family(self, model_name: str) -> str:
        """Extract model family from full model name"""
        if pd.isna(model_name):
            return 'unknown'

        model_lower = str(model_name).lower()
        if 'claude' in model_lower:
            return 'Claude'
        elif 'chatgpt' in model_lower or 'gpt' in model_lower or 'openai' in model_lower:
            return 'ChatGPT'
        elif 'gemini' in model_lower:
            return 'Gemini'
        elif 'llama' in model_lower:
            return 'Llama'
        else:
            return 'Other'

    def wilson_ci(self, successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate Wilson confidence interval for binomial proportion"""
        if total == 0:
            return 0.0, 0.0

        p = successes / total
        z = stats.norm.ppf(1 - (1 - confidence) / 2)

        denominator = 1 + (z**2 / total)
        center = (p + (z**2) / (2 * total)) / denominator
        margin = z * np.sqrt((p * (1 - p) + (z**2) / (4 * total)) / total) / denominator

        return max(0, center - margin), min(1, center + margin)

    def plot_dumbbell_arbitration(self, df: pd.DataFrame) -> None:
        """Create dumbbell chart showing external vs internal arbitration by scenario"""
        print("📊 Creating dumbbell arbitration chart...")

        # Create scenario_model identifier
        df['scenario_model'] = df['scenario'] + '_' + df['model']

        # Group by scenario_model and calculate means
        grouped = df.groupby('scenario_model').agg({
            'x_arbitration_rate': 'mean',
            'i_arbitration_rate': 'mean',
            'theatre_exposure_index': lambda x: (x == 1).sum()  # Count hidden theatre cases
        }).reset_index()

        # Add trial counts
        trial_counts = df.groupby('scenario_model').size().reset_index(name='n_trials')
        grouped = grouped.merge(trial_counts, on='scenario_model')

        # Calculate hidden theatre gap and filter out rows with NaN
        grouped = grouped.dropna(subset=['x_arbitration_rate', 'i_arbitration_rate'])
        grouped['hidden_theatre_gap'] = grouped['x_arbitration_rate'] - grouped['i_arbitration_rate']

        # Sort by absolute gap size (largest visual gaps at top)
        grouped['abs_gap'] = grouped['hidden_theatre_gap'].abs()
        grouped = grouped.sort_values('abs_gap', ascending=False)


        # Calculate overall hidden theatre percentage from actual TEI data
        total_hidden = (df['theatre_exposure_index'] == 1).sum()
        total_valid = df['theatre_exposure_index'].notna().sum()
        overall_hidden_pct = (total_hidden / total_valid * 100) if total_valid > 0 else 0

        # Create the plot with better aspect ratio
        fig, ax = plt.subplots(figsize=(12, max(8, len(grouped) * 0.3)))

        y_positions = list(range(len(grouped)))  # Same order as DataFrame

        # Draw connecting lines
        for i, (_, row) in enumerate(grouped.iterrows()):
            x_val = row['x_arbitration_rate']
            i_val = row['i_arbitration_rate']
            y_pos = y_positions[i]  # Use the correct y position
            ax.plot([i_val, x_val], [y_pos, y_pos], 'k-', alpha=0.6, linewidth=1.5)


        # Plot points
        ax.scatter(grouped['x_arbitration_rate'], y_positions,
                  c='darkred', s=80, alpha=0.8, label='External', zorder=3)
        ax.scatter(grouped['i_arbitration_rate'], y_positions,
                  c='steelblue', s=80, alpha=0.8, label='Internal', zorder=3)

        # Add vertical shaded zone for "hidden theatre band" (high external + low internal)
        ax.axvspan(0.0, 0.1, alpha=0.1, color='blue', label='Low Internal Zone')
        ax.axvspan(0.3, 1.0, alpha=0.1, color='red', label='High External Zone')


        # Create labels with short model names for readability
        labels = []
        for _, row in grouped.iterrows():
            scenario_model = row['scenario_model']
            parts = scenario_model.split('_')
            if len(parts) >= 3:
                scenario = parts[0]
                topic = parts[1]
                model_full = '_'.join(parts[2:])

                # Extract short model name
                model_short = self.extract_model_family(model_full).lower()
                label = f"{scenario}·{topic}·{model_short}"
            else:
                label = scenario_model.replace('_', '·')
            labels.append(label)

        # Remove delta annotations for cleaner plot
        # (Gap information is already shown by the visual spacing between dots)

        # Customize plot
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()  # Invert so largest gaps appear at top
        ax.set_xlabel('Arbitration Rate', fontsize=12)
        ax.set_title(f'External vs Internal Arbitration by Scenario\n{overall_hidden_pct:.0f}% of trials show hidden theatre (TEI = +1)',
                    fontsize=14, fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend(loc='lower right', fontsize=10)

        # Add subtitle
        ax.text(0.5, -0.08, 'Hidden theatre = high external + low internal arbitration',
               transform=ax.transAxes, ha='center', va='top', fontsize=11,
               style='italic', color='gray')

        plt.tight_layout()
        plt.savefig(self.output_dir / "dumbbell_arbitration.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Dumbbell arbitration chart saved")

    def plot_tei_mix_by_model(self, df: pd.DataFrame) -> None:
        """Create stacked bars showing TEI distribution by model family"""
        print("📊 Creating TEI mix by model chart...")

        # Extract model families
        df['model_family'] = df['model'].apply(self.extract_model_family)

        # Filter to valid TEI data
        valid_tei = df['theatre_exposure_index'].notna()
        df_tei = df[valid_tei].copy()

        if len(df_tei) == 0:
            print("⚠️ No valid TEI data found - skipping TEI mix chart")
            return

        # Group by model family and count TEI values
        tei_counts = df_tei.groupby(['model_family', 'theatre_exposure_index']).size().unstack(fill_value=0)

        # Ensure all TEI values are present
        for tei_val in [-1, 0, 1]:
            if tei_val not in tei_counts.columns:
                tei_counts[tei_val] = 0

        # Reorder columns and calculate percentages
        tei_counts = tei_counts[[-1, 0, 1]]  # Surface-only, Aligned, Hidden
        tei_percentages = tei_counts.div(tei_counts.sum(axis=1), axis=0) * 100

        # Calculate totals for Wilson CIs
        totals = tei_counts.sum(axis=1)
        hidden_counts = tei_counts[1]

        # Calculate Wilson CIs for hidden theatre proportion
        wilson_cis = []
        for family in tei_percentages.index:
            n_total = totals[family]
            n_hidden = hidden_counts[family]
            ci_lower, ci_upper = self.wilson_ci(n_hidden, n_total)
            wilson_cis.append((ci_lower * 100, ci_upper * 100))

        # Sort by hidden theatre percentage descending
        hidden_pct = tei_percentages[1]
        sort_order = hidden_pct.sort_values(ascending=False).index
        tei_percentages = tei_percentages.loc[sort_order]
        wilson_cis = [wilson_cis[tei_percentages.index.get_loc(family)] for family in sort_order]
        totals = totals[sort_order]

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Define colors
        colors = ['lightcoral', 'lightgray', 'darkgreen']  # Surface-only, Aligned, Hidden
        labels = ['Surface Only', 'Aligned', 'Hidden Theatre']

        # Create stacked bars
        bottom = np.zeros(len(tei_percentages))
        bar_segments = []

        for i, (tei_val, color, label) in enumerate(zip([-1, 0, 1], colors, labels)):
            values = tei_percentages[tei_val].values
            bars = ax.bar(range(len(tei_percentages)), values, bottom=bottom,
                         color=color, alpha=0.8, label=label)
            bar_segments.append((bars, values, tei_val))
            bottom += values

        # Add percentage labels on bars (for segments > 8%)
        for bars, values, tei_val in bar_segments:
            for bar, pct in zip(bars, values):
                if pct > 8:  # Only label significant segments
                    height = bar.get_height()
                    y_pos = bar.get_y() + height / 2
                    ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                           f'{pct:.0f}%', ha='center', va='center',
                           fontsize=9, fontweight='bold', color='white')

        # Add Wilson CIs for hidden theatre only (the green segments)
        hidden_bars = bar_segments[2][0]  # Hidden theatre bars
        hidden_values = tei_percentages[1].values

        for i, (bar, ci) in enumerate(zip(hidden_bars, wilson_cis)):
            ci_lower_pct, ci_upper_pct = ci

            # Calculate the center y-position of the hidden theatre segment
            # This is: surface% + aligned% + (hidden% / 2)
            surface_pct = tei_percentages.iloc[i][-1]  # Surface-only (-1)
            aligned_pct = tei_percentages.iloc[i][0]   # Aligned (0)
            hidden_pct = tei_percentages.iloc[i][1]    # Hidden (1)

            y_center = surface_pct + aligned_pct + (hidden_pct / 2)
            x_center = bar.get_x() + bar.get_width() / 2

            # Calculate the vertical span of the CI bracket relative to the center
            ci_half_span = (ci_upper_pct - ci_lower_pct) / 2
            y_lower = y_center - ci_half_span
            y_upper = y_center + ci_half_span

            # Draw CI bracket positioned at the center of the hidden segment
            ax.plot([x_center, x_center], [y_lower, y_upper], 'k-', linewidth=2, alpha=0.8)
            ax.plot([x_center - 0.1, x_center + 0.1], [y_lower, y_lower], 'k-', linewidth=2, alpha=0.8)
            ax.plot([x_center - 0.1, x_center + 0.1], [y_upper, y_upper], 'k-', linewidth=2, alpha=0.8)

        # Add sample sizes below x-axis
        x_positions = range(len(tei_percentages))
        for i, (family, total) in enumerate(zip(tei_percentages.index, totals)):
            ax.text(i, -8, f'n={total}', ha='center', va='top', fontsize=9, color='gray')

        # Customize plot
        ax.set_xticks(x_positions)
        ax.set_xticklabels(tei_percentages.index, fontsize=11)
        ax.set_ylabel('Percentage', fontsize=12)
        ax.set_title('Theatre Exposure Index Distribution by Model Family\nHidden Theatre share with Wilson 95% CIs',
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.subplots_adjust(right=0.85)  # Make space for legend
        plt.savefig(self.output_dir / "tei_mix_by_model.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ TEI mix by model chart saved")

    def plot_efe_waterfalls(self, df: pd.DataFrame) -> None:
        """Create EFE waterfall grid for top hidden theatre cases"""
        print("📊 Creating EFE waterfall grid...")

        # Create scenario_model identifier and calculate hidden theatre gap
        df['scenario_model'] = df['scenario'] + '_' + df['model']

        # Group by scenario_model and calculate means, focusing on cases with EFE data
        efe_data = df.dropna(subset=['efe_R', 'efe_E', 'efe_Ghat']).copy()

        if len(efe_data) == 0:
            print("⚠️ No valid EFE data found - skipping EFE waterfall")
            return

        grouped = efe_data.groupby('scenario_model').agg({
            'x_arbitration_rate': 'mean',
            'i_arbitration_rate': 'mean',
            'efe_R': 'mean',
            'efe_E': 'mean',
            'efe_Ghat': 'mean'
        }).reset_index()

        # Calculate hidden theatre gap and sort by absolute gap size (same as dumbbell chart)
        grouped['hidden_theatre_gap'] = grouped['x_arbitration_rate'] - grouped['i_arbitration_rate']
        grouped['abs_gap'] = grouped['hidden_theatre_gap'].abs()
        grouped = grouped.sort_values('abs_gap', ascending=False)

        # Select top 8 cases for 2x4 grid
        top_cases = grouped.head(8)

        if len(top_cases) == 0:
            print("⚠️ No cases available for EFE waterfall")
            return

        # Create 2x4 grid with space for legend
        fig, axes = plt.subplots(2, 4, figsize=(16, 9))
        fig.suptitle(r'EFE Proxy Decomposition for Top Hidden Theatre Cases' + '\n' + r'$\widehat{G} = R - E$ (lower is better)',
                    fontsize=16, fontweight='bold')

        # Flatten axes for easier iteration
        axes_flat = axes.flatten()

        # Find global min/max for consistent scaling using calculated values
        all_values = []
        for _, row in top_cases.iterrows():
            expected_G = row['efe_R'] - row['efe_E']
            all_values.extend([0, row['efe_R'], -row['efe_E'], expected_G])

        x_min, x_max = min(all_values), max(all_values)
        x_range = x_max - x_min
        x_margin = x_range * 0.1
        xlim = (x_min - x_margin, x_max + x_margin)

        # Create waterfalls for each case
        for i, (_, row) in enumerate(top_cases.iterrows()):
            if i >= 8:  # Safety check
                break

            ax = axes_flat[i]

            # Show R - E = G decomposition clearly
            R_value = row['efe_R']
            E_value = row['efe_E']  # This is the raw E value (positive)
            G_value = row['efe_Ghat']  # Should equal R - E

            # Verify the math
            expected_G = R_value - E_value
            raw_G = G_value


            # Create bars showing the subtraction visually
            bar_height = 0.2
            bar_positions = [0.3, 0, -0.3]  # R at top, -E in middle, G at bottom

            # R bar: positive green bar from 0 to +R
            ax.barh(bar_positions[0], R_value, color='green', alpha=0.8, height=bar_height)
            ax.text(R_value + 0.01, bar_positions[0], f'R=+{R_value:.3f}',
                   ha='left', va='center', fontweight='bold', fontsize=9)

            # -E bar: red bar from +R back toward 0 (leftward subtraction)
            ax.barh(bar_positions[1], -E_value, left=R_value, color='red', alpha=0.8, height=bar_height)
            ax.text(R_value - E_value/2, bar_positions[1], f'-E=-{abs(E_value):.3f}',
                   ha='center', va='center', fontweight='bold', fontsize=9, color='black')

            # G bar: final result showing R - E
            ax.barh(bar_positions[2], expected_G, color='steelblue', alpha=0.8, height=bar_height)
            ax.text(expected_G + 0.01, bar_positions[2], f'Ĝ={expected_G:.3f}',
                   ha='left', va='center', fontweight='bold', fontsize=9)

            # Use short model names in titles
            scenario_model = row['scenario_model']
            parts = scenario_model.split('_')
            if len(parts) >= 3:
                scenario = parts[0]
                topic = parts[1]
                model_full = '_'.join(parts[2:])
                model_short = self.extract_model_family(model_full).lower()
                title = f"{scenario}·{topic}·{model_short}"
            else:
                title = scenario_model.replace('_', '·')

            # Customize subplot
            max_value = max(R_value, expected_G) * 1.2
            ax.set_xlim(0, max_value)
            ax.set_ylim(-0.5, 0.5)
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_yticks(bar_positions)
            ax.set_yticklabels(['Response\nCost (+R)', 'Evidence\nSubtraction (-E)', 'Total EFE\n(Ĝ)'], fontsize=8)
            ax.grid(True, axis='x', alpha=0.3)
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

            # Only show x-axis labels on bottom row
            if i >= 4:
                ax.set_xlabel('EFE Value', fontsize=9)
            else:
                ax.set_xticklabels([])

        # Hide unused subplots
        for i in range(len(top_cases), 8):
            axes_flat[i].set_visible(False)

        # Add legend at bottom of figure with matching colors
        legend_colors = ['green', 'red', 'steelblue']
        legend_labels = ['+R (Response cost)', '-E (Evidence subtraction)', r'$\widehat{G}$ (Total EFE = R - E)']
        legend_handles = [plt.Rectangle((0,0),1,1, color=color, alpha=0.8) for color in legend_colors]

        fig.legend(legend_handles, legend_labels, loc='lower center', ncol=3,
                  bbox_to_anchor=(0.5, 0.02), fontsize=11, frameon=True, fancybox=True)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make space for legend
        plt.savefig(self.output_dir / "efe_waterfalls.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ EFE waterfall grid saved")

    def generate_all_plots(self, df: pd.DataFrame) -> None:
        """Generate all three focused plots"""
        print("📈 Generating all focused plots...")

        # Set consistent style
        plt.style.use('default')
        sns.set_palette("deep")

        # Generate plots
        self.plot_dumbbell_arbitration(df)
        self.plot_tei_mix_by_model(df)
        self.plot_efe_waterfalls(df)

        print("✅ All focused plots generated successfully")

    def run_visualization(self):
        """Main execution method"""
        print("🎨 Starting Focused Theatre Detection Visualization")
        print("=" * 50)

        try:
            # Load data
            df = self.load_data()

            # Generate all plots
            self.generate_all_plots(df)

            print("=" * 50)
            print(f"🎉 Visualization complete! Plots saved to: {self.output_dir}")
            print("📁 Generated files:")
            print("   - dumbbell_arbitration.png")
            print("   - tei_mix_by_model.png")
            print("   - efe_waterfalls.png")

        except Exception as e:
            print(f"❌ Error during visualization: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Generate focused theatre detection plots for paper')
    parser.add_argument('--results_dir', required=True, help='Directory containing analysis results')
    parser.add_argument('--output_dir', help='Output directory for plots (default: results_dir/plots)')

    args = parser.parse_args()

    visualizer = FocusedTheatreVisualizer(args.results_dir, args.output_dir)
    visualizer.run_visualization()


if __name__ == '__main__':
    main()