"""
Quantitative Text Analysis Script for LLM Responses

Analyzes measurable text features like token counts, pronoun ratios, 
structural elements, and linguistic patterns across response datasets.

Usage:
    python quantitative-analysis.py input_responses.csv [--output summary.csv]
"""

import pandas as pd
import numpy as np
import re
import argparse
import logging
from typing import Dict, List, Tuple
from collections import Counter
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuantitativeAnalyzer:
    """Analyzes quantitative features of text responses"""
    
    def __init__(self):
        # Define pronoun categories
        self.first_person_pronouns = {'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves'}
        self.third_person_pronouns = {'it', 'its', 'itself', 'this', 'that', 'these', 'those'}
        
        # AI-specific phrases
        self.ai_disclaimers = [
            r"as an ai",
            r"i am an ai",
            r"i'm an ai", 
            r"i don't have",
            r"i am not",
            r"i'm not",
            r"i cannot",
            r"i can't",
            r"i don't",
            r"as a language model",
            r"i'm a language model"
        ]
        
        # Structural patterns
        self.numbered_list_pattern = r'^\s*\d+[\.\)]\s+'
        self.bullet_pattern = r'^\s*[-*•]\s+'
        
    def analyze_response(self, response_text: str) -> Dict:
        """Analyze a single response and return quantitative metrics"""
        if pd.isna(response_text) or not isinstance(response_text, str):
            return self._empty_metrics()
            
        text = response_text.strip()
        if not text:
            return self._empty_metrics()
            
        metrics = {}
        
        # Basic counts
        metrics.update(self._count_tokens(text))
        
        # Pronoun analysis
        metrics.update(self._analyze_pronouns(text))
        
        # Structural features
        metrics.update(self._analyze_structure(text))
        
        # AI-specific language
        metrics.update(self._analyze_ai_language(text))
        
        # Sentence analysis
        metrics.update(self._analyze_sentences(text))
        
        return metrics
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics for invalid/missing text"""
        return {
            'word_count': 0,
            'char_count': 0,
            'char_count_no_spaces': 0,
            'sentence_count': 0,
            'avg_sentence_length': 0.0,
            'first_person_pronouns': 0,
            'third_person_pronouns': 0,
            'pronoun_ratio_1st_to_3rd': 0.0,
            'total_pronouns': 0,
            'numbered_list_items': 0,
            'bullet_list_items': 0,
            'has_numbered_list': False,
            'has_bullet_list': False,
            'ai_disclaimer_count': 0,
            'unique_ai_disclaimers': 0,
            'i_dont_count': 0,
            'i_am_not_count': 0,
            'as_an_ai_count': 0,
            'question_count': 0,
            'exclamation_count': 0,
            'avg_word_length': 0.0
        }
    
    def _count_tokens(self, text: str) -> Dict:
        """Count basic text units"""
        words = text.split()
        
        return {
            'word_count': len(words),
            'char_count': len(text),
            'char_count_no_spaces': len(text.replace(' ', '')),
            'avg_word_length': np.mean([len(word.strip('.,!?;:')) for word in words]) if words else 0.0
        }
    
    def _analyze_pronouns(self, text: str) -> Dict:
        """Analyze pronoun usage patterns"""
        # Convert to lowercase and extract words
        words = re.findall(r'\b\w+\b', text.lower())
        
        first_person_count = sum(1 for word in words if word in self.first_person_pronouns)
        third_person_count = sum(1 for word in words if word in self.third_person_pronouns)
        total_pronouns = first_person_count + third_person_count
        
        # Calculate ratio (avoid division by zero)
        if third_person_count > 0:
            ratio = first_person_count / third_person_count
        elif first_person_count > 0:
            ratio = float('inf')  # All first person, no third person
        else:
            ratio = 0.0  # No pronouns
            
        return {
            'first_person_pronouns': first_person_count,
            'third_person_pronouns': third_person_count,
            'pronoun_ratio_1st_to_3rd': ratio,
            'total_pronouns': total_pronouns
        }
    
    def _analyze_structure(self, text: str) -> Dict:
        """Analyze structural elements like lists"""
        lines = text.split('\n')
        
        numbered_items = 0
        bullet_items = 0
        
        for line in lines:
            if re.match(self.numbered_list_pattern, line):
                numbered_items += 1
            elif re.match(self.bullet_pattern, line):
                bullet_items += 1
                
        return {
            'numbered_list_items': numbered_items,
            'bullet_list_items': bullet_items,
            'has_numbered_list': numbered_items > 0,
            'has_bullet_list': bullet_items > 0
        }
    
    def _analyze_ai_language(self, text: str) -> Dict:
        """Analyze AI-specific language patterns"""
        text_lower = text.lower()
        
        # Count specific patterns
        disclaimer_counts = {}
        total_disclaimers = 0
        unique_disclaimers = 0
        
        for pattern in self.ai_disclaimers:
            count = len(re.findall(pattern, text_lower))
            disclaimer_counts[pattern] = count
            if count > 0:
                total_disclaimers += count
                unique_disclaimers += 1
        
        return {
            'ai_disclaimer_count': total_disclaimers,
            'unique_ai_disclaimers': unique_disclaimers,
            'i_dont_count': disclaimer_counts.get(r"i don't", 0),
            'i_am_not_count': disclaimer_counts.get(r"i am not", 0) + disclaimer_counts.get(r"i'm not", 0),
            'as_an_ai_count': disclaimer_counts.get(r"as an ai", 0)
        }
    
    def _analyze_sentences(self, text: str) -> Dict:
        """Analyze sentence-level features"""
        # Simple sentence splitting (can be improved with nltk)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        sentence_count = len(sentences)
        avg_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0.0
        
        # Count punctuation
        question_count = text.count('?')
        exclamation_count = text.count('!')
        
        return {
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_length,
            'question_count': question_count,
            'exclamation_count': exclamation_count
        }

def analyze_dataset(input_file: str, output_file: str = None) -> pd.DataFrame:
    """
    Analyze a dataset of responses and generate quantitative metrics
    
    Args:
        input_file: Path to CSV with response data
        output_file: Optional path for output CSV
        
    Returns:
        DataFrame with quantitative metrics
    """
    logger.info(f"Loading response data from {input_file}...")
    
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        sys.exit(1)
    
    # Validate required columns
    required_cols = ['response_text']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        logger.info(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    logger.info(f"Loaded {len(df)} responses")
    
    # Initialize analyzer
    analyzer = QuantitativeAnalyzer()
    
    # Analyze each response
    logger.info("Analyzing responses...")
    results = []
    
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            logger.info(f"Processing response {idx + 1}/{len(df)}")
            
        # Get base info
        result = {
            'response_id': f"{row.get('prompt_id', 'unknown')}_{row.get('run_id', idx)}",
            'prompt_id': row.get('prompt_id', 'unknown'),
            'run_id': row.get('run_id', idx)
        }
        
        # Add quantitative metrics
        metrics = analyzer.analyze_response(row['response_text'])
        result.update(metrics)
        
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Generate summary statistics
    logger.info("Generating summary statistics...")
    summary_stats = generate_summary_statistics(results_df)
    
    # Save results
    if output_file:
        results_df.to_csv(output_file, index=False)
        logger.info(f"Saved detailed results to {output_file}")
        
        # Save summary
        summary_file = output_file.replace('.csv', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(summary_stats)
        logger.info(f"Saved summary statistics to {summary_file}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("QUANTITATIVE ANALYSIS SUMMARY")
    print("="*60)
    print(summary_stats)
    
    return results_df

def generate_summary_statistics(df: pd.DataFrame) -> str:
    """Generate human-readable summary statistics"""
    
    # Filter out metadata columns for numeric analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    metadata_cols = ['run_id']
    analysis_cols = [col for col in numeric_cols if col not in metadata_cols]
    
    summary = []
    summary.append(f"Dataset Overview:")
    summary.append(f"  Total responses: {len(df)}")
    summary.append(f"  Unique prompts: {df['prompt_id'].nunique()}")
    summary.append("")
    
    # Basic text metrics
    summary.append("Text Volume Metrics:")
    for col in ['word_count', 'char_count', 'sentence_count']:
        if col in df.columns:
            summary.append(f"  {col.replace('_', ' ').title()}:")
            summary.append(f"    Mean: {df[col].mean():.1f}")
            summary.append(f"    Median: {df[col].median():.1f}")
            summary.append(f"    Range: {df[col].min():.0f} - {df[col].max():.0f}")
    summary.append("")
    
    # Pronoun analysis
    summary.append("Pronoun Usage:")
    summary.append(f"  Responses with first-person pronouns: {(df['first_person_pronouns'] > 0).sum()}/{len(df)} ({(df['first_person_pronouns'] > 0).mean()*100:.1f}%)")
    summary.append(f"  Mean first-person pronoun count: {df['first_person_pronouns'].mean():.1f}")
    summary.append(f"  Mean third-person pronoun count: {df['third_person_pronouns'].mean():.1f}")
    
    # Handle infinite ratios for display
    finite_ratios = df['pronoun_ratio_1st_to_3rd'][df['pronoun_ratio_1st_to_3rd'] != float('inf')]
    if len(finite_ratios) > 0:
        summary.append(f"  Mean 1st/3rd pronoun ratio (finite values): {finite_ratios.mean():.2f}")
    summary.append("")
    
    # Structural features
    summary.append("Structural Features:")
    summary.append(f"  Responses with numbered lists: {df['has_numbered_list'].sum()}/{len(df)} ({df['has_numbered_list'].mean()*100:.1f}%)")
    summary.append(f"  Responses with bullet lists: {df['has_bullet_list'].sum()}/{len(df)} ({df['has_bullet_list'].mean()*100:.1f}%)")
    summary.append("")
    
    # AI language patterns
    summary.append("AI Language Patterns:")
    summary.append(f"  Responses with AI disclaimers: {(df['ai_disclaimer_count'] > 0).sum()}/{len(df)} ({(df['ai_disclaimer_count'] > 0).mean()*100:.1f}%)")
    summary.append(f"  Mean AI disclaimer count: {df['ai_disclaimer_count'].mean():.1f}")
    summary.append(f"  'I don't' frequency: {df['i_dont_count'].sum()} total occurrences")
    summary.append(f"  'I am not' frequency: {df['i_am_not_count'].sum()} total occurrences")
    summary.append(f"  'As an AI' frequency: {df['as_an_ai_count'].sum()} total occurrences")
    summary.append("")
    
    # Sentence complexity
    summary.append("Sentence Complexity:")
    summary.append(f"  Mean sentence length (words): {df['avg_sentence_length'].mean():.1f}")
    summary.append(f"  Mean word length (characters): {df['avg_word_length'].mean():.1f}")
    summary.append(f"  Questions per response: {df['question_count'].mean():.1f}")
    summary.append(f"  Exclamations per response: {df['exclamation_count'].mean():.1f}")
    summary.append("")
    
    # Comparative analysis by prompt type (if identifiable)
    factual_responses = df[df['prompt_id'].str.contains('factual', case=False, na=False)]
    if len(factual_responses) > 0:
        non_factual_responses = df[~df['prompt_id'].str.contains('factual', case=False, na=False)]
        
        summary.append("Factual vs Non-Factual Comparison:")
        summary.append(f"  Factual responses ({len(factual_responses)}):")
        summary.append(f"    Mean word count: {factual_responses['word_count'].mean():.1f}")
        summary.append(f"    First-person pronoun usage: {(factual_responses['first_person_pronouns'] > 0).mean()*100:.1f}%")
        summary.append(f"    AI disclaimer usage: {(factual_responses['ai_disclaimer_count'] > 0).mean()*100:.1f}%")
        
        summary.append(f"  Non-factual responses ({len(non_factual_responses)}):")
        summary.append(f"    Mean word count: {non_factual_responses['word_count'].mean():.1f}")
        summary.append(f"    First-person pronoun usage: {(non_factual_responses['first_person_pronouns'] > 0).mean()*100:.1f}%")
        summary.append(f"    AI disclaimer usage: {(non_factual_responses['ai_disclaimer_count'] > 0).mean()*100:.1f}%")
    
    return "\n".join(summary)

def main():
    parser = argparse.ArgumentParser(description='Analyze quantitative features of LLM responses')
    parser.add_argument('input_file', help='Path to input CSV file with response data')
    parser.add_argument('--output', '-o', help='Path to output CSV file for detailed results')
    
    args = parser.parse_args()
    
    # Generate default output filename if not provided
    if not args.output:
        input_path = Path(args.input_file)
        args.output = str(input_path.parent / f"{input_path.stem}_quantitative_analysis.csv")
    
    # Run analysis
    results_df = analyze_dataset(args.input_file, args.output)
    
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main()
