"""
Metrics calculation module for transcription evaluation.

Provides functions for:
- Word Error Rate (WER) calculation
- Diarization Error Rate (DER) calculation
- System performance metrics (time, memory, GPU usage)
- Comparative analysis and reporting
"""

import time
import psutil
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


def calculate_wer(reference: str, hypothesis: str, normalize_punctuation: bool = True) -> Dict[str, float]:
    """
    Calculate Word Error Rate (WER) between reference and hypothesis transcripts.

    Args:
        reference: Ground truth transcript text
        hypothesis: Generated transcript text to evaluate
        normalize_punctuation: If True, apply comprehensive text normalization before WER calculation
                              (default: True). This includes:
                              - Expanding contractions: "it's" → "it is", "don't" → "do not", etc.
                              - Handling possessives/contractions: "dog's" → "dog is"
                              - Removing punctuation: commas, periods, etc.
                              - Normalizing case: all text → lowercase

    Returns:
        Dictionary containing:
            - wer: Word Error Rate (0.0 to 1.0+)
            - mer: Match Error Rate
            - wil: Word Information Lost
            - substitutions: Number of word substitutions
            - deletions: Number of word deletions
            - insertions: Number of word insertions
            - hits: Number of correct words
            - punctuation_normalized: Whether normalization was applied
            - punctuation_diff_count: Number of punctuation differences (if normalized)

    Normalization Strategy (when normalize_punctuation=True):
        1. Expand known contractions (it's, he's, don't, can't, etc.)
        2. Expand remaining 's patterns to " is" (handles both possessives and contractions)
        3. Remove punctuation marks (,;:.!?"()[]{}-)
        4. Normalize to lowercase
        5. Clean up extra whitespace

    Examples:
        >>> # Simple WER test
        >>> ref = "the quick brown fox"
        >>> hyp = "the quick brown dog"
        >>> result = calculate_wer(ref, hyp)
        >>> print(f"WER: {result['wer']:.2%}")

        >>> # Contraction normalization
        >>> ref = "it's raining today"
        >>> hyp = "it is raining today"
        >>> result = calculate_wer(ref, hyp)
        >>> print(f"WER: {result['wer']:.2%}")  # 0.00% with normalization
    """
    try:
        from jiwer import process_words
        import re

        # Store original texts for punctuation comparison
        ref_original = reference
        hyp_original = hypothesis

        # Normalize punctuation if requested
        if normalize_punctuation:
            # STEP 1: Expand contractions FIRST (while apostrophes still exist)
            # This makes "it's" → "it is", "don't" → "do not", etc.
            contraction_map = {
                r"\bit's\b": "it is",
                r"\bhe's\b": "he is",
                r"\bshe's\b": "she is",
                r"\bthat's\b": "that is",
                r"\bwhat's\b": "what is",
                r"\bwhere's\b": "where is",
                r"\bwho's\b": "who is",
                r"\bthere's\b": "there is",
                r"\bhere's\b": "here is",
                r"\bhow's\b": "how is",
                r"\blet's\b": "let us",
                r"\bdon't\b": "do not",
                r"\bdoesn't\b": "does not",
                r"\bdidn't\b": "did not",
                r"\bwon't\b": "will not",
                r"\bwouldn't\b": "would not",
                r"\bshouldn't\b": "should not",
                r"\bcouldn't\b": "could not",
                r"\bcan't\b": "cannot",
                r"\bain't\b": "am not",
                r"\baren't\b": "are not",
                r"\bisn't\b": "is not",
                r"\bwasn't\b": "was not",
                r"\bweren't\b": "were not",
                r"\bhaven't\b": "have not",
                r"\bhasn't\b": "has not",
                r"\bhadn't\b": "had not",
                r"\bmustn't\b": "must not",
                r"\bi'm\b": "i am",
                r"\byou're\b": "you are",
                r"\bwe're\b": "we are",
                r"\bthey're\b": "they are",
                r"\bi've\b": "i have",
                r"\byou've\b": "you have",
                r"\bwe've\b": "we have",
                r"\bthey've\b": "they have",
                r"\bi'll\b": "i will",
                r"\byou'll\b": "you will",
                r"\bhe'll\b": "he will",
                r"\bshe'll\b": "she will",
                r"\bwe'll\b": "we will",
                r"\bthey'll\b": "they will",
                r"\bi'd\b": "i would",
                r"\byou'd\b": "you would",
                r"\bhe'd\b": "he would",
                r"\bshe'd\b": "she would",
                r"\bwe'd\b": "we would",
                r"\bthey'd\b": "they would",
            }

            # Apply contraction expansion to both texts (case-insensitive)
            ref_normalized = reference
            hyp_normalized = hypothesis
            for contraction, expansion in contraction_map.items():
                ref_normalized = re.sub(contraction, expansion, ref_normalized, flags=re.IGNORECASE)
                hyp_normalized = re.sub(contraction, expansion, hyp_normalized, flags=re.IGNORECASE)

            # STEP 2: Handle ambiguous "'s" (could be possessive OR contraction)
            # After expanding known contractions (it's, he's, she's, etc.), any remaining
            # "'s" is either:
            # - A possessive: "the dog's bone" → strip apostrophe → "the dogs bone"
            # - An unknown contraction: "the dog's hungry" → expand → "the dog is hungry"
            # We'll expand ALL remaining "'s" to "is" to handle both cases
            ref_normalized = re.sub(r"(\w)'s\b", r"\1 is", ref_normalized, flags=re.IGNORECASE)
            hyp_normalized = re.sub(r"(\w)'s\b", r"\1 is", hyp_normalized, flags=re.IGNORECASE)

            # STEP 3: Replace hyphens/dashes with spaces (to preserve word boundaries)
            # This makes "decision-making" → "decision making" instead of "decisionmaking"
            # Handles: - (hyphen), — (em dash), – (en dash)
            ref_normalized = re.sub(r'[\-—–]+', ' ', ref_normalized)
            hyp_normalized = re.sub(r'[\-—–]+', ' ', hyp_normalized)

            # STEP 4: Remove other common punctuation marks
            # This removes: , . ; : ! ? " ( ) [ ] { } ... etc.
            punctuation_pattern = r'[,\.;:!?\"\(\)\[\]\{\}…]'
            ref_normalized = re.sub(punctuation_pattern, '', ref_normalized)
            hyp_normalized = re.sub(punctuation_pattern, '', hyp_normalized)

            # STEP 5: Clean up extra spaces left by punctuation removal and hyphen replacement
            ref_normalized = re.sub(r'\s+', ' ', ref_normalized).strip()
            hyp_normalized = re.sub(r'\s+', ' ', hyp_normalized).strip()

            # STEP 6: Normalize case (make everything lowercase for comparison)
            ref_normalized = ref_normalized.lower()
            hyp_normalized = hyp_normalized.lower()

            # Count punctuation differences
            ref_punct = re.findall(punctuation_pattern, reference)
            hyp_punct = re.findall(punctuation_pattern, hypothesis)
            punct_diff = abs(len(ref_punct) - len(hyp_punct))

            # Use normalized text for WER calculation
            reference = ref_normalized
            hypothesis = hyp_normalized
        else:
            punct_diff = 0

        # Calculate detailed measures using jiwer v4 API
        output = process_words(reference, hypothesis)

        result = {
            'wer': output.wer,
            'mer': output.mer,
            'wil': output.wil,
            'substitutions': output.substitutions,
            'deletions': output.deletions,
            'insertions': output.insertions,
            'hits': output.hits,
            'punctuation_normalized': normalize_punctuation,
        }

        if normalize_punctuation:
            result['punctuation_diff_count'] = punct_diff

        return result

    except ImportError:
        raise ImportError("jiwer library not installed. Run: pip install jiwer")


def calculate_wer_from_files(reference_file: str, hypothesis_file: str,
                              text_column: str = 'Transcription') -> Dict[str, float]:
    """
    Calculate WER from CSV or Excel files containing transcripts.

    Args:
        reference_file: Path to ground truth transcript file (CSV or Excel)
        hypothesis_file: Path to generated transcript file (CSV or Excel)
        text_column: Name of column containing transcript text

    Returns:
        WER metrics dictionary (same as calculate_wer)
    """
    # Load files
    ref_ext = Path(reference_file).suffix.lower()
    hyp_ext = Path(hypothesis_file).suffix.lower()

    if ref_ext in ['.xlsx', '.xls']:
        ref_df = pd.read_excel(reference_file)
    elif ref_ext == '.csv':
        ref_df = pd.read_csv(reference_file)
    else:
        raise ValueError(f"Unsupported file format: {ref_ext}")

    if hyp_ext in ['.xlsx', '.xls']:
        hyp_df = pd.read_excel(hypothesis_file)
    elif hyp_ext == '.csv':
        hyp_df = pd.read_csv(hypothesis_file)
    else:
        raise ValueError(f"Unsupported file format: {hyp_ext}")

    # Extract text
    if text_column not in ref_df.columns:
        raise ValueError(f"Column '{text_column}' not found in reference file")
    if text_column not in hyp_df.columns:
        raise ValueError(f"Column '{text_column}' not found in hypothesis file")

    # Combine all text (join segments)
    ref_text = ' '.join(ref_df[text_column].dropna().astype(str))
    hyp_text = ' '.join(hyp_df[text_column].dropna().astype(str))

    return calculate_wer(ref_text, hyp_text)


def calculate_der(reference_rttm: str, hypothesis_rttm: str,
                  collar: float = 0.25, skip_overlap: bool = False) -> Dict[str, float]:
    """
    Calculate Diarization Error Rate (DER) between reference and hypothesis RTTM files.

    Args:
        reference_rttm: Path to ground truth RTTM file
        hypothesis_rttm: Path to generated RTTM file
        collar: Collar size in seconds (tolerance around boundaries)
        skip_overlap: Whether to skip overlapping speech in evaluation

    Returns:
        Dictionary containing:
            - der: Overall Diarization Error Rate
            - false_alarm: Percentage of false alarm time
            - missed_speech: Percentage of missed speech time
            - speaker_confusion: Percentage of speaker confusion time
    """
    try:
        from pyannote.core import Annotation
        from pyannote.metrics.diarization import DiarizationErrorRate

        # Load RTTM files
        reference = Annotation.from_rttm(reference_rttm)
        hypothesis = Annotation.from_rttm(hypothesis_rttm)

        # Calculate DER
        metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
        der_value = metric(reference, hypothesis)

        # Get detailed components
        components = metric.compute_components(reference, hypothesis)

        return {
            'der': der_value,
            'false_alarm': components['false alarm'],
            'missed_speech': components['missed detection'],
            'speaker_confusion': components['confusion'],
        }
    except ImportError:
        raise ImportError("pyannote.metrics not installed. Run: pip install pyannote.metrics")


class PerformanceTimer:
    """Context manager for timing code execution."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        print(f"{self.name} took {self.elapsed:.2f} seconds ({self.elapsed/60:.2f} minutes)")


class SystemMetrics:
    """Track system resource usage during processing."""

    def __init__(self):
        self.metrics = {
            'cpu_percent': [],
            'memory_mb': [],
            'gpu_memory_mb': [],
            'gpu_utilization': [],
        }
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self.peak_memory = self.start_memory

    def update(self):
        """Update current metrics."""
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.Process().memory_info().rss / 1024 / 1024

        self.metrics['cpu_percent'].append(cpu)
        self.metrics['memory_mb'].append(mem)
        self.peak_memory = max(self.peak_memory, mem)

        # Try to get GPU metrics if available
        try:
            import torch
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_util = torch.cuda.utilization()
                self.metrics['gpu_memory_mb'].append(gpu_mem)
                self.metrics['gpu_utilization'].append(gpu_util)
        except:
            pass

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        summary = {
            'avg_cpu_percent': sum(self.metrics['cpu_percent']) / len(self.metrics['cpu_percent']) if self.metrics['cpu_percent'] else 0,
            'peak_memory_mb': self.peak_memory,
            'memory_increase_mb': self.peak_memory - self.start_memory,
        }

        if self.metrics['gpu_memory_mb']:
            summary['avg_gpu_memory_mb'] = sum(self.metrics['gpu_memory_mb']) / len(self.metrics['gpu_memory_mb'])
            summary['peak_gpu_memory_mb'] = max(self.metrics['gpu_memory_mb'])

        if self.metrics['gpu_utilization']:
            summary['avg_gpu_utilization'] = sum(self.metrics['gpu_utilization']) / len(self.metrics['gpu_utilization'])

        return summary


def generate_comparison_report(results: List[Dict], output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Generate comparison report from multiple test results.

    Args:
        results: List of dictionaries containing test results
            Each dict should have keys: 'name', 'wer', 'processing_time', etc.
        output_file: Optional path to save report (CSV or Excel)

    Returns:
        DataFrame with comparison data
    """
    df = pd.DataFrame(results)

    # Calculate derived metrics if processing_time and audio_duration available
    if 'processing_time' in df.columns and 'audio_duration' in df.columns:
        df['real_time_factor'] = df['processing_time'] / df['audio_duration']

    # Sort by WER (best first)
    if 'wer' in df.columns:
        df = df.sort_values('wer')

    # Save if requested
    if output_file:
        ext = Path(output_file).suffix.lower()
        if ext == '.csv':
            df.to_csv(output_file, index=False)
        elif ext in ['.xlsx', '.xls']:
            df.to_excel(output_file, index=False)
        else:
            raise ValueError(f"Unsupported output format: {ext}")
        print(f"Report saved to: {output_file}")

    return df


def save_metrics_json(metrics: Dict, output_file: str):
    """
    Save metrics dictionary to JSON file.

    Args:
        metrics: Dictionary containing metrics
        output_file: Path to output JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {output_file}")


def load_metrics_json(input_file: str) -> Dict:
    """
    Load metrics from JSON file.

    Args:
        input_file: Path to JSON file

    Returns:
        Metrics dictionary
    """
    with open(input_file, 'r') as f:
        return json.load(f)


# Example usage
if __name__ == "__main__":
    # Test WER calculation
    ref = "the quick brown fox jumps over the lazy dog"
    hyp = "the quick brown dog jumps over a lazy cat"

    print("Testing WER calculation:")
    result = calculate_wer(ref, hyp)
    print(f"WER: {result['wer']:.2%}")
    print(f"Substitutions: {result['substitutions']}")
    print(f"Deletions: {result['deletions']}")
    print(f"Insertions: {result['insertions']}")

    # Test timer
    print("\nTesting performance timer:")
    with PerformanceTimer("Test operation") as timer:
        time.sleep(1)

    # Test system metrics
    print("\nTesting system metrics:")
    metrics = SystemMetrics()
    for i in range(5):
        metrics.update()
        time.sleep(0.2)
    summary = metrics.get_summary()
    print(f"Average CPU: {summary['avg_cpu_percent']:.1f}%")
    print(f"Peak Memory: {summary['peak_memory_mb']:.1f} MB")
