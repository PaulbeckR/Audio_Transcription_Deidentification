"""
Analyze word-level errors between ground truth and WhisperX transcripts.
Shows exactly which words were wrong (substitutions, deletions, insertions).
"""
import sys
from pathlib import Path
import pandas as pd
from jiwer import process_words, visualize_alignment
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.evaluation.metrics import calculate_wer

config = Config()
PROJECT_PATH = Path(__file__).parent

def normalize_text(text):
    """Apply same normalization as WER calculation"""
    # Expand contractions
    contraction_map = {
        r"\bit's\b": "it is", r"\bhe's\b": "he is", r"\bshe's\b": "she is",
        r"\bthat's\b": "that is", r"\bwhat's\b": "what is", r"\bwhere's\b": "where is",
        r"\bwho's\b": "who is", r"\bthere's\b": "there is", r"\bhere's\b": "here is",
        r"\bhow's\b": "how is", r"\blet's\b": "let us", r"\bdon't\b": "do not",
        r"\bdoesn't\b": "does not", r"\bdidn't\b": "did not", r"\bwon't\b": "will not",
        r"\bwouldn't\b": "would not", r"\bshouldn't\b": "should not",
        r"\bcouldn't\b": "could not", r"\bcan't\b": "cannot", r"\bain't\b": "am not",
        r"\baren't\b": "are not", r"\bisn't\b": "is not", r"\bwasn't\b": "was not",
        r"\bweren't\b": "were not", r"\bhaven't\b": "have not", r"\bhasn't\b": "has not",
        r"\bhadn't\b": "had not", r"\bmustn't\b": "must not", r"\bi'm\b": "i am",
        r"\byou're\b": "you are", r"\bwe're\b": "we are", r"\bthey're\b": "they are",
        r"\bi've\b": "i have", r"\byou've\b": "you have", r"\bwe've\b": "we have",
        r"\bthey've\b": "they have", r"\bi'll\b": "i will", r"\byou'll\b": "you will",
        r"\bhe'll\b": "he will", r"\bshe'll\b": "she will", r"\bwe'll\b": "we will",
        r"\bthey'll\b": "they will", r"\bi'd\b": "i would", r"\byou'd\b": "you would",
        r"\bhe'd\b": "he would", r"\bshe'd\b": "she would", r"\bwe'd\b": "we would",
        r"\bthey'd\b": "they would",
    }

    normalized = text
    for contraction, expansion in contraction_map.items():
        normalized = re.sub(contraction, expansion, normalized, flags=re.IGNORECASE)

    # Handle remaining 's patterns
    normalized = re.sub(r"(\w)'s\b", r"\1 is", normalized, flags=re.IGNORECASE)

    # Remove punctuation
    punctuation_pattern = r'[,\.;:!?\"\(\)\[\]\{\}\-—–…]'
    normalized = re.sub(punctuation_pattern, '', normalized)

    # Clean up whitespace and lowercase
    normalized = re.sub(r'\s+', ' ', normalized).strip().lower()

    return normalized


def analyze_errors(experiment_name):
    """Analyze word errors for a specific experiment"""
    print("="*80)
    print(f"WORD ERROR ANALYSIS: {experiment_name}")
    print("="*80)

    # Paths
    output_folder = PROJECT_PATH / config.paths.output_dirs['whisperx'] / experiment_name
    model_name = config.models.whisperx.model
    device = config.get_device()

    whisperx_csv = output_folder / f"{experiment_name}_{model_name}_{device}.csv"

    # Get ground truth path from experiment config
    if config.experiment.transcript_truth:
        truth_path = PROJECT_PATH / config.paths.transcripts / config.experiment.transcript_truth
    else:
        print(f"\nNo ground truth configured for this experiment")
        return

    # Load data
    print(f"\nLoading data...")
    print(f"  Ground Truth: {truth_path.name}")
    print(f"  WhisperX:     {whisperx_csv.name}")

    truth_df = pd.read_csv(truth_path)
    whisperx_df = pd.read_csv(whisperx_csv)

    # Combine transcriptions
    truth_text = ' '.join(truth_df['Transcription'].dropna().astype(str))
    whisperx_text = ' '.join(whisperx_df['Transcription'].dropna().astype(str))

    print(f"\nOriginal texts:")
    print(f"  Truth words:    {len(truth_text.split())}")
    print(f"  WhisperX words: {len(whisperx_text.split())}")

    # Normalize
    truth_normalized = normalize_text(truth_text)
    whisperx_normalized = normalize_text(whisperx_text)

    print(f"\nNormalized texts:")
    print(f"  Truth words:    {len(truth_normalized.split())}")
    print(f"  WhisperX words: {len(whisperx_normalized.split())}")

    # Calculate WER with alignment
    output = process_words(truth_normalized, whisperx_normalized)

    print(f"\n" + "="*80)
    print(f"OVERALL METRICS")
    print("="*80)
    print(f"WER:           {output.wer:.2%}")
    print(f"Substitutions: {output.substitutions}")
    print(f"Deletions:     {output.deletions}")
    print(f"Insertions:    {output.insertions}")
    print(f"Hits:          {output.hits}")

    # Show alignment visualization
    print(f"\n" + "="*80)
    print(f"WORD-LEVEL ALIGNMENT (showing errors)")
    print("="*80)
    print("\nLegend:")
    print("  S = Substitution (wrong word)")
    print("  D = Deletion (missing word)")
    print("  I = Insertion (extra word)")
    print("  H = Hit (correct word)")
    print("\n" + "-"*80)

    # Get alignment
    alignment_str = visualize_alignment(output, show_measures=False)

    # Parse and display errors only
    lines = alignment_str.strip().split('\n')

    # Extract error positions
    errors = []
    ref_words = truth_normalized.split()
    hyp_words = whisperx_normalized.split()

    # Use jiwer's alignment output - AlignmentChunk objects
    for chunk in output.alignments:
        chunk_type = chunk.type
        ref_start = chunk.ref_start_idx
        ref_end = chunk.ref_end_idx
        hyp_start = chunk.hyp_start_idx
        hyp_end = chunk.hyp_end_idx

        if chunk_type == 'substitute':
            ref_word = ' '.join(ref_words[ref_start:ref_end]) if ref_start < len(ref_words) else ''
            hyp_word = ' '.join(hyp_words[hyp_start:hyp_end]) if hyp_start < len(hyp_words) else ''
            errors.append(f"SUBSTITUTION: '{ref_word}' → '{hyp_word}'")
        elif chunk_type == 'delete':
            ref_word = ' '.join(ref_words[ref_start:ref_end]) if ref_start < len(ref_words) else ''
            errors.append(f"DELETION: '{ref_word}' → [missing]")
        elif chunk_type == 'insert':
            hyp_word = ' '.join(hyp_words[hyp_start:hyp_end]) if hyp_start < len(hyp_words) else ''
            errors.append(f"INSERTION: [extra] → '{hyp_word}'")

    # Display errors
    if errors:
        print(f"\nFound {len(errors)} errors:\n")
        for i, error in enumerate(errors[:50], 1):  # Show first 50 errors
            print(f"{i:3d}. {error}")

        if len(errors) > 50:
            print(f"\n... and {len(errors) - 50} more errors")
    else:
        print("\nNo errors found - perfect transcription!")

    # Error type summary
    print(f"\n" + "="*80)
    print(f"ERROR TYPE BREAKDOWN")
    print("="*80)
    subs = sum(1 for e in errors if e.startswith('SUBSTITUTION'))
    dels = sum(1 for e in errors if e.startswith('DELETION'))
    inss = sum(1 for e in errors if e.startswith('INSERTION'))

    print(f"Substitutions: {subs:3d} ({subs/len(errors)*100:.1f}%)" if errors else "Substitutions: 0")
    print(f"Deletions:     {dels:3d} ({dels/len(errors)*100:.1f}%)" if errors else "Deletions: 0")
    print(f"Insertions:    {inss:3d} ({inss/len(errors)*100:.1f}%)" if errors else "Insertions: 0")

    return errors


if __name__ == "__main__":
    # Analyze both experiments
    print("\n\n")

    # GPT_test
    print("\n" + "#"*80)
    print("# EXPERIMENT 1: GPT_TEST")
    print("#"*80)

    # Temporarily change config to gpt_test
    original_experiment = config.experiment.experiment_name
    config.experiment.experiment_name = "gpt_test_wx-lv2"
    config.experiment.transcript_truth = "gpt_test_truth.csv"

    errors_gpt1 = analyze_errors("gpt_test_wx-lv2")

    # GPT_test2
    print("\n\n")
    print("\n" + "#"*80)
    print("# EXPERIMENT 2: GPT_TEST2")
    print("#"*80)

    config.experiment.experiment_name = "gpt_test2_wx-lv2"
    config.experiment.transcript_truth = "gpt_test2_truth.csv"

    errors_gpt2 = analyze_errors("gpt_test2_wx-lv2")

    # Restore
    config.experiment.experiment_name = original_experiment

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)