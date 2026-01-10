"""
Show the actual words that were transcribed incorrectly.
Displays substitutions and insertions with context.
"""
import sys
from pathlib import Path
import pandas as pd
from jiwer import process_words
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config

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

    # Replace hyphens/dashes with spaces (to preserve word boundaries)
    # This makes "decision-making" → "decision making" instead of "decisionmaking"
    normalized = re.sub(r'[\-—–]+', ' ', normalized)

    # Remove other punctuation
    punctuation_pattern = r'[,\.;:!?\"\(\)\[\]\{\}…]'
    normalized = re.sub(punctuation_pattern, '', normalized)

    # Clean up whitespace and lowercase
    normalized = re.sub(r'\s+', ' ', normalized).strip().lower()

    return normalized


def show_errors(experiment_name, truth_file, max_errors=100, save_to_file=True):
    """Show actual error words with context"""
    output_lines = []

    def print_and_save(text):
        """Print to console and save to output list"""
        print(text)
        output_lines.append(text)

    print_and_save("\n" + "="*80)
    print_and_save(f"ACTUAL WORD ERRORS: {experiment_name}")
    print_and_save("="*80)

    # Paths
    output_folder = PROJECT_PATH / config.paths.output_dirs['whisperx'] / experiment_name
    model_name = config.models.whisperx.model
    device = config.get_device()

    whisperx_csv = output_folder / f"{experiment_name}_{model_name}_{device}.csv"
    truth_path = PROJECT_PATH / config.paths.transcripts / truth_file

    # Load data
    truth_df = pd.read_csv(truth_path)
    whisperx_df = pd.read_csv(whisperx_csv)

    # Combine transcriptions
    truth_text = ' '.join(truth_df['Transcription'].dropna().astype(str))
    whisperx_text = ' '.join(whisperx_df['Transcription'].dropna().astype(str))

    # Normalize
    truth_normalized = normalize_text(truth_text)
    whisperx_normalized = normalize_text(whisperx_text)

    # Process with jiwer
    output = process_words(truth_normalized, whisperx_normalized)

    ref_words = truth_normalized.split()
    hyp_words = whisperx_normalized.split()

    # Extract errors
    substitutions = []
    insertions = []
    deletions = []

    # output.alignments is a list of lists - get the first (and only) list
    alignment_chunks = output.alignments[0] if output.alignments else []

    for chunk in alignment_chunks:
        chunk_type = chunk.type
        ref_start = chunk.ref_start_idx
        ref_end = chunk.ref_end_idx
        hyp_start = chunk.hyp_start_idx
        hyp_end = chunk.hyp_end_idx

        if chunk_type == 'substitute':
            ref_word = ' '.join(ref_words[ref_start:ref_end])
            hyp_word = ' '.join(hyp_words[hyp_start:hyp_end])

            # Get context (3 words before and after)
            context_start = max(0, ref_start - 3)
            context_end = min(len(ref_words), ref_end + 3)
            context = ' '.join(ref_words[context_start:context_end])

            substitutions.append({
                'truth': ref_word,
                'whisperx': hyp_word,
                'context': context
            })

        elif chunk_type == 'insert':
            hyp_word = ' '.join(hyp_words[hyp_start:hyp_end])

            # Get context from hypothesis
            context_start = max(0, hyp_start - 3)
            context_end = min(len(hyp_words), hyp_end + 3)
            context = ' '.join(hyp_words[context_start:context_end])

            insertions.append({
                'word': hyp_word,
                'context': context
            })

        elif chunk_type == 'delete':
            ref_word = ' '.join(ref_words[ref_start:ref_end])

            # Get context
            context_start = max(0, ref_start - 3)
            context_end = min(len(ref_words), ref_end + 3)
            context = ' '.join(ref_words[context_start:context_end])

            deletions.append({
                'word': ref_word,
                'context': context
            })

    # Display results
    print_and_save(f"\nSUMMARY:")
    print_and_save(f"  Substitutions: {len(substitutions)}")
    print_and_save(f"  Insertions:    {len(insertions)}")
    print_and_save(f"  Deletions:     {len(deletions)}")

    # Show substitutions
    if substitutions:
        print_and_save("\n" + "-"*80)
        print_and_save("SUBSTITUTIONS (wrong words):")
        print_and_save("-"*80)
        for i, sub in enumerate(substitutions[:max_errors], 1):
            print_and_save(f"\n{i}. TRUTH: '{sub['truth']}' -> WHISPERX: '{sub['whisperx']}'")
            print_and_save(f"   Context: ...{sub['context']}...")

    # Show insertions
    if insertions:
        print_and_save("\n" + "-"*80)
        print_and_save("INSERTIONS (extra words added by WhisperX):")
        print_and_save("-"*80)
        for i, ins in enumerate(insertions[:max_errors], 1):
            print_and_save(f"\n{i}. EXTRA: '{ins['word']}'")
            print_and_save(f"   Context: ...{ins['context']}...")

    # Show deletions
    if deletions:
        print_and_save("\n" + "-"*80)
        print_and_save("DELETIONS (words missing from WhisperX):")
        print_and_save("-"*80)
        for i, dele in enumerate(deletions[:max_errors], 1):
            print_and_save(f"\n{i}. MISSING: '{dele['word']}'")
            print_and_save(f"   Context: ...{dele['context']}...")

    # Save to file if requested
    if save_to_file:
        output_folder = PROJECT_PATH / config.paths.output_dirs['whisperx'] / experiment_name
        error_report_file = output_folder / f"{experiment_name}_error_analysis.txt"
        with open(error_report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        print_and_save(f"\nError analysis saved to: {error_report_file}")

    return {
        'substitutions': substitutions,
        'insertions': insertions,
        'deletions': deletions
    }


if __name__ == "__main__":
    print("\n" + "#"*80)
    print("DETAILED WORD ERROR ANALYSIS")
    print("#"*80)

    # Analyze GPT_test
    print("\n\n>>> ANALYZING GPT_TEST <<<")
    errors1 = show_errors("gpt_test_wx-lv2", "gpt_test_truth.csv", max_errors=30)

    # Analyze GPT_test2
    print("\n\n>>> ANALYZING GPT_TEST2 <<<")
    errors2 = show_errors("gpt_test2_wx-lv2", "gpt_test2_truth.csv", max_errors=30)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nShowing first 30 errors of each type per experiment.")
    print("Most errors are INSERTIONS (extra words WhisperX added).")