"""
Prepare simple dataset for Whisper fine-tuning (without Audio feature).

This version avoids audio backend conflicts by storing just file paths
and loading audio during training.

Usage:
    python src/prepare_hf_dataset_simple.py
"""

import sys
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config

config = Config()
PROJECT_PATH = Path(__file__).parent.parent


def load_aligned_data(json_path: Path):
    """Load aligned training data from JSON file."""
    print(f"\nLoading aligned data: {json_path.name}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"  Segments: {data['num_segments']}")
    print(f"  Words: {data['num_words']}")
    print(f"  Duration: {data['duration_seconds']:.2f}s")

    return data


def prepare_simple_dataset(aligned_files: list):
    """
    Prepare simple dataset without Audio feature.

    Returns list of examples with audio paths and text.
    """
    print("\nPreparing simple dataset...")

    examples = []

    for aligned_file in aligned_files:
        data = load_aligned_data(aligned_file)

        audio_path = data['audio_file']

        # Extract each segment as a training example
        for segment in data['segments']:
            examples.append({
                'audio_path': audio_path,
                'text': segment['text'].strip(),
                'start': segment.get('start', 0.0),
                'end': segment.get('end', 0.0)
            })

    print(f"\n  Total examples: {len(examples)}")
    print(f"  Unique audio files: {len(set(ex['audio_path'] for ex in examples))}")

    return examples


def save_simple_dataset(examples: list, output_dir: Path):
    """Save dataset as simple JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "training_data.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f"\nOK: Saved dataset to: {output_file}")
    print(f"    Format: List of {len(examples)} examples")
    print(f"    Fields: audio_path, text, start, end")

    return output_file


if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("PREPARE SIMPLE DATASET FOR WHISPER FINE-TUNING")
    print("#" * 80)

    # Define paths
    training_data_dir = PROJECT_PATH / "training_data"
    output_dir = PROJECT_PATH / "datasets" / "whisper_finetuning_simple"

    # Find all aligned JSON files
    aligned_files = list(training_data_dir.glob("*_aligned.json"))

    if not aligned_files:
        print("\nERROR: No aligned training data found!")
        print(f"   Expected files in: {training_data_dir}")
        print("   Run src/create_training_data.py first to generate aligned data.")
        sys.exit(1)

    print(f"\nFound {len(aligned_files)} aligned data file(s):")
    for f in aligned_files:
        print(f"  - {f.name}")

    # Prepare simple dataset
    examples = prepare_simple_dataset(aligned_files)

    # Save dataset
    output_file = save_simple_dataset(examples, output_dir)

    # Summary
    print("\n" + "=" * 80)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 80)
    print(f"\nDataset saved to: {output_file}")
    print("\nNext steps:")
    print("  1. Run src/finetune_whisper_simple.py to start fine-tuning")
    print("  2. Monitor with: tensorboard --logdir models/whisper-finetuned-interviews/logs")