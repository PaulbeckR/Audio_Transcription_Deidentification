"""
Prepare Hugging Face Dataset for Whisper fine-tuning.

This script converts the aligned training data (created by create_training_data.py)
into a Hugging Face Dataset format suitable for fine-tuning Whisper.

Usage:
    python src/prepare_hf_dataset.py
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# CRITICAL: Apply torch.load patch for consistency (datasets library may use torch internally)
# This fixes PyTorch 2.6+ compatibility issues
from src.core.torch_utils import patch_torch_load
patch_torch_load()

from src.config import Config

config = Config()
PROJECT_PATH = Path(__file__).parent.parent.parent


def load_aligned_data(json_path: Path):
    """Load aligned training data from JSON file."""
    print(f"\nLoading aligned data: {json_path.name}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"  Segments: {data['num_segments']}")
    print(f"  Words: {data['num_words']}")
    print(f"  Duration: {data['duration_seconds']:.2f}s")

    return data


def prepare_dataset_dict(aligned_files: list):
    """
    Prepare dataset dictionary for Hugging Face Dataset.

    Format expected by Whisper fine-tuning:
    {
        'audio': [audio_path, audio_path, ...],
        'sentence': [transcription, transcription, ...]
    }
    """
    print("\nPreparing Hugging Face dataset...")

    dataset_dict = {
        'audio': [],
        'sentence': []
    }

    total_segments = 0

    for aligned_file in aligned_files:
        data = load_aligned_data(aligned_file)

        audio_path = data['audio_file']

        # Extract each segment as a training example
        for segment in data['segments']:
            # Use full audio file path
            # Note: Hugging Face will handle audio loading and segmentation
            dataset_dict['audio'].append(audio_path)
            dataset_dict['sentence'].append(segment['text'].strip())
            total_segments += 1

    print(f"\n  Total training examples: {total_segments}")
    print(f"  Unique audio files: {len(set(dataset_dict['audio']))}")

    return dataset_dict


def save_dataset_dict(dataset_dict: dict, output_path: Path):
    """Save dataset dictionary as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_dict, f, indent=2, ensure_ascii=False)

    print(f"\nOK: Saved dataset dictionary: {output_path}")


def create_simple_json_dataset(dataset_dict: dict):
    """
    Create simple JSON dataset format (audio paths + text).

    This avoids HuggingFace Audio feature and torchcodec dependency.
    Audio is loaded on-the-fly during training using whisperx.

    Args:
        dataset_dict: Dict with 'audio' (paths) and 'sentence' (text) lists

    Returns:
        List of example dicts with 'audio_path' and 'text' keys
    """
    print("\nCreating simple JSON dataset (no Audio feature, no torchcodec)...")

    examples = []
    for audio_path, text in zip(dataset_dict['audio'], dataset_dict['sentence']):
        examples.append({
            'audio_path': str(audio_path),  # Store path as string
            'text': text
        })

    print(f"  OK: Created {len(examples)} examples")
    print(f"  Format: audio_path (string) + text (string)")
    print(f"  Audio loading: On-the-fly during training with whisperx")

    return examples


def split_simple_dataset(examples: list, train_ratio: float = 0.9):
    """
    Split example list into train and validation sets.

    Args:
        examples: List of example dicts
        train_ratio: Fraction for training (default: 0.9)

    Returns:
        Tuple of (train_examples, val_examples)
    """
    if not examples:
        return None, None

    total_size = len(examples)

    # For very small datasets (<100 examples), don't split
    if total_size < 100:
        print(f"\nWARNING: Dataset too small ({total_size} examples) - using all for training")
        return examples, None

    # Split
    split_idx = int(total_size * train_ratio)

    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    print(f"\nDataset split:")
    print(f"  Train: {len(train_examples)} examples")
    print(f"  Validation: {len(val_examples)} examples")

    return train_examples, val_examples


def save_simple_json(examples: list, output_dir: Path, name: str = "train"):
    """
    Save simple JSON dataset to disk.

    Args:
        examples: List of example dicts
        output_dir: Output directory
        name: Dataset split name (train/validation/test)
    """
    if not examples:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.json"

    print(f"\nSaving {name} dataset to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f"  OK: Saved {len(examples)} examples")


if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("PREPARE HUGGING FACE DATASET FOR WHISPER FINE-TUNING")
    print("#" * 80)

    # Define paths
    training_data_dir = PROJECT_PATH / "training_data"
    dataset_output_dir = PROJECT_PATH / "datasets" / "whisper_finetuning"

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

    # Prepare dataset dictionary
    dataset_dict = prepare_dataset_dict(aligned_files)

    # Save dataset dictionary (for reference)
    dict_output = dataset_output_dir / "dataset_dict.json"
    save_dataset_dict(dataset_dict, dict_output)

    # Create simple JSON dataset (avoids Audio feature / torchcodec)
    examples = create_simple_json_dataset(dataset_dict)

    # Split dataset (if large enough)
    train_examples, val_examples = split_simple_dataset(examples, train_ratio=0.9)

    # Save datasets as simple JSON
    save_simple_json(train_examples, dataset_output_dir, name="train")
    if val_examples is not None:
        save_simple_json(val_examples, dataset_output_dir, name="validation")

    # Summary
    print("\n" + "=" * 80)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 80)

    print(f"\nDataset saved to: {dataset_output_dir}")
    print(f"  Format: Simple JSON (audio_path + text)")
    print(f"  Train: {dataset_output_dir / 'train.json'}")
    if val_examples:
        print(f"  Validation: {dataset_output_dir / 'validation.json'}")

    print("\nNext steps:")
    print("  1. Install fine-tuning dependencies:")
    print("     pip install -r requirements_finetuning.txt")
    print("  2. Run training:")
    print("     python src/scripts/run_training.py --experiment baseline_lora")
    print("\nNote: Using custom dataset loader (no torchcodec dependency)")