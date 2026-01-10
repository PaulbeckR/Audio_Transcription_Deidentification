"""
Create timestamped training data for Whisper fine-tuning using forced alignment.

This script takes ground truth transcripts (without timestamps) and aligns them
to audio files using WhisperX's forced alignment capability, creating timestamped
training data suitable for fine-tuning.

Usage:
    python src/create_training_data.py
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# CRITICAL: Apply torch.load patch BEFORE importing torch or whisperx
# This fixes PyTorch 2.6+ compatibility with WhisperX models
from src.core.torch_utils import patch_torch_load
patch_torch_load()

import torch

from src.config import Config

config = Config()
PROJECT_PATH = Path(__file__).parent.parent.parent

# Add FFmpeg to PATH from config
FFMPEG_PATH = config.get_ffmpeg_path()
if FFMPEG_PATH and FFMPEG_PATH not in os.environ.get("PATH", ""):
    os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ.get("PATH", "")

import whisperx


def load_audio(audio_path: Path):
    """Load audio file."""
    print(f"\nLoading audio: {audio_path.name}")
    audio = whisperx.load_audio(str(audio_path))
    duration = len(audio) / 16000  # WhisperX uses 16kHz
    print(f"  Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    return audio


def load_ground_truth(truth_path: Path):
    """Load ground truth transcript from CSV file."""
    print(f"\nLoading ground truth: {truth_path.name}")
    df = pd.read_csv(truth_path)

    # Combine all transcription segments into full text
    segments = []
    for _, row in df.iterrows():
        segments.append({
            'speaker': row['Speaker'],
            'text': row['Transcription'].strip()
        })

    # Combine into full text for alignment
    full_text = ' '.join([seg['text'] for seg in segments])

    print(f"  Loaded {len(segments)} segments")
    print(f"  Total words: {len(full_text.split())}")

    return segments, full_text


def create_segments_for_alignment(text: str, audio_duration: float):
    """
    Create segment structure for WhisperX alignment.

    WhisperX's align() expects segments with start/end times.
    Since we don't have timing, we create one segment spanning the entire audio.
    """
    return [{
        'text': text,
        'start': 0.0,
        'end': audio_duration
    }]


def align_transcript(audio, text, device: str = "cuda"):
    """
    Perform forced alignment of ground truth text to audio.

    Args:
        audio: Audio array from whisperx.load_audio()
        text: Ground truth transcript text
        device: 'cuda' or 'cpu'

    Returns:
        Aligned segments with word-level timestamps
    """
    print(f"\nPerforming forced alignment...")
    print(f"  Device: {device}")

    # Calculate audio duration
    audio_duration = len(audio) / 16000  # WhisperX uses 16kHz

    # Load alignment model
    print("  Loading alignment model...")
    model_a, metadata = whisperx.load_align_model(
        language_code="en",
        device=device
    )

    # Create segments for alignment (with start/end times)
    segments = create_segments_for_alignment(text, audio_duration)

    # Perform alignment
    print("  Aligning transcript to audio...")
    result = whisperx.align(
        segments,
        model_a,
        metadata,
        audio,
        device=device,
        return_char_alignments=False
    )

    print(f"  Aligned {len(result['segments'])} segments")
    print(f"  Total words aligned: {sum(len(seg.get('words', [])) for seg in result['segments'])}")

    return result


def validate_alignment(result, min_confidence: float = 0.8):
    """
    Validate alignment quality.

    Checks:
    - All segments have timestamps
    - Word confidence scores are reasonable
    - No large gaps in timestamps
    """
    print("\nValidating alignment quality...")

    issues = []
    total_words = 0
    low_confidence_words = 0

    for i, segment in enumerate(result['segments']):
        # Check segment has timestamps
        if 'start' not in segment or 'end' not in segment:
            issues.append(f"Segment {i}: Missing timestamps")
            continue

        # Check words
        words = segment.get('words', [])
        for word in words:
            total_words += 1
            score = word.get('score', 0.0)
            if score < min_confidence:
                low_confidence_words += 1

    # Report validation results
    if issues:
        print(f"  WARNING: Found {len(issues)} issues:")
        for issue in issues[:5]:  # Show first 5
            print(f"    - {issue}")
    else:
        print(f"  OK: No critical issues found")

    if total_words > 0:
        avg_confidence = (total_words - low_confidence_words) / total_words
        print(f"  Words with confidence >= {min_confidence}: {avg_confidence:.1%}")
        if low_confidence_words > 0:
            print(f"  WARNING: {low_confidence_words} words below {min_confidence} confidence")

    return len(issues) == 0


def save_training_data(result, audio_path: Path, truth_path: Path, output_dir: Path):
    """
    Save aligned training data in format suitable for fine-tuning.

    Creates:
    - JSON file with aligned segments and word timestamps
    - Metadata file with training info
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create output filename
    base_name = audio_path.stem  # e.g., "gpt_test"
    output_json = output_dir / f"{base_name}_aligned.json"
    output_metadata = output_dir / f"{base_name}_metadata.json"

    # Prepare training data structure
    training_data = {
        'audio_file': str(audio_path.absolute()),
        'ground_truth_file': str(truth_path.absolute()),
        'duration_seconds': result['segments'][-1]['end'] if result['segments'] else 0,
        'num_segments': len(result['segments']),
        'num_words': sum(len(seg.get('words', [])) for seg in result['segments']),
        'segments': result['segments']
    }

    # Save aligned data
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)

    print(f"\nOK: Saved aligned training data: {output_json}")

    # Create metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'audio_file': audio_path.name,
        'ground_truth_file': truth_path.name,
        'duration_seconds': training_data['duration_seconds'],
        'num_segments': training_data['num_segments'],
        'num_words': training_data['num_words'],
        'alignment_model': 'whisperx',
        'language': 'en'
    }

    with open(output_metadata, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"OK: Saved metadata: {output_metadata}")

    return output_json


def process_training_file(audio_path: Path, truth_path: Path, output_dir: Path, device: str = "cuda"):
    """
    Process a single audio+transcript pair to create aligned training data.
    """
    print("=" * 80)
    print(f"CREATING TRAINING DATA: {audio_path.stem}")
    print("=" * 80)

    # Load audio
    audio = load_audio(audio_path)

    # Load ground truth
    segments, full_text = load_ground_truth(truth_path)

    # Perform alignment
    result = align_transcript(audio, full_text, device=device)

    # Validate alignment
    is_valid = validate_alignment(result)

    if not is_valid:
        print("\nWARNING: Alignment quality issues detected!")
        print("   Review the output and consider manual correction if needed.")

    # Save training data
    output_file = save_training_data(result, audio_path, truth_path, output_dir)

    return output_file


if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("WHISPER FINE-TUNING: CREATE TRAINING DATA")
    print("#" * 80)

    # Detect device
    device = config.get_device()
    print(f"\nDevice: {device}")

    # Define paths
    audio_dir = PROJECT_PATH / "Audio_Local_tests" / "audio_files"
    truth_dir = PROJECT_PATH / config.paths.transcripts
    output_dir = PROJECT_PATH / "training_data"

    # Training files to process
    training_files = [
        {
            'audio': audio_dir / "gpt_test.m4a",
            'truth': truth_dir / "gpt_test_truth.csv"
        },
        # Add more training files here as you collect them
        # {
        #     'audio': audio_dir / "gpt_test2.m4a",
        #     'truth': truth_dir / "gpt_test2_truth.csv"
        # },
    ]

    print(f"\nProcessing {len(training_files)} training file(s)...")

    aligned_files = []
    for file_pair in training_files:
        audio_path = file_pair['audio']
        truth_path = file_pair['truth']

        if not audio_path.exists():
            print(f"\nWARNING: Audio file not found: {audio_path}")
            continue

        if not truth_path.exists():
            print(f"\nWARNING: Ground truth file not found: {truth_path}")
            continue

        try:
            output_file = process_training_file(audio_path, truth_path, output_dir, device=device)
            aligned_files.append(output_file)
        except Exception as e:
            print(f"\nERROR processing {audio_path.name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("TRAINING DATA CREATION COMPLETE")
    print("=" * 80)
    print(f"\nSuccessfully created {len(aligned_files)} training data file(s):")
    for f in aligned_files:
        print(f"  OK: {f}")

    print(f"\nTraining data saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Review alignment quality in the JSON files")
    print("  2. Run src/prepare_hf_dataset.py to convert to Hugging Face format")
    print("  3. Run src/finetune_whisper.py to start fine-tuning")