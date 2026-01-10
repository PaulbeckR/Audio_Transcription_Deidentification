"""
Builds clean transcript file = xlsx or doc. 

This script tests WhisperX with forced alignment on hamlet_test.m4a.wav
and compares results with the baseline Whisper approach.


Output:
- Word-level aligned transcript with speaker labels
"""

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path
import time


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load configuration
from src.config import Config
config = Config()

# Add FFmpeg to PATH from config
FFMPEG_PATH = config.get_ffmpeg_path()
if FFMPEG_PATH and FFMPEG_PATH not in os.environ.get("PATH", ""):
    os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ.get("PATH", "")

import torch
import pandas as pd

# Get paths from config
PROJECT_PATH = Path(__file__).parent.parent

# Get model settings from config
MODEL = config.models.whisperx.model
model_name = MODEL

# Detect device from config
device_type = config.get_device()
device_label = device_type.upper()

# Output files (with whisperx-specific names)

# Output folder from config (use whisperx output directory)
OUTPUT_FOLDER = PROJECT_PATH / config.paths.output_dirs['whisperx'] / config.experiment.experiment_name
OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)

INPUT_CSV = OUTPUT_FOLDER / f"{config.experiment.experiment_name}_{model_name}_{device_type}.csv"
OUTPUT_TXT = OUTPUT_FOLDER / f"{config.experiment.experiment_name}_{model_name}_{device_type}_clean.txt"


print("="*80)
print(f"BUILD CLEAN TRANSCRIPT - {config.experiment.experiment_name} ({device_label})")
print("="*80)
print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Input CSV: {INPUT_CSV}")
print(f"Output TXT: {OUTPUT_TXT}")

# Takes CSV File, groups transcript data by speaker, and outputs a clean transcript file
def build_clean_transcript(input_csv: Path, output_file: Path):
    """
    Build a clean transcript file from WhisperX CSV output.
    Groups consecutive segments by the same speaker.

    Args:
        input_csv: Path to WhisperX CSV output (Speaker, Start Time, End Time, Duration, Transcription)
        output_file: Path to output text file
    """
    # Read CSV into DataFrame
    df = pd.read_csv(input_csv)

    # Open output file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        current_speaker = None
        speaker_text = []  # Accumulate text for current speaker

        for index, row in df.iterrows():
            speaker = row['Speaker']
            transcription = row['Transcription'].strip()  # Remove leading/trailing whitespace

            if speaker != current_speaker:
                # Write accumulated text from previous speaker
                if current_speaker is not None and speaker_text:
                    f_out.write(f"{current_speaker}:\n")
                    f_out.write(' '.join(speaker_text))
                    f_out.write("\n\n")  # Double newline between speakers
                    speaker_text = []

                current_speaker = speaker

            # Add transcription to current speaker's text
            speaker_text.append(transcription)

        # Write final speaker's text
        if current_speaker is not None and speaker_text:
            f_out.write(f"{current_speaker}:\n")
            f_out.write(' '.join(speaker_text))
            f_out.write("\n")

    print(f"\nClean transcript built successfully: {output_file}")
    return output_file


# Main execution
if __name__ == "__main__":
    # Check if input CSV exists
    if not INPUT_CSV.exists():
        print(f"\nError: Input CSV not found: {INPUT_CSV}")
        print("Please run test_whisperx.py first to generate the transcript.")
        sys.exit(1)

    # Build clean transcript
    print("\nBuilding clean transcript...")
    output_path = build_clean_transcript(INPUT_CSV, OUTPUT_TXT)

    print("\n" + "="*80)
    print("BUILD COMPLETE")
    print("="*80)
    print(f"\nClean transcript saved to:")
    print(f"  {output_path}")
    print(f"\nYou can now open this file to read the grouped transcript.")
