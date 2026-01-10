"""
Check transcript files for correct format before data preparation.

This script validates that your transcript CSV files have the required format
for fine-tuning preparation.

Usage:
    python check_transcripts.py

Requirements:
    - Transcripts should be CSV files
    - Must have 'Speaker' and 'Transcription' columns
    - Files should be in training_data/raw/interviews/transcripts/
"""

import pandas as pd
from pathlib import Path
import sys

def check_transcript_file(csv_path: Path):
    """
    Check a single transcript file for correct format.

    Args:
        csv_path: Path to CSV file

    Returns:
        Tuple of (success: bool, message: str, stats: dict)
    """
    try:
        # Try to read CSV
        df = pd.read_csv(csv_path)

        # Check required columns
        required_cols = ['Speaker', 'Transcription']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            return False, f"Missing columns: {missing_cols}. Has: {list(df.columns)}", None

        # Check for empty data
        if len(df) == 0:
            return False, "File is empty", None

        # Get statistics
        num_segments = len(df)
        speakers = df['Speaker'].unique().tolist()
        num_speakers = len(speakers)

        # Estimate word count
        word_count = df['Transcription'].str.split().str.len().sum()

        # Check for null values
        null_speakers = df['Speaker'].isnull().sum()
        null_transcriptions = df['Transcription'].isnull().sum()

        if null_speakers > 0 or null_transcriptions > 0:
            return False, f"Found null values: {null_speakers} speakers, {null_transcriptions} transcriptions", None

        stats = {
            'segments': num_segments,
            'speakers': speakers,
            'num_speakers': num_speakers,
            'words': word_count
        }

        return True, "Format OK", stats

    except pd.errors.EmptyDataError:
        return False, "File is empty or corrupted", None
    except pd.errors.ParserError as e:
        return False, f"CSV parsing error: {e}", None
    except Exception as e:
        return False, f"Error: {e}", None


def main():
    """Main function to check all transcripts."""
    print("=" * 80)
    print("TRANSCRIPT FORMAT CHECKER")
    print("=" * 80)

    # Check if directory exists
    transcript_dir = Path("training_data/raw/interviews/transcripts")

    if not transcript_dir.exists():
        print(f"\n❌ Directory not found: {transcript_dir}")
        print("\nPlease create the directory and copy your transcript files:")
        print(f"  mkdir -p {transcript_dir}")
        print(f"  cp /path/to/your/transcripts/*.csv {transcript_dir}/")
        sys.exit(1)

    # Get all CSV files
    csv_files = sorted(transcript_dir.glob("*.csv"))

    if not csv_files:
        print(f"\n❌ No CSV files found in {transcript_dir}")
        print("\nPlease copy your transcript files to this directory:")
        print(f"  cp /path/to/your/transcripts/*.csv {transcript_dir}/")
        sys.exit(1)

    print(f"\nFound {len(csv_files)} transcript files")
    print("=" * 80)

    # Check each file
    results = {
        'success': [],
        'failed': []
    }

    total_segments = 0
    total_words = 0
    all_speakers = set()

    for i, csv_file in enumerate(csv_files, 1):
        success, message, stats = check_transcript_file(csv_file)

        if success:
            results['success'].append(csv_file.name)
            total_segments += stats['segments']
            total_words += stats['words']
            all_speakers.update(stats['speakers'])

            print(f"\n✅ [{i}/{len(csv_files)}] {csv_file.name}")
            print(f"   Segments: {stats['segments']}")
            print(f"   Words: {stats['words']:,}")
            print(f"   Speakers: {stats['speakers']}")
        else:
            results['failed'].append((csv_file.name, message))
            print(f"\n❌ [{i}/{len(csv_files)}] {csv_file.name}")
            print(f"   Error: {message}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n✅ Successful: {len(results['success'])} files")
    print(f"❌ Failed: {len(results['failed'])} files")

    if results['success']:
        print(f"\nTotal segments: {total_segments:,}")
        print(f"Total words: {total_words:,}")
        print(f"Estimated duration: ~{total_words / 150 / 60:.1f} hours")
        print(f"  (assuming 150 words/minute average speech rate)")
        print(f"\nUnique speakers found: {sorted(all_speakers)}")

    if results['failed']:
        print("\n" + "-" * 80)
        print("FAILED FILES (need fixing):")
        print("-" * 80)
        for filename, error in results['failed']:
            print(f"\n❌ {filename}")
            print(f"   {error}")

    print("\n" + "=" * 80)

    if results['failed']:
        print("\n⚠️  Some files have errors. Please fix them before proceeding.")
        print("\nRequired format:")
        print("  - CSV file")
        print("  - Columns: 'Speaker', 'Transcription'")
        print("  - No null/empty values")
        print("\nExample:")
        print("  Speaker,Transcription")
        print('  Interviewer,"Question text here"')
        print('  Interviewee,"Response text here"')
        sys.exit(1)
    else:
        print("\n✅ All transcript files are correctly formatted!")
        print("\nNext steps:")
        print("  1. Verify audio files are in: training_data/raw/interviews/audio/")
        print("  2. Ensure audio and transcript filenames match")
        print("  3. Run data preparation:")
        print("\n     python src/data/create_training_data.py \\")
        print("       --audio-dir training_data/raw/interviews/audio \\")
        print("       --transcript-dir training_data/raw/interviews/transcripts \\")
        print("       --output training_data/aligned/interviews")


if __name__ == "__main__":
    main()
