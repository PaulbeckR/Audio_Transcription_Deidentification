"""
WhisperX Test Script for hamlet_test

This script tests WhisperX with forced alignment on hamlet_test.m4a.wav
and compares results with the baseline Whisper approach.

Important: Run this with .venv_whisperx environment!
Usage: .venv_whisperx\Scripts\python.exe test_whisperx.py

Output:
- Word-level aligned transcript with speaker labels
- Timing and performance metrics
- JSON output for comparison
"""

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path
import time

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# CRITICAL: Set environment variable to force weights_only=False for PyTorch 2.6+
# This fixes the "Unsupported global: GLOBAL omegaconf.listconfig.ListConfig" error
# when loading Pyannote/WhisperX models
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

# Load configuration
from src.config import Config
config = Config()

# Add FFmpeg to PATH from config
FFMPEG_PATH = config.get_ffmpeg_path()
if FFMPEG_PATH and FFMPEG_PATH not in os.environ.get("PATH", ""):
    os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ.get("PATH", "")

import torch
import pandas as pd

# Monkey-patch torch.load to force weights_only=False for WhisperX/Pyannote compatibility
from src.torch_utils import patch_torch_load
patch_torch_load()

# Monkey-patch for torchaudio compatibility with WhisperX
# WhisperX/faster-whisper expects old torchaudio API
import torchaudio
if not hasattr(torchaudio, 'AudioMetaData'):
    from collections import namedtuple
    AudioMetaData = namedtuple('AudioMetaData', [
        'sample_rate', 'num_frames', 'num_channels', 'bits_per_sample', 'encoding'
    ])
    torchaudio.AudioMetaData = AudioMetaData

if not hasattr(torchaudio, 'list_audio_backends'):
    # Provide a dummy function for list_audio_backends
    def list_audio_backends():
        return ['soundfile', 'sox_io']
    torchaudio.list_audio_backends = list_audio_backends

if not hasattr(torchaudio, 'get_audio_backend'):
    # Provide a dummy function for get_audio_backend
    def get_audio_backend():
        return 'soundfile'
    torchaudio.get_audio_backend = get_audio_backend

import whisperx
from whisperx.diarize import DiarizationPipeline


# Get model settings from config
MODEL = config.models.whisperx.model
model_name = MODEL

# Get paths from config
PROJECT_PATH = Path(__file__).parent
AUDIO_FILE = PROJECT_PATH / config.paths.audio_files / config.experiment.audio_file

# Ground truth is optional (may be None/null)
if config.experiment.transcript_truth:
    GROUND_TRUTH = PROJECT_PATH / config.paths.transcripts / config.experiment.transcript_truth
else:
    GROUND_TRUTH = None

# Output folder from config (use whisperx output directory)
OUTPUT_FOLDER = PROJECT_PATH / config.paths.output_dirs['whisperx'] / config.experiment.experiment_name
OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)

# Detect device from config
device_type = config.get_device()
device_label = device_type.upper()

# Output files (with whisperx-specific names)

OUTPUT_TRANSCRIPT = OUTPUT_FOLDER / f"{config.experiment.experiment_name}_{model_name}_{device_type}.json"
OUTPUT_CSV = OUTPUT_FOLDER / f"{config.experiment.experiment_name}_{model_name}_{device_type}.csv"
OUTPUT_METRICS = OUTPUT_FOLDER / f"{config.experiment.experiment_name}_metrics_{model_name}_{device_type}.json"

print("="*80)
print(f"WHISPERX TEST - {config.experiment.experiment_name} ({device_label})")
print("="*80)
print(f"\nTest Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nInput Audio: {AUDIO_FILE}")
print(f"Ground Truth: {GROUND_TRUTH}")
print(f"Output JSON: {OUTPUT_TRANSCRIPT}")
print(f"Output CSV: {OUTPUT_CSV}")
print(f"Output Metrics: {OUTPUT_METRICS}")

# Verify input file exists
if not AUDIO_FILE.exists():
    print(f"\nERROR: Audio file not found: {AUDIO_FILE}")
    sys.exit(1)

# for WhisperX, we need a ground truth file for WER calculation
if GROUND_TRUTH and GROUND_TRUTH.exists():
    has_ground_truth = True
else:
    if GROUND_TRUTH:
        print(f"\nWARNING: Ground truth file not found: {GROUND_TRUTH}")
    else:
        print(f"\nWARNING: No ground truth file configured")
    print("WER calculation will be skipped.")
    has_ground_truth = False

# Get audio duration
from pydub import AudioSegment
audio = AudioSegment.from_file(str(AUDIO_FILE))
audio_duration = len(audio) / 1000.0  # Convert to seconds
print(f"\nAudio Duration: {audio_duration:.2f} seconds ({audio_duration/60:.2f} minutes)")

from src.metrics import (
    PerformanceTimer,
    SystemMetrics,
    calculate_wer_from_files,
    save_metrics_json
)

# Initialize metrics
all_metrics = {
    'test_info': {
        'test_name': f'{device_label} WhisperX {model_name} - {config.experiment.experiment_name}',
        'date': datetime.now().isoformat(),
        'audio_file': str(AUDIO_FILE),
        'ground_truth': str(GROUND_TRUTH) if (GROUND_TRUTH and GROUND_TRUTH.exists()) else None,
        'audio_duration_seconds': audio_duration,
        'device': device_type,
        'model': model_name,
        'whisperx_version': '3.7.4',
        'pytorch_version': torch.__version__,
    },
    'timings': {},
    'accuracy': {},  # For WER calculation
}

# Check GPU
print("\n" + "="*80)
print("STEP 1: Device Detection")
print("="*80)
if torch.cuda.is_available():
    print(f"Device: {device_type}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"PyTorch Version: {torch.__version__}")
else:
    print(f"Device: CPU")
    print("No GPU available - running on CPU")

# ============================================================================
# STEP 2: Load WhisperX Model
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Loading WhisperX Model")
print("="*80)

start_time = time.time()
print(f"Loading Whisper model ({model_name})...")

# Get compute type and batch size from config
compute_type = config.models.whisperx.compute_type
batch_size = config.models.whisperx.batch_size

try:
    model = whisperx.load_model(
        model_name,
        device=device_type,
        compute_type=compute_type,
        language=config.models.whisperx.language
    )
    load_time = time.time() - start_time
    all_metrics['timings']['model_load'] = load_time
    print(f"Model loaded in {load_time:.2f} seconds")
except Exception as e:
    print(f"ERROR loading model: {e}")
    sys.exit(1)

# ============================================================================
# STEP 3: Transcribe Audio
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Transcribing Audio")
print("="*80)

start_time = time.time()
print(f"Transcribing {AUDIO_FILE.name}...")
print(f"Batch size: {batch_size}, Compute type: {compute_type}")

try:
    result = model.transcribe(
        str(AUDIO_FILE),
        batch_size=batch_size
    )
    transcribe_time = time.time() - start_time
    all_metrics['timings']['transcription'] = transcribe_time

    print(f"Transcription complete in {transcribe_time:.2f} seconds")
    print(f"Number of segments: {len(result['segments'])}")

    # Show sample
    if result['segments']:
        print(f"\nSample segment:")
        sample = result['segments'][0]
        print(f"  Time: {sample['start']:.2f}s - {sample['end']:.2f}s")
        print(f"  Text: {sample['text']}")

except Exception as e:
    print(f"ERROR during transcription: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STEP 4: Align Whisper Output
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Forced Alignment")
print("="*80)

start_time = time.time()
print("Loading alignment model...")

try:
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device_type
    )

    print("Aligning transcript to audio...")
    result_aligned = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        str(AUDIO_FILE),
        device=device_type,
        return_char_alignments=config.processing.alignment.return_char_alignments
    )

    align_time = time.time() - start_time
    all_metrics['timings']['alignment'] = align_time

    print(f"Alignment complete in {align_time:.2f} seconds")
    print(f"Aligned segments: {len(result_aligned['segments'])}")

    # Show sample with word-level timestamps
    if result_aligned['segments'] and 'words' in result_aligned['segments'][0]:
        print(f"\nSample aligned segment with word timestamps:")
        sample = result_aligned['segments'][0]
        print(f"  Segment: {sample['start']:.2f}s - {sample['end']:.2f}s")
        print(f"  Text: {sample['text']}")
        if sample.get('words'):
            print(f"  Words: {len(sample['words'])} words")
            for word in sample['words'][:5]:  # Show first 5 words
                print(f"    '{word['word']}': {word.get('start', 'N/A'):.2f}s - {word.get('end', 'N/A'):.2f}s")

except Exception as e:
    print(f"ERROR during alignment: {e}")
    import traceback
    traceback.print_exc()
    result_aligned = result  # Fall back to unaligned
    all_metrics['timings']['alignment'] = 0

# ============================================================================
# STEP 5: Diarization (Speaker Assignment)
# ============================================================================
print("\n" + "="*80)
print("STEP 5: Speaker Diarization")
print("="*80)

start_time = time.time()
print("Loading diarization model...")

try:
    diarize_model = DiarizationPipeline(
        use_auth_token=config.get_hf_token(),
        device=device_type
    )

    print("Performing diarization...")
    diarize_segments = diarize_model(str(AUDIO_FILE))

    print("Assigning speakers to segments...")
    result_diarized = whisperx.assign_word_speakers(
        diarize_segments,
        result_aligned
    )

    diarize_time = time.time() - start_time
    all_metrics['timings']['diarization'] = diarize_time

    print(f"Diarization complete in {diarize_time:.2f} seconds")

    # Count speakers
    speakers = set()
    for segment in result_diarized['segments']:
        if 'speaker' in segment:
            speakers.add(segment['speaker'])
    print(f"Detected {len(speakers)} speakers: {sorted(speakers)}")

    # Show sample with speaker labels
    if result_diarized['segments']:
        print(f"\nSample with speaker labels:")
        for seg in result_diarized['segments'][:3]:
            speaker = seg.get('speaker', 'UNKNOWN')
            print(f"  [{speaker}] {seg['start']:.2f}s: {seg['text'][:50]}...")

except Exception as e:
    print(f"ERROR during diarization: {e}")
    import traceback
    traceback.print_exc()
    result_diarized = result_aligned  # Fall back to aligned without speakers
    all_metrics['timings']['diarization'] = 0

# ============================================================================
# STEP 6: Save Results
# ============================================================================
print("\n" + "="*80)
print("STEP 6: Saving Results")
print("="*80)

# Save full JSON output
print(f"Saving JSON to {OUTPUT_TRANSCRIPT}...")
with open(OUTPUT_TRANSCRIPT, 'w', encoding='utf-8') as f:
    json.dump(result_diarized, f, indent=2, ensure_ascii=False)

# Convert to CSV format for comparison with baseline
print(f"Converting to CSV format...")
csv_rows = []
for segment in result_diarized['segments']:
    speaker = segment.get('speaker', 'UNKNOWN')
    start = segment['start']
    end = segment['end']
    duration = end - start
    text = segment['text'].strip()

    csv_rows.append({
        'Speaker': speaker,
        'Start Time': start,
        'End Time': end,
        'Duration': duration,
        'Transcription': text
    })

df = pd.DataFrame(csv_rows)
df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
print(f"Saved CSV to {OUTPUT_CSV}")
print(f"Total segments: {len(csv_rows)}")

# Calculate total time and performance metrics
total_time = sum([
    all_metrics['timings'].get('model_load', 0),
    all_metrics['timings'].get('transcription', 0),
    all_metrics['timings'].get('alignment', 0),
    all_metrics['timings'].get('diarization', 0)
])
all_metrics['timings']['total'] = total_time

all_metrics['performance'] = {
    'real_time_factor': total_time / audio_duration if audio_duration > 0 else 0,
    'processing_speed': f"{audio_duration / total_time:.2f}x realtime" if total_time > 0 else "N/A"
}

all_metrics['output_stats'] = {
    'num_segments': len(csv_rows),
    'num_speakers': len(speakers) if 'speakers' in locals() else 0,
    'total_words': sum(len(seg.get('words', [])) for seg in result_diarized['segments']),
}


# Calculate WER for base model
if has_ground_truth:
    print("\nCalculating WER for model...")
    try:
        wer_base = calculate_wer_from_files(
            str(GROUND_TRUTH),
            str(OUTPUT_CSV),
            text_column='Transcription'
        )
        all_metrics['accuracy']['whisper_model'] = wer_base
        print(f"  WER: {wer_base['wer']:.2%}")
        print(f"  Substitutions: {wer_base['substitutions']}")
        print(f"  Deletions: {wer_base['deletions']}")
        print(f"  Insertions: {wer_base['insertions']}")
    except Exception as e:
        import traceback
        print(f"  Error calculating WER: {e}")
        traceback.print_exc()

# Save metrics
print(f"Saving metrics to {OUTPUT_METRICS}...")
with open(OUTPUT_METRICS, 'w', encoding='utf-8') as f:
    json.dump(all_metrics, f, indent=2)

# ============================================================================
# STEP 7: Summary
# ============================================================================
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print(f"\nTiming Breakdown:")
print(f"  Model Load:     {all_metrics['timings'].get('model_load', 0):8.2f}s")
print(f"  Transcription:  {all_metrics['timings'].get('transcription', 0):8.2f}s")
print(f"  Alignment:      {all_metrics['timings'].get('alignment', 0):8.2f}s")
print(f"  Diarization:    {all_metrics['timings'].get('diarization', 0):8.2f}s")
print(f"  -------------------------")
print(f"  Total:          {total_time:8.2f}s ({total_time/60:.2f} min)")

print(f"\nPerformance:")
print(f"  Audio Duration:      {audio_duration:8.2f}s ({audio_duration/60:.2f} min)")
print(f"  Processing Time:     {total_time:8.2f}s ({total_time/60:.2f} min)")
print(f"  Real-Time Factor:    {all_metrics['performance']['real_time_factor']:.2f}x")
print(f"  Speed:               {all_metrics['performance']['processing_speed']}")

print(f"\nOutput Statistics:")
print(f"  Segments:   {all_metrics['output_stats']['num_segments']}")
print(f"  Speakers:   {all_metrics['output_stats']['num_speakers']}")
print(f"  Words:      {all_metrics['output_stats']['total_words']}")

print(f"\nFiles Generated:")
print(f"  [OK] {OUTPUT_TRANSCRIPT}")
print(f"  [OK] {OUTPUT_CSV}")
print(f"  [OK] {OUTPUT_METRICS}")


