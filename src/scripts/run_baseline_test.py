"""
CPU Baseline Test Script for hamlet_test

This script runs the full transcription pipeline on hamlet_test.m4a.wav
and collects comprehensive metrics for baseline performance.

Output:
- Transcript CSV file
- RTTM file
- Metrics JSON file
- Console output with timing and metrics
"""

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

# Load configuration
from src.config import Config
config = Config()

# Add FFmpeg to PATH from config
FFMPEG_PATH = config.get_ffmpeg_path()
if FFMPEG_PATH and FFMPEG_PATH not in os.environ.get("PATH", ""):
    os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ.get("PATH", "")

# Suppress torchaudio backend and AudioMetaData deprecation warnings from pyannote.audio
warnings.filterwarnings("ignore", category=UserWarning, message=".*torchaudio.*backend.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*AudioMetaData.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*speechbrain.pretrained.*deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only.*")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import whisper
from pyannote.audio import Pipeline
import torch
from pydub import AudioSegment
import pandas as pd
import csv

from src.evaluation.metrics import (
    PerformanceTimer,
    SystemMetrics,
    calculate_wer_from_files,
    save_metrics_json
)
from src.core.utils import get_device, get_audio_duration, format_time
from helper_functions import (
    convert_to_wav,
    milliseconds_until_sound,
    extract_end_time
)

# Get paths from config
PROJECT_PATH = Path(__file__).parent
AUDIO_FILE = PROJECT_PATH / config.paths.audio_files / "hamlet_test.m4a.wav"
GROUND_TRUTH = PROJECT_PATH / config.paths.transcripts / "hamlet_test_truth.xlsx"

# Output folders from config
WAV_FOLDER = PROJECT_PATH / config.paths.temp['wav_files']
RTTM_FOLDER = PROJECT_PATH / "Audio_Local_tests" / "rttm_files"  # Not in config yet
TEMP_AUDIO_CLIPS = PROJECT_PATH / config.paths.temp['audio_clips']
OUTPUT_FOLDER = PROJECT_PATH / config.paths.output_dirs['whisper'] / "hamlet_test"

# Create output folders
OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)
WAV_FOLDER.mkdir(exist_ok=True, parents=True)
RTTM_FOLDER.mkdir(exist_ok=True, parents=True)

# Get model and device from config
MODEL = config.models.whisper.model
device_type = "gpu" if torch.cuda.is_available() else "cpu"
device_label = device_type.upper()

# Output files (with device-specific names)
OUTPUT_TRANSCRIPT = OUTPUT_FOLDER / f"hamlet_test_whisper_{MODEL}_{device_type}.csv"
OUTPUT_RTTM = RTTM_FOLDER / f"hamlet_test_whisper_{MODEL}_{device_type}.rttm"
OUTPUT_METRICS = OUTPUT_FOLDER / f"whisper_metrics_{MODEL}_{device_type}.json"
OUTPUT_WAV = WAV_FOLDER / "hamlet_test_baseline.wav"

print("="*80)
print(f"{device_label} BASELINE TEST - HAMLET_TEST")
print("="*80)
print(f"\nTest Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nInput Audio: {AUDIO_FILE}")
print(f"Ground Truth: {GROUND_TRUTH}")
print(f"Output Transcript: {OUTPUT_TRANSCRIPT}")
print(f"Output Metrics: {OUTPUT_METRICS}")

# Verify input files exist
if not AUDIO_FILE.exists():
    print(f"\nERROR: Audio file not found: {AUDIO_FILE}")
    print("Please ensure hamlet_test.m4a.wav is in Audio_Local_tests/audio_files/")
    sys.exit(1)

if not GROUND_TRUTH.exists():
    print(f"\nWARNING: Ground truth file not found: {GROUND_TRUTH}")
    print("WER calculation will be skipped.")
    has_ground_truth = False
else:
    has_ground_truth = True

# Initialize metrics collection
all_metrics = {
    'test_info': {
        'test_name': f'{device_label} ,"Whisper", {MODEL} - hamlet_test',
        'date': datetime.now().isoformat(),
        'audio_file': str(AUDIO_FILE),
        'ground_truth': str(GROUND_TRUTH) if has_ground_truth else None,
    },
    'timings': {},
    'system_metrics': {},
    'accuracy': {},
}

# Get audio duration
print("\n" + "="*80)
print("STEP 1: Audio Information")
print("="*80)
audio_duration = get_audio_duration(str(AUDIO_FILE))
if audio_duration:
    print(f"Audio Duration: {format_time(audio_duration)}")
    all_metrics['test_info']['audio_duration_seconds'] = audio_duration
else:
    print("Could not determine audio duration")

# Device detection
print("\n" + "="*80)
print("STEP 2: Device Detection")
print("="*80)
device = get_device(prefer_gpu=True, verbose=True)
all_metrics['test_info']['device'] = str(device)

# Initialize system metrics
sys_metrics = SystemMetrics()

# ============================================================================
# STEP 3: Audio Preprocessing
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Audio Preprocessing")
print("="*80)

with PerformanceTimer("Audio Preprocessing") as timer:
    # Load audio
    print("Loading audio file...")
    audio = AudioSegment.from_file(str(AUDIO_FILE))

    # Convert to sample rate from config
    sample_rate = config.processing.audio.sample_rate
    print(f"Converting to {sample_rate}Hz...")
    audio = audio.set_frame_rate(sample_rate)

    # Trim silence from start
    print("Trimming silence from start...")
    start_trim = milliseconds_until_sound(audio)
    trimmed = audio[start_trim:]
    print(f"Trimmed {start_trim/1000:.2f} seconds of silence")

    # Export
    print(f"Exporting to {OUTPUT_WAV}")
    trimmed.export(str(OUTPUT_WAV), format='wav')

    sys_metrics.update()

all_metrics['timings']['preprocessing'] = timer.elapsed
print(f"Preprocessing complete: {format_time(timer.elapsed)}")

# ============================================================================
# STEP 4: Speaker Diarization (Pyannote)
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Speaker Diarization (Pyannote)")
print("="*80)

with PerformanceTimer("Pyannote Diarization") as timer:
    print("Loading Pyannote pipeline...")
    try:
        pipeline = Pipeline.from_pretrained(
            config.models.pyannote.model,
            use_auth_token=config.get_hf_token() or True,
        )
    except Exception as e:
        print(f"Error loading Pyannote pipeline: {e}")
        print("\nNote: You may need to:")
        print("1. Accept pyannote/speaker-diarization-3.1 user conditions at https://hf.co/pyannote/speaker-diarization-3.1")
        print("2. Create access token at https://hf.co/settings/tokens")
        print("3. Set environment variable: HF_TOKEN=your_token")
        sys.exit(1)

    print(f"Moving pipeline to {device}...")
    pipeline.to(device)

    print("Running diarization (this may take several minutes on CPU)...")
    diarization = pipeline(str(OUTPUT_WAV), num_speakers=2)

    print(f"Writing RTTM file to {OUTPUT_RTTM}")
    with open(OUTPUT_RTTM, "w") as rttm:
        diarization.write_rttm(rttm)

    sys_metrics.update()

all_metrics['timings']['diarization'] = timer.elapsed
print(f"Diarization complete: {format_time(timer.elapsed)}")

# ============================================================================
# STEP 5: Process RTTM and Segment Audio
# ============================================================================
print("\n" + "="*80)
print("STEP 5: Process RTTM and Segment Audio")
print("="*80)

with PerformanceTimer("RTTM Processing") as timer:
    # Read and group RTTM lines
    print(f"Reading RTTM file: {OUTPUT_RTTM}")
    with open(OUTPUT_RTTM, 'r') as f:
        dlines = [line.strip() for line in f if line.strip()]

    print(f"Found {len(dlines)} RTTM entries")

    # Group by speaker
    groups = []
    g = []
    lastend = 0
    rtm_length = 0

    for d in dlines:
        start_time, end_time, duration = extract_end_time(d)
        rtm_length += end_time - start_time

        # If new speaker or segment too long
        if g and (g[0].split()[7] != d.split()[7]) or rtm_length > 20:
            groups.append(g)
            g = []
            rtm_length = 0

        g.append(d)

        end_time_ms = end_time * 1000
        if lastend > end_time_ms:
            groups.append(g)
            g = []
        else:
            lastend = end_time_ms

    if g:
        groups.append(g)

    print(f"Grouped into {len(groups)} speaker segments")

    # Segment audio by speaker
    print("Segmenting audio by speaker turns...")
    audio_full = AudioSegment.from_file(str(OUTPUT_WAV), format="wav")
    start_stop_list = []

    # Clear temp folder first
    for temp_file in TEMP_AUDIO_CLIPS.glob("*.wav"):
        temp_file.unlink()

    for gidx, g in enumerate(groups):
        start, _, _ = extract_end_time(g[0])
        _, end, _ = extract_end_time(g[-1])
        duration = (end - start) / 1000
        start_time = int(start * 1000)
        end_time = int(end * 1000)

        audio_seg = audio_full[start_time:end_time]

        filename = f"{gidx}.wav"
        use_name = TEMP_AUDIO_CLIPS / filename
        audio_seg.export(str(use_name), format="wav")

        start_stop_list.append([g[0].split()[7], start_time, end_time, duration])

    print(f"Created {len(start_stop_list)} audio segments")
    sys_metrics.update()

all_metrics['timings']['rttm_processing'] = timer.elapsed
all_metrics['test_info']['num_segments'] = len(start_stop_list)
print(f"RTTM processing complete: {format_time(timer.elapsed)}")

# ============================================================================
# STEP 6A: Whisper Transcription 
# ============================================================================
print("\n" + "="*80)
print(f"STEP 6A: Whisper Transcription {MODEL}")
print("="*80)

output_base = OUTPUT_FOLDER / f"hamlet_test_whisper_{MODEL}_{device_type}.csv"

with PerformanceTimer("Whisper Base Model") as timer:
    print(f"Loading Whisper {MODEL} model...")
    # Use in_memory setting from config
    in_memory = config.models.whisper.in_memory
    try:
        model = whisper.load_model(MODEL, device=device, in_memory=in_memory)
    except (TypeError, OSError) as e:
        print(f"  Warning: Could not load with in_memory={in_memory} ({e})")
        print(f"  Trying standard load...")
        model = whisper.load_model(MODEL, device=device)

    print(f"Transcribing {len(start_stop_list)} segments...")
    with open(output_base, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Speaker', 'Start Time', 'End Time', 'Duration', 'Transcription'])

        for i, line_items in enumerate(start_stop_list):
            audiof = TEMP_AUDIO_CLIPS / f'{i}.wav'

            if i % 10 == 0:
                print(f"  Segment {i+1}/{len(start_stop_list)}...")
                sys_metrics.update()

            result = model.transcribe(
                audio=str(audiof),
                language=config.models.whisper.language,
                word_timestamps=config.models.whisper.word_timestamps
            )

            if result['text'].strip():
                writer.writerow([
                    line_items[0],  # Speaker
                    line_items[1],  # Start time
                    line_items[2],  # End time
                    line_items[3],  # Duration
                    result["text"]
                ])

    sys_metrics.update()

all_metrics['timings']['whisper_model'] = timer.elapsed
print(f"Whisper {MODEL} complete: {format_time(timer.elapsed)}")
print(f"Output saved to: {output_base}")

# Calculate WER for base model
if has_ground_truth:
    print("\nCalculating WER for model...")
    try:
        wer_base = calculate_wer_from_files(
            str(GROUND_TRUTH),
            str(output_base),
            text_column='Transcription'
        )
        all_metrics['accuracy']['whisper_model'] = wer_base
        print(f"  WER: {wer_base['wer']:.2%}")
        print(f"  Substitutions: {wer_base['substitutions']}")
        print(f"  Deletions: {wer_base['deletions']}")
        print(f"  Insertions: {wer_base['insertions']}")
    except Exception as e:
        print(f"  Error calculating WER: {e}")

# ============================================================================
# STEP 6B: Whisper Transcription (Large Model)
# ============================================================================
# print("\n" + "="*80)
# print("STEP 6B: Whisper Transcription (Large Model)")
# print("="*80)

# output_large = OUTPUT_FOLDER / f"hamlet_test_whisper_large_{device_type}.csv"
# operation_name = f"Whisper {MODEL} Model"

# with PerformanceTimer(operation_name) as timer:
#     print(f"Loading Whisper {MODEL} model (this will download ~3GB if not cached)...")
#     model = whisper.load_model(MODEL, device=device)

#     print(f"Transcribing {len(start_stop_list)} segments...")
#     with open(output_large, "w", newline='', encoding="utf-8") as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['Speaker', 'Start Time', 'End Time', 'Duration', 'Transcription'])

#         for i, line_items in enumerate(start_stop_list):
#             audiof = TEMP_AUDIO_CLIPS / f'{i}.wav'

#             if i % 10 == 0:
#                 print(f"  Segment {i+1}/{len(start_stop_list)}...")
#                 sys_metrics.update()

#             result = model.transcribe(audio=str(audiof), language='en', word_timestamps=True)

#             if result['text'].strip():
#                 writer.writerow([
#                     line_items[0],  # Speaker
#                     line_items[1],  # Start time
#                     line_items[2],  # End time
#                     line_items[3],  # Duration
#                     result["text"]
#                 ])

#     sys_metrics.update()

# all_metrics['timings']['whisper_large'] = timer.elapsed
# print(f"Whisper {MODEL} complete: {format_time(timer.elapsed)}")
# print(f"Output saved to: {output_large}")

# # Calculate WER for large model
# if has_ground_truth:
#     print("\nCalculating WER for large model...")
#     try:
#         wer_large = calculate_wer_from_files(
#             str(GROUND_TRUTH),
#             str(output_large),
#             text_column='Transcription'
#         )
#         all_metrics['accuracy']['whisper_large'] = wer_large
#         print(f"  WER: {wer_large['wer']:.2%}")
#         print(f"  Substitutions: {wer_large['substitutions']}")
#         print(f"  Deletions: {wer_large['deletions']}")
#         print(f"  Insertions: {wer_large['insertions']}")
#     except Exception as e:
#         print(f"  Error calculating WER: {e}")

# ============================================================================
# STEP 7: Cleanup and Final Metrics
# ============================================================================
print("\n" + "="*80)
print("STEP 7: Cleanup and Final Metrics")
print("="*80)

# Clean up temp audio clips
print("Cleaning up temporary audio clips...")
for temp_file in TEMP_AUDIO_CLIPS.glob("*.wav"):
    temp_file.unlink()
print(f"Deleted {len(start_stop_list)} temporary files")

# Get system metrics summary
sys_summary = sys_metrics.get_summary()
all_metrics['system_metrics'] = sys_summary

# Calculate total time and real-time factor
total_time = sum(all_metrics['timings'].values())
all_metrics['timings']['total'] = total_time

if audio_duration:
    rtf = total_time / audio_duration
    all_metrics['performance'] = {
        'real_time_factor': rtf,
        'processing_speed': f"{audio_duration / total_time:.2f}x slower than realtime" if rtf > 1 else f"{total_time / audio_duration:.2f}x realtime"
    }

# Save metrics to JSON
print(f"\nSaving metrics to {OUTPUT_METRICS}")
save_metrics_json(all_metrics, str(OUTPUT_METRICS))

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TEST COMPLETE - SUMMARY")
print("="*80)

print(f"\nTotal Processing Time: {format_time(total_time)}")
if audio_duration:
    print(f"Audio Duration: {format_time(audio_duration)}")
    print(f"Real-Time Factor: {rtf:.2f}x (processing took {rtf:.2f}x the audio duration)")

print("\n--- Stage Timings ---")
for stage, elapsed in all_metrics['timings'].items():
    if stage != 'total':
        pct = (elapsed / total_time) * 100
        print(f"{stage:20s}: {format_time(elapsed):>12s} ({pct:>5.1f}%)")

print("\n--- System Resources ---")
print(f"Peak Memory: {sys_summary['peak_memory_mb']:.1f} MB")
print(f"Avg CPU: {sys_summary['avg_cpu_percent']:.1f}%")

if has_ground_truth and 'whisper_model' in all_metrics['accuracy']:
    print("\n--- Accuracy (WER) ---")
    print(f"Whisper {MODEL}:  {all_metrics['accuracy']['whisper_model']['wer']:.2%}")
    # if 'whisper_large' in all_metrics['accuracy']:
    #     print(f"Whisper Large: {all_metrics['accuracy']['whisper_large']['wer']:.2%}")
    #     improvement = all_metrics['accuracy']['whisper_base']['wer'] - all_metrics['accuracy']['whisper_large']['wer']
    #     print(f"Improvement:   {improvement:.2%} (lower is better)")

print("\n--- Output Files ---")
print(f"Whisper {MODEL}:  {output_base}")
#print(f"Whisper Large: {output_large}")
print(f"RTTM File:     {OUTPUT_RTTM}")
print(f"Metrics JSON:  {OUTPUT_METRICS}")

print(f"\nTest Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
