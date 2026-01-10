"""
Evaluate fine-tuned Whisper model against baseline.

Compares the fine-tuned model with the baseline Whisper large-v2 on test sets.

Usage:
    python src/evaluate_finetuned.py
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import os 
import warnings
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.metrics import calculate_wer
from src.utils import convert_to_wav

config = Config()
PROJECT_PATH = Path(__file__).parent.parent

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# CRITICAL: Set environment variable to force weights_only=False for PyTorch 2.6+
# This fixes the "Unsupported global: GLOBAL omegaconf.listconfig.ListConfig" error
# when loading Pyannote/WhisperX models
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

print("Importing dependencies...")
try:
    import torch
    # IMPORTANT: Patch torch.load BEFORE importing whisperx or transformers
    from src.torch_utils import patch_torch_load
    patch_torch_load()

    import whisperx
    from transformers import pipeline
except ImportError as e:
    print(f"\nERROR: Missing dependency: {e}")
    sys.exit(1)

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



def load_ground_truth(csv_path: Path):
    """Load ground truth transcript from CSV."""
    import pandas as pd

    df = pd.read_csv(csv_path)

    # Try different column names that might contain the transcription
    text_column = None
    for col in ['Transcription', 'Text', 'transcription', 'text', 'transcript']:
        if col in df.columns:
            text_column = col
            break

    if text_column is None:
        raise ValueError(f"Could not find transcription column in {csv_path}. Columns: {df.columns.tolist()}")

    # Concatenate all text
    full_text = " ".join(df[text_column].astype(str).tolist())
    return full_text


def print_gpu_memory(stage: str):
    """Print current GPU memory usage for debugging."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        free, total = torch.cuda.mem_get_info()
        free_gb = free / 1024**3
        total_gb = total / 1024**3
        print(f"  [{stage}] GPU Memory: {allocated:.2f}GB allocated | {reserved:.2f}GB reserved | {free_gb:.2f}GB/{total_gb:.2f}GB free")


def transcribe_with_whisperx_baseline(audio_path: Path, device: str = "cuda"):
    """Transcribe using baseline WhisperX (large-v2)."""
    print("\n" + "=" * 80)
    print("BASELINE: WhisperX (openai/whisper-large-v2)")
    print("=" * 80)
    # WhisperX/faster-whisper expects old torchaudio API
    import torchaudio
    import gc

    print_gpu_memory("Before baseline")

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
    # Load model
    print("Loading baseline model...")
    model = whisperx.load_model("large-v2", device=device, compute_type="float16")
    print_gpu_memory("After model load")

    # Load and transcribe audio
    print(f"Transcribing: {audio_path.name}")
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, language="en")
    print_gpu_memory("After transcription")

    # Extract full text
    full_text = " ".join([seg['text'] for seg in result['segments']])

    # Clean up GPU memory
    del model
    del audio
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print_gpu_memory("After cleanup")

    return full_text, result


def transcribe_with_finetuned(audio_path: Path, model_path: Path, device: str = "cuda"):
    """Transcribe using fine-tuned model via HuggingFace pipeline."""
    import gc

    print("\n" + "=" * 80)
    print("FINE-TUNED: LoRA Adapter")
    print("=" * 80)

    print_gpu_memory("Before fine-tuned")

    print(f"Loading fine-tuned model from: {model_path}")

    # HuggingFace pipeline requires WAV format - convert if needed
    if audio_path.suffix.lower() != '.wav':
        wav_path = audio_path.with_suffix('.wav')
        if not wav_path.exists():
            print(f"Converting {audio_path.suffix} to WAV format...")
            convert_to_wav(str(audio_path), str(wav_path))
        audio_path = wav_path

    # Load fine-tuned model via HuggingFace pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=str(model_path),
        device=0 if device == "cuda" else -1,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    print_gpu_memory("After model load")

    # Transcribe
    print(f"Transcribing: {audio_path.name}")
    result = pipe(str(audio_path), return_timestamps=True)
    print_gpu_memory("After transcription")

    full_text = result['text']

    # Clean up GPU memory
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print_gpu_memory("After cleanup")

    return full_text, result


def evaluate_model(audio_path: Path, truth_path: Path, model_path: Path = None, device: str = "cuda"):
    """Evaluate model(s) on a test file."""
    print("\n" + "#" * 80)
    print("EVALUATION: Fine-Tuned vs Baseline")
    print("#" * 80)
    print(f"\nTest File: {audio_path.name}")
    print(f"Ground Truth: {truth_path.name}")

    # Load ground truth
    print("\nLoading ground truth...")
    ground_truth = load_ground_truth(truth_path)
    print(f"  Ground truth length: {len(ground_truth)} characters")

    results = {}

    # Evaluate baseline
    try:
        baseline_text, baseline_result = transcribe_with_whisperx_baseline(audio_path, device)
        baseline_metrics = calculate_wer(ground_truth, baseline_text)

        results['baseline'] = {
            'text': baseline_text,
            'wer': baseline_metrics['wer'],
            'metrics': baseline_metrics,
            'length': len(baseline_text)
        }

        print(f"\nBaseline WER: {baseline_metrics['wer']:.2%}")
    except Exception as e:
        print(f"\nERROR in baseline: {e}")
        import traceback
        traceback.print_exc()
        results['baseline'] = {'error': str(e)}

    # Evaluate fine-tuned
    if model_path and model_path.exists():
        try:
            finetuned_text, finetuned_result = transcribe_with_finetuned(audio_path, model_path, device)
            finetuned_metrics = calculate_wer(ground_truth, finetuned_text)

            results['finetuned'] = {
                'text': finetuned_text,
                'wer': finetuned_metrics['wer'],
                'metrics': finetuned_metrics,
                'length': len(finetuned_text)
            }

            print(f"\nFine-tuned WER: {finetuned_metrics['wer']:.2%}")

            # Calculate improvement
            if 'baseline' in results and 'wer' in results['baseline']:
                improvement = results['baseline']['wer'] - finetuned_metrics['wer']
                improvement_pct = (improvement / results['baseline']['wer']) * 100 if results['baseline']['wer'] > 0 else 0

                print(f"\n" + "=" * 80)
                print("IMPROVEMENT")
                print("=" * 80)
                print(f"WER Reduction: {improvement:.2%} absolute")
                print(f"Relative Improvement: {improvement_pct:.1f}%")

                if improvement > 0:
                    print("OK: Fine-tuned model is BETTER")
                elif improvement < 0:
                    print("WARNING: Fine-tuned model is WORSE")
                else:
                    print("NEUTRAL: No change")

                results['improvement'] = {
                    'absolute': improvement,
                    'relative_pct': improvement_pct
                }

        except Exception as e:
            print(f"\nERROR in fine-tuned: {e}")
            import traceback
            traceback.print_exc()
            results['finetuned'] = {'error': str(e)}
    else:
        print(f"\nWARNING: Fine-tuned model not found at {model_path}")

    return results


if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("FINE-TUNED MODEL EVALUATION")
    print("#" * 80)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Configuration
    DEVICE = config.get_device()
    FINETUNED_MODEL = PROJECT_PATH / "models" / "whisper-finetuned-interviews" / "final_model"

    # Test files
    test_files = [
        {
            'name': 'GPT Test 2 (Unseen Data - Generalization Test)',
            'audio': PROJECT_PATH / config.paths.audio_files / "GPT_test2.m4a",
            'truth': PROJECT_PATH / config.paths.transcripts / "gpt_test2_truth.csv"
        },
        {
            'name': 'GPT Test (Training Data - Overfitting Check)',
            'audio': PROJECT_PATH / config.paths.audio_files / "GPT_test.m4a",
            'truth': PROJECT_PATH / config.paths.transcripts / "gpt_test_truth.csv"
        }
    ]

    print(f"\nConfiguration:")
    print(f"  Device: {DEVICE}")
    print(f"  Fine-tuned model: {FINETUNED_MODEL}")
    print(f"  Test files: {len(test_files)}")

    # Check if fine-tuned model exists
    if not FINETUNED_MODEL.exists():
        print(f"\nERROR: Fine-tuned model not found!")
        print(f"  Expected location: {FINETUNED_MODEL}")
        print(f"  Run src/finetune_whisper_simple.py first to create the model.")
        sys.exit(1)

    # Run evaluations
    all_results = {}

    for idx, test in enumerate(test_files, 1):
        print(f"\n{'='*80}")
        print(f"PROCESSING TEST FILE {idx}/{len(test_files)}: {test['name']}")
        print(f"{'='*80}")

        if not test['audio'].exists():
            print(f"\nWARNING: Audio file not found: {test['audio']}")
            continue
        if not test['truth'].exists():
            print(f"\nWARNING: Ground truth not found: {test['truth']}")
            continue

        try:
            results = evaluate_model(
                audio_path=test['audio'],
                truth_path=test['truth'],
                model_path=FINETUNED_MODEL,
                device=DEVICE
            )

            all_results[test['name']] = results
            print(f"\nTest {idx}/{len(test_files)} completed successfully")

        except Exception as e:
            print(f"\nCRITICAL ERROR in test {idx}/{len(test_files)}: {e}")
            import traceback
            traceback.print_exc()
            all_results[test['name']] = {'critical_error': str(e)}
            print(f"\nContinuing to next test file...")

        # Force GPU cleanup between tests
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print_gpu_memory("Between tests")

    # Save results
    output_file = PROJECT_PATH / config.paths.output_dirs['pretrain_test'] / "results" / f"finetuning_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_file}")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
