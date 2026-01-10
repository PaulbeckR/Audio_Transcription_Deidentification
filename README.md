# Audio Transcription & Deidentification

> **High-accuracy audio transcription pipeline with speaker diarization, timestamp alignment, and Whisper fine-tuning for interview transcripts.**

---

## Overview

This project provides an end-to-end pipeline for processing long-form interview audio files into clean, accurate, speaker-attributed transcripts. The system combines state-of-the-art speech recognition (Whisper/WhisperX), speaker diarization (Pyannote), and forced alignment to generate high-quality transcriptions optimized for interview content.

**Key Features:**
- **GPU-accelerated transcription** using Whisper and WhisperX
- **Speaker diarization** with Pyannote Audio
- **Word-level timestamp alignment** for precise transcription
- **Fine-tuning pipeline** for domain-specific accuracy improvements
- **Comprehensive evaluation metrics** (WER, DER, timing accuracy)
- **Experiment tracking** with TensorBoard integration
- **Configuration-driven** workflow for reproducible experiments

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Audio_Transcription_Deidentification

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install base dependencies
pip install -r requirements.txt

# (Optional) Install fine-tuning dependencies
pip install -r requirements_finetuning.txt
```

**For detailed installation instructions**, see:
- [Notes/Setup/INSTALLATION_GUIDE.md](Notes/Setup/INSTALLATION_GUIDE.md)
- [Notes/Setup/GPU_SETUP.md](Notes/Setup/GPU_SETUP.md)
- [Notes/Setup/DUAL_ENVIRONMENT_SETUP.md](Notes/Setup/DUAL_ENVIRONMENT_SETUP.md)

### 2. Run Baseline Transcription

```bash
# Run baseline test on sample audio
python run_baseline_test.py
```

### 3. Fine-Tune Whisper (Optional)

```bash
# Quick test experiment (1 epoch)
python src/scripts/run_training.py --experiment quick_test

# Full baseline fine-tuning (5 epochs)
python src/scripts/run_training.py --experiment baseline_lora

# Monitor training with TensorBoard
tensorboard --logdir models/whisper-finetuned
```

**For complete training workflow**, see [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

---

## Project Structure

```
Audio_Transcription_Deidentification/
├── src/                          # Source code (reorganized Dec 2025)
│   ├── core/                     # Core utilities and configuration
│   │   ├── config.py             # Configuration management
│   │   ├── paths.py              # Path resolution
│   │   └── torch_utils.py        # PyTorch compatibility patches
│   ├── data/                     # Data preparation and processing
│   │   ├── create_training_data.py    # Audio alignment for training
│   │   └── prepare_hf_dataset.py      # HuggingFace dataset creation
│   ├── evaluation/               # Evaluation and metrics
│   │   ├── calculate_wer.py      # WER calculation utilities
│   │   └── evaluate_model.py     # Model evaluation scripts
│   ├── models/                   # Model management
│   │   └── model_loader.py       # Model loading and setup
│   ├── training/                 # Fine-tuning pipeline
│   │   ├── dataset.py            # Custom PyTorch datasets
│   │   ├── finetune.py           # Main fine-tuning logic
│   │   ├── lora.py               # LoRA configuration
│   │   └── callbacks.py          # Training callbacks
│   └── scripts/                  # Entry point scripts
│       └── run_training.py       # Training orchestration
│
├── config/                       # Configuration files
│   ├── paths.yaml                # File paths and directories
│   ├── training.yaml             # Training hyperparameters
│   └── experiments.yaml          # Experiment definitions
│
├── models/                       # Model storage
│   └── whisper-finetuned/        # Fine-tuned model checkpoints
│
├── prepared_datasets/            # Training data
│   └── whisper_finetuning/       # Prepared training datasets
│       ├── train.json            # Training examples (71 samples)
│       └── validation.json       # Validation examples (8 samples)
│
├── Audio_Local_tests/            # Test data and baselines
│   ├── audio_files/              # Test audio samples
│   ├── transcription_test/       # Ground truth transcripts
│   └── baseline_output/          # Baseline model outputs
│
├── Notes/                        # Documentation and notes
│   ├── Setup/                    # Installation guides
│   ├── Week1-ModelComparisons/   # Week 1 baseline experiments
│   ├── Week2-FinetuneAndReorg/   # Week 2 fine-tuning & reorganization
│   └── Troubleshooting/          # Common issues and solutions
│
├── old_code/                     # Legacy code (pre-reorganization)
│   └── finetune_whisper_simple.py  # Proven working fine-tuning (Dec 29)
│
├── run_baseline_test.py          # Quick baseline test script
├── requirements.txt              # Base dependencies
├── requirements_finetuning.txt   # Fine-tuning dependencies
│
└── README.md                     # This file
```

**For detailed structure**, see [Notes/File_Structure.md](Notes/File_Structure.md)

---

## Documentation

### Getting Started
- **[WORKSPACE_READY.md](WORKSPACE_READY.md)** - Complete workspace overview and status
- **[Notes/QUICK_START_GUIDE.md](Notes/QUICK_START_GUIDE.md)** - Quick start for new users
- **[Notes/Setup/INSTALLATION_GUIDE.md](Notes/Setup/INSTALLATION_GUIDE.md)** - Detailed installation

### Data & Training
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Complete training workflow
- **[DATA_OVERVIEW.md](DATA_OVERVIEW.md)** - Dataset information and statistics
- **[PREPARE_YOUR_DATA.md](PREPARE_YOUR_DATA.md)** - How to prepare your own data
- **[config/README.md](config/README.md)** - Configuration file reference

### Experiment Results
- **[Notes/Week1-ModelComparisons/](Notes/Week1-ModelComparisons/)** - Baseline model comparisons
  - [WEEK1_FINAL_REPORT.md](Notes/Week1-ModelComparisons/WEEK1_FINAL_REPORT.md)
  - [GPU_BASELINE_RESULTS.md](Notes/Week1-ModelComparisons/GPU_BASELINE_RESULTS.md)
- **[Notes/Week2-FinetuneAndReorg/](Notes/Week2-FinetuneAndReorg/)** - Fine-tuning experiments
  - [FINETUNING_RESULTS.md](Notes/Week2-FinetuneAndReorg/FINETUNING_RESULTS.md)
  - [REORGANIZATION_COMPLETE.md](Notes/Week2-FinetuneAndReorg/REORGANIZATION_COMPLETE.md)

### Troubleshooting
- **[Notes/Troubleshooting/TORCHCODEC_SOLUTION.md](Notes/Troubleshooting/TORCHCODEC_SOLUTION.md)** - Torchcodec dependency solution
- **[Notes/Troubleshooting/CODEBASE_ISSUES.md](Notes/Troubleshooting/CODEBASE_ISSUES.md)** - Known issues and fixes
- **[Notes/Setup/WHISPERX_INSTALLATION_ISSUE.md](Notes/Setup/WHISPERX_INSTALLATION_ISSUE.md)** - WhisperX installation

### Reference
- **[Notes/RESOURCES.md](Notes/RESOURCES.md)** - External resources and links
- **[Notes/Methods_Algorithms.md](Notes/Methods_Algorithms.md)** - Technical background
- **[Notes/TODO.md](Notes/TODO.md)** - Project roadmap and tasks

---

## Key Features

### 1. Baseline Transcription

High-accuracy baseline transcription using WhisperX:
- **Model**: Whisper Large-v3 (best quality)
- **Forced Alignment**: Word-level timestamps using WhisperX
- **Speaker Diarization**: Pyannote Audio 3.0
- **GPU Acceleration**: Full CUDA support
- **Metrics**: WER, DER, timing accuracy

**Performance** (from Week 1 experiments):
- **WER**: 3.64% (hamlet_test.wav baseline)
- **DER**: 0.00% (perfect speaker attribution)
- **Speed**: ~0.15x realtime on RTX 3050 (6GB)

See [Notes/Week1-ModelComparisons/GPU_BASELINE_RESULTS.md](Notes/Week1-ModelComparisons/GPU_BASELINE_RESULTS.md)

### 2. Fine-Tuning Pipeline

Domain-adaptive fine-tuning using LoRA (Low-Rank Adaptation):
- **Method**: LoRA for parameter-efficient fine-tuning
- **Dataset**: Custom interview transcripts (JSON format)
- **Configuration**: YAML-based experiment management
- **Monitoring**: TensorBoard integration
- **Checkpointing**: Automatic best model selection

**Experiment Types**:
- `quick_test`: 1 epoch, rapid validation
- `baseline_lora`: 5 epochs, full training
- `lr_sweep`: Learning rate comparison
- `rank_sweep`: LoRA rank optimization

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) and [config/experiments.yaml](config/experiments.yaml)

### 3. Custom Dataset Support

Flexible data preparation pipeline:
- **Input**: Audio files + transcripts
- **Alignment**: WhisperX forced alignment with word timestamps
- **Format**: Simple JSON (avoids HuggingFace torchcodec dependency)
- **Splits**: Configurable train/validation/test splits

**Data Format**:
```json
[
  {
    "audio_path": "path/to/audio.wav",
    "text": "transcription text"
  }
]
```

See [PREPARE_YOUR_DATA.md](PREPARE_YOUR_DATA.md)

### 4. Evaluation & Metrics

Comprehensive evaluation suite:
- **Word Error Rate (WER)**: Transcription accuracy
- **Diarization Error Rate (DER)**: Speaker attribution accuracy
- **Timing Metrics**: Alignment precision
- **Comparison Reports**: Before/after fine-tuning

Scripts:
- `src/evaluation/calculate_wer.py`: WER calculation utilities
- `src/evaluation/evaluate_model.py`: Full evaluation pipeline

---

## Configuration

All configuration is managed through YAML files in `config/`:

### Main Configuration Files

**[config/paths.yaml](config/paths.yaml)** - File paths and directories:
```yaml
data:
  audio_files: "Audio_Local_tests/audio_files"
  transcription_test: "Audio_Local_tests/transcription_test"

models:
  whisper_model: "large-v3"
  output_dir: "models/whisper-finetuned"
```

**[config/training.yaml](config/training.yaml)** - Training hyperparameters:
```yaml
strategy:
  method: "lora"  # LoRA, full, or freeze_encoder

hyperparameters:
  batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-5
  num_epochs: 5

lora:
  rank: 32
  alpha: 64
  target_modules: ["q_proj", "v_proj"]
```

**[config/experiments.yaml](config/experiments.yaml)** - Experiment definitions:
```yaml
baseline_lora:
  name: "baseline-lora-r32"
  description: "Baseline LoRA fine-tuning"
  enabled: true
  dataset:
    input_dir: "prepared_datasets/whisper_finetuning"
```

See [config/README.md](config/README.md) for complete configuration reference.

---

## Usage Examples

### Run Baseline Evaluation

```bash
# Quick baseline test
python run_baseline_test.py

# Evaluate specific audio file
python src/evaluation/evaluate_model.py \
  --audio Audio_Local_tests/audio_files/hamlet_test.wav \
  --truth Audio_Local_tests/transcription_test/hamlet_truth.csv
```

### Fine-Tune Whisper

```bash
# List available experiments
python src/scripts/run_training.py --list

# Run specific experiment
python src/scripts/run_training.py --experiment baseline_lora

# Run with custom config override
python src/scripts/run_training.py \
  --experiment baseline_lora \
  --epochs 3 \
  --learning-rate 2e-5
```

### Prepare Training Data

```bash
# Create aligned training data from audio + transcripts
python src/data/create_training_data.py \
  --audio-dir path/to/audio \
  --transcript-dir path/to/transcripts \
  --output training_data/aligned

# Convert to HuggingFace dataset format
python src/data/prepare_hf_dataset.py \
  --input training_data/aligned \
  --output prepared_datasets/my_dataset \
  --train-split 0.8 \
  --val-split 0.1 \
  --test-split 0.1
```

### Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir models/whisper-finetuned

# View at http://localhost:6006
```

---

## Hardware Requirements

### Minimum (CPU-only baseline)
- **CPU**: 4 cores, 8 threads
- **RAM**: 8GB
- **Storage**: 10GB
- **Performance**: Slow (~1.0x realtime)

### Recommended (GPU training)
- **GPU**: NVIDIA RTX 3050 (6GB VRAM) or better
- **CPU**: 8 cores
- **RAM**: 16GB
- **Storage**: 50GB (models + datasets)
- **Performance**: Fast (~0.15x realtime)

### Tested Configurations
- **Desktop**: RTX 3050 6GB + Ryzen 7 5700G (current development machine)
- **Laptop**: CPU-only (baseline testing)

See [Notes/Setup/GPU_SETUP.md](Notes/Setup/GPU_SETUP.md) for GPU setup details.

---

## Project Status

### Current Phase: Fine-Tuning & Optimization (Week 2-3)

**Completed:**
- ✅ Week 1: Baseline model comparisons (WhisperX GPU evaluation)
- ✅ Codebase reorganization (Dec 2025 - Jan 2026)
- ✅ Custom dataset pipeline (avoids torchcodec dependency)
- ✅ LoRA fine-tuning implementation
- ✅ Experiment tracking configuration
- ✅ TensorBoard integration

**In Progress:**
- 🔄 Testing fine-tuning pipeline end-to-end
- 🔄 Preparing full interview dataset (41 transcripts, 30-41 hours)
- 🔄 Hyperparameter optimization

**Next Steps:**
- Fine-tune on full interview dataset
- Evaluate fine-tuned model performance
- Compare against baseline WER
- Document optimal hyperparameters

See [Notes/CURRENT_STATUS.md](Notes/CURRENT_STATUS.md) and [Notes/TODO.md](Notes/TODO.md)

---

## Technology Stack

### Core Libraries
- **Whisper** (OpenAI) - Speech recognition model
- **WhisperX** - Forced alignment + diarization integration
- **Pyannote Audio** - Speaker diarization
- **PyTorch** - Deep learning framework
- **HuggingFace Transformers** - Model fine-tuning
- **PEFT** - LoRA implementation

### Utilities
- **jiwer** - WER/CER calculation
- **pydub** - Audio manipulation
- **ffmpeg** - Audio processing backend
- **psutil** - System monitoring
- **PyYAML** - Configuration management
- **TensorBoard** - Training visualization

See [Notes/RESOURCES.md](Notes/RESOURCES.md) for complete list with links.

---

## Contributing

This is a research project for academic use. Contributions and suggestions are welcome.

**Before contributing:**
1. Review [Notes/File_Structure.md](Notes/File_Structure.md)
2. Check [Notes/TODO.md](Notes/TODO.md) for open tasks
3. See [Notes/Troubleshooting/CODEBASE_ISSUES.md](Notes/Troubleshooting/CODEBASE_ISSUES.md) for known issues

---

## Privacy & Data

**Important**: This project processes sensitive interview data.

- Audio files and transcripts are **NOT** included in the repository
- All data is processed **locally only**
- No cloud services or external APIs are used (except HuggingFace model downloads)
- Ground truth test data (`hamlet_test.wav`) is safe for development/testing

**Data Location**: All sensitive data should be in:
- `Audio_Local_tests/` (test data)
- `training_data/` (aligned training data)
- `prepared_datasets/` (processed datasets)

These directories are in `.gitignore`.

---

## License

[Specify license here - MIT, Apache 2.0, etc.]

**Dependencies** are subject to their respective licenses:
- Whisper: MIT License
- WhisperX: BSD-2-Clause License
- Pyannote: MIT License
- PyTorch: BSD License

---

## Acknowledgments

This project builds upon excellent open-source work:
- **OpenAI Whisper** - Robust speech recognition
- **WhisperX** (Max Bain et al.) - Timestamp alignment
- **Pyannote Audio** (Hervé Bredin et al.) - Speaker diarization
- **HuggingFace** - Transformers and model hosting

---

## Contact & Support

**Issues**: Report bugs and issues in [Notes/Troubleshooting/](Notes/Troubleshooting/)

**Questions**: See documentation in [Notes/](Notes/) or check [Notes/RESOURCES.md](Notes/RESOURCES.md)

---

**Last Updated**: 2026-01-09
**Version**: 1.4.25 (Post-Reorganization)
**Python**: 3.10+
**PyTorch**: 2.4.1
**CUDA**: 12.1 (if using GPU)
