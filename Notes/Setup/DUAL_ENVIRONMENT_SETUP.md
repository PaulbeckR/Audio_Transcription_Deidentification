# Dual Environment Setup - Main vs WhisperX

**Date:** December 23, 2025

---

## Overview

We now have two separate Python virtual environments to handle different transcription approaches:

1. **`.venv`** - Main pipeline with CUDA GPU support
2. **`.venv_whisperx`** - WhisperX environment (CPU-only for now)

---

## Environment Details

### Main Environment (`.venv`)

**Purpose:** Production transcription with GPU acceleration

**Python Version:** 3.10.0

**Key Packages:**
- PyTorch: 2.5.1+cu121 (CUDA enabled)
- TorchAudio: 2.5.1+cu121
- TorchVision: 0.20.1+cu121
- Pyannote Audio: 3.1.1
- OpenAI Whisper: Latest from GitHub
- jiwer: 3.0.0+ (WER calculation)

**GPU Support:** ✅ YES
- CUDA Version: 12.1
- Tested on: NVIDIA GeForce RTX 3050 6GB
- Verified speedup: 1.44x overall (6.18x for diarization, 1.27x-2.01x for Whisper)

**Activation:**
```bash
.venv\Scripts\activate  # Windows
```

**Use Cases:**
- Production transcription pipeline
- GPU-accelerated baseline tests
- Fine-tuning preparation
- WER/DER calculations
- Batch processing

---

### WhisperX Environment (`.venv_whisperx`)

**Purpose:** Testing WhisperX for advanced forced alignment

**Python Version:** 3.10.0

**Key Packages:**
- PyTorch: 2.8.0+cpu (CPU-only)
- TorchAudio: 2.8.0
- WhisperX: 3.7.4
- Pyannote Audio: 3.4.0 (newer version)
- All WhisperX dependencies

**GPU Support:** ❌ NO (PyTorch 2.8.0 CUDA not available yet)

**Activation:**
```bash
.venv_whisperx\Scripts\activate  # Windows
```

**Use Cases:**
- WhisperX alignment testing
- Word-level timestamp extraction
- Comparison with Whisper's built-in word timestamps
- Evaluating alignment quality

---

## Why Two Environments?

### The PyTorch Version Conflict

WhisperX requires PyTorch ~=2.8.0, but:
- PyTorch 2.8.0 is only available as CPU-only version
- Latest CUDA PyTorch is 2.6.0 (we use 2.5.1+cu121)
- Installing WhisperX breaks GPU support in the main environment

### Solution: Separate Environments

**Advantages:**
1. ✅ Maintain GPU acceleration for production pipeline
2. ✅ Test WhisperX independently without breaking main setup
3. ✅ Compare both approaches side-by-side
4. ✅ Easy to switch between environments
5. ✅ No dependency conflicts

**Disadvantages:**
1. ⚠️ WhisperX runs on CPU only (slower)
2. ⚠️ Duplicate packages (more disk space)
3. ⚠️ Need to remember which environment to use

---

## Usage Guide

### Running Scripts in Main Environment

```bash
# Activate main environment
.venv\Scripts\activate

# Run GPU-accelerated baseline test
python run_baseline_test.py

# Calculate WER
python calculate_wer_simple.py

# Test GPU
python test_gpu.py
```

### Running Scripts in WhisperX Environment

```bash
# Activate WhisperX environment
.venv_whisperx\Scripts\activate

# Test WhisperX
python test_whisperx.py  # (to be created)

# Run WhisperX alignment
python scripts/align_with_whisperx.py  # (to be created)
```

### Running Without Activation

```bash
# Main environment
.venv\Scripts\python.exe run_baseline_test.py

# WhisperX environment
.venv_whisperx\Scripts\python.exe test_whisperx.py
```

---

## Environment Comparison

| Feature | Main (.venv) | WhisperX (.venv_whisperx) |
|---------|-------------|---------------------------|
| **PyTorch** | 2.5.1+cu121 | 2.8.0+cpu |
| **GPU Support** | ✅ Yes (CUDA 12.1) | ❌ No (CPU only) |
| **Whisper** | ✅ OpenAI Whisper | ✅ Via WhisperX |
| **WhisperX** | ❌ No | ✅ Yes (3.7.4) |
| **Pyannote** | 3.1.1 | 3.4.0 |
| **Word Timestamps** | Via `word_timestamps=True` | Via forced alignment |
| **Speed** | Fast (GPU accelerated) | Slow (CPU only) |
| **Production Ready** | ✅ Yes | ⚠️ Testing only |

---

## Switching Between Environments

### Deactivate Current Environment
```bash
deactivate
```

### Activate Desired Environment
```bash
# Main
.venv\Scripts\activate

# WhisperX
.venv_whisperx\Scripts\activate
```

### Check Which Environment is Active
```bash
# Windows
where python
# Should show path to either .venv or .venv_whisperx

# Or check PyTorch version
python -c "import torch; print(torch.__version__)"
# Main: 2.5.1+cu121
# WhisperX: 2.8.0+cpu
```

---

##Future Improvements

### Option 1: Wait for PyTorch CUDA 2.8.0+
- WhisperX could run on GPU when PyTorch releases CUDA builds for 2.8.0+
- Would enable full GPU acceleration for forced alignment
- Timeline: Unknown (depends on PyTorch project)

### Option 2: Use Docker (Already Created)
- Docker containers provide better isolation
- Can have different CUDA versions in separate containers
- See `docker/` directory for Dockerfiles and instructions

### Option 3: Use WSL2
- Run WhisperX in Windows Subsystem for Linux
- Access to Linux GPU drivers and CUDA toolkit
- Better compatibility with Python ML ecosystem

---

## File Structure

```
Audio_Transcription_Deidentification/
├── .venv/                     # Main environment (GPU)
│   ├── Scripts/
│   │   ├── python.exe
│   │   ├── pip.exe
│   │   └── activate
│   └── Lib/
├── .venv_whisperx/            # WhisperX environment (CPU)
│   ├── Scripts/
│   │   ├── python.exe
│   │   ├── pip.exe
│   │   └── activate
│   └── Lib/
├── requirements.txt           # Main environment dependencies
├── run_baseline_test.py       # Use with .venv
├── test_whisperx.py          # Use with .venv_whisperx (TBD)
└── Notes/
    ├── DUAL_ENVIRONMENT_SETUP.md  # This file
    └── WHISPERX_INSTALLATION_ISSUE.md
```

---

## Maintenance

### Updating Main Environment
```bash
.venv\Scripts\activate
pip install --upgrade package_name
pip freeze > requirements.txt  # Update requirements if needed
```

### Updating WhisperX Environment
```bash
.venv_whisperx\Scripts\activate
pip install --upgrade whisperx
# No need for requirements.txt - just reinstall whisperx fresh
```

### Removing Environments
```bash
# Remove WhisperX environment (if no longer needed)
Remove-Item -Recurse -Force .venv_whisperx

# Remove main environment (careful!)
Remove-Item -Recurse -Force .venv
```

---

## Next Steps

1. ✅ Create test script for WhisperX (`test_whisperx.py`)
2. ✅ Run WhisperX on hamlet_test and evaluate alignment
3. ✅ Compare WhisperX alignment vs Whisper word timestamps
4. ✅ Decide which approach to use for training data preparation
5. ⏳ If WhisperX proves valuable, consider Docker/WSL2 for GPU support

---

## Summary

We now have:
- **Production environment** (`.venv`) with proven GPU acceleration (1.44x speedup)
- **Testing environment** (`.venv_whisperx`) with WhisperX for alignment evaluation
- **Flexibility** to use whichever approach works best
- **No conflicts** between different PyTorch versions

This dual-environment approach allows us to test WhisperX while maintaining our working GPU pipeline!
