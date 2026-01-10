# Installation Guide

**Purpose**: Complete setup for Audio Transcription & Deidentification project
**Date**: 2025-12-23
**System**: Windows

---

## Prerequisites

- Python 3.8 or higher
- Git (already installed - confirmed by repository access)
- ~15GB free disk space (for models and dependencies)
- Internet connection for downloads

---

## Step 1: Verify Python Installation

```bash
python --version
# Should show Python 3.8 or higher
```

**If Python not installed**:
1. Download from https://www.python.org/downloads/
2. Run installer
3. ✓ Check "Add Python to PATH"
4. Click "Install Now"

---

## Step 2: Create Virtual Environment (Recommended)

**Why**: Isolates project dependencies from system Python

```bash
# Navigate to project directory
cd c:\Users\rpaul\Documents\GitHub\Audio_Transcription_Deidentification

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Your prompt should now show (.venv)
```

**To deactivate later**: `deactivate`

---

## Step 3: Install Core Dependencies

### Option A: Install from requirements.txt (Recommended)

```bash
# Make sure virtual environment is activated
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: This will take 10-20 minutes and download ~5GB

### Option B: Install Manually (if requirements.txt has issues)

```bash
# Core ML libraries
pip install torch torchvision torchaudio

# Whisper
pip install git+https://github.com/openai/whisper.git

# Pyannote
pip install pyannote.audio==3.3.2

# Audio processing
pip install pydub ffmpeg-python

# Data processing
pip install pandas openpyxl

# Metrics (our new additions)
pip install jiwer psutil

# Utilities
pip install tqdm librosa soundfile
```

---

## Step 4: Install FFmpeg

**Required for**: Audio format conversion

### Windows Installation

**Option A: Chocolatey (Easiest)**
```bash
# If you have Chocolatey package manager
choco install ffmpeg
```

**Option B: Manual Download**
1. Go to https://ffmpeg.org/download.html#build-windows
2. Click "Windows builds from gyan.dev"
3. Download "ffmpeg-release-essentials.zip"
4. Extract to `C:\ffmpeg`
5. Add to PATH:
   - Search "Environment Variables" in Windows
   - Edit "Path" variable
   - Add `C:\ffmpeg\bin`
   - Click OK
6. Restart command prompt

**Verify Installation**:
```bash
ffmpeg -version
```

---

## Step 5: Set Up Hugging Face Authentication

**Required for**: Pyannote speaker diarization models

1. Create account at https://huggingface.co/join
2. Accept model conditions:
   - Go to https://hf.co/pyannote/speaker-diarization-3.1
   - Click "Agree and access repository"
3. Create access token:
   - Go to https://hf.co/settings/tokens
   - Click "New token"
   - Name it (e.g., "audio-transcription")
   - Set role to "read"
   - Copy the token

4. Save token as environment variable:

**Windows (Command Prompt)**:
```bash
# Set for current session
set HF_TOKEN=your_token_here

# Set permanently
setx HF_TOKEN "your_token_here"
```

**Or create `.env` file** in project root:
```
HF_TOKEN=your_token_here
```

---

## Step 6: Verify Installation

### Test 1: Check Installed Packages

```bash
pip list | findstr /I "whisper pyannote torch jiwer"
```

**Expected output**:
```
jiwer              3.x.x
openai-whisper     ...
pyannote.audio     3.3.2
pyannote.metrics   3.2.1
torch              2.4.1
...
```

### Test 2: Test GPU Setup

```bash
python test_gpu.py
```

**Expected**: Should detect GPU if available, or report CPU-only

### Test 3: Test Imports

```bash
python -c "import whisper; print('Whisper OK')"
python -c "import pyannote.audio; print('Pyannote OK')"
python -c "from src.metrics import calculate_wer; print('Metrics OK')"
python -c "from pydub import AudioSegment; print('Pydub OK')"
```

**All should print "OK"**

---

## Step 7: Download Models (Optional Pre-download)

Models will auto-download on first use, but you can pre-download to save time later:

### Whisper Models

```bash
python -c "import whisper; whisper.load_model('base')"
python -c "import whisper; whisper.load_model('large')"
```

**Sizes**:
- base: ~140 MB
- large: ~3 GB

### Pyannote Models

Will download on first run (~1-2 GB total)

---

## Installation Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'whisper'"

**Solution**:
1. Ensure virtual environment is activated: `.venv\Scripts\activate`
2. Install whisper: `pip install git+https://github.com/openai/whisper.git`
3. Verify: `python -c "import whisper"`

### Issue: "FFmpeg not found"

**Solution**:
1. Install FFmpeg (see Step 4)
2. Verify: `ffmpeg -version`
3. Restart command prompt after installation

### Issue: "Pyannote authentication failed"

**Solution**:
1. Accept model agreement at https://hf.co/pyannote/speaker-diarization-3.1
2. Create and set HF_TOKEN (see Step 5)
3. Restart Python session

### Issue: "CUDA out of memory" (if using GPU)

**Solution**:
1. Use smaller models (base instead of large)
2. Process shorter audio segments
3. Close other GPU applications
4. Fall back to CPU if needed

### Issue: "pip install fails with compiler error"

**Solution**:
1. Update pip: `python -m pip install --upgrade pip`
2. Install build tools: `pip install setuptools wheel`
3. For Windows: Install Visual C++ Build Tools if needed
   - Download from https://visualstudio.microsoft.com/downloads/
   - Select "Desktop development with C++"

---

## After Installation

### Run Baseline Test

```bash
# Activate environment
.venv\Scripts\activate

# Run test
python run_baseline_test.py
```

**Expected**: Processes hamlet_test audio file and generates metrics

**Time**: 15-30 minutes on CPU for full test

---

## Optional: WhisperX Installation

**Note**: WhisperX has complex dependencies on Windows and may require:
- Visual Studio Build Tools
- FFmpeg development libraries
- May not install cleanly on all systems

### Try Installation

```bash
pip install whisperx
```

**If it fails** (common on Windows):
- You can still proceed without WhisperX
- Focus on Whisper + Pyannote for Week 1
- Consider WhisperX on Linux/Mac or cloud environment later

**Alternative**: Use Google Colab for WhisperX testing
- Free GPU available
- Can test WhisperX there
- Transfer learnings back to local setup

---

## Recommended Installation Order

1. ✅ Python (prerequisite)
2. ✅ Create virtual environment
3. ✅ Install core dependencies (requirements.txt)
4. ✅ Install FFmpeg
5. ✅ Set up Hugging Face token
6. ✅ Verify with test scripts
7. ✅ Run baseline test on hamlet_test
8. ⧖ WhisperX (optional, try but don't block on it)

---

## Quick Start Script

Save this as `setup.bat` and run it:

```batch
@echo off
echo Audio Transcription Project - Setup Script
echo ==========================================

echo.
echo Step 1: Creating virtual environment...
python -m venv .venv

echo.
echo Step 2: Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Step 3: Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Step 4: Installing dependencies...
pip install -r requirements.txt

echo.
echo Step 5: Testing installation...
python -c "import whisper; print('✓ Whisper installed')"
python -c "import pyannote.audio; print('✓ Pyannote installed')"
python -c "from src.metrics import calculate_wer; print('✓ Metrics module ready')"

echo.
echo ==========================================
echo Setup complete!
echo.
echo Next steps:
echo 1. Set your HF_TOKEN environment variable
echo 2. Run: python test_gpu.py
echo 3. Run: python run_baseline_test.py
echo ==========================================
pause
```

---

## Environment Activation Reminder

**Every time you work on the project**:

```bash
# Navigate to project
cd c:\Users\rpaul\Documents\GitHub\Audio_Transcription_Deidentification

# Activate environment
.venv\Scripts\activate

# Your prompt should show (.venv)
# Now you can run scripts
```

---

## Storage Requirements

**Total space needed**: ~15 GB

Breakdown:
- Python packages: ~5 GB
- Whisper models (base + large): ~3.2 GB
- Pyannote models: ~1-2 GB
- Working files and outputs: ~1 GB
- Buffer: ~3-5 GB

**Check free space**:
```bash
dir C:\
```

---

## What to Install First (Minimum Viable)

If you want to start testing quickly:

```bash
# Create venv
python -m venv .venv
.venv\Scripts\activate

# Core only
pip install torch whisper pyannote.audio pydub jiwer psutil pandas openpyxl

# Test
python test_gpu.py
```

Then add other packages as needed.

---

**Last Updated**: 2025-12-23
**Status**: Ready for fresh installation
