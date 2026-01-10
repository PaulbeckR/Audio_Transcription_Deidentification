@echo off
echo ============================================================
echo Audio Transcription Project - Setup Script
echo ============================================================

echo.
echo Step 1: Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    echo Please ensure Python is installed and in PATH
    pause
    exit /b 1
)

echo.
echo Step 2: Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Step 3: Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Step 4: Installing dependencies from requirements.txt...
echo This will take 10-20 minutes and download ~5GB...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo WARNING: Some packages may have failed to install
    echo You can continue and install missing packages manually
    pause
)

echo.
echo Step 5: Testing core installations...
python -c "import sys; print('Python:', sys.version)"

echo Testing Whisper...
python -c "import whisper; print('✓ Whisper installed')" 2>nul
if errorlevel 1 (
    echo ✗ Whisper not found - installing...
    pip install git+https://github.com/openai/whisper.git
)

echo Testing Pyannote...
python -c "import pyannote.audio; print('✓ Pyannote installed')" 2>nul
if errorlevel 1 (
    echo ✗ Pyannote not found - check installation
)

echo Testing our metrics module...
python -c "from src.metrics import calculate_wer; print('✓ Metrics module ready')" 2>nul
if errorlevel 1 (
    echo ✗ Metrics module error - check src/metrics.py
)

echo Testing PyDub...
python -c "from pydub import AudioSegment; print('✓ Pydub ready')" 2>nul
if errorlevel 1 (
    echo ✗ Pydub not found
)

echo.
echo Step 6: Testing FFmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo ✗ FFmpeg not found
    echo   Please install FFmpeg:
    echo   1. Download from https://ffmpeg.org/download.html
    echo   2. Add to PATH environment variable
    echo   3. Restart command prompt
) else (
    echo ✓ FFmpeg installed
)

echo.
echo Step 7: Testing GPU availability...
python test_gpu.py

echo.
echo ============================================================
echo Setup Status
echo ============================================================
echo.
echo If all core packages installed successfully:
echo   1. Set your HF_TOKEN environment variable
echo      setx HF_TOKEN "your_token_here"
echo   2. Accept Pyannote conditions:
echo      https://hf.co/pyannote/speaker-diarization-3.1
echo   3. Run baseline test:
echo      python run_baseline_test.py
echo.
echo See Notes\INSTALLATION_GUIDE.md for detailed instructions
echo ============================================================
pause
