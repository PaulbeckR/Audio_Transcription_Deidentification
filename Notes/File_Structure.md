# Project File Structure

**Last Updated**: 2025-12-23

---

## Directory Overview

```
Audio_Transcription_Deidentification/
│
├── src/                              # NEW - Modular Python code
│   ├── __init__.py                   # Package initialization
│   ├── metrics.py                    # WER/DER calculation, performance tracking
│   ├── utils.py                      # Device detection, helper functions
│   ├── preprocessing.py              # Future: Audio conversion, trimming
│   ├── diarization.py                # Future: Pyannote processing
│   ├── transcription.py              # Future: Whisper processing
│   └── alignment.py                  # Future: WhisperX/MFA alignment
│
├── Notes/                            # Project documentation
│   ├── PROJECT_PLAN.md               # Overall roadmap and phases
│   ├── WEEK1_PLAN.md                 # Week 1 detailed implementation plan
│   ├── RESOURCES.md                  # Links, papers, documentation
│   ├── TODO.md                       # Master task list
│   ├── GPU_SETUP.md                  # GPU installation and troubleshooting
│   ├── BASELINE_CPU_METRICS.md       # Future: CPU baseline results
│   ├── GPU_METRICS.md                # Future: GPU performance results
│   ├── WHISPERX_SETUP.md             # Future: WhisperX installation
│   ├── WEEK1_RESULTS.md              # Future: Week 1 analysis
│   ├── Methods_Algorithms.md         # Technical background research
│   ├── Ideas_notes.md                # Ongoing discoveries and ideas
│   └── File_Structure.md             # This file
│
├── notebooks/                        # Jupyter notebooks (experimental)
│   ├── transcript_notebook.ipynb     # Main transcription pipeline
│   ├── create_rttm.ipynb             # RTTM generation and cleaning
│   ├── create_pretraining_data.ipynb # Training data preparation
│   ├── prepare_completed_transcripts.ipynb  # Transcript formatting
│   ├── test_notebook.ipynb           # General testing
│   └── TEST_TRUTH.ipynb              # Truth data processing
│
├── Audio_Local_tests/                # Local test environment (safe for Claude)
│   ├── audio_files/                  # Test audio files
│   │   └── hamlet_test.m4a.wav       # Primary test file
│   ├── audio_clips/                  # Temporary audio segments
│   ├── wav_files/                    # Converted WAV files
│   ├── rttm_files/                   # Diarization outputs
│   ├── clean_rttm_files/             # Processed RTTM files
│   └── transcription_test/           # Ground truth transcripts
│       └── hamlet_test_truth.xlsx    # Validated transcript
│
├── config/                           # Future: Configuration files
│   ├── paths.yaml                    # Future: All folder paths
│   └── models.yaml                   # Future: Model settings
│
├── scripts/                          # Future: Executable scripts
│   ├── process_single_file.py        # Future: Process one audio file
│   ├── process_folder.py             # Future: Batch processing
│   ├── align_transcripts.py          # Future: Alignment workflow
│   └── calculate_metrics.py          # Future: Metrics computation
│
├── helper_functions.py               # Legacy helper functions
├── one_and_done.py                   # Legacy: Speaker processing utility
├── test_gpu.py                       # GPU verification script
├── WEEK1_STATUS.md                   # Week 1 progress tracker
├── README.md                         # Project overview
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git exclusions
└── Audio_Transcription_Deidentification.code-workspace  # VS Code workspace

```

---

## File Descriptions

### Source Code (src/)

**metrics.py**
- `calculate_wer()` - WER from text strings
- `calculate_wer_from_files()` - WER from CSV/Excel files
- `calculate_der()` - Diarization Error Rate from RTTM
- `PerformanceTimer` - Context manager for timing
- `SystemMetrics` - CPU/GPU/memory tracking
- `generate_comparison_report()` - Multi-result comparison
- `save_metrics_json()` / `load_metrics_json()` - Metrics persistence

**utils.py**
- `get_device()` - Automatic CPU/GPU detection
- `get_gpu_info()` - GPU specifications
- `validate_audio_file()` - File validation
- `ensure_directory()` - Directory creation
- `get_audio_duration()` - Audio length calculation
- `format_time()` - Human-readable time strings
- `clear_temp_folder()` - Temporary file cleanup

### Notebooks

**transcript_notebook.ipynb**
- Main transcription pipeline
- Audio preprocessing
- Pyannote diarization
- Whisper transcription
- Output to CSV
- Currently uses CPU (hardcoded)
- Needs update to use device detection

**create_rttm.ipynb**
- RTTM file generation
- Speaker segment merging
- RTTM cleaning and consolidation
- Batch RTTM processing
- Comparison with validated transcripts
- Currently uses CPU
- Needs GPU enablement

**create_pretraining_data.ipynb**
- Training data preparation workflow
- May be corrupted (JSON parse error noted)
- Needs investigation

**prepare_completed_transcripts.ipynb**
- Formats existing transcripts
- Converts from Word docs
- Standardizes speaker labels
- Exports to text format

### Legacy Files

**helper_functions.py**
- `convert_to_wav()` - Audio format conversion
- `millisec()` - Time conversion
- `extract_end_time()` - RTTM parsing
- `milliseconds_until_sound()` - Silence detection
- `move_file()` - File operations
- `analyze_chunks()` - Audio segmentation
- `compute_dynamic_thresholds()` - VAD thresholds

**one_and_done.py**
- Speaker label processing
- Excel file handling
- Ground truth formatting

### Test Data

**hamlet_test.m4a.wav**
- Primary test file
- ~5 minutes duration
- Multiple speakers
- Safe for development/testing

**hamlet_test_truth.xlsx**
- Validated ground truth transcript
- Speaker labels
- Text content
- Used for WER/DER calculation

---

## Data Flow

### Current Workflow

```
Input Audio File (.m4a, .wav)
    ↓
1. Preprocessing (helper_functions.py)
    - Convert to WAV
    - Trim silence
    - Downsample to 16kHz
    ↓
2. Diarization (Pyannote)
    - Generate RTTM file
    - Identify speaker segments
    ↓
3. RTTM Processing (create_rttm.ipynb)
    - Clean and merge segments
    - Group by speaker turns
    ↓
4. Audio Segmentation
    - Split by speaker/time
    - Save temporary clips
    ↓
5. Transcription (Whisper)
    - Process each segment
    - Generate text
    ↓
6. Output (CSV)
    - Speaker labels
    - Timestamps
    - Transcription text
```

### Future Workflow (After Refactoring)

```
Input Audio File
    ↓
1. src/preprocessing.py
    - Load config from config/paths.yaml
    - Validate file
    - Convert format
    ↓
2. src/diarization.py
    - Auto device detection
    - Run Pyannote
    - Clean RTTM
    ↓
3. src/transcription.py OR src/alignment.py
    - Use selected model (Whisper/WhisperX)
    - Process on GPU/CPU
    - Word-level timestamps
    ↓
4. src/metrics.py
    - Calculate WER/DER
    - Log performance
    ↓
Output + Metrics Report
```

---

## Configuration Files (Planned)

### config/paths.yaml (Future)
```yaml
project:
  root: "C:/Users/rpaul/Documents/GitHub/Audio_Transcription_Deidentification"

local_testing:
  audio_files: "Audio_Local_tests/audio_files"
  transcription_test: "Audio_Local_tests/transcription_test"
  wav_files: "Audio_Local_tests/wav_files"
  rttm_files: "Audio_Local_tests/rttm_files"

production:
  input_folder: "TBD"
  output_folder: "TBD"
  temp_folder: "temp"
```

### config/models.yaml (Future)
```yaml
whisper:
  model_size: "large"
  device: "auto"  # auto-detect GPU/CPU
  fp16: true
  language: "en"

pyannote:
  model: "pyannote/speaker-diarization-3.1"
  num_speakers: 2
  device: "auto"

whisperx:
  model_size: "large-v2"
  device: "auto"
  batch_size: 16
  compute_type: "float16"
```

---

## Folder Conventions

### Naming Conventions

**Audio Files**:
- Format: `{ID}_S{session}_{date}.{ext}`
- Example: `PC1-1025-01_S1_2022.02.28.m4a`

**Output Files**:
- RTTM: `{ID}.rttm` or `clean_{ID}.rttm`
- Transcript: `{ID}.csv` or `{ID}_v2.csv`
- Truth: `{ID}_truth.xlsx`

**Temporary Files**:
- Audio clips: `segment_{n}.wav` or `{n}.wav`
- Cleaned after processing

### File Extensions

- `.m4a`, `.wav`, `.mp3` - Audio files
- `.rttm` - Speaker diarization output
- `.csv` - Transcript output
- `.xlsx`, `.xls` - Ground truth transcripts
- `.txt` - Formatted transcripts
- `.json` - Metrics and config
- `.yaml` - Configuration files
- `.py` - Python source code
- `.ipynb` - Jupyter notebooks
- `.md` - Markdown documentation

---

## Git Structure

### Tracked Files
- Source code (src/, scripts/)
- Documentation (Notes/, README.md)
- Configuration templates
- Notebooks (experimental code)
- requirements.txt

### Ignored Files (.gitignore)
- Audio files (*.m4a, *.wav, *.mp3)
- Output files (*.rttm, transcripts/*.csv)
- Temporary files (audio_clips/*)
- Virtual environments (.venv/)
- IDE files (.vscode/, *.code-workspace)
- Python cache (__pycache__/, *.pyc)
- Model checkpoints (*.pt, *.pth)

---

## Maintenance

### Weekly
- Update TODO.md with progress
- Document new findings in Ideas_notes.md
- Keep metrics in Notes/

### Monthly
- Review and update PROJECT_PLAN.md
- Update requirements.txt versions
- Archive old notebooks if refactored

### As Needed
- Update this file when structure changes
- Document breaking changes in README.md
- Tag releases when major milestones reached

---

**What Files Do What?**

| File | Purpose | Status |
|------|---------|--------|
| src/metrics.py | WER/DER calculation | ✓ Ready |
| src/utils.py | Helper utilities | ✓ Ready |
| transcript_notebook.ipynb | Main pipeline | ⧖ Needs GPU update |
| create_rttm.ipynb | RTTM processing | ⧖ Needs GPU update |
| test_gpu.py | GPU verification | ✓ Ready |
| helper_functions.py | Legacy helpers | ✓ Working (to migrate) |
| one_and_done.py | Speaker processing | ✓ Working |

---

**Last Updated**: 2025-12-23
