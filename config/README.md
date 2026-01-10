# Configuration System

This directory contains YAML configuration files for the Audio Transcription Deidentification project.

## Overview

The configuration system centralizes all settings in three YAML files:

- **`paths.yaml`** - File paths and directories
- **`models.yaml`** - Model selection and parameters
- **`processing.yaml`** - Processing pipeline settings

## Usage

### Basic Usage

```python
from src.config import Config

# Load configuration
config = Config()

# Access settings
audio_dir = config.paths.audio_files
model_name = config.models.whisperx.model
batch_size = config.models.whisperx.batch_size
sample_rate = config.processing.audio.sample_rate
```

### Using in Scripts

```python
from src.config import load_config

config = load_config()

# Get device (auto-detects CUDA availability)
device = config.get_device()

# Get FFmpeg path
ffmpeg_path = config.get_ffmpeg_path()

# Get Hugging Face token from environment
hf_token = config.get_hf_token()
```

## Configuration Files

### paths.yaml

Defines all file paths used in the project:

```yaml
project_root: "."
audio_files: "Audio_Local_tests/audio_files"
transcripts: "Audio_Local_tests/transcription_test"
output: "Audio_Local_tests/baseline_output"

training:
  root: "training_data"
  aligned: "training_data/aligned"
  inventory: "training_data/inventory.json"

external:
  ffmpeg: "C:\\path\\to\\ffmpeg\\bin"
```

**Key Sections:**
- **Base directories** - Audio files, transcripts, output
- **Training data paths** - Aligned data, inventory
- **Temporary paths** - Audio clips, WAV conversions
- **External dependencies** - FFmpeg, Hugging Face token

### models.yaml

Configures model selection and parameters:

```yaml
whisperx:
  enabled: true
  model: "large-v2"  # Best WER: 25.01%
  device: "cuda"
  batch_size: 4  # Optimized for 6GB GPU
  language: "en"

whisper:
  model: "large-v2"
  device: "cuda"
  in_memory: true  # Prevents OSError on Windows

pyannote:
  model: "pyannote/speaker-diarization-3.1"
  device: "cuda"
```

**Key Sections:**
- **WhisperX configuration** - Primary transcription model
- **Whisper fallback** - Alternative transcription
- **Pyannote diarization** - Speaker detection
- **Auto-device** - Device selection preferences
- **Memory management** - Batch size recommendations

### processing.yaml

Controls processing pipeline behavior:

```yaml
audio:
  sample_rate: 16000
  trim_silence: true
  target_format: "wav"

alignment:
  return_char_alignments: false
  min_word_confidence: 0.0

diarization:
  enabled: true
  min_speakers: null  # auto-detect
  max_speakers: null

output:
  format: "json"
  json:
    include_word_timestamps: true
    include_speaker_labels: true

metrics:
  calculate_wer: true
  track_timing: true
```

**Key Sections:**
- **Audio preprocessing** - Sample rate, format conversion
- **Alignment settings** - Word-level timestamp configuration
- **Diarization settings** - Speaker detection parameters
- **Output format** - JSON/CSV/TXT options
- **Metrics** - WER calculation, performance tracking
- **Batch processing** - Parallel processing settings
- **Training export** - Data preparation for fine-tuning

## Environment Variables

The configuration system respects these environment variables:

- `HF_TOKEN` - Hugging Face authentication token (required for Pyannote)
- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` - Set to "1" for WhisperX compatibility

## Model Selection Guide

Based on comprehensive testing ([Notes/MODEL_COMPARISON.md](../Notes/MODEL_COMPARISON.md)):

### Recommended: WhisperX large-v2
- **WER:** 25.01% (best accuracy)
- **Speed:** 2.26x realtime
- **Features:** Forced alignment, integrated diarization
- **Use case:** Training data preparation, high-accuracy transcription

### Alternative: WhisperX medium.en
- **WER:** 28.30%
- **Speed:** 6.46x realtime (fastest)
- **Use case:** Batch processing, speed-critical applications

### Not Recommended:
- **Whisper large models** - Too slow (0.6x realtime)
- **large-v3** - No advantage over large-v2

## Customization

### Changing the Model

Edit `config/models.yaml`:

```yaml
whisperx:
  model: "medium.en"  # Switch to faster model
  batch_size: 8       # Increase batch size
```

### Adding New Paths

Edit `config/paths.yaml`:

```yaml
custom:
  my_output: "custom_output_dir"
```

Access in code:

```python
config = Config()
my_path = config.paths.custom['my_output']
```

### Adjusting Processing

Edit `config/processing.yaml`:

```yaml
audio:
  sample_rate: 48000  # Higher quality
  channels: 2         # Stereo

batch:
  enabled: true       # Enable batch processing
  max_workers: 2      # Parallel workers
```

## Testing Configuration

Run the test script to verify configuration:

```bash
python test_config.py
```

Expected output:
```
Loading configuration...

=== Paths Configuration ===
Project Root: .
Audio Files: Audio_Local_tests/audio_files
...

[OK] Configuration system loaded successfully!
```

## Migration Guide

### Updating Existing Scripts

**Before (hardcoded):**
```python
MODEL = "large-v2"
AUDIO_DIR = "Audio_Local_tests/audio_files"
batch_size = 4
```

**After (using config):**
```python
from src.config import Config

config = Config()
MODEL = config.models.whisperx.model
AUDIO_DIR = config.paths.audio_files
batch_size = config.models.whisperx.batch_size
```

### Benefits
- ✅ Centralized configuration
- ✅ Environment-agnostic paths
- ✅ Type-safe access
- ✅ Easy model switching
- ✅ No hardcoded values

## File Structure

```
config/
├── README.md           # This file
├── paths.yaml          # Path configuration
├── models.yaml         # Model configuration
└── processing.yaml     # Processing configuration

src/
└── config.py           # Configuration loader module

test_config.py          # Configuration test script
```

## Troubleshooting

### FileNotFoundError: Configuration file not found

Make sure you're running scripts from the project root directory:

```bash
cd c:\Users\rpaul\Documents\GitHub\Audio_Transcription_Deidentification
python test_config.py
```

### CUDA not available

The config system automatically falls back to CPU if `auto_device.fallback_to_cpu: true` in `models.yaml`.

To force CPU mode:
```yaml
whisperx:
  device: "cpu"
```

### Missing environment variables

Set required environment variables:

```bash
# Windows
set HF_TOKEN=your_token_here
set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# Linux/Mac
export HF_TOKEN=your_token_here
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

## Next Steps

1. **Update existing scripts** - Migrate `test_whisperx.py` and `run_baseline_test.py` to use config
2. **Transcript inventory** - Create script to scan available data
3. **Alignment pipeline** - Build batch processing with WhisperX
4. **Training data export** - Prepare aligned data for fine-tuning

See [Notes/CODEBASE_REORG_PLAN.md](../Notes/CODEBASE_REORG_PLAN.md) for detailed roadmap.
