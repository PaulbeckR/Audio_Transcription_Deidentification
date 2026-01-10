# Experiment Configuration Guide

## Quick Start

To run a test with different files, simply edit `config/experiment.yaml`:

```yaml
audio_file: "your_audio.m4a"
transcript_truth: "your_ground_truth.txt"  # or null if no ground truth
experiment_name: "my_test_name"
```

Then run your test script - **no code changes needed!**

```bash
# WhisperX
.venv_whisperx\Scripts\python.exe test_whisperx.py

# Whisper: Need to reverify environment - installsed whisperx accidentally
.venv\Scripts\python.exe run_baseline_test.py
```

## Configuration Fields

### `audio_file` (required)
- **Type**: String
- **Description**: Filename of audio file to process
- **Location**: Will look in `Audio_Local_tests/audio_files/`
- **Example**: `"hamlet_test.m4a.wav"`, `"gpt_test.m4a"`

### `transcript_truth` (optional)
- **Type**: String or `null`
- **Description**: Ground truth transcript for WER calculation
- **Location**: Will look in `Audio_Local_tests/transcription_test/`
- **Example**: `"hamlet_test_truth.xlsx"`, `"gpt_test_truth.txt"`, `null`
- **Note**: Set to `null` if you don't have ground truth

### `experiment_name` (required)
- **Type**: String
- **Description**: Name for this test run (used in output folders/filenames)
- **Example**: `"hamlet_test"`, `"gpt_test_wx-lv2"`, `"pretrain_sample"`

## How It Works

### File Paths

The experiment config automatically constructs full paths:

```python
# Audio file
AUDIO_FILE = "Audio_Local_tests/audio_files/{audio_file}"

# Ground truth (if specified)
GROUND_TRUTH = "Audio_Local_tests/transcription_test/{transcript_truth}"

# Output folder
OUTPUT_FOLDER = "Audio_Local_tests/baseline_output/whisperx/{experiment_name}"
```

### Output Files

All outputs use the experiment name:

**WhisperX outputs:**
- `{experiment_name}_{model}_{device}.json` - Full transcript with alignment
- `{experiment_name}_{model}_{device}.csv` - CSV format transcript
- `{experiment_name}_metrics_{model}_{device}.json` - Performance metrics

**Example with `experiment_name: "gpt_test_wx-lv2"`:**
- `gpt_test_wx-lv2_large-v2_cuda.json`
- `gpt_test_wx-lv2_large-v2_cuda.csv`
- `gpt_test_wx-lv2_metrics_large-v2_cuda.json`

## Example Configurations

### Test with Ground Truth
```yaml
audio_file: "hamlet_test.m4a.wav"
transcript_truth: "hamlet_test_truth.xlsx"
experiment_name: "hamlet_baseline"
```

### Test without Ground Truth
```yaml
audio_file: "new_recording.wav"
transcript_truth: null
experiment_name: "new_test_run"
```

### Pretraining Data Test
```yaml
audio_file: "pretrain_sample.m4a"
transcript_truth: null  # No ground truth for pretraining
experiment_name: "pretrain_sample_01"
```

### GPT-Generated Audio Test
```yaml
audio_file: "gpt_test.m4a"
transcript_truth: "gpt_test_truth.txt"
experiment_name: "gpt_test_wx-lv2"
```

## Workflow

1. **Prepare your audio file**
   - Place in `Audio_Local_tests/audio_files/`
   - Note the filename

2. **Prepare ground truth** (optional)
   - Place in `Audio_Local_tests/transcription_test/`
   - Can be `.xlsx`, `.txt`, or `.csv`
   - Note the filename

3. **Edit `config/experiment.yaml`**
   ```yaml
   audio_file: "your_file.wav"
   transcript_truth: "your_truth.txt"  # or null
   experiment_name: "descriptive_name"
   ```

4. **Run test**
   ```bash
   .venv_whisperx\Scripts\python.exe test_whisperx.py
   ```

5. **Check outputs**
   - Look in `Audio_Local_tests/baseline_output/whisperx/{experiment_name}/`
   - Files will be named `{experiment_name}_*`

## Tips

### Naming Conventions

Use descriptive experiment names that indicate:
- **Source**: `hamlet`, `gpt`, `pretrain`
- **Model**: `wx-lv2` (WhisperX large-v2), `w-base` (Whisper base)
- **Version**: `v1`, `v2`, `test1`

Examples:
- `hamlet_test` - Original baseline
- `gpt_test_wx-lv2` - GPT audio with WhisperX large-v2
- `pretrain_sample_01` - First pretraining sample
- `podcast_ep1_test` - Podcast episode 1

### Multiple Tests

To run multiple tests sequentially:

1. Edit `experiment.yaml` for test 1
2. Run `test_whisperx.py`
3. Edit `experiment.yaml` for test 2
4. Run `test_whisperx.py` again

Each test creates separate output folders based on `experiment_name`.

### Ground Truth Formats

Supported formats:
- **Excel (`.xlsx`)**: Must have a column named "Transcription" or "Text"
- **Text (`.txt`)**: Plain text, one line or full transcript
- **CSV (`.csv`)**: Must have appropriate text column

### No Ground Truth

If you don't have ground truth:
```yaml
transcript_truth: null
```

The script will:
- Skip WER calculation
- Still produce transcripts and metrics
- Show warning: "No ground truth file configured"

## Integration with Other Configs

The experiment config works alongside:

### `models.yaml` - Model Selection
```yaml
whisperx:
  model: "large-v2"  # Change model here
  batch_size: 4
```

### `paths.yaml` - Directory Structure
```yaml
audio_files: "Audio_Local_tests/audio_files"
transcripts: "Audio_Local_tests/transcription_test"
output_dirs:
  whisperx: "Audio_Local_tests/baseline_output/whisperx"
```

### `processing.yaml` - Processing Settings
```yaml
audio:
  sample_rate: 16000
alignment:
  return_char_alignments: false
```

## Troubleshooting

### "Audio file not found"
- Check filename spelling in `experiment.yaml`
- Ensure file is in `Audio_Local_tests/audio_files/`
- Include file extension: `"file.wav"` not `"file"`

### "Ground truth file not found"
- Check filename in `experiment.yaml`
- Ensure file is in `Audio_Local_tests/transcription_test/`
- Or set to `null` if no ground truth

### Output folder conflicts
- Each `experiment_name` creates a new subfolder
- Use unique names to avoid overwriting results
- Or use `skip_if_exists: true` in settings (future feature)

## See Also

- [config/README.md](README.md) - Main configuration guide
- [models.yaml](models.yaml) - Model selection and parameters
- [paths.yaml](paths.yaml) - Directory structure
- [processing.yaml](processing.yaml) - Processing settings