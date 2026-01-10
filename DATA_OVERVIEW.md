# Data Overview - Current Test Files and Training Data

**Last Updated**: January 7, 2026
**Purpose**: Document current test files and training data used for baseline and fine-tuning

---

## Current Data Status

### Test Data (For Baseline Evaluation)
Located in: `Audio_Local_tests/`

### Training Data (For Fine-Tuning)
Located in: `training_data/` and `prepared_datasets/whisper_finetuning/`

---

## 1. Test Audio Files

### Location: `Audio_Local_tests/audio_files/`

| File | Type | Purpose | Status |
|------|------|---------|--------|
| `GPT_test.m4a` | Interview (GPT-generated) | Baseline testing | ✅ Active |
| `GPT_test2.m4a` | Interview (GPT-generated) | Current experiment | ✅ Active |
| `hamlet_test.m4a` | Shakespeare reading | Baseline testing | ✅ Available |

**Audio Specifications**:
- Sample rate: Various (converted to 16kHz for processing)
- Format: M4A (converted to WAV for Whisper)
- Type: Interview conversations

---

## 2. Ground Truth Transcripts

### Location: `Audio_Local_tests/transcription_test/`

| File | Audio File | Format | Segments | Words | Description |
|------|------------|--------|----------|-------|-------------|
| `gpt_test_truth.csv` | GPT_test.m4a | CSV | ~79 | ~1045 | Speaker-labeled interview transcript |
| `gpt_test2_truth.csv` | GPT_test2.m4a | CSV | ~79 | ~1045 | Current experiment ground truth |
| `hamlet_test_truth.xlsx` | hamlet_test.m4a | Excel | N/A | N/A | Shakespeare reading |

**CSV Format**:
```csv
Speaker,Transcription
Interviewer,"Before we get started, I want to briefly explain..."
Interviewee,"That sounds good to me. I think having that structure..."
```

**Content Type**:
- Two-person interview conversations (Interviewer + Interviewee)
- Medical/caregiving scenario (GPT test 2 mentions caring for father "Robert" with Parkinson's)
- Speaker turns clearly labeled

---

## 3. Training Data (Current Fine-Tuning Dataset)

### Aligned Training Data

**Location**: `training_data/`

| File | Purpose | Created | Size |
|------|---------|---------|------|
| `gpt_test_aligned.json` | Timestamped aligned segments | Dec 28, 2025 | Force-aligned with WhisperX |
| `gpt_test_metadata.json` | Dataset metadata | Dec 28, 2025 | Stats and info |

**Metadata** (from `gpt_test_metadata.json`):
```json
{
  "created_at": "2025-12-28T12:56:32",
  "audio_file": "gpt_test.m4a",
  "ground_truth_file": "gpt_test_truth.csv",
  "duration_seconds": 383.508,
  "num_segments": 79,
  "num_words": 1045,
  "alignment_model": "whisperx",
  "language": "en"
}
```

**Data Stats**:
- **Duration**: 383.5 seconds (~6.4 minutes)
- **Segments**: 79 speaker turns
- **Words**: 1,045 total words
- **Alignment**: WhisperX forced alignment with timestamps

---

## 4. HuggingFace Dataset (Fine-Tuning Format)

### Location: `prepared_datasets/whisper_finetuning/`

**Structure**:
```
prepared_datasets/whisper_finetuning/
├── train/
│   ├── data-00000-of-00001.arrow  (252MB)
│   ├── dataset_info.json
│   └── state.json
└── whisper_finetuning_simple/
    └── ... (alternative format)
```

**Dataset Info**:
```json
{
  "features": {
    "audio": {
      "sampling_rate": 16000,
      "_type": "Audio"
    },
    "sentence": {
      "dtype": "string",
      "_type": "Value"
    }
  }
}
```

**Format**: HuggingFace Arrow format
- Audio at 16kHz
- Text transcriptions
- Ready for Whisper fine-tuning
- Size: ~252MB (includes audio data)

---

## 5. Baseline Test Results

### Location: `Audio_Local_tests/baseline_output/`

**WhisperX Results**:
```
baseline_output/whisperx/
├── gpt_test_wx-lv2/
│   └── gpt_test_wx-lv2_large-v2_cuda.csv
├── gpt_test2_wx-lv2/
│   └── gpt_test2_wx-lv2_large-v2_cuda.csv
└── hamlet_test/
    ├── hamlet_test_whisperx_large-v2_cuda.csv
    ├── hamlet_test_whisperx_large-v3_cuda.csv
    └── hamlet_test_whisperx_medium.en_cuda.csv
```

**Whisper Results** (for comparison):
```
baseline_output/whisper/hamlet_test/
├── hamlet_test_whisper_large-v2_gpu.csv
├── hamlet_test_whisper_large-v3_gpu.csv
└── hamlet_test_whisper_medium.en_gpu.csv
```

**Baseline WER** (from Notes/MODEL_COMPARISON.md):
- WhisperX large-v2: **25.01% WER** (best)
- Whisper large-v2: 27.60% WER
- WhisperX medium.en: 28.30% WER

---

## 6. Fine-Tuning Results (Dec 29, 2025)

### Training Configuration
- **Base Model**: openai/whisper-large-v2
- **Method**: LoRA (rank=32, alpha=64)
- **Training Data**: GPT test set (79 segments, 1045 words)
- **Epochs**: 5
- **Batch Size**: 2 (effective 16 with gradient accumulation)
- **Learning Rate**: 1e-5

### Results
- ✅ Training completed successfully
- ✅ Model saved to `models/whisper-finetuned-interviews/final_model/`
- Loss: 6.24 → 6.08 (2.6% reduction)

**Note**: This was fine-tuned on the SAME data used for baseline (GPT test set) - essentially a proof-of-concept that the pipeline works. For production, you'll want to use different/larger interview datasets.

---

## 7. Your Interview Transcript Data (To Be Added)

### What You Have
You mentioned having **complete interview transcripts** ready for fine-tuning.

### Recommended Directory Structure

```
training_data/
├── raw/                           # Your original interview data
│   ├── interviews/
│   │   ├── audio/
│   │   │   ├── interview_001.m4a
│   │   │   ├── interview_002.m4a
│   │   │   └── ...
│   │   └── transcripts/
│   │       ├── interview_001_truth.csv
│   │       ├── interview_002_truth.csv
│   │       └── ...
│
├── aligned/                       # After forced alignment
│   └── interviews/
│       ├── interview_001_aligned.json
│       ├── interview_002_aligned.json
│       └── ...
│
└── inventory.json                # Metadata about all interviews
```

### Expected Format (CSV)
```csv
Speaker,Transcription
Interviewer,"Question or statement..."
Interviewee,"Response..."
Interviewer,"Follow-up..."
```

---

## 8. Data Preparation Workflow

### For Your Interview Data

#### Step 1: Organize Raw Data
```bash
# Place your interview audio in:
training_data/raw/interviews/audio/

# Place your transcripts in:
training_data/raw/interviews/transcripts/
```

#### Step 2: Create Aligned Training Data
```bash
python src/data/create_training_data.py \
  --audio-dir training_data/raw/interviews/audio \
  --transcript-dir training_data/raw/interviews/transcripts \
  --output training_data/aligned/interviews
```

This will:
- Load each audio file
- Load corresponding transcript
- Use WhisperX to force-align text to audio
- Create timestamped training data
- Save as JSON files

#### Step 3: Convert to HuggingFace Dataset
```bash
python src/data/prepare_hf_dataset.py \
  --input training_data/aligned/interviews \
  --output prepared_datasets/whisper_finetuning_interviews \
  --train-split 0.8 \
  --val-split 0.1 \
  --test-split 0.1
```

This will:
- Load all aligned JSON files
- Split into train/validation/test
- Convert to HuggingFace Arrow format
- Save ready for fine-tuning

#### Step 4: Update Configuration
Edit `config/training.yaml`:
```yaml
dataset:
  input_dir: "prepared_datasets/whisper_finetuning_interviews"
```

Or create new experiment in `config/experiments.yaml`:
```yaml
my_interviews:
  name: "interview-dataset"
  description: "Fine-tuning on real interview transcripts"
  dataset:
    input_dir: "prepared_datasets/whisper_finetuning_interviews"
```

#### Step 5: Run Training
```bash
python src/scripts/run_training.py --experiment baseline_lora
```

---

## 9. Data Quality Considerations

### Current GPT Test Data Characteristics
- ✅ Clean speech (GPT-generated, likely read aloud)
- ✅ Clear speaker labels
- ✅ Good audio quality
- ⚠️ Small dataset (79 segments, ~6 minutes)
- ⚠️ Limited vocabulary/domain
- ⚠️ May not represent real interview variability

### For Your Real Interview Data
Consider:
- **Audio quality**: Background noise, microphone quality
- **Speaker overlap**: Do speakers talk over each other?
- **Accents/dialects**: Regional variations
- **Technical terms**: Domain-specific vocabulary
- **Audio length**: Longer interviews need chunking
- **Transcription accuracy**: Ground truth quality matters

### Recommended Dataset Size
- **Minimum**: 30-60 minutes (proof of concept)
- **Good**: 2-5 hours (noticeable improvement)
- **Better**: 10+ hours (significant domain adaptation)
- **Best**: 50+ hours (production quality)

---

## 10. Current vs Future Data

### Current (GPT Test Set)
```
Dataset: GPT test (proof of concept)
Duration: ~6 minutes
Segments: 79
Words: 1,045
Purpose: Validate pipeline, baseline testing
Status: ✅ Working, used for Dec 29 fine-tuning
```

### Future (Your Interview Data)
```
Dataset: Real interview transcripts
Duration: TBD (you have "complete transcripts")
Segments: TBD
Words: TBD
Purpose: Production fine-tuning
Status: ⏳ Ready to prepare
```

---

## 11. Data Files Reference

### Audio Files
- **Test Audio**: `Audio_Local_tests/audio_files/*.m4a`
- **Converted Audio**: `Audio_Local_tests/wav_files/*.wav`

### Transcripts
- **Ground Truth**: `Audio_Local_tests/transcription_test/*_truth.csv`
- **Baseline Output**: `Audio_Local_tests/baseline_output/whisperx/*.csv`

### Training Data
- **Aligned**: `training_data/*_aligned.json`
- **Metadata**: `training_data/*_metadata.json`
- **HF Dataset**: `prepared_datasets/whisper_finetuning/train/`

### Models
- **Fine-tuned**: `models/whisper-finetuned-interviews/final_model/`
- **Checkpoints**: `models/whisper-finetuned-interviews/checkpoint-*/`

---

## 12. Next Steps for Your Data

### Immediate Actions
1. ✅ Review current test data structure (this document)
2. ➡️ **Organize your interview audio files**
3. ➡️ **Organize your interview transcripts** (CSV format with Speaker column)
4. ➡️ **Run data preparation pipeline** (steps 1-3 above)
5. ➡️ **Start baseline training** on your data

### Questions to Answer
- How many interview audio files do you have?
- What's the total duration?
- What format are your transcripts in? (CSV, TXT, etc.)
- Are speakers already labeled in transcripts?
- What's the audio quality like?

### Data Preparation Checklist
- [ ] Count total interview files
- [ ] Check total audio duration
- [ ] Verify transcript format matches expected CSV structure
- [ ] Ensure speaker labels are consistent
- [ ] Check audio file formats (M4A, WAV, MP3)
- [ ] Organize into `training_data/raw/interviews/` structure
- [ ] Run alignment script
- [ ] Convert to HuggingFace format
- [ ] Run first training experiment

---

## Summary

**Current Test Data**: GPT-generated interview (~6 min, 79 segments, 1045 words)
- Used for baseline evaluation (25.01% WER with WhisperX large-v2)
- Used for proof-of-concept fine-tuning (Dec 29, 2025)
- Pipeline validated and working ✅

**Your Interview Data**: Ready to be prepared
- Complete interview transcripts available
- Need to organize, align, and convert to HF format
- Will provide real-world fine-tuning data
- Expected to significantly improve domain-specific performance

**Workspace Status**: ✅ Ready to process your interview data
- Data preparation scripts in `src/data/`
- Configuration system ready
- Training pipeline validated
- Documentation complete

**Your Next Command**:
```bash
# After organizing your interview files:
python src/data/create_training_data.py \
  --audio-dir training_data/raw/interviews/audio \
  --transcript-dir training_data/raw/interviews/transcripts \
  --output training_data/aligned/interviews
```

---

**Questions?** See:
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Complete workflow
- [WORKSPACE_READY.md](WORKSPACE_READY.md) - Getting started
- `src/data/create_training_data.py` - Data preparation script
