# Preparing Your 41 Interview Transcripts for Fine-Tuning

**Your Dataset**: 41 audio files × 45-60 minutes each = **30-41 hours of audio**
**Status**: ✅ **Excellent size** for production fine-tuning
**Expected Result**: Significant domain-specific improvement

---

## Dataset Overview

### Your Data
- **Files**: 41 interview transcripts with audio
- **Duration per file**: 45-60 minutes
- **Total duration**: ~30-41 hours
- **Expected segments**: ~15,000-20,000 speaker turns (estimated)
- **Expected words**: ~300,000-400,000 words (estimated)

### Comparison to Current Test Data
- **Current (GPT test)**: 6 minutes, 79 segments, 1,045 words
- **Your data**: **~300x larger** - professional training dataset! 🎉

---

## Preparation Workflow

### Phase 1: Organization (30 minutes)

#### Step 1: Set Location for cloud-based files

#### Step 2: Identify training vs validation set for transcribed files. 

**Naming Convention**: Will need to confirm naming is consistent.
```
training_data/raw/interviews/
├── audio/
│   ├── PC1-1025-01_S1_2022.02.28.m4a
│   ├── ...
│   ├── ...
│   └── PC1-1739-01_S3_2022.03.22.m4a
└── transcripts/
    ├── PC1-1025-01_S1_2022.02.28.doc
    ├── ...
    ├── ...
    └── PC1-1739_01_S3_2022.03.22.doc
```



---

### Phase 2: Verify Transcript Format 

#### Required CSV Format
Your transcripts should be formatted like this:

```csv
Speaker,Transcription
Interviewer,"Before we get started, I want to briefly explain..."
Interviewee,"That sounds good to me. I think having that structure..."
Interviewer,"Let's start with the basics..."
Interviewee,"I guess the main problem is..."
```

#### Quick Check Script

Create `check_transcripts.py`:
```python
import pandas as pd
from pathlib import Path

transcript_dir = Path("...") 

for csv_file in sorted(transcript_dir.glob("*.csv")):
    try:
        df = pd.read_csv(csv_file)

        # Check required columns
        if 'Speaker' not in df.columns or 'Transcription' not in df.columns:
            print(f"❌ {csv_file.name}: Missing required columns")
            print(f"   Has: {list(df.columns)}")
            continue

        # Check data
        num_segments = len(df)
        speakers = df['Speaker'].unique()

        print(f"✅ {csv_file.name}: {num_segments} segments, speakers: {speakers}")

    except Exception as e:
        print(f"❌ {csv_file.name}: Error - {e}")
```

Run it:
```bash
python check_transcripts.py
```

**If your transcripts are in a different format**, will need to format transcripts using previous code.

---

### Phase 3: Force Alignment (LONGEST - Plan 10-15 hours)

This step aligns your ground truth transcripts to audio using WhisperX.

**Time estimate**:
- ~15-20 minutes per file × 41 files = **10-14 hours total**
- Can run overnight or in batches

#### Option A: Process All Files at Once
```bash
python src/data/create_training_data.py \
  --audio-dir training_data/raw/interviews/audio \
  --transcript-dir training_data/raw/interviews/transcripts \
  --output training_data/aligned/interviews
```

**This will**:
- Process all 41 files sequentially
- Create `PC1-1025-01_S1_aligned.json`, `PC1-1025-01_S2.json`, etc.
- Take 10-14 hours total
- Safe to run overnight

#### Option B: Process in Batches (Recommended)

If you want to test first or process in smaller batches:

**Batch 1: Test with 3 files**
```bash
# Copy just 3 files to test
mkdir -p training_data/raw/interviews_test/audio
mkdir -p training_data/raw/interviews_test/transcripts

cp training_data/raw/interviews/audio/interview_001.m4a training_data/raw/interviews_test/audio/
cp training_data/raw/interviews/audio/interview_002.m4a training_data/raw/interviews_test/audio/
cp training_data/raw/interviews/audio/interview_003.m4a training_data/raw/interviews_test/audio/

cp training_data/raw/interviews/transcripts/interview_001_truth.csv training_data/raw/interviews_test/transcripts/
cp training_data/raw/interviews/transcripts/interview_002_truth.csv training_data/raw/interviews_test/transcripts/
cp training_data/raw/interviews/transcripts/interview_003_truth.csv training_data/raw/interviews_test/transcripts/

# Process test batch
python src/data/create_training_data.py \
  --audio-dir training_data/raw/interviews_test/audio \
  --transcript-dir training_data/raw/interviews_test/transcripts \
  --output training_data/aligned/interviews_test
```

**After verifying it works**, process all 41 files.

#### Progress Monitoring

The script will show progress:
```
Processing interview_001.m4a...
  Loading audio: interview_001.m4a
  Duration: 2847.32 seconds (47.46 minutes)
  Loading ground truth: interview_001_truth.csv
  Full text: 8,234 words
  Performing forced alignment...
  Alignment complete: 234 segments
  Saved: training_data/aligned/interviews/interview_001_aligned.json

Processing interview_002.m4a...
...
```

---

### Phase 4: Convert to HuggingFace Dataset (1-2 hours)

After alignment completes, convert to training format:

```bash
python src/data/prepare_hf_dataset.py \
  --input training_data/aligned/interviews \
  --output prepared_datasets/whisper_finetuning_interviews \
  --train-split 0.8 \
  --val-split 0.1 \
  --test-split 0.1
```

**This will**:
- Load all 41 aligned JSON files
- Split into train (33 files), validation (4 files), test (4 files)
- Convert to HuggingFace Arrow format
- Create `prepared_datasets/whisper_finetuning_interviews/`

**Expected output**:
```
prepared_datasets/whisper_finetuning_interviews/
├── train/           # ~33 files worth of data
├── validation/      # ~4 files for validation
└── test/            # ~4 files for final evaluation
```

**Dataset size estimate**: 10-15 GB (with audio)

---

### Phase 5: Update Configuration (5 minutes)

Edit `config/experiments.yaml`:

```yaml
# Add your dataset
datasets:
  gpt_test:
    name: "gpt_test_set"
    description: "GPT-generated test set (79 segments, 1045 words)"
    input_dir: "datasets/whisper_finetuning"

  interview_production:  # NEW
    name: "production_interviews"
    description: "41 real interview transcripts (~30-41 hours)"
    input_dir: "prepared_datasets/whisper_finetuning_interviews"
    enabled: true
    stats:
      num_files: 41
      duration_hours: 35  # Approximate
      segments: 16000  # Estimate

# Set as active dataset
active_dataset: "interview_production"
```

Or update `config/training.yaml`:
```yaml
dataset:
  input_dir: "prepared_datasets/whisper_finetuning_interviews"  # Point to your data
```

---

### Phase 6: Run Baseline Training (2-6 hours)

#### Quick Test First (Recommended)
```bash
# Test with quick_test experiment (1 epoch)
python src/scripts/run_training.py --experiment quick_test
```

**This will**:
- Use minimal LoRA (rank 8)
- Train for 1 epoch
- Complete in ~30-60 minutes
- Verify pipeline works with your data

#### Full Training
```bash
# Run proven baseline configuration
python src/scripts/run_training.py --experiment baseline_lora
```

**Training estimates** (41 hours of data):
- **Epochs**: 5
- **Time per epoch**: ~40-60 minutes (depends on GPU)
- **Total training time**: 3-5 hours
- **GPU memory**: 6GB (RTX 3050 should work)

#### Monitor Training
```bash
tensorboard --logdir models/whisper-finetuned-lora-r32/logs
# Open http://localhost:6006
```

---

## Recommended Workflow Schedule

### Day 1: Setup & Test (3 hours)
1. ✅ Organize 41 files into directory structure (30 min)
2. ✅ Verify transcript formats with check script (30 min)
3. ✅ Process 3 test files through alignment (1 hour)
4. ✅ Verify aligned output looks good (15 min)
5. ✅ Convert test batch to HF format (15 min)
6. ✅ Run quick_test training on test batch (30 min)

### Day 2: Full Data Preparation (Overnight)
7. ✅ Start alignment on all 41 files (10-14 hours - **run overnight**)

### Day 3: Training
8. ✅ Verify alignment completed successfully (30 min)
9. ✅ Convert all aligned data to HF format (1-2 hours)
10. ✅ Update configuration (5 min)
11. ✅ Run baseline_lora training (3-5 hours)

### Day 4: Evaluation & Iteration
12. ✅ Evaluate results on test set
13. ✅ Compare to baseline
14. ✅ Try different experiments (lora_large, high_lr, etc.)

---

## Expected Results

### With 30-41 Hours of Data

**Baseline (before fine-tuning)**:
- WhisperX large-v2: ~25% WER (general domain)

**After fine-tuning** (expected):
- Interview-specific: **15-20% WER** (40% relative improvement)
- Better handling of:
  - Domain-specific terminology
  - Speaker patterns
  - Interview conversation style
  - Your specific audio characteristics

**Why this will work well**:
- ✅ Large dataset (30-41 hours)
- ✅ Domain-specific (interviews)
- ✅ Consistent format
- ✅ Good quality transcripts
- ✅ Proven training pipeline

---

## Disk Space Requirements

Plan for:
- **Raw audio**: 41 files × ~500MB = ~20 GB
- **Aligned JSON**: 41 files × ~50MB = ~2 GB
- **HF dataset**: ~10-15 GB (includes audio)
- **Model checkpoints**: ~5 GB
- **Total needed**: ~40-50 GB free space

---

## Troubleshooting

### If Alignment is Too Slow
- Process in batches of 5-10 files
- Use GPU (CUDA) - much faster than CPU
- Run overnight for large batches

### If You Run Out of Memory During Alignment
- Process files one at a time
- Close other applications
- Reduce WhisperX batch_size in code

### If Training is Too Slow
- Reduce batch size in config
- Use quick_test experiment first
- Train on subset of data initially

### If Disk Space is Low
- Process in batches
- Delete intermediate files after conversion
- Use external drive for datasets

---

## Quick Start Commands

```bash
# 1. Organize files (do this manually)
mkdir -p training_data/raw/interviews/{audio,transcripts}

# 2. Check transcripts
python check_transcripts.py

# 3. Align data (SLOW - 10-14 hours)
python src/data/create_training_data.py \
  --audio-dir training_data/raw/interviews/audio \
  --transcript-dir training_data/raw/interviews/transcripts \
  --output training_data/aligned/interviews

# 4. Convert to HF format
python src/data/prepare_hf_dataset.py \
  --input training_data/aligned/interviews \
  --output prepared_datasets/whisper_finetuning_interviews \
  --train-split 0.8 --val-split 0.1 --test-split 0.1

# 5. List experiments
python src/scripts/list_experiments.py

# 6. Run training
python src/scripts/run_training.py --experiment baseline_lora

# 7. Monitor
tensorboard --logdir models/whisper-finetuned-lora-r32/logs
```

---

## Next Steps

1. **Right now**: Organize your 41 audio + transcript files into the directory structure
2. **Verify formats**: Run the check script to ensure transcripts are formatted correctly
3. **Test batch**: Process 3 files to verify pipeline works
4. **Overnight**: Run alignment on all 41 files
5. **Next day**: Convert to HF format and start training

---

## Questions Before You Start?

- What format are your transcripts currently in?
- Do they have Speaker labels already?
- What's the audio format? (M4A, WAV, MP3?)
- Do you want to start with a test batch of 3-5 files first?

---

**You're in great shape!** 30-41 hours of domain-specific data is excellent for fine-tuning. This should give you production-quality results. 🚀

**Your next action**: Organize the 41 files into the directory structure and run the check script to verify transcript formats.
