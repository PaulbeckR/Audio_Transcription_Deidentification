# Pretraining Data Experiment - Detailed Plan

**Created:** 2025-12-27
**Objective:** Test whether adding pretraining/fine-tuning data from GPT_test improves model performance on GPT_test2
**Model:** WhisperX large-v2 (baseline: 25.01% WER on hamlet_test)
**Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU

---

## Executive Summary

This experiment will test the impact of domain-specific fine-tuning on Whisper's performance. We'll use GPT_test as training data to create a fine-tuned model, then evaluate whether it performs better on GPT_test2 (same domain, different speakers) compared to the baseline model. The goal is to use gpt_test and gpt_test2 to build infra related to pre-train and then formally test on real-life data.

**Key Question:** Does fine-tuning on one interview improve transcription accuracy on similar interviews with different speakers?

---

## Background Context

### Interview Characteristics
- **Format:** Structured interviews (problem-defining, goal-setting)
- **Interviewers:** 4 total, majority from 3 interviewers
- **Interviewees:** All different individuals (1-3 sessions each)
- **Language Pattern:** High overlap/consistency due to structured format
- **Domain:** Clinical/therapeutic interviews with specific vocabulary and phrasing

### Current Model Status
- **Selected Model:** WhisperX large-v2
- **Baseline Performance:** 25.01% WER (hamlet_test)
- **Speed:** 2.26x realtime
- **Features:** Word-level timestamps, integrated diarization

### Challenge
- Pre-transcribed ground truth files **do not have timestamps**
- Need to create timestamped training data through forced alignment

---

## Test Files Overview

### GPT_test
- **Audio:** `Audio_Local_tests/audio_files/GPT_test.m4a` (3.2 MB)
- **Ground Truth:** `Audio_Local_tests/transcription_test/gpt_test_truth.txt`
- **Format:** CSV with Speaker, Transcription columns
- **Speakers:** Interviewer, Interviewee (2 speakers)
- **Use:** Training/fine-tuning data source

### GPT_test2
- **Audio:** `Audio_Local_tests/audio_files/GPT_test2.m4a` (to be confirmed)
- **Ground Truth:** `Audio_Local_tests/transcription_test/gpt_test2_truth.csv`
- **Format:** CSV (to be verified)
- **Use:** Evaluation data (unseen speakers, same domain)

---

## Experimental Design

### Phase 1: Baseline Establishment (No Fine-tuning)

**Purpose:** Establish baseline performance before any fine-tuning

#### Step 1.1: Transcribe GPT_test with Baseline Model
```bash
# Use .venv_whisperx environment
# Run WhisperX large-v2 on GPT_test.m4a
```

**Metrics to Collect:**
- Word Error Rate (WER)
- Substitutions, Deletions, Insertions
- Processing time
- GPU memory usage
- Number of segments generated
- Speaker detection accuracy
- (Potentially) Transcription confidence: Subgroup of high vs low confidence transcriptions

**Outputs:**
- `Audio_Local_tests/baseline_output/GPT_test_baseline.csv`
- `Audio_Local_tests/baseline_output/GPT_test_baseline_metrics.json`

#### Step 1.2: Transcribe GPT_test2 with Baseline Model
```bash
# Same process for GPT_test2
```

**Metrics to Collect:** (same as 1.1)

**Outputs:**
- `Audio_Local_tests/baseline_output/GPT_test2_baseline.csv`
- `Audio_Local_tests/baseline_output/GPT_test2_baseline_metrics.json`

#### Step 1.3: Analyze Baseline Results
- Calculate WER for both files
- Identify common error patterns
- Note domain-specific vocabulary issues
- Compare to hamlet_test baseline (25.01% WER)

---

### Phase 2: Create Aligned Training Data

**Purpose:** Convert GPT_test ground truth (no timestamps) into timestamped training data

#### Step 2.1: WhisperX Full Pipeline on GPT_test

**Process:**
1. **Transcription:** Use WhisperX large-v2 to transcribe GPT_test.m4a
2. **Forced Alignment:** Align ground truth text to audio using WhisperX
   - Input: `gpt_test_truth.txt` text
   - Output: Word-level timestamps
3. **Diarization:** Identify speaker segments
4. **Quality Check:** Verify alignment accuracy

**WhisperX Forced Alignment:**
```python
import whisperx

# Load alignment model
align_model, metadata = whisperx.load_align_model(
    language_code="en",
    device="cuda"
)

# Align ground truth to audio
result_aligned = whisperx.align(
    transcript_segments,  # From ground truth
    align_model,
    metadata,
    audio,
    device="cuda"
)
```

**Output Format (JSON):**
```json
{
  "audio_file": "GPT_test.m4a",
  "duration": 180.5,
  "segments": [
    {
      "speaker": "Interviewer",
      "start": 0.0,
      "end": 12.5,
      "text": "Before we dive in today, I want to explain...",
      "words": [
        {"word": "Before", "start": 0.0, "end": 0.4, "score": 0.95},
        {"word": "we", "start": 0.45, "end": 0.6, "score": 0.98}
      ]
    }
  ]
}
```

#### Step 2.2: Quality Validation

**Checks:**
- [ ] All ground truth text successfully aligned
- [ ] Timestamp gaps < 1 second (reasonable)
- [ ] Speaker labels match ground truth
- [ ] Word-level confidence scores > 0.8 (average)

**Manual Review:**
- Sample 10 random segments
- Verify timestamps match audio
- Check for alignment drift

#### Step 2.3: Format for Fine-tuning

**Required Format:** Depends on chosen fine-tuning method

**Option A: Hugging Face Transformers Format**
```python
# Convert to Hugging Face dataset format
from datasets import Dataset, Audio

dataset = Dataset.from_dict({
    "audio": [audio_path],
    "text": [transcription],
    "speaker": [speaker_label],
    "timestamps": [word_timestamps]
})
```

**Option B: Whisper Native Format**
- Audio segments (.wav or .flac)
- Corresponding .txt files with transcriptions
- Optional: .json metadata

**Outputs:**
- `training_data/GPT_test/aligned_transcript.json`
- `training_data/GPT_test/audio_segments/` (if needed)
- `training_data/GPT_test/metadata.json`

---

### Phase 3: Fine-tune Whisper Model

**Purpose:** Adapt WhisperX large-v2 to interview domain using GPT_test data

#### Step 3.1: Research Fine-tuning Approaches

**Options to Investigate:**

1. **Hugging Face Transformers (Recommended)**
   - Well-documented
   - Supports LoRA (parameter-efficient fine-tuning)
   - GPU-friendly for 6GB VRAM
   - Example: https://huggingface.co/blog/fine-tune-whisper

2. **OpenAI Whisper (Limited)**
   - No official fine-tuning support
   - Requires custom implementation

3. **Parameter-Efficient Methods**
   - LoRA (Low-Rank Adaptation)
   - Adapter layers
   - Reduces memory requirements

**Decision Criteria:**
- GPU memory constraints (6GB)
- Documentation quality
- Community support
- Training time

#### Step 3.2: Set Up Fine-tuning Environment

**Dependencies:**
```bash
# Install in .venv_whisperx or create new env
pip install transformers datasets accelerate
pip install peft  # For LoRA
pip install evaluate jiwer
```

**Configuration:**
- Learning rate: 1e-5 (start conservatively)
- Batch size: 4-8 (adjust for 6GB VRAM)
- Epochs: 3-5
- Warmup steps: 100-200
- Gradient accumulation: 2-4 (if needed)

#### Step 3.3: Prepare Training Dataset

**Split Strategy:**

Since GPT_test is small, use it entirely for training. We'll evaluate on:
- GPT_test (check overfitting)
- GPT_test2 (generalization to new speakers)
- hamlet_test (verify no catastrophic forgetting)

**Data Augmentation (Optional):**
- Speed perturbation (0.9x, 1.1x)
- Add background noise
- Pitch shifting
- (Only if needed for better generalization)

#### Step 3.4: Run Fine-tuning

**Training Script Outline:**
```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

# Load base model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")

# Configure training
training_args = Seq2SeqTrainingArguments(
    output_dir="./models/whisper-finetuned-gpt",
    per_device_train_batch_size=8,
    learning_rate=1e-5,
    num_train_epochs=5,
    fp16=True,  # Use mixed precision for 6GB GPU
    gradient_accumulation_steps=2,
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    load_best_model_at_end=True,
)

# Train
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=processor.feature_extractor,
)

trainer.train()
```

**Monitoring:**
- Track training loss
- Monitor GPU memory usage
- Watch for overfitting signs
- Save checkpoints regularly

**Expected Training Time:**
- Small dataset: 30-60 minutes
- Depends on epochs and dataset size

#### Step 3.5: Save Fine-tuned Model

**Outputs:**
- `models/whisper-finetuned-gpt/` - Model weights
- `models/whisper-finetuned-gpt/config.json` - Configuration
- `training_logs/` - Training metrics and plots

---

### Phase 4: Evaluation

**Purpose:** Compare fine-tuned model to baseline across multiple test sets

#### Step 4.1: Evaluate on GPT_test (Training Data)

**Purpose:** Check for overfitting

**Process:**
1. Transcribe GPT_test with fine-tuned model
2. Calculate WER
3. Compare to baseline GPT_test WER

**Expected Result:**
- Should see WER improvement
- If WER → 0%, likely overfit
- Target: Moderate improvement (5-15% WER reduction)

#### Step 4.2: Evaluate on GPT_test2 (Target Evaluation)

**Purpose:** Measure generalization to new speakers in same domain

**Process:**
1. Transcribe GPT_test2 with fine-tuned model
2. Calculate WER
3. **Compare to baseline GPT_test2 WER**
4. Analyze error types (substitutions, deletions, insertions)

**Success Criteria:**
- ✅ **WER improvement ≥ 5%** (e.g., 30% → 25%)
- ✅ Fewer domain-specific vocabulary errors
- ✅ Better handling of structured interview patterns

**Failure Indicators:**
- ❌ No improvement or worse WER
- ❌ Overfitting (great on GPT_test, poor on GPT_test2)

#### Step 4.3: Evaluate on hamlet_test (Generalization Check)

**Purpose:** Ensure model didn't "forget" general transcription ability

**Process:**
1. Transcribe hamlet_test with fine-tuned model
2. Calculate WER
3. Compare to baseline hamlet_test (25.01% WER)

**Success Criteria:**
- ✅ WER within 2% of baseline (e.g., 25.01% → 27% acceptable)
- ❌ **Catastrophic forgetting** if WER > 30%

#### Step 4.4: Detailed Error Analysis

**Compare Error Patterns:**

| Error Type | Baseline GPT_test | Fine-tuned GPT_test | Baseline GPT_test2 | Fine-tuned GPT_test2 |
|------------|-------------------|---------------------|--------------------|-----------------------|
| Substitutions | ? | ? | ? | ? |
| Deletions | ? | ? | ? | ? |
| Insertions | ? | ? | ? | ? |
| **Total WER** | ? | ? | ? | ? |

**Domain-Specific Analysis:**
- Clinical terminology accuracy
- Interviewer-specific phrase recognition
- Speaker transition handling
- Silence/pause handling

---

### Phase 5: Analysis & Recommendations

#### Step 5.1: Quantitative Analysis

**Metrics Summary:**
```
Baseline Model (WhisperX large-v2):
- GPT_test WER: X.XX%
- GPT_test2 WER: X.XX%
- hamlet_test WER: 25.01%

Fine-tuned Model:
- GPT_test WER: X.XX% (Δ -X.XX%)
- GPT_test2 WER: X.XX% (Δ -X.XX%)
- hamlet_test WER: X.XX% (Δ +/-X.XX%)
```

**Statistical Significance:**
- Calculate 95% confidence intervals
- Perform paired t-test if multiple samples available

#### Step 5.2: Qualitative Analysis

**Questions to Answer:**
1. Did fine-tuning improve domain-specific vocabulary?
2. Which error types decreased most?
3. Did speaker generalization work (Interviewer → Interviewer)?
4. Are there trade-offs (improved interviews, worse on general content)?

#### Step 5.3: Recommendations

**If Successful (≥5% WER improvement on GPT_test2):**
- ✅ Proceed with larger-scale fine-tuning
- ✅ Collect more interview transcripts for training
- ✅ Create domain-adapted production model
- ✅ Document optimal training hyperparameters

**If Marginal (1-5% improvement):**
- ⚠️ Collect more training data (current dataset too small)
- ⚠️ Try data augmentation
- ⚠️ Experiment with learning rate/epochs

**If Unsuccessful (<1% improvement or worse):**
- ❌ Model may be saturated (large-v2 already very good)
- ❌ Domain overlap insufficient for transfer learning
- ❌ Consider prompt engineering instead of fine-tuning
- ❌ Focus on post-processing corrections

---

## Implementation Timeline

### Phase 1: Baseline & Setup 
- [x] GPT_test files confirmed
- [x] Run baseline transcriptions (GPT_test, GPT_test2)
- [x] Calculate baseline WERs
- [x] Research fine-tuning methods

### Phase 2: Alignment & Training Prep 
- [x] Create aligned training data from GPT_test
- [x] Quality check alignments
- [x] Set up fine-tuning environment
- [x] Prepare training dataset

### Phase 3: Fine-tuning 
- [x] Configure training parameters
- [x] Run fine-tuning (monitor closely)
- [x] Save best checkpoint
- [ ] Validate model loads correctly

### Phase 4: Evaluation & Analysis 
- [ ] Test on GPT_test, GPT_test2, hamlet_test
- [ ] Calculate all metrics
- [ ] Error analysis
- [ ] Document findings

### Phase 5: Documentation & Next Steps 
- [ ] Create PRETRAINING_EXPERIMENT.md with results
- [ ] Update TODO.md
- [ ] Recommendations for full dataset

**Total Estimated Time:** 

---

## Technical Considerations

### Memory Constraints (6GB GPU)
- Use gradient accumulation to simulate larger batch sizes
- Enable mixed precision (fp16)
- Consider LoRA for parameter-efficient fine-tuning
- Monitor GPU memory during training

### Overfitting Prevention
- Keep epochs low (3-5 max)
- Use dropout if available
- Early stopping based on validation loss
- Don't train to 100% accuracy on GPT_test

### Data Format Issues
- Ground truth has no timestamps → Need forced alignment
- CSV format → Parse carefully (Speaker, Transcription columns)
- Ensure speaker labels are consistent

---

## Success Metrics

### Must Have (Minimum Viable)
- ✅ Baseline WERs calculated for GPT_test and GPT_test2
- ✅ Fine-tuning completes without errors

- ✅ Results documented with comparison

### Should Have (Quality Threshold)
- ✅ ≥5% WER improvement on GPT_test2
- ✅ No catastrophic forgetting (hamlet_test WER < 30%)
- ✅ Detailed error analysis completed
- ✅ Clear recommendations for next steps

### Nice to Have (Stretch Goals)
- ✅ ≥10% WER improvement on GPT_test2
- ✅ Visualization of error distributions
- ✅ Speaker-specific performance breakdown
- ✅ Automated fine-tuning pipeline script

---

## Risks & Mitigations

### Risk 1: Insufficient Training Data
- **Risk:** GPT_test too small for meaningful fine-tuning
- **Mitigation:** Use data augmentation, LoRA (requires less data)
- **Contingency:** Collect more interviews before fine-tuning

### Risk 2: Overfitting
- **Risk:** Model memorizes GPT_test, fails on GPT_test2
- **Mitigation:** Early stopping, dropout, low epochs
- **Contingency:** Reduce training epochs, add regularization

### Risk 3: Alignment Quality Issues
- **Risk:** Poor forced alignment creates noisy training data
- **Mitigation:** Manual review of alignment samples
- **Contingency:** Use WhisperX's confidence scores to filter

### Risk 4: GPU Memory Limitations
- **Risk:** OOM errors during fine-tuning
- **Mitigation:** Reduce batch size, use gradient accumulation
- **Contingency:** Use LoRA, or fine-tune on cloud GPU

### Risk 5: No Improvement
- **Risk:** Fine-tuning doesn't help (large-v2 already optimal)
- **Mitigation:** Try different learning rates, data augmentation
- **Contingency:** Focus on post-processing, prompt engineering

---

## File Structure

```
Audio_Local_tests/
├── audio_files/
│   ├── GPT_test.m4a
│   └── GPT_test2.m4a
├── transcription_test/
│   ├── gpt_test_truth.txt
│   └── gpt_test2_truth.csv
├── baseline_output/
│   ├── GPT_test_baseline.csv
│   ├── GPT_test_baseline_metrics.json
│   ├── GPT_test2_baseline.csv
│   └── GPT_test2_baseline_metrics.json

training_data/
└── GPT_test/
    ├── aligned_transcript.json
    ├── audio_segments/ (if needed)
    └── metadata.json

models/
└── whisper-finetuned-gpt/
    ├── pytorch_model.bin
    ├── config.json
    ├── tokenizer/
    └── training_args.bin

training_logs/
├── training_loss.png
├── metrics.json
└── training.log

Notes/
└── PRETRAINING_EXPERIMENT.md  (Results document)
```

---

## Next Steps After Experiment

### If Successful:
1. **Scale Up Training Data**
   - Identify all available validated interview transcripts
   - Align all transcripts using WhisperX
   - Create comprehensive training dataset (hours of data)

2. **Systematic Fine-tuning**
   - Train on larger dataset
   - Use train/val/test splits properly
   - Optimize hyperparameters

3. **Production Deployment**
   - Save best model for production use
   - Create inference pipeline
   - Document model performance

### If Unsuccessful:
1. **Alternative Approaches**
   - Prompt engineering (context-aware prompts)
   - Post-processing rules for domain vocabulary
   - Vocabulary injection techniques

2. **Data Collection**
   - Prioritize collecting more interview transcripts
   - Ensure diversity in speakers and scenarios
   - Improve ground truth quality

3. **Model Exploration**
   - Try Whisper large-v3 (if not already tested)
   - Explore other ASR models (Wav2Vec2, etc.)
   - Consider ensemble approaches

---

## References

- Hugging Face Whisper Fine-tuning: https://huggingface.co/blog/fine-tune-whisper
- WhisperX Paper: https://arxiv.org/abs/2303.00747
- LoRA (Parameter-Efficient Fine-tuning): https://arxiv.org/abs/2106.09685
- Whisper Model: https://github.com/openai/whisper

---

**Document Status:** Planning Complete - Ready for Implementation
**Next Action:** Run Phase 1 baseline transcriptions