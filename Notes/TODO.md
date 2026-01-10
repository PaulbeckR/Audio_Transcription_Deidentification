# Project TODO List

**Last Updated**: 2025-12-27
**Current Phase**: Week 2 - Pretraining Data Testing & Fine-tuning Prep
**Branch**: ComprehensiveRestructure

---

## Current Status (Updated 2025-12-27)

###  Completed - Week 1
- [x] GPU enabled and tested (NVIDIA GeForce RTX 3050 6GB)
- [x] WhisperX installed in separate .venv_whisperx environment
- [x] Comprehensive model evaluation completed
- [x] **Model Selected: WhisperX large-v2** (25.01% WER, 2.26x realtime)
- [x] Baseline metrics established (see Notes/Week1/MODEL_COMPARISON.md)
- [x] Configuration system created (config/models.yaml)

###  Current Priority - Pretraining Data Experiment

**New Test Files Added:**
- `GPT_test.m4a` - Initial test audio (~3.2MB)
- `GPT_test2.m4a` - Second test audio (to be confirmed)
- `gpt_test_truth.txt` - Ground truth transcript (CSV format, 2 speakers)
- `gpt_test2_truth.csv` - Ground truth transcript (to be confirmed)

**Objective:** Test whether adding pretraining data from GPT_test improves model performance on GPT_test2.

---

## Active Tasks - Pretraining Data Pipeline

### Phase 1: Baseline Transcription (No Pretraining)
- [ ] Run WhisperX large-v2 on GPT_test.m4a
  - [ ] Generate baseline transcript
  - [ ] Calculate WER against gpt_test_truth.txt
  - [ ] Record baseline metrics (time, memory, accuracy)
- [ ] Run WhisperX large-v2 on GPT_test2.m4a
  - [ ] Generate baseline transcript
  - [ ] Calculate WER against gpt_test2_truth.csv
  - [ ] Record baseline metrics
- [ ] Save baseline results for comparison

### Phase 2: Create Aligned Pretraining Data from GPT_test
- [ ] Process GPT_test through full pipeline:
  - [ ] WhisperX transcription with word-level timestamps
  - [ ] WhisperX diarization (speaker identification)
  - [ ] WhisperX forced alignment
- [ ] Generate time-stamped transcript file:
  - [ ] Word-level timestamps
  - [ ] Speaker labels
  - [ ] Confidence scores
  - [ ] JSON or specialized format for fine-tuning
- [ ] Quality check alignment:
  - [ ] Compare to ground truth
  - [ ] Verify timestamp accuracy
  - [ ] Validate speaker assignments
- [ ] Prepare pretraining dataset:
  - [ ] Format for Whisper fine-tuning
  - [ ] Include audio segments
  - [ ] Create metadata file

### Phase 3: Fine-tune Model with GPT_test Data
- [ ] Research Whisper fine-tuning process:
  - [ ] Hugging Face Transformers approach
  - [ ] Official OpenAI approach (if available)
  - [ ] Required data format
- [ ] Set up fine-tuning environment:
  - [ ] Install fine-tuning dependencies
  - [ ] Configure training parameters
  - [ ] Set up GPU training
- [ ] Fine-tune WhisperX/Whisper on GPT_test data:
  - [ ] Create training dataset
  - [ ] Configure hyperparameters
  - [ ] Run training
  - [ ] Monitor loss/metrics
  - [ ] Save fine-tuned model
- [ ] Document fine-tuning process

### Phase 4: Evaluate Fine-tuned Model
- [ ] Test fine-tuned model on GPT_test:
  - [ ] Transcribe with fine-tuned model
  - [ ] Calculate WER
  - [ ] Compare to baseline GPT_test results
  - [ ] Verify no overfitting
- [ ] Test fine-tuned model on GPT_test2:
  - [ ] Transcribe with fine-tuned model
  - [ ] Calculate WER
  - [ ] **Compare to baseline GPT_test2 results**
  - [ ] Measure improvement (if any)
- [ ] Optionally retest on hamlet_test:
  - [ ] Verify model generalization
  - [ ] Check for performance degradation
  - [ ] Compare to baseline hamlet_test WER

### Phase 5: Analysis & Documentation
- [ ] Create comparison matrix:
  - Baseline GPT_test WER
  - Baseline GPT_test2 WER
  - Fine-tuned GPT_test WER
  - Fine-tuned GPT_test2 WER
  - Improvement percentages
- [ ] Analyze results:
  - [ ] Did pretraining help?
  - [ ] How much improvement?
  - [ ] Overfitting concerns?
  - [ ] Generalization to new speakers?
- [ ] Document findings:
  - [ ] Create Notes/PRETRAINING_EXPERIMENT.md
  - [ ] Include methodology
  - [ ] Include results and analysis
  - [ ] Recommendations for full dataset

---

## Week 1 Tasks (Completed)

---

## Week 2 Tasks (Preview)

### Configuration Management
- [ ] Create config/ directory
- [ ] Create config/paths.yaml with all folder paths
- [ ] Create config/models.yaml with model settings
- [ ] Create src/config.py to load configurations
- [ ] Update notebooks to use config files
- [ ] Remove all hardcoded paths from codebase

### Training Data Preparation
- [ ] Identify all validated transcripts
- [ ] Choose alignment tool (WhisperX vs MFA) based on Week 1
- [ ] Align all validated transcripts to audio
- [ ] Create standardized training data format
- [ ] Validate alignment quality on sample
- [ ] Export to training-ready format

### Code Organization
- [ ] Create src/preprocessing.py
- [ ] Create src/diarization.py
- [ ] Create src/transcription.py
- [ ] Create src/alignment.py (if using separate tool)
- [ ] Refactor notebook code into modules
- [ ] Create unit tests for key functions

---

## Week 3 Tasks (Preview)

### Fine-Tuning
- [ ] Prepare Whisper fine-tuning dataset
- [ ] Set up training environment
- [ ] Configure training hyperparameters
- [ ] Run fine-tuning
- [ ] Evaluate fine-tuned model
- [ ] Compare to baseline

---

## Week 4 Tasks (Preview)

### Production Pipeline
- [ ] Create scripts/process_single_file.py
- [ ] Create scripts/process_folder.py
- [ ] Implement batch processing
- [ ] Add error handling and logging
- [ ] Create progress tracking
- [ ] Build folder monitoring system

---

## Ongoing Tasks

### Documentation
- [ ] Keep Notes/ files updated
- [ ] Document all major decisions
- [ ] Track metrics over time
- [ ] Update README as project evolves

### Code Quality
- [ ] Add docstrings to functions
- [ ] Create type hints where applicable
- [ ] Ensure consistent code style
- [ ] Remove deprecated code

### Testing
- [ ] Test on hamlet_test regularly
- [ ] Validate against ground truth
- [ ] Check for regressions
- [ ] Document edge cases

---

## Completed Tasks

_Tasks will be moved here as they are completed during Week 1 implementation_

---

## Blocked/On Hold

_Note any blocked tasks or dependencies here_

---

## Questions & Decisions Needed

1. **WhisperX vs Whisper**: Wait for Week 1 testing results
2. **Model size for production**: Decide after speed/accuracy comparison
3. **Pyannote standalone vs WhisperX diarization**: Test both in Week 1
4. **Configuration format**: YAML vs JSON vs .env - decide in Week 2
5. **Training data format**: Depends on chosen fine-tuning approach

---

## Notes

- All work done on `ComprehensiveRestructure` branch
- hamlet_test is safe for testing/sharing
- Keep Box folder access minimal (filenames only)
- Document all metric calculations for reproducibility
