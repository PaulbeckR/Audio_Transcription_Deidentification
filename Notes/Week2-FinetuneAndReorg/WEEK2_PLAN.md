# Week 2 Implementation Plan

**Goal:** Prepare for fine-tuning by aligning transcripts and building configuration system

**Duration:** Week 2 (Phase 2 from PROJECT_PLAN.md)

**Status:** Not Started

---

## Overview

Week 2 focuses on three main objectives:
1. **Align all validated transcripts** using WhisperX for word-level timestamps
2. **Create configuration management system** for paths, models, and parameters
3. **Build training dataset** in standardized format for fine-tuning

This bridges the gap between baseline testing (Week 1) and fine-tuning (Week 3).

---

## Task Breakdown

### Task 1: WhisperX Setup and Testing
**Duration:** 2-3 hours
**Priority:** High

**Objective:** Get WhisperX working and validate it produces better alignment than base Whisper

**Subtasks:**
1. **Install WhisperX** (if not already done)
   - Address Windows installation challenges (av library, FFmpeg dev libs)
   - Consider WSL2 or cloud environment if Windows issues persist
   - Verify GPU support is enabled

2. **Create WhisperX test script**
   - Test on hamlet_test.m4a.wav
   - Compare with existing Whisper outputs
   - Validate word-level timestamp accuracy

3. **Evaluate WhisperX alignment**
   - Load WhisperX output and ground truth
   - Check timestamp precision
   - Compare speaker assignment quality vs Pyannote

**Success Criteria:**
- [x] WhisperX installed and running
- [x] Word-level timestamps generated for hamlet_test
- [x] Alignment quality assessed
- [x] Decision made: Use WhisperX or continue with Whisper + Pyannote: WhisperX chosen

**Deliverables:**
- `test_whisperx.py` - Test script
- `Notes/WHISPERX_EVALUATION.md` - Evaluation report
- WhisperX output files for hamlet_test

**Blockers/Risks:**
- Windows installation issues → Mitigation: Use WSL2 or cloud GPU
- Poor alignment quality → Contingency: Use Whisper + manual alignment tools

---

### Task 2: Configuration System Design
**Duration:** 3-4 hours
**Priority:** High

**Objective:** Create a clean configuration system to eliminate hardcoded paths and settings

**Subtasks:**
1. **Design configuration structure**
   - Create `config/` directory
   - Define YAML schema for paths, models, processing parameters
   - Plan environment-specific configs (dev, production)

2. **Create configuration files**
   - `config/paths.yaml` - All file paths and directories
   - `config/models.yaml` - Model settings (sizes, devices, parameters)
   - `config/processing.yaml` - Pipeline parameters (batch size, thresholds, etc.)

3. **Build configuration loader**
   - `src/config.py` - Configuration loading and validation
   - Support environment variable overrides
   - Provide sensible defaults

4. **Update existing code**
   - Replace hardcoded paths in notebooks
   - Update `run_baseline_test.py` to use configs
   - Update helper functions to accept config objects

**Success Criteria:**
- [x] Config files created and documented
- [x] Config loader functional with validation
- [x] At least one script (baseline test) converted to use configs
- [x] Paths work correctly across different environments

**Deliverables:**
- `config/paths.yaml`
- `config/models.yaml`
- `config/processing.yaml`
- `src/config.py` - Configuration management module
- `Notes/CONFIG_GUIDE.md` - Usage documentation

**Example Configuration Structure:**
```yaml
# config/paths.yaml
project:
  root: "."
  audio_files: "Audio_Local_tests/audio_files"
  transcripts: "Audio_Local_tests/transcription_test"
  rttm_files: "Audio_Local_tests/rttm_files"
  output: "Audio_Local_tests/baseline_output"

training:
  aligned_data: "training_data/aligned"
  raw_transcripts: "training_data/raw"
  validated: "training_data/validated"
```

```yaml
# config/models.yaml
whisper:
  default_size: "large"
  device: "auto"  # auto-detect GPU/CPU
  language: "en"
  compute_type: "float16"  # or int8 for quantization

pyannote:
  model: "pyannote/speaker-diarization-3.1"
  min_speakers: 2
  max_speakers: 4

whisperx:
  enabled: false  # toggle WhisperX vs Whisper
  batch_size: 16
  align_model: "WAV2VEC2_ASR_BASE_960H"
```

---

### Task 3: Transcript Inventory and Validation
**Duration:** 2 hours
**Priority:** Medium

**Objective:** Identify all existing validated transcripts and assess quality for training

**Subtasks:**
1. **Scan for existing transcripts**
   - Search Audio_Local_tests/ for all .xlsx, .csv transcript files
   - Identify which have ground truth vs hypothesis
   - Catalog audio files with corresponding transcripts

2. **Create transcript inventory**
   - Build spreadsheet/JSON listing all available data
   - Include: filename, duration, has_ground_truth, quality_rating, speakers
   - Calculate total validated transcript hours

3. **Quality assessment**
   - Review sample transcripts for accuracy
   - Identify any with obvious errors or incomplete data
   - Mark high-quality transcripts for training dataset

**Success Criteria:**
- [ ] Complete inventory of existing transcripts
- [ ] Total hours of validated data calculated
- [ ] Quality ratings assigned
- [ ] Priority list for alignment created

**Deliverables:**
- `training_data/inventory.json` - Complete data inventory
- `Notes/TRAINING_DATA_INVENTORY.md` - Human-readable summary
- List of transcripts ready for alignment

---

### Task 4: Alignment Pipeline
**Duration:** 4-5 hours
**Priority:** High

**Objective:** Create automated pipeline to align all validated transcripts with word-level timestamps

**Subtasks:**
1. **Design alignment format**
   - Choose output format (JSON, CSV, or specialized format)
   - Include: word, start_time, end_time, speaker, confidence
   - Plan for compatibility with Hugging Face dataset format

2. **Build alignment script**
   - `scripts/align_transcripts.py` - Batch alignment tool
   - Process all transcripts from inventory
   - Use WhisperX or fallback to Whisper
   - Save aligned outputs in consistent format

3. **Validate alignment quality**
   - Sample check alignments against ground truth
   - Calculate timestamp accuracy metrics
   - Identify any problematic alignments

4. **Create training dataset structure**
   - Organize aligned data in Hugging Face format or similar
   - Include audio segments, transcripts, speaker labels
   - Split into train/validation/test sets

**Success Criteria:**
- [ ] Alignment pipeline functional
- [ ] All validated transcripts aligned
- [ ] Quality validation completed
- [ ] Training dataset structured and ready

**Deliverables:**
- `scripts/align_transcripts.py` - Alignment pipeline
- `training_data/aligned/` - Aligned transcript outputs
- `training_data/dataset_manifest.json` - Dataset metadata
- `Notes/ALIGNMENT_REPORT.md` - Quality assessment

**Output Format Example:**
```json
{
  "audio_file": "hamlet_test.m4a.wav",
  "duration": 756.736,
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.0,
      "end": 5.2,
      "words": [
        {"word": "Who's", "start": 0.0, "end": 0.3, "confidence": 0.95},
        {"word": "there", "start": 0.35, "end": 0.6, "confidence": 0.98}
      ],
      "text": "Who's there?"
    }
  ]
}
```

---

### Task 5: Notebook Refactoring (Optional)
**Duration:** 2-3 hours
**Priority:** Low (can defer to Week 4)

**Objective:** Begin converting notebooks to modular scripts for better maintainability

**Subtasks:**
1. **Audit existing notebooks**
   - Review transcript_notebook.ipynb, create_rttm.ipynb
   - Identify reusable code sections
   - Note dependencies and inputs/outputs

2. **Extract core functions**
   - Move notebook logic to src/ modules
   - Create: `src/transcription.py`, `src/diarization.py`
   - Keep notebooks as demonstration/testing tools

3. **Create unified pipeline script**
   - `scripts/process_audio.py` - End-to-end processing
   - Use configuration system
   - Support both CPU and GPU
   - Include progress tracking

**Success Criteria:**
- [ ] Core functions extracted to src/
- [ ] At least one notebook converted to script
- [ ] Pipeline script functional

**Deliverables:**
- `src/transcription.py`
- `src/diarization.py`
- `scripts/process_audio.py`

**Note:** This can be deferred to Week 4 (Phase 4) if time is limited.

---

### Task 6: Documentation Updates
**Duration:** 1-2 hours
**Priority:** Medium

**Objective:** Update documentation to reflect Week 1 results and Week 2 changes

**Subtasks:**
1. **Update README.md**
   - Add Week 1 results summary
   - Update installation instructions
   - Document GPU setup steps
   - Add quick start guide

2. **Update File_Structure.md**
   - Document new config/ directory
   - Add training_data/ structure
   - Update src/ module descriptions

3. **Create usage guides**
   - How to run baseline tests
   - How to use configuration system
   - How to align new transcripts

**Success Criteria:**
- [ ] README.md reflects current state
- [ ] File structure documented
- [ ] Usage guides created

**Deliverables:**
- Updated `README.md`
- Updated `Notes/File_Structure.md`
- `Notes/USAGE_GUIDE.md`

---

## Week 2 Deliverables Summary

**Code:**
- [x] `src/config.py` - Configuration management
- [x] `scripts/align_transcripts.py` - Alignment pipeline
- [x] `test_whisperx.py` - WhisperX testing
- [x] Updated `run_baseline_test.py` using configs

**Configuration:**
- [x] `config/paths.yaml`
- [x] `config/models.yaml`
- [x] `config/processing.yaml`

**Data:**
- [ ] `training_data/inventory.json` - Data inventory
- [ ] `training_data/aligned/` - Aligned transcripts
- [ ] `training_data/dataset_manifest.json` - Dataset metadata

**Documentation:**
- [ ] `Notes/WEEK2_RESULTS.md` - Week 2 summary
- [x] `Notes/WHISPERX_EVALUATION.md` - WhisperX assessment
- [x] `Notes/CONFIG_GUIDE.md` - Configuration usage
- [ ] `Notes/TRAINING_DATA_INVENTORY.md` - Data summary
- [ ] `Notes/ALIGNMENT_REPORT.md` - Alignment quality
- [ ] Updated `README.md`

---

## Success Criteria

**Must Have:**
- Configuration system functional and documented
- Transcript inventory complete
- At least hamlet_test aligned with word-level timestamps
- Decision made on WhisperX vs Whisper
- Ready to begin fine-tuning in Week 3

**Nice to Have:**
- All validated transcripts aligned
- Notebooks refactored to scripts
- Training dataset in Hugging Face format
- Comprehensive documentation updates

---

## Timeline

**Day 1-2:** WhisperX setup and testing (Task 1)
**Day 2-3:** Configuration system design and implementation (Task 2)
**Day 3-4:** Transcript inventory and alignment pipeline (Tasks 3-4)
**Day 5:** Documentation and validation (Task 6)
**Day 6-7:** Buffer for blockers, optional notebook refactoring (Task 5)

---

## Risks and Mitigations

**Risk 1: WhisperX installation fails on Windows**
- Mitigation: Use WSL2, cloud environment, or proceed without WhisperX
- Contingency: Use Whisper + Montreal Forced Aligner or similar

**Risk 2: Limited training data available**
- Mitigation: Start with what exists, plan data collection
- Contingency: Use data augmentation, synthetic data generation

**Risk 3: Alignment quality poor**
- Mitigation: Sample validation before processing all data
- Contingency: Manual correction tools, use higher confidence threshold

**Risk 4: Configuration system too complex**
- Mitigation: Start simple, iterate based on needs
- Contingency: Keep hardcoded paths as fallback

---

## Questions to Answer This Week

1. Should we use WhisperX or continue with Whisper + Pyannote?
2. How many hours of validated transcripts do we have?
3. What is the quality of our existing alignments?
4. What format should training data be in for fine-tuning?
5. Do we need to collect more training data before fine-tuning?

---

## Next Steps After Week 2

**Week 3: Fine-Tuning**
- Prepare Whisper fine-tuning dataset
- Train domain-adapted model
- Evaluate WER improvements
- Select best model for production

**Week 4: Production Pipeline**
- Refactor remaining notebooks
- Build batch processing system
- Create automation scripts
- Comprehensive testing
