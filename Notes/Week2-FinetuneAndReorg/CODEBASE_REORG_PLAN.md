# Codebase Reorganization Plan

**Date:** December 24, 2025
**Context:** Preparing for Week 2 - Training data preparation with WhisperX large-v2

---

## Current State

### ✅ What's Working Well
- `src/metrics.py` - WER/DER calculation (jiwer v4 compatible)
- `src/utils.py` - Device detection, helpers
- `run_baseline_test.py` - Whisper testing with model switching
- `test_whisperx.py` - WhisperX testing (PyTorch 2.8.0+cu126)
- Dual environments (`.venv` for Whisper, `.venv_whisperx` for WhisperX)

### ⚠️ What Needs Fixing
1. **No configuration system** - paths/models hardcoded everywhere
2. **Notebooks doing too much** - should extract to `src/` modules
3. **Scripts not integrated** - `run_baseline_test.py` and `test_whisperx.py` standalone
4. **No batch processing** - can't process multiple files easily
5. **Training data format unclear** - need standardized output for fine-tuning

---

## Reorganization Priorities

### Priority 1: Configuration System (HIGH) 🔴
**Why First:** Eliminates hardcoded paths, enables easy environment switching

**Tasks:**
1. Create `config/` directory structure
2. Create `config/paths.yaml` - all file paths
3. Create `config/models.yaml` - model settings (WhisperX large-v2 as default)
4. Create `config/processing.yaml` - pipeline parameters
5. Create `src/config.py` - config loader with validation
6. Update `run_baseline_test.py` to use configs
7. Update `test_whisperx.py` to use configs

**Estimated Time:** 3-4 hours

**Deliverables:**
- `config/paths.yaml`
- `config/models.yaml`
- `config/processing.yaml`
- `src/config.py`
- Updated test scripts using configs

---

### Priority 2: Transcript Inventory (MEDIUM) 🟡
**Why Second:** Need to know what data we have before building pipelines

**Tasks:**
1. Create `scripts/inventory_transcripts.py`
2. Scan `Audio_Local_tests/` for all audio + transcript pairs
3. Calculate total duration of validated data
4. Identify which files are ready for alignment
5. Create `training_data/inventory.json`
6. Generate `Notes/TRAINING_DATA_INVENTORY.md` report

**Estimated Time:** 2 hours

**Deliverables:**
- `scripts/inventory_transcripts.py`
- `training_data/inventory.json`
- `Notes/TRAINING_DATA_INVENTORY.md`

---

### Priority 3: Alignment Pipeline (HIGH) 🔴
**Why Third:** Core Week 2 objective - prepare training data

**Tasks:**
1. Create `src/alignment.py` module
   - WhisperX wrapper functions
   - Batch processing logic
   - Output formatting for training
2. Create `scripts/align_transcripts.py`
   - Process files from inventory
   - Use WhisperX large-v2
   - Save in standardized format
3. Define training data format (JSON with word-level timestamps)
4. Process at least `hamlet_test` through alignment
5. Validate alignment quality

**Estimated Time:** 4-5 hours

**Deliverables:**
- `src/alignment.py`
- `scripts/align_transcripts.py`
- `training_data/aligned/` - aligned outputs
- `Notes/ALIGNMENT_REPORT.md`

---

### Priority 4: Modular Transcription (MEDIUM) 🟡
**Why Fourth:** Extract working code from notebooks and scripts

**Tasks:**
1. Create `src/transcription.py`
   - Extract Whisper logic from `run_baseline_test.py`
   - Support multiple models (medium.en, large-v2, large-v3)
   - Device auto-detection
2. Create `src/diarization.py`
   - Extract Pyannote logic from notebooks
   - RTTM processing
   - Speaker segment handling
3. Update `run_baseline_test.py` to use `src/transcription.py`
4. Keep notebooks as examples/demos

**Estimated Time:** 3-4 hours

**Deliverables:**
- `src/transcription.py`
- `src/diarization.py`
- Refactored `run_baseline_test.py`

---

### Priority 5: Unified Pipeline Script (LOW) 🟢
**Why Last:** Nice to have, but not blocking training data prep

**Tasks:**
1. Create `scripts/process_audio.py`
   - End-to-end pipeline (diarization → transcription → alignment)
   - Support both Whisper and WhisperX modes
   - Batch folder processing
   - Progress tracking
2. Command-line interface with arguments
3. Integration with config system

**Estimated Time:** 2-3 hours

**Deliverables:**
- `scripts/process_audio.py`
- `Notes/USAGE_GUIDE.md`

---

## Recommended Execution Order

### Day 1: Configuration Foundation
- **Morning:** Create config directory and YAML files
- **Afternoon:** Build `src/config.py` loader
- **Evening:** Update `run_baseline_test.py` and `test_whisperx.py` to use configs

### Day 2: Data Preparation
- **Morning:** Build transcript inventory script
- **Afternoon:** Run inventory, analyze results
- **Evening:** Design alignment output format

### Day 3: Alignment Pipeline
- **Morning:** Create `src/alignment.py` module
- **Afternoon:** Build `scripts/align_transcripts.py`
- **Evening:** Process `hamlet_test` and validate

### Day 4: Modularization (Optional)
- Extract transcription and diarization to `src/`
- Refactor existing scripts to use modules
- Update documentation

### Day 5: Buffer/Polish
- Handle any blockers
- Write documentation
- Prepare Week 2 results summary

---

## Configuration Design

### config/paths.yaml
```yaml
# Base paths
project_root: "."
audio_files: "Audio_Local_tests/audio_files"
transcripts: "Audio_Local_tests/transcription_test"
output: "Audio_Local_tests/baseline_output"

# Training data paths
training:
  root: "training_data"
  aligned: "training_data/aligned"
  inventory: "training_data/inventory.json"

# Temporary paths
temp:
  audio_clips: "Audio_Local_tests/audio_clips"
  wav_files: "Audio_Local_tests/wav_files"
```

### config/models.yaml
```yaml
# WhisperX configuration (RECOMMENDED based on testing)
whisperx:
  enabled: true
  model: "large-v2"  # Best WER: 25.01%
  device: "cuda"
  compute_type: "float16"
  batch_size: 4  # For 6GB GPU
  language: "en"

# Whisper fallback
whisper:
  model: "large-v2"
  device: "cuda"
  language: "en"

# Pyannote diarization
pyannote:
  model: "pyannote/speaker-diarization-3.1"
  device: "cuda"
```

### config/processing.yaml
```yaml
# Audio preprocessing
audio:
  sample_rate: 16000
  trim_silence: true

# Alignment settings
alignment:
  return_char_alignments: false

# Output format
output:
  format: "json"  # or "csv"
  include_confidence: true
  include_word_timestamps: true
```

---

## Training Data Format

### Proposed Structure
```json
{
  "metadata": {
    "audio_file": "hamlet_test.m4a.wav",
    "duration": 756.736,
    "model": "whisperx-large-v2",
    "date_processed": "2025-12-24",
    "num_speakers": 4
  },
  "segments": [
    {
      "id": 0,
      "speaker": "SPEAKER_00",
      "start": 3.24,
      "end": 4.34,
      "text": "He will come straight.",
      "words": [
        {"word": "He", "start": 3.24, "end": 3.42, "score": 0.95},
        {"word": "will", "start": 3.44, "end": 3.64, "score": 0.98},
        {"word": "come", "start": 3.72, "end": 3.96, "score": 0.97},
        {"word": "straight.", "start": 3.98, "end": 4.34, "score": 0.96}
      ]
    }
  ]
}
```

### Why This Format?
- ✅ Compatible with Hugging Face datasets
- ✅ Preserves all alignment information
- ✅ Easy to convert to other formats
- ✅ Includes metadata for tracking
- ✅ Word-level timestamps for fine-tuning

---

## Success Criteria

### Configuration System ✓
- [ ] All paths centralized in YAML
- [ ] Model settings configurable
- [ ] Scripts load from config
- [ ] Environment-agnostic paths

### Transcript Inventory ✓
- [ ] Complete list of available data
- [ ] Total hours calculated
- [ ] Quality ratings assigned
- [ ] Ready for alignment

### Alignment Pipeline ✓
- [ ] WhisperX large-v2 integration
- [ ] Batch processing functional
- [ ] Standardized output format
- [ ] At least `hamlet_test` processed

### Code Quality ✓
- [ ] Modular, reusable functions
- [ ] No hardcoded paths
- [ ] Documented with docstrings
- [ ] Type hints where applicable

---

## Migration Strategy

### Phase 1: Additive (Don't Break Existing)
1. Create new config system alongside existing code
2. Build new modules in `src/` without touching notebooks
3. Create new scripts in `scripts/` for batch processing
4. Keep `run_baseline_test.py` and `test_whisperx.py` working

### Phase 2: Integration (Week 3+)
1. Update test scripts to use new modules
2. Deprecate hardcoded paths
3. Migrate notebook logic to scripts
4. Archive old notebooks

### Phase 3: Cleanup (Week 4+)
1. Remove deprecated code
2. Consolidate duplicate functionality
3. Finalize documentation
4. Tag stable release

---

## Risks & Mitigations

**Risk 1: Breaking existing functionality**
- Mitigation: Keep old code working, add new system alongside
- Test each change incrementally

**Risk 2: Config complexity**
- Mitigation: Start simple, iterate based on needs
- Provide sensible defaults

**Risk 3: Time estimation**
- Mitigation: Prioritize critical path (config → inventory → alignment)
- Defer nice-to-haves (unified pipeline, full refactor)

**Risk 4: Training data format**
- Mitigation: Research Hugging Face dataset format first
- Make format conversion easy if needed

---

## Next Steps

**Immediate Action:**
1. Create `config/` directory
2. Write `paths.yaml` with current structure
3. Write `models.yaml` with WhisperX large-v2 settings
4. Build basic `config.py` loader

**Then:**
5. Update `test_whisperx.py` to use config
6. Test to ensure nothing breaks
7. Move to transcript inventory

---

## Questions to Answer

1. **How many validated transcripts do we have?** → Inventory will tell us
2. **What format for training data?** → JSON with word timestamps (above)
3. **WhisperX or Whisper?** → ✅ **WhisperX large-v2** (25.01% WER)
4. **Batch size for 6GB GPU?** → 4 (tested and working)
5. **Training data structure?** → Segments with word arrays (above)

---

**Ready to start:** Configuration system is the foundation for everything else!
