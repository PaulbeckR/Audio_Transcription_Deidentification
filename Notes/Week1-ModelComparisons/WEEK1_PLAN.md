# Week 1 Implementation Plan - Quick Wins & Baseline Metrics

**Branch**: ComprehensiveRestructure
**Timeline**: Week 1
**Status**: In Progress

---

## Objectives

1. Establish baseline performance metrics with current CPU configuration
2. Enable GPU acceleration across all processing
3. Test WhisperX for alignment capabilities
4. Compare models and document findings
5. Create foundation for ongoing metrics tracking

---

## Tasks Breakdown

### Task 1: Implement WER Calculation System
**Duration**: 2-3 hours
**Priority**: High

**Subtasks**:
- [ ] Install jiwer library
- [ ] Create src/metrics.py module
- [ ] Implement WER calculation function
- [ ] Implement DER calculation function (using pyannote.metrics)
- [ ] Create comparison report generator
- [ ] Add system metrics tracking (time, memory, CPU/GPU usage)

**Deliverables**:
- Functional WER/DER calculation module
- Metrics report template

---

### Task 2: Baseline CPU Metrics (hamlet_test)
**Duration**: 2-3 hours
**Priority**: High

**Subtasks**:
- [ ] Run full pipeline on hamlet_test with CPU
- [ ] Record processing time for each stage:
  - [ ] Audio preprocessing
  - [ ] Pyannote diarization
  - [ ] RTTM cleaning
  - [ ] Whisper transcription (base model)
  - [ ] Whisper transcription (large model)
- [ ] Calculate WER against hamlet_test_truth.xlsx
- [ ] Calculate DER if ground truth RTTM available
- [ ] Document memory usage
- [ ] Document CPU utilization
- [ ] Save all metrics to baseline report

**Deliverables**:
- Notes/BASELINE_CPU_METRICS.md with full results
- CSV/JSON metrics file for future comparison

---

### Task 3: GPU Enablement
**Duration**: 1-2 hours
**Priority**: High

**Subtasks**:
- [ ] Verify CUDA installation and GPU availability
- [ ] Update transcript_notebook.ipynb for GPU
  - [ ] Modify Pyannote device selection
  - [ ] Modify Whisper device selection
- [ ] Update create_rttm.ipynb for GPU
  - [ ] Modify Pyannote device selection
  - [ ] Modify Whisper device selection
- [ ] Create helper function for device detection
- [ ] Test GPU functionality with small audio clip
- [ ] Document GPU specifications (model, memory, CUDA version)

**Deliverables**:
- GPU-enabled notebooks
- Device detection utility function
- GPU compatibility documentation

---

### Task 4: GPU Performance Metrics (hamlet_test)
**Duration**: 2-3 hours
**Priority**: High

**Subtasks**:
- [ ] Run full pipeline on hamlet_test with GPU
- [ ] Record processing time for each stage:
  - [ ] Audio preprocessing
  - [ ] Pyannote diarization (GPU)
  - [ ] RTTM cleaning
  - [ ] Whisper transcription (base model, GPU)
  - [ ] Whisper transcription (large model, GPU)
- [ ] Calculate WER against hamlet_test_truth.xlsx
- [ ] Calculate DER if ground truth RTTM available
- [ ] Document GPU memory usage
- [ ] Document GPU utilization
- [ ] Compare against CPU baseline
- [ ] Calculate speedup factor

**Deliverables**:
- Notes/GPU_METRICS.md with full results
- Comparison chart: CPU vs GPU performance
- Speedup analysis

---

### Task 5: Install & Configure WhisperX
**Duration**: 1-2 hours
**Priority**: Medium

**Subtasks**:
- [ ] Install WhisperX and dependencies
  ```bash
  pip install whisperx
  ```
- [ ] Verify installation
- [ ] Test basic transcription on hamlet_test
- [ ] Configure for GPU usage
- [ ] Document installation process
- [ ] Note any compatibility issues

**Deliverables**:
- WhisperX installed and functional
- Installation documentation in Notes/WHISPERX_SETUP.md

---

### Task 6: WhisperX Alignment Testing
**Duration**: 3-4 hours
**Priority**: Medium

**Subtasks**:
- [ ] Test WhisperX transcription on hamlet_test (CPU)
  - [ ] Record processing time
  - [ ] Calculate WER against ground truth
  - [ ] Examine word-level timestamps
- [ ] Test WhisperX transcription on hamlet_test (GPU)
  - [ ] Record processing time
  - [ ] Calculate WER against ground truth
  - [ ] Compare against CPU version
- [ ] Test WhisperX forced alignment feature
  - [ ] Use hamlet_test_truth.xlsx text
  - [ ] Align to hamlet_test audio
  - [ ] Evaluate timestamp quality
  - [ ] Compare timestamps to Pyannote RTTM
- [ ] Test WhisperX diarization
  - [ ] Compare speaker labels to ground truth
  - [ ] Calculate DER
  - [ ] Compare to standalone Pyannote

**Deliverables**:
- WhisperX metrics document
- Alignment quality assessment
- Recommendations for WhisperX vs current pipeline

---

### Task 7: Comparative Analysis & Documentation
**Duration**: 2-3 hours
**Priority**: Medium

**Subtasks**:
- [ ] Create comparison matrix:
  - [ ] Whisper Base (CPU)
  - [ ] Whisper Large (CPU)
  - [ ] Whisper Base (GPU)
  - [ ] Whisper Large (GPU)
  - [ ] WhisperX (CPU)
  - [ ] WhisperX (GPU)
- [ ] Metrics to compare:
  - [ ] WER
  - [ ] Processing time
  - [ ] Memory usage
  - [ ] Ease of use
  - [ ] Alignment quality
- [ ] Create visualization (table/chart)
- [ ] Write recommendations
- [ ] Update PROJECT_PLAN.md with findings
- [ ] Update README.md with Week 1 results

**Deliverables**:
- Notes/WEEK1_RESULTS.md with complete analysis
- Model selection recommendations
- Updated project documentation

---

## Testing Protocol

### Test File: hamlet_test
- **Location**: Audio_Local_tests/audio_files/hamlet_test.m4a.wav
- **Ground Truth**: Audio_Local_tests/transcription_test/hamlet_test_truth.xlsx
- **Duration**: ~5 minutes (first segment of full audio)
- **Speakers**: Multiple (test file from Hamlet play)

### Metrics Collection Template

For each test run, collect:
```
Configuration:
- Model: [Whisper Base/Large/WhisperX]
- Device: [CPU/GPU]
- Batch Size: [if applicable]
- Additional Settings: [...]

Performance:
- Total Time: [seconds]
- Preprocessing Time: [seconds]
- Diarization Time: [seconds]
- Transcription Time: [seconds]
- Real-Time Factor: [processing_time / audio_duration]

Accuracy:
- WER: [percentage]
- DER: [percentage]
- Speaker Confusion Rate: [if available]

Resources:
- Peak Memory: [MB]
- Average GPU/CPU Utilization: [%]
- GPU Memory: [MB, if applicable]
```

---

## Expected Outcomes

### Performance Improvements
- **CPU → GPU Speedup**: 10-50x (based on literature)
- **Pyannote**: 7 minutes → 10-30 seconds
- **Whisper**: Expect 3-5x speedup

### Accuracy Improvements
- **Whisper Base → Large**: ~20% WER reduction expected
- **WhisperX**: Potential 10-15% additional improvement from alignment

### Decision Points
1. **Continue with Whisper or switch to WhisperX?**
   - Compare WER, speed, ease of integration

2. **Which model size for production?**
   - Balance accuracy vs speed

3. **Pyannote standalone or WhisperX diarization?**
   - Compare DER, consistency

---

## Dependencies & Installation

### Required Packages
```bash
# Already installed (from requirements.txt)
- whisper (openai/whisper)
- pyannote.audio==3.3.2
- torch, torchaudio
- pydub, ffmpeg

# New installations needed
pip install jiwer              # WER calculation
pip install whisperx           # WhisperX
pip install pyannote.metrics   # DER calculation (may already be installed)
pip install psutil             # System metrics
pip install pyyaml             # For future config management
```

### GPU Requirements
- CUDA-compatible GPU (detected in environment)
- CUDA toolkit (verify with `nvidia-smi`)
- PyTorch with CUDA support (verify with `torch.cuda.is_available()`)

---

## Risk Mitigation

**Risk**: GPU not available or incompatible
- **Mitigation**: Verify CUDA before starting GPU tasks
- **Fallback**: Document CPU as baseline, investigate GPU issues separately

**Risk**: WhisperX installation issues
- **Mitigation**: Test in separate virtual environment first
- **Fallback**: Continue with Whisper + separate alignment if needed

**Risk**: Metrics calculation errors
- **Mitigation**: Manually verify first calculation
- **Fallback**: Use online WER calculators for validation

---

## Success Criteria

Week 1 is successful if:
- [ ] WER calculation is functional and validated
- [ ] Complete baseline metrics documented (CPU)
- [ ] GPU acceleration is working
- [ ] Achieved measurable speedup (minimum 5x)
- [ ] WhisperX tested with initial assessment
- [ ] Comparative analysis completed
- [ ] Clear recommendation for Phase 2 direction

---

## Deliverables Checklist

Documentation:
- [ ] Notes/BASELINE_CPU_METRICS.md
- [ ] Notes/GPU_METRICS.md
- [ ] Notes/WHISPERX_SETUP.md
- [ ] Notes/WEEK1_RESULTS.md
- [ ] Updated PROJECT_PLAN.md
- [ ] Updated README.md

Code:
- [ ] src/metrics.py (WER, DER, system metrics)
- [ ] Updated transcript_notebook.ipynb (GPU-enabled)
- [ ] Updated create_rttm.ipynb (GPU-enabled)
- [ ] Helper function for device detection

Data:
- [ ] Baseline metrics (CSV/JSON)
- [ ] GPU metrics (CSV/JSON)
- [ ] WhisperX metrics (CSV/JSON)
- [ ] Comparison matrix (CSV/table)

---

## Next Steps (Week 2 Preview)

Based on Week 1 results:
1. Align all validated transcripts with chosen tool
2. Create configuration management system
3. Begin training data preparation
4. Plan fine-tuning approach
