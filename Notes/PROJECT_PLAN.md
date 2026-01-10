# Audio Transcription & Deidentification - Project Plan

## Project Overview

**Goal**: Process audio interview files into high-accuracy transcripts using speaker diarization and speech-to-text, with capability for continuous model improvement through fine-tuning.

**Current State**:
- Working pipeline: Pyannote (diarization) + Whisper (transcription)
- Performance issues: Slow processing on CPU, moderate accuracy
- Alignment challenges: RTTM speaker boundaries don't match human transcripts

**Target State**:
- GPU-accelerated processing (10-50x speedup)
- Multiple model options (WhisperAI, WhisperX) with comparative evaluation
- Automated metrics tracking (WER, DER, speed, memory)
- Fine-tuned models for interview domain
- Batch processing pipeline for folder automation

---

## Audio Context

- **Format**: Long-form interviews (2 speakers)
- **Participants**: 3 interviewers total, various interviewees (1-3 sessions each)
- **Sessions**: Labeled S1, S2, S3 by topic/curriculum
- **Characteristics**:
  - Moderate vocabulary overlap across sessions
  - Consistent interviewer voice characteristics
  - Opportunity for domain-specific fine-tuning

---

## Key Challenges & Solutions

### Challenge 1: Speed
- **Problem**: Pyannote takes 6+ hours for short audio on CPU
- **Solution**: GPU acceleration (CUDA)
- **Expected Impact**: 10-50x speedup

### Challenge 2: Accuracy
- **Problem**: High error rate, speaker misalignment
- **Solutions**:
  - Larger Whisper models (base → large)
  - Fine-tuning on domain-specific data
  - WhisperX for better alignment
- **Expected Impact**: 30-50% WER reduction

### Challenge 3: Training Data
- **Problem**: Existing transcripts lack timestamps
- **Solution**: Forced alignment with WhisperX or Montreal Forced Aligner
- **Expected Impact**: Automated timestamp generation

### Challenge 4: Evaluation
- **Problem**: No quantified metrics for improvements
- **Solution**: Automated WER, DER, speed, memory tracking
- **Expected Impact**: Data-driven optimization decisions

---

## Implementation Phases

### Phase 1: Quick Wins & Baseline Metrics (Week 1) ⚡
**Status**: In Progress (Branch: ComprehensiveRestructure)

**Objectives**:
1. Establish baseline metrics (CPU performance)
2. Enable GPU acceleration
3. Test WhisperX alignment
4. Compare models and configurations

**Deliverables**:
- WER calculation implementation
- CPU baseline metrics (hamlet_test)
- GPU-enabled code
- GPU performance metrics (hamlet_test)
- WhisperX alignment test results
- Comparative analysis document

### Phase 2: Alignment & Training Data (Week 2)
**Objectives**:
1. Align all validated transcripts with WhisperX
2. Create configuration management system
3. Build training dataset

**Deliverables**:
- Aligned training data in standard format
- config/paths.yaml, config/models.yaml
- Updated codebase using configs
- Training data quality report

### Phase 3: Fine-Tuning (Week 3)
**Objectives**:
1. Prepare Whisper fine-tuning dataset
2. Train domain-adapted model
3. Evaluate improvements

**Deliverables**:
- Fine-tuned Whisper model
- Training metrics and logs
- Evaluation report with WER comparison
- Model selection recommendations

### Phase 4: Production Pipeline (Week 4)
**Objectives**:
1. Refactor notebooks to modular code
2. Build batch processing system
3. Create automation scripts

**Deliverables**:
- src/ module structure
- Batch processing scripts
- Folder monitoring system
- User documentation

### Phase 5: Iteration & Improvement (Ongoing)
**Objectives**:
1. Process new audio files
2. Review and correct outputs
3. Expand training dataset
4. Monitor metrics over time

**Deliverables**:
- Processed transcripts
- Updated training datasets
- Performance tracking dashboard
- Continuous improvement reports

---

## Model Comparison Framework

### Models to Evaluate

1. **Whisper (OpenAI)**
   - Models: base, medium, large, large-v3
   - Pros: Proven accuracy, multilingual
   - Cons: No built-in alignment, separate diarization needed

2. **WhisperX**
   - Features: Whisper + forced alignment + optional diarization
   - Pros: Word-level timestamps, integrated pipeline
   - Cons: Additional dependency, memory intensive

3. **Future Considerations**
   - NVIDIA NeMo (GPU-optimized)
   - Faster Whisper (optimized inference)
   - SpeechBrain (research framework)

### Evaluation Metrics

**Accuracy**:
- Word Error Rate (WER) - primary metric
- Character Error Rate (CER) - supplementary
- Diarization Error Rate (DER) - speaker identification

**Performance**:
- Processing time (seconds per minute of audio)
- Real-time factor (processing time / audio duration)
- GPU memory usage (MB)
- CPU utilization (%)

**Reliability**:
- Error consistency across files
- Speaker confusion rate
- Silence handling accuracy

---

## Technical Architecture

### Current Stack
- **Diarization**: Pyannote Audio 3.3.2
- **Transcription**: OpenAI Whisper (base model)
- **Audio Processing**: pydub, ffmpeg
- **Compute**: CPU (will migrate to GPU)

### Planned Additions
- **WhisperX**: Alignment and enhanced transcription
- **jiwer**: WER/CER calculation
- **PyYAML**: Configuration management
- **psutil**: System metrics monitoring

### Hardware Requirements
- **GPU**: CUDA-compatible (detected in environment)
- **Memory**: 16GB+ RAM recommended for large models
- **Storage**: ~10GB for models, expandable for training data

---

## Data Privacy & Security

**Requirements**:
- All processing must be local (no cloud APIs)
- Audio files and transcripts are confidential
- hamlet_test is the only safe file for sharing/testing

**Compliance**:
- No data uploaded to external services
- Models run locally (Whisper, Pyannote, WhisperX all support offline)
- Git excludes sensitive data via .gitignore
- Future: PHI detection with SpaCy/NLTK for deidentification

---

## Success Criteria

### Week 1
- [ ] WER calculation functional
- [ ] Baseline metrics documented (CPU)
- [ ] GPU acceleration working
- [ ] 10x+ speedup achieved
- [ ] WhisperX tested and evaluated

### Month 1
- [ ] All validated transcripts aligned
- [ ] Configuration system implemented
- [ ] Fine-tuned model trained
- [ ] 30%+ WER improvement from baseline
- [ ] Automated batch processing working

### Month 3
- [ ] 20+ hours of training data
- [ ] WER < 5% on interview domain
- [ ] Full automation pipeline
- [ ] Documentation complete
- [ ] Reproducible workflow established

---

## Resources & References

### Documentation
- Whisper: https://github.com/openai/whisper
- WhisperX: https://github.com/m-bain/whisperX
- Pyannote: https://github.com/pyannote/pyannote-audio
- jiwer: https://github.com/jitsi/jiwer

### Academic References
- Recent Methods in Speech Segmentation: link.springer.com/chapter/10.1007/978-3-031-70259-4_21
- Whisper paper: https://arxiv.org/abs/2212.04356
- Pyannote paper: https://arxiv.org/abs/2104.04045

### Internal Notes
- [Methods_Algorithms.md](Methods_Algorithms.md) - Technical background
- [Ideas_notes.md](Ideas_notes.md) - Ongoing ideas and discoveries
- [File_Structure.md](File_Structure.md) - Codebase organization

---

## Risk Management

### Risk 1: GPU Compatibility Issues
- **Mitigation**: Test on hamlet_test first, maintain CPU fallback
- **Contingency**: Use cloud GPU if local fails (with data restrictions)

### Risk 2: WhisperX Performance
- **Mitigation**: Compare against baseline before full adoption
- **Contingency**: Continue with Whisper + separate alignment

### Risk 3: Training Data Quality
- **Mitigation**: Manual validation of alignment samples
- **Contingency**: Use smaller, high-quality dataset vs larger noisy one

### Risk 4: Model Size vs Speed Tradeoff
- **Mitigation**: Test multiple model sizes with metrics
- **Contingency**: Offer multiple processing modes (fast/accurate)

---

## Change Log

**2025-12-23**: Initial project plan created
- Established 5-phase implementation roadmap
- Defined Week 1 quick wins
- Created model comparison framework
- Documented current state and challenges
