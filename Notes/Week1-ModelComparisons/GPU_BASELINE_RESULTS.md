# GPU Baseline Test Results - Complete Analysis

**Test Date:** December 23, 2025
**Audio File:** hamlet_test.m4a.wav (12.6 minutes)
**Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU
**CUDA Version:** 12.7

---

## Executive Summary

GPU acceleration achieved **1.44x overall speedup** while maintaining identical accuracy to CPU processing.

**Key Findings:**
- **Diarization speedup: 6.18x** (279.6s → 45.3s)
- **Whisper Base speedup: 2.01x** (105.2s → 52.4s)
- **Whisper Large speedup: 1.27x** (1742.7s → 1374.7s)
- **Total time: 35.5 min (CPU) → 24.6 min (GPU)**
- **Accuracy: Identical** (Large model: 27.66% WER on both CPU and GPU)

---

## Detailed Performance Comparison

### Overall Pipeline Performance

| Metric | CPU | GPU | Speedup |
|--------|-----|-----|---------|
| **Total Time** | 2128.6s (35.5 min) | 1473.4s (24.6 min) | **1.44x** |
| **Real-Time Factor** | 2.81x | 1.95x | 1.44x faster |
| **Processing Speed** | 0.36x slower than realtime | 0.51x slower than realtime | 44% improvement |

### Stage-by-Stage Breakdown

| Stage | CPU Time | GPU Time | Speedup | % of Total (CPU) | % of Total (GPU) |
|-------|----------|----------|---------|------------------|------------------|
| **Preprocessing** | 0.70s | 0.71s | 1.0x | 0.03% | 0.05% |
| **Diarization (Pyannote)** | 279.64s | 45.26s | **6.18x** | 13.1% | 3.1% |
| **RTTM Processing** | 0.32s | 0.26s | 1.2x | 0.01% | 0.02% |
| **Whisper Base** | 105.21s | 52.44s | **2.01x** | 4.9% | 3.6% |
| **Whisper Large** | 1742.74s | 1374.73s | **1.27x** | 81.9% | 93.3% |

### Key Insights

1. **Pyannote Diarization** benefits most from GPU (6.18x speedup)
2. **Whisper Base** shows good GPU acceleration (2.01x speedup)
3. **Whisper Large** shows modest GPU acceleration (1.27x speedup)
   - Still dominates total processing time (93% on GPU vs 82% on CPU)
   - Appears to be memory-bandwidth limited on 6GB GPU
4. **Bottleneck shifted** from diarization (13% CPU) to Whisper Large (93% GPU)

---

## Accuracy Comparison

### Word Error Rate (WER)

| Model | CPU WER | GPU WER | Difference |
|-------|---------|---------|------------|
| **Whisper Base** | 40.93% | 40.63% | 0.29% (negligible) |
| **Whisper Large** | 27.66% | 27.66% | 0.00% (identical) |

### Detailed Error Breakdown - Whisper Large

| Metric | CPU | GPU |
|--------|-----|-----|
| **Total Words** | 1760 | 1760 |
| **Correct (Hits)** | 1289 | 1289 |
| **Substitutions** | 381 | 381 |
| **Deletions** | 33 | 33 |
| **Insertions** | 57 | 57 |
| **Total Errors** | 471 | 471 |

**Conclusion:** GPU produces identical accuracy to CPU for Whisper Large model.

### Model Comparison (GPU Results)

| Metric | Base Model | Large Model | Improvement |
|--------|------------|-------------|-------------|
| **WER** | 40.63% | 27.66% | **31.9% relative** |
| **Total Errors** | 692 | 471 | 221 fewer errors |
| **Processing Time** | 52.4s | 1374.7s | 26.2x slower |

**Recommendation:** Use Whisper Large for production despite slower processing - the 31.9% accuracy improvement justifies the time cost.

---

## System Resource Usage

| Metric | CPU | GPU |
|--------|-----|-----|
| **Average CPU Usage** | 34.9% | 35.3% |
| **Peak Memory** | 6478 MB | 4562 MB |
| **Memory Increase** | 6116 MB | 3991 MB |

**Note:** GPU processing used 30% less RAM than CPU processing, likely due to model weights stored in GPU memory instead of system RAM.

---

## Projected Speedup for Longer Audio

For a 1-hour (3600s) audio file, using Whisper Large model:

**CPU Projection:**
- Diarization: ~1329s (22 min)
- Whisper Large: ~8274s (138 min)
- **Total: ~160 minutes (2.7 hours)**

**GPU Projection:**
- Diarization: ~215s (3.6 min)
- Whisper Large: ~6527s (109 min)
- **Total: ~113 minutes (1.9 hours)**

**Savings: 47 minutes per hour of audio (29% faster)**

---

## Hardware Specifications

### GPU Details
- **Model:** NVIDIA GeForce RTX 3050 Laptop GPU
- **Memory:** 6GB GDDR6
- **CUDA Cores:** 2048
- **CUDA Version:** 12.7
- **Driver Version:** 566.07

### Software Stack
- **Python:** 3.x (in virtual environment)
- **PyTorch:** 2.4.1+cu121 (CUDA 12.1)
- **Pyannote Audio:** 3.1.1
- **OpenAI Whisper:** Latest
- **CUDA Toolkit:** 12.1 (via PyTorch)

---

## Recommendations

### For Current Hardware (RTX 3050 6GB)

1. **Use GPU for all processing** - 1.44x speedup with identical accuracy
2. **Whisper Large is still the bottleneck** - occupies 93% of processing time
3. **Consider batch processing** - GPU can handle multiple files more efficiently than CPU
4. **Monitor GPU memory** - 6GB is sufficient but leaves little headroom for larger models

### For Future Improvements

1. **Upgrade to RTX 4060/4070 (8-12GB)** - Would significantly improve Whisper Large performance
2. **Test Whisper Medium** - May offer better speed/accuracy tradeoff on current GPU
3. **Implement WhisperX** - Could provide better GPU utilization with batched inference
4. **Consider quantization** - INT8/FP16 models could improve speed without much accuracy loss

### Production Pipeline Recommendation

**Current Setup (RTX 3050 6GB):**
```
Audio Input → Pyannote (GPU) → Whisper Large (GPU) → Output
Expected: ~25 minutes per 12.6 min audio (2.0x realtime)
```

**For High-Volume Processing:**
- Consider cloud GPU (A100, V100) for >10x speedup
- Or queue-based processing with current GPU for overnight batch jobs

---

## Files Generated

**Output Files:**
- `baseline_metrics_gpu.json` - Complete GPU performance metrics
- `hamlet_test_whisper_base_gpu.csv` - Base model transcription (70 segments)
- `hamlet_test_whisper_large_gpu.csv` - Large model transcription (70 segments)
- `wer_results_gpu.json` - GPU WER calculations
- `hamlet_test_baseline_gpu.rttm` - Diarization output

**Comparison Files:**
- `baseline_metrics_cpu.json` - CPU baseline (for comparison)
- `wer_results.json` - CPU WER results (for comparison)

---

## Conclusion

GPU acceleration on the RTX 3050 6GB provides a meaningful **44% speedup** while maintaining perfect accuracy. The investment in GPU setup is justified for production use.

**Next Steps:**
1. ✅ Week 1 Complete: CPU baseline, GPU baseline, WER calculations documented
2. Proceed to Week 2: Configuration system, training data preparation
3. Test WhisperX for potential additional speedup
4. Explore model quantization for faster inference

**Cost-Benefit Analysis:**
- Setup time: ~2 hours (one-time)
- Speedup: 1.44x (ongoing benefit)
- Accuracy: Identical
- **Verdict: Recommended for production deployment**
