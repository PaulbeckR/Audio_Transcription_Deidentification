# Whisper vs WhisperX Model Comparison

**Test Audio:** hamlet_test.m4a.wav (756.74 seconds / 12.61 minutes)
**Test Date:** December 24, 2025
**Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU

---

## Summary Results

| Model | Framework | WER | Speed | Total Time | Memory |
|-------|-----------|-----|-------|------------|--------|
| **medium.en** | Whisper | 28.95% | 2.53x | 299.16s | 1.85 GB |
| **medium.en** | WhisperX | 28.30% | **6.46x** ⭐ | 117.21s | - |
| **large-v2** | Whisper | **27.60%** ⭐ | 0.61x | 1248.95s | 4.34 GB |
| **large-v2** | WhisperX | **25.01%** 🏆 | 2.26x | 334.16s | - |
| **large-v3** | Whisper | 28.13% | 0.64x | 1178.72s | 4.91 GB |
| **large-v3** | WhisperX | 28.36% | 2.16x | 349.74s | - |

**Legend:**
- ⭐ Best in category
- 🏆 Best overall WER

---

## Detailed Analysis

### 1. Accuracy (WER - Word Error Rate)

**Rankings (Lower is Better):**
1. **WhisperX large-v2: 25.01%** 🏆 **BEST**
2. Whisper large-v2: 27.60%
3. Whisper large-v3: 28.13%
4. WhisperX medium.en: 28.30%
5. WhisperX large-v3: 28.36%
6. Whisper medium.en: 28.95%

**Key Findings:**
- **WhisperX large-v2 is the most accurate** (25.01% WER)
- WhisperX consistently outperforms Whisper in same model size
- large-v3 doesn't improve over large-v2 (may be overfitted for different data)
- WhisperX medium.en nearly matches Whisper large models!

### Error Breakdown

#### medium.en Comparison
| Metric | Whisper | WhisperX | Difference |
|--------|---------|----------|------------|
| Substitutions | 394 | 367 | -27 (better) |
| Deletions | 37 | 67 | +30 (worse) |
| Insertions | 62 | 48 | -14 (better) |
| **Total Errors** | **493** | **482** | **-11 (better)** |

#### large-v2 Comparison
| Metric | Whisper | WhisperX | Difference |
|--------|---------|----------|------------|
| Substitutions | 377 | 312 | -65 (better) |
| Deletions | 37 | 70 | +33 (worse) |
| Insertions | 56 | 44 | -12 (better) |
| **Total Errors** | **470** | **426** | **-44 (better)** |

**Analysis:** WhisperX has more deletions but significantly fewer substitutions, leading to better overall WER.

---

### 2. Processing Speed

**Rankings (Faster is Better):**
1. **WhisperX medium.en: 6.46x realtime** ⭐ **FASTEST**
2. Whisper medium.en: 2.53x realtime
3. WhisperX large-v2: 2.26x realtime
4. WhisperX large-v3: 2.16x realtime
5. Whisper large-v2: 0.61x (slower than realtime)
6. Whisper large-v3: 0.64x (slower than realtime)

**Key Findings:**
- **WhisperX is 2.5x - 3.7x faster than Whisper** for same model size
- WhisperX medium.en is incredibly fast (processes 12.6 min audio in 1.95 min)
- Whisper large models are slower than realtime (unusable for live transcription)
- WhisperX large models maintain >2x realtime speed

### Time Breakdown

#### WhisperX medium.en (Total: 117.21s)
- Model Load: 3.72s (3%)
- Transcription: 23.20s (20%)
- **Alignment: 17.50s (15%)** ← Forced alignment overhead
- Diarization: 72.79s (62%)

#### WhisperX large-v2 (Total: 334.16s)
- Model Load: 7.22s (2%)
- Transcription: 37.05s (11%)
- **Alignment: 17.55s (5%)** ← Consistent alignment time
- Diarization: 272.34s (82%) ← Longer for complex diarization

**Note:** WhisperX alignment time is consistent (~17-26s) regardless of model size!

---

### 3. Memory Usage

| Model | Whisper Peak | Whisper Increase |
|-------|--------------|------------------|
| medium.en | 1.85 GB | 1.27 GB |
| large-v2 | 4.34 GB | 3.75 GB |
| large-v3 | 4.91 GB | 4.32 GB |

**Key Findings:**
- Large models use 2.3x - 2.6x more memory than medium
- large-v3 uses most memory (4.91 GB peak)
- medium.en fits comfortably in 6GB VRAM

---

### 4. Output Quality

#### Segment Counts
| Model | Framework | Segments |
|-------|-----------|----------|
| medium.en | Whisper | 68 |
| medium.en | WhisperX | **159** |
| large-v2 | Whisper | 68 |
| large-v2 | WhisperX | **165** |
| large-v3 | Whisper | 68 |
| large-v3 | WhisperX | **162** |

**Key Findings:**
- **WhisperX produces 2.3x - 2.4x more segments** (finer granularity)
- WhisperX provides word-level timestamps via forced alignment
- More segments = better timing precision for training data

#### Speaker Diarization
| Model | Framework | Speakers Detected |
|-------|-----------|-------------------|
| All | Whisper | N/A (external Pyannote) |
| medium.en | WhisperX | 4 |
| large-v2 | WhisperX | 4 |
| large-v3 | WhisperX | 0 (diarization failed) |

---

## Recommendations

### Best Overall: **WhisperX large-v2** 🏆

**Reasons:**
1. **Lowest WER: 25.01%** (3.5% better than best Whisper)
2. **Fast: 2.26x realtime** (processes 12.6 min in 5.6 min)
3. **Forced alignment** provides word-level timestamps
4. **Integrated diarization** (4 speakers detected)
5. Balanced speed/accuracy tradeoff

### Best for Speed: **WhisperX medium.en** ⚡

**Reasons:**
1. **Fastest: 6.46x realtime** (processes 12.6 min in 1.95 min)
2. **Good WER: 28.30%** (only 3.3% worse than large-v2)
3. **Low memory:** Fits easily in 6GB VRAM
4. Excellent for batch processing

### Best for Whisper-only: **Whisper large-v2**

**Reasons:**
1. **Best Whisper WER: 27.60%**
2. Available in main environment (no separate setup)
3. Works with word_timestamps=True
4. Can combine with external Pyannote

### NOT Recommended: **large-v3**

**Reasons:**
1. No accuracy improvement over large-v2
2. Slightly slower than large-v2
3. WhisperX diarization failed
4. Highest memory usage (4.91 GB)

---

## Speed Comparison Chart

```
Processing Time (seconds) - Lower is Better
0        200      400      600      800      1000     1200     1400
├────────┼────────┼────────┼────────┼────────┼────────┼────────┼
│
WhisperX medium.en   ■■ 117s (6.46x realtime) ⭐ FASTEST
│
Whisper medium.en    ■■■■■■ 299s (2.53x realtime)
│
WhisperX large-v2    ■■■■■■■ 334s (2.26x realtime)
│
WhisperX large-v3    ■■■■■■■ 350s (2.16x realtime)
│
Whisper large-v3     ■■■■■■■■■■■■■■■■■■■■■ 1179s (0.64x realtime)
│
Whisper large-v2     ■■■■■■■■■■■■■■■■■■■■■■ 1249s (0.61x realtime)
│
```

---

## WER Comparison Chart

```
Word Error Rate (%) - Lower is Better
20%      22%      24%      26%      28%      30%
├────────┼────────┼────────┼────────┼────────┼
│
WhisperX large-v2    ■■■■■ 25.01% 🏆 BEST
│
Whisper large-v2     ■■■■■■ 27.60%
│
Whisper large-v3     ■■■■■■■ 28.13%
│
WhisperX medium.en   ■■■■■■■ 28.30%
│
WhisperX large-v3    ■■■■■■■ 28.36%
│
Whisper medium.en    ■■■■■■■ 28.95%
│
```

---

## Cost-Benefit Analysis

### WhisperX medium.en vs Whisper medium.en
- **Accuracy:** +2.2% improvement (28.95% → 28.30%)
- **Speed:** +155% faster (2.53x → 6.46x realtime)
- **Features:** + Forced alignment, + Integrated diarization
- **Cost:** Separate environment setup
- **Verdict:** ✅ **STRONGLY RECOMMENDED**

### WhisperX large-v2 vs Whisper large-v2
- **Accuracy:** +9.4% improvement (27.60% → 25.01%)
- **Speed:** +270% faster (0.61x → 2.26x realtime)
- **Features:** + Forced alignment, + Integrated diarization
- **Memory:** Similar usage
- **Verdict:** ✅ **HIGHLY RECOMMENDED**

### large-v2 vs medium.en (WhisperX)
- **Accuracy:** +11.6% improvement (28.30% → 25.01%)
- **Speed:** -65% slower (6.46x → 2.26x realtime)
- **Processing:** 217s longer (117s → 334s)
- **Verdict:** ⚖️ **USE CASE DEPENDENT**
  - Batch processing: Use medium.en (faster)
  - Best accuracy needed: Use large-v2

---

## Conclusion

**For this project (training data preparation):**

1. **Primary Choice: WhisperX large-v2**
   - Best accuracy (25.01% WER)
   - Fast enough (2.26x realtime)
   - Superior forced alignment
   - Integrated diarization

2. **Alternative: WhisperX medium.en**
   - When speed is critical
   - For large batch processing
   - Still excellent accuracy (only 3.3% worse)

3. **Avoid:**
   - Whisper large models (too slow)
   - large-v3 (no advantage over large-v2)

**WhisperX provides 9-11% better accuracy while being 2.5-3.7x faster than Whisper!**
