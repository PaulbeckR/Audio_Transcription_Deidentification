# WhisperX Installation Issue - Windows GPU Incompatibility

**Date:** December 23, 2025
**Issue:** WhisperX installation breaks GPU support on Windows

---

## Problem Summary

WhisperX has a dependency conflict that makes it incompatible with CUDA-enabled PyTorch on Windows:

1. **WhisperX 3.7.4** requires `torch~=2.8.0`
2. **WhisperX 3.6.0** requires `torch~=2.9.1`
3. **PyTorch 2.8.0 and 2.9.1 are only available as CPU-only** versions
4. **Latest CUDA PyTorch is 2.6.0+cu124** (or 2.5.1+cu121 which we use)

### Installation Attempts

**Attempt 1:** Install WhisperX 3.7.4
- Result: Installed PyTorch 2.8.0 (CPU-only), broke GPU support

**Attempt 2:** Install WhisperX 3.6.0
- Result: Installed PyTorch 2.9.1 (CPU-only), broke GPU support

**Attempt 3:** Install PyTorch 2.8.0+cu124
- Result: Version doesn't exist - CUDA builds stop at 2.6.0

---

## Root Cause

WhisperX's PyTorch version requirements (`~=2.8.0` or `~=2.9.1`) are **ahead of CUDA-enabled PyTorch releases**. The PyTorch project hasn't released CUDA versions for 2.8.0+ yet, only CPU versions.

This appears to be a packaging issue with WhisperX where the developers are using the latest PyTorch CPU versions without considering CUDA availability.

---

## Alternative Solutions

### Option 1: Use WSL2 (Windows Subsystem for Linux)
- Install Ubuntu in WSL2
- Install CUDA toolkit in WSL2
- Install WhisperX with GPU support in Linux environment
- **Pros:** Full GPU support, proper Linux environment
- **Cons:** Requires WSL2 setup, file path management complexity

### Option 2: Use Cloud/Docker Environment
- Run WhisperX in Google Colab, AWS, or Azure with GPU
- **Pros:** Guaranteed compatibility, access to better GPUs
- **Cons:** Data transfer concerns (PHI/sensitive audio), cost

### Option 3: Install from Source with Modified Dependencies
- Clone WhisperX repository
- Modify `setup.py` to use `torch>=2.0,<3.0` instead of `~=2.8.0`
- Install from local source
- **Pros:** Can work with our CUDA PyTorch
- **Cons:** May have compatibility issues, maintenance burden

### Option 4: Use Alternative Alignment Tools
Instead of WhisperX, use:
- **Montreal Forced Aligner (MFA)** - Industry standard for alignment
- **Wav2Vec2 + Custom alignment** - Use Facebook's Wav2Vec2
- **Gentle** - Kaldi-based forced aligner
- **aeneas** - Python forced alignment library

**Pros:** Avoid WhisperX dependency hell
**Cons:** Additional integration work, may not be as optimized

### Option 5: Use Whisper + Pyannote (Current Approach)
Continue with our current pipeline:
- Pyannote for diarization (works great on GPU - 6.18x speedup)
- Whisper for transcription (works on GPU - 1.27x-2.01x speedup)
- Post-process to get word-level timestamps from Whisper's output
- **Pros:** Already working, proven GPU acceleration, good accuracy
- **Cons:** No word-level forced alignment out of the box

---

## Recommended Approach

**For Week 2:** Proceed with **Option 5** (current Whisper + Pyannote pipeline) and defer WhisperX testing.

**Justification:**
1. We already have a working GPU-accelerated pipeline (1.44x speedup proven)
2. Whisper provides word-level timestamps with `word_timestamps=True` flag
3. We can extract and use these for training data alignment
4. Avoids breaking our CUDA setup
5. WhisperX can be revisited later if needed (via WSL2 or cloud)

**For Training Data Preparation:**
- Use Whisper's built-in `word_timestamps=True` feature
- Extract word-level timing from Whisper JSON output
- Combine with Pyannote speaker diarization
- This gives us: `word | start_time | end_time | speaker | confidence`

---

## Whisper Word Timestamps Example

Whisper already provides word-level timestamps when using `word_timestamps=True`:

```python
import whisper

model = whisper.load_model("large", device="cuda")
result = model.transcribe(
    audio="hamlet_test.wav",
    language='en',
    word_timestamps=True  # Enable word-level timestamps
)

# Access word-level data
for segment in result['segments']:
    print(f"Segment: {segment['start']}-{segment['end']}")
    for word_data in segment.get('words', []):
        print(f"  {word_data['word']}: {word_data['start']}-{word_data['end']}")
```

This provides the alignment functionality we need without WhisperX!

---

## Next Steps

1. ✅ Document WhisperX installation issue
2. **Update Week 2 Task 1:** Change from "Test WhisperX" to "Extract word-level timestamps from Whisper"
3. Create script to combine:
   - Whisper word timestamps
   - Pyannote speaker labels
   - Generate aligned training data format
4. Proceed with Week 2 Tasks 2-4 using Whisper-based approach

---

## Future Consideration

If WhisperX becomes critical:
- Set up WSL2 environment for testing
- Or use cloud environment for alignment pipeline
- Re-evaluate when PyTorch CUDA 2.8.0+ is released

For now, Whisper's built-in word timestamps + Pyannote diarization provides everything we need for fine-tuning preparation.
