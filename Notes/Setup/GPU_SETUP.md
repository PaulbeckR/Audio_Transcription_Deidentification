# GPU Setup Guide

**Date**: 2025-12-23
**Status**: GPU Not Currently Configured
**Action Required**: Install PyTorch with CUDA support

---

## Current Status

**GPU Verification Results**:
- CUDA Available: **NO**
- Current PyTorch: CPU-only version (torch==2.4.1)
- GPU Hardware: To be determined

**Diagnosis**:
The current PyTorch installation is CPU-only. To enable GPU acceleration, we need to:
1. Verify NVIDIA GPU is present
2. Install CUDA toolkit (if not present)
3. Reinstall PyTorch with CUDA support

---

## Step 1: Verify GPU Hardware

### Check for NVIDIA GPU (Windows)

**Option A: Device Manager**
1. Press `Win + X` and select "Device Manager"
2. Expand "Display adapters"
3. Look for NVIDIA GPU (e.g., "NVIDIA GeForce RTX 3080")

**Option B: Command Line**
```bash
# Check if nvidia-smi is available (means NVIDIA drivers installed)
nvidia-smi
```

**Expected Output (if GPU present)**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx.xx    Driver Version: 535.xx.xx    CUDA Version: 12.2    |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ... WDDM  | 00000000:01:00.0  On |                  N/A |
...
```

**If nvidia-smi not found**:
- Either no NVIDIA GPU present, OR
- NVIDIA drivers not installed

---

## Step 2: Install CUDA Toolkit (if needed)

### Check CUDA Version

If `nvidia-smi` works, check the CUDA version in top right corner.

### Install CUDA Toolkit

**Download**: https://developer.nvidia.com/cuda-downloads

**Recommended Version**: CUDA 12.1 or 11.8 (check PyTorch compatibility)

**Installation**:
1. Download installer for Windows
2. Run installer (typically 2-3 GB download)
3. Follow installation wizard
4. Restart computer after installation

**Verify Installation**:
```bash
nvcc --version
```

---

## Step 3: Install PyTorch with CUDA Support

### Check PyTorch Compatibility

Visit: https://pytorch.org/get-started/locally/

Select:
- PyTorch Build: Stable
- Your OS: Windows
- Package: Pip
- Language: Python
- Compute Platform: CUDA 12.1 (or your CUDA version)

### Installation Command

**For CUDA 12.1** (most recent):
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8** (if older GPU):
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Important**: This will reinstall PyTorch. Your current version will be replaced.

### Verify Installation

```bash
python test_gpu.py
```

**Expected Output**:
```
GPU VERIFICATION TEST
1. PyTorch CUDA Availability:
   CUDA Available: True
   CUDA Version: 12.1
   Number of GPUs: 1
   GPU 0: NVIDIA GeForce RTX 3080
   - Total Memory: 10.00 GB
   ...
```

---

## Step 4: Install CUDA-Accelerated Dependencies

### Update Torchaudio
```bash
pip install torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Verify Whisper Works with GPU
```python
import whisper
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base", device=device)
print(f"Whisper loaded on: {device}")
```

### Verify Pyannote Works with GPU
```python
from pyannote.audio import Pipeline
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
pipeline.to(device)
print(f"Pyannote loaded on: {device}")
```

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution 1**: Reduce batch size
```python
# For Whisper
result = model.transcribe(audio, fp16=False)  # Disable FP16

# For Pyannote
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
# Process shorter segments
```

**Solution 2**: Use smaller models
- Whisper: Use `small` or `medium` instead of `large`
- Monitor GPU memory: `torch.cuda.memory_allocated()`

**Solution 3**: Clear GPU cache
```python
import torch
torch.cuda.empty_cache()
```

### Issue: "RuntimeError: CUDA error: no kernel image available"

**Cause**: PyTorch CUDA version doesn't match GPU compute capability

**Solution**:
1. Check GPU compute capability: https://developer.nvidia.com/cuda-gpus
2. Install compatible PyTorch version
3. Older GPUs may need CUDA 11.8 instead of 12.1

### Issue: Multiple CUDA versions installed

**Solution**: Ensure PATH points to correct CUDA version
```bash
# Check which CUDA is being used
nvcc --version

# Windows: Check PATH environment variable
echo %PATH%
```

---

## Performance Expectations

### Before GPU (CPU):
- Pyannote: ~7 minutes per 5-min audio
- Whisper large: ~5-10 minutes per 5-min audio
- Total: ~12-17 minutes per 5-min audio

### After GPU:
- Pyannote: ~10-30 seconds per 5-min audio (10-40x faster)
- Whisper large: ~30-60 seconds per 5-min audio (5-10x faster)
- Total: ~40-90 seconds per 5-min audio

**Overall speedup**: 10-25x expected

---

## Alternative: Use CPU for Now

If GPU setup is problematic, we can:

1. **Continue with CPU for Week 1**
   - Document CPU baseline metrics
   - Skip GPU comparison for now
   - Focus on WhisperX and accuracy improvements

2. **Use smaller models**
   - Whisper `base` or `medium` instead of `large`
   - Trade some accuracy for speed

3. **Process overnight**
   - Batch process files during downtime
   - CPU is slower but works

4. **Cloud GPU (future option)**
   - Google Colab (free tier with GPU)
   - AWS/Azure GPU instances
   - **Note**: Must handle data privacy considerations

---

## Next Steps

### If GPU Available:
1. [ ] Run `nvidia-smi` and document GPU model
2. [ ] Note CUDA version from nvidia-smi
3. [ ] Install PyTorch with CUDA support
4. [ ] Run `python test_gpu.py` to verify
5. [ ] Update notebooks to use GPU
6. [ ] Run hamlet_test with GPU
7. [ ] Document speedup

### If No GPU:
1. [ ] Document that system is CPU-only
2. [ ] Adjust Week 1 plan to focus on CPU metrics only
3. [ ] Consider GPU options for future:
   - New hardware
   - Cloud GPU (with data privacy review)
   - Continue with CPU (acceptable for moderate usage)

---

## Hardware Recommendations (Future)

If purchasing GPU hardware:

**Budget Option** (~$300-400):
- NVIDIA RTX 3060 (12GB VRAM)
- Good for Whisper large + Pyannote

**Recommended** (~$600-800):
- NVIDIA RTX 4070 (12GB VRAM)
- Excellent performance/price

**High-End** (~$1200+):
- NVIDIA RTX 4080 or 4090 (16-24GB VRAM)
- Can handle largest models + batch processing

**Key Spec**: VRAM (GPU memory)
- Minimum 8GB for this project
- 12GB recommended
- 16GB+ for large batches or future fine-tuning

---

## Documentation Status

- [ ] GPU model identified
- [ ] CUDA version confirmed
- [ ] PyTorch with CUDA installed
- [ ] Test script passes
- [ ] Whisper works on GPU
- [ ] Pyannote works on GPU
- [ ] Baseline speedup measured

**Last Updated**: 2025-12-23
