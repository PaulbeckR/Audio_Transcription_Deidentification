# Torch Utils Module - Centralized Compatibility Patches

**Created**: December 29, 2025
**Purpose**: Centralized PyTorch compatibility utilities for WhisperX and HuggingFace models

---

## Overview

The `src/torch_utils.py` module provides a centralized location for PyTorch compatibility patches, specifically addressing the `weights_only` loading issue in PyTorch 2.6+.

## Problem

**PyTorch 2.6+ Breaking Change**:
- Default behavior changed: `torch.load()` now defaults to `weights_only=True` for security
- WhisperX and Pyannote models require `weights_only=False` to load properly
- Error: `"Unsupported global: GLOBAL omegaconf.listconfig.ListConfig"`

## Solution

Created a reusable utility module that monkey-patches `torch.load()` to force `weights_only=False`.

### Module: `src/torch_utils.py`

```python
from src.torch_utils import patch_torch_load

# Call BEFORE importing whisperx or transformers
patch_torch_load()
```

**Key Features**:
- Safe: Only patches if not already patched
- Reversible: `restore_torch_load()` available if needed
- Transparent: Preserves all other torch.load behavior
- Well-documented: Clear comments on why and how

---

## Usage Examples

### In `evaluate_finetuned.py`

```python
import torch
from src.torch_utils import patch_torch_load
patch_torch_load()

# Now safe to import models
import whisperx
from transformers import pipeline
```

### In `test_whisperx.py`

```python
import torch
from src.torch_utils import patch_torch_load
patch_torch_load()

# Now safe to import WhisperX
import whisperx
```

### In `finetune_whisper_simple.py` (if needed)

```python
import torch
from src.torch_utils import patch_torch_load
patch_torch_load()

from transformers import WhisperForConditionalGeneration
```

---

## Benefits

1. **DRY Principle**: Single source of truth for the patch
2. **Maintainability**: Update in one place, applies everywhere
3. **Consistency**: Same behavior across all scripts
4. **Documentation**: Centralized explanation of why patch is needed
5. **Safety**: Explicit, documented approach vs scattered monkey-patches

---

## Migration

### Before (Duplicated Code)

Each script had its own implementation:

```python
# test_whisperx.py
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# evaluate_finetuned.py
# ... duplicate code ...

# Other scripts
# ... more duplicates ...
```

### After (Centralized)

All scripts use the same utility:

```python
from src.torch_utils import patch_torch_load
patch_torch_load()
```

---

## Files Updated

1. ✓ Created `src/torch_utils.py`
2. ✓ Updated `test_whisperx.py` to use shared module
3. ✓ Updated `evaluate_finetuned.py` to use shared module
4. Future: Update other scripts as needed

---

## Technical Notes

- **Order matters**: Must call `patch_torch_load()` BEFORE importing whisperx or transformers
- **Thread-safe**: Patch applies globally but checks for existing patch
- **No performance impact**: Just a wrapper function
- **Security**: Safe because we trust HuggingFace model sources

---

## Alternative Approaches Considered

1. **Environment variable**: `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`
   - Not reliable across all PyTorch versions

2. **Explicit parameter**: Pass `weights_only=False` at every call
   - Not feasible (library code makes torch.load calls internally)

3. **Downgrade PyTorch**: Use older version
   - Loses security improvements and new features

4. **Wait for library updates**: WhisperX/Pyannote to support new PyTorch
   - Unknown timeline, blocks current work

**Chosen approach** (monkey-patch) is the most pragmatic solution.