# Codebase Issues and Inconsistencies Report

**Date**: January 7, 2026
**Analysis**: Senior programmer code review
**Status**: Issues identified, no changes made yet

---

## Critical Issue #1: Missing Dependencies ❌

### Problem: HuggingFace `datasets` Package Not Installed

**Error**:
```
ImportError: cannot import name 'load_from_disk' from 'datasets' (unknown location)
```

**Root Cause**:
1. The HuggingFace `datasets` package is **not installed** in the virtual environment
2. There is a **local `prepared_datasets/` directory** that shadows the package import
3. Python finds the local `prepared_datasets/` directory first and treats it as a namespace package

**Affected Files**:
- `src/training/finetune.py` (line ~44): `from datasets import load_from_disk, DatasetDict`
- `src/data/prepare_hf_dataset.py` (likely uses datasets API)

**Evidence**:
```bash
$ pip show datasets
WARNING: Package(s) not found: datasets

$ ls prepared_datasets/
whisper_finetuning/  whisper_finetuning_simple/

$ python -c "import datasets; print(datasets.__file__)"
None  # <-- Namespace package, not the real HuggingFace datasets
```

**Solution Required**:
```bash
pip install datasets  # HuggingFace datasets library
```

**Additional Missing Packages** (Required for fine-tuning):
```bash
pip install transformers  # For WhisperForConditionalGeneration
pip install peft          # For LoRA (get_peft_model, LoraConfig)
pip install evaluate      # For WER metric computation
pip install accelerate    # For training optimization
```

---

## Critical Issue #2: Import Path Inconsistencies ⚠️

### Problem: Scripts Reference Old Module Locations

After reorganization, several scripts still import from old paths that no longer exist.

### Affected Files:

#### `src/scripts/run_baseline_test.py`
**Current (BROKEN)**:
```python
from src.metrics import (
    calculate_wer,
    PerformanceTimer,
    SystemMetrics,
    generate_comparison_report
)
from src.utils import get_device, get_audio_duration, format_time
```

**Should Be**:
```python
from src.evaluation.metrics import (
    calculate_wer,
    PerformanceTimer,
    SystemMetrics,
    generate_comparison_report
)
from src.core.utils import get_device, get_audio_duration, format_time
```

#### `src/scripts/analyze_word_errors.py`
**Current (BROKEN)**:
```python
from src.metrics import calculate_wer
```

**Should Be**:
```python
from src.evaluation.metrics import calculate_wer
```

#### `src/scripts/test_gpu.py`
**Current (BROKEN)**:
```python
from src.utils import get_device, get_gpu_info
```

**Should Be**:
```python
from src.core.utils import get_device, get_gpu_info
```

**Impact**: These scripts will fail with `ModuleNotFoundError` when run.

---

## Issue #3: Naming Inconsistency - "experiments" vs "experiment"

### Problem: Inconsistent Config File Naming

**Current State**:
- Configuration expects: `experiment.yaml` (singular)
- File created: `experiments.yaml` (plural)
- Config loader tries to load: `experiment.yaml`

**In `src/config.py` line ~316**:
```python
self._experiment_data = self._load_yaml("experiment.yaml")  # SINGULAR
```

**But we created**:
```
config/experiments.yaml  # PLURAL
```

**Evidence**:
Check the config directory:
```bash
$ ls config/*.yaml
config/experiments.yaml    # <-- PLURAL (new file)
config/models.yaml
config/paths.yaml
config/processing.yaml
config/training.yaml
```

**Impact**:
- If there's no `config/experiment.yaml` (singular), the config loader will fail
- The new `experiments.yaml` file won't be loaded
- Experiments configuration won't work

**Clarification Needed**:
1. Is there an old `experiment.yaml` file that we need to keep?
2. Should we rename `experiments.yaml` → `experiment.yaml`?
3. Or update the config loader to use `experiments.yaml`?

---

## Issue #4: Module Organization - Some Scripts Not Updated

### Files That Reference Old Locations

**Files in `old_code/` directory**:
These files were archived but may still be referenced or used:
- `evaluate_finetuned.py` (in old_code) - Was this intentionally replaced?
- `finetune_whisper.py` (in old_code) - Replaced by `src/training/finetune.py`
- `finetune_whisper_simple.py` (in old_code) - Consolidated

**Question**: Are these files safe to ignore, or do they need updating?

---

## Issue #5: Missing Requirements File for Fine-Tuning

### Problem: No Requirements File for Training Dependencies

**Current `requirements.txt`** includes:
- ✅ PyTorch, torchaudio
- ✅ whisper (OpenAI)
- ✅ pyannote
- ✅ Basic ML packages

**Missing for Fine-Tuning**:
- ❌ `datasets` (HuggingFace)
- ❌ `transformers` (HuggingFace)
- ❌ `peft` (Parameter-Efficient Fine-Tuning / LoRA)
- ❌ `evaluate` (HuggingFace metrics)
- ❌ `accelerate` (Training acceleration)

**Should Create**: `requirements_finetuning.txt` or add to main requirements

---

## Issue #6: Directory Naming Collision

### Problem: Local `prepared_datasets/` Directory Shadows Python Package

**Conflict**:
```
prepared_datasets/                    # LOCAL DIRECTORY (data storage)
    ├── whisper_finetuning/
    └── ...

vs.

from datasets import ...     # PYTHON PACKAGE (HuggingFace)
```

**Python's Import Resolution**:
1. Current directory
2. PYTHONPATH
3. Site-packages

Since there's a local `prepared_datasets/` directory, Python treats it as a package and doesn't look for the HuggingFace `datasets` package.

**Solution Options**:
1. **Rename local directory** (RECOMMENDED):
   ```bash
   mv prepared_datasets/ data_storage/  # or prepared_prepared_datasets/ or hf_prepared_datasets/
   ```
   Then update all references in code and config.

2. **Install datasets package** (PARTIAL FIX):
   ```bash
   pip install datasets
   ```
   But the shadowing will still cause issues in some contexts.

**Best Practice**: Local directories should never have the same name as Python packages you import.

---

## Issue #7: Config Path References

### Files That May Need Path Updates

If we rename `prepared_datasets/` → `prepared_prepared_datasets/`:

**Files to Check**:
- `config/paths.yaml` - May reference "datasets" directory
- `config/training.yaml` - References `prepared_datasets/whisper_finetuning`
- `config/experiments.yaml` - References dataset directories
- Any scripts that hardcode dataset paths

---

## Issue #8: Inconsistent Documentation

### Documentation References Old Structure

**Files That May Need Updates**:
- `DATA_OVERVIEW.md` - References `prepared_datasets/` directory
- `PREPARE_YOUR_DATA.md` - Has paths to `prepared_datasets/whisper_finetuning_interviews`
- `TRAINING_GUIDE.md` - References dataset paths
- `WORKSPACE_READY.md` - Shows old structure

If we rename `prepared_datasets/`, all these docs need updates.

---

## Summary of Issues

### Critical (Must Fix Before Training)
1. ❌ **Missing Python packages**: datasets, transformers, peft, evaluate, accelerate
2. ❌ **Import path inconsistencies**: 4 scripts importing from old locations
3. ❌ **Naming collision**: `prepared_datasets/` directory shadows HuggingFace package

### High Priority (Should Fix Soon)
4. ⚠️ **Config file naming**: `experiment.yaml` vs `experiments.yaml` mismatch
5. ⚠️ **Missing requirements**: No requirements_finetuning.txt

### Medium Priority (Polish)
6. ℹ️ **Documentation updates**: If directories renamed
7. ℹ️ **Old code cleanup**: Verify archived files aren't needed

---

## Recommended Fix Order

### Step 1: Fix Package Dependencies
```bash
pip install datasets transformers peft evaluate accelerate
```

### Step 2: Fix Import Paths
Update these files:
- `src/scripts/run_baseline_test.py`
- `src/scripts/analyze_word_errors.py`
- `src/scripts/test_gpu.py`

Change:
```python
from src.metrics import ...  →  from src.evaluation.metrics import ...
from src.utils import ...    →  from src.core.utils import ...
```

### Step 3: Resolve Naming Issues
**Option A**: Rename local directory (RECOMMENDED)
```bash
mv datasets prepared_datasets
# Update all config files and docs
```

**Option B**: Just install packages and hope for best
```bash
pip install datasets  # May still have shadowing issues
```

### Step 4: Fix Config File Name
**Either**:
- Rename `config/experiments.yaml` → `config/experiment.yaml`
- OR update `src/config.py` to load `experiments.yaml`

### Step 5: Create Requirements File
```bash
# Create requirements_finetuning.txt with:
datasets>=2.14.0
transformers>=4.35.0
peft>=0.7.0
evaluate>=0.4.0
accelerate>=0.24.0
```

---

## Files Requiring Updates

### Python Files (Import Paths)
- [ ] `src/scripts/run_baseline_test.py`
- [ ] `src/scripts/analyze_word_errors.py`
- [ ] `src/scripts/test_gpu.py`
- [ ] `src/config.py` (experiment.yaml vs experiments.yaml)

### Config Files (If Directory Renamed)
- [ ] `config/paths.yaml`
- [ ] `config/training.yaml`
- [ ] `config/experiments.yaml`

### Documentation (If Directory Renamed)
- [ ] `DATA_OVERVIEW.md`
- [ ] `PREPARE_YOUR_DATA.md`
- [ ] `TRAINING_GUIDE.md`
- [ ] `WORKSPACE_READY.md`
- [ ] `REORGANIZATION_COMPLETE.md`

### Requirements
- [ ] Create `requirements_finetuning.txt`
- [ ] Or update `requirements.txt` with training dependencies

---

## Questions for Resolution

1. **Directory Naming**: Should we rename `prepared_datasets/` to avoid shadowing?
   - Recommended: `prepared_prepared_datasets/` or `hf_prepared_datasets/` or `training_prepared_datasets/`

2. **Config File Name**: Keep `experiments.yaml` or rename to `experiment.yaml`?
   - What's the original file (if any)?

3. **Old Code**: Are files in `old_code/` still needed?
   - `evaluate_finetuned.py` - Was this replaced by new evaluation?

4. **Requirements**: Should training deps be in main requirements.txt or separate file?
   - Recommended: Separate `requirements_finetuning.txt`

---

## Testing After Fixes

```bash
# Test imports
python -c "from src.core import get_device; print('OK')"
python -c "from src.evaluation import calculate_wer; print('OK')"
python -c "from datasets import load_from_disk; print('OK')"
python -c "from transformers import WhisperProcessor; print('OK')"
python -c "from peft import LoraConfig; print('OK')"

# Test config loading
python -c "from src.config import Config; c = Config(); print('OK')"

# Test scripts
python src/scripts/list_experiments.py
python src/scripts/test_gpu.py
```

---

**Status**: Issues identified and documented. Awaiting decisions on:
1. Directory renaming strategy
2. Config file naming preference
3. Requirements file approach

**Next Action**: Make decisions on the questions above, then implement fixes.
