# Audio Transcription Deidentification - Reorganization Plan

**Date**: January 7, 2026
**Status**: Ready for Implementation
**Goal**: Create clean, modular, config-driven codebase ready for deep analysis on real data

---

## Executive Summary

This reorganization transforms the codebase into a fully config-driven, modular system that eliminates hardcoded parameters and enables systematic experimentation with:
- ✅ **LoRA vs Full Fine-tuning** comparison
- ✅ **Hyperparameter sweeps** (learning rate, batch size, epochs, LoRA rank)
- ✅ **Dataset variations** (test sets, full interviews, train/val/test splits)
- ✅ **Model comparisons** (baseline vs fine-tuned, different configurations)

---

## Current State Assessment

### Strengths ✅
- Centralized YAML configuration system ([config.py](../src/config.py))
- Type-safe configuration with dataclasses
- Clear separation: config, utils, metrics, training
- GPU-optimized for 6GB VRAM (RTX 3050)
- Comprehensive WER calculation with normalization
- Proven LoRA fine-tuning pipeline (Dec 29, 2025 results: 25.01% WER)

### Issues Identified ⚠️

**1. Hardcoded Training Parameters**
- LoRA config in [finetune_whisper.py:152-159](../src/finetune_whisper.py#L152-L159)
  ```python
  lora_config = {
      "r": 32,  # HARDCODED
      "lora_alpha": 64,  # HARDCODED
      "target_modules": ["q_proj", "v_proj"],  # HARDCODED
  }
  ```
- Training hyperparameters in [finetune_whisper.py:303-306](../src/finetune_whisper.py#L303-L306)
  ```python
  BATCH_SIZE = 2  # HARDCODED
  NUM_EPOCHS = 5  # HARDCODED
  LEARNING_RATE = 1e-5  # HARDCODED
  ```

**2. No LoRA On/Off Switch**
- Cannot easily switch between LoRA and full fine-tuning
- No comparison mechanism for different training strategies

**3. Missing Configuration Files**
- No `training.yaml` for fine-tuning parameters
- No `experiments.yaml` for iteration and sweeps
- No dataset management configuration

**4. Script Organization**
- Test scripts in root (`run_baseline_test.py`, `analyze_word_errors.py`)
- Duplicate functionality (`finetune_whisper.py` vs `finetune_whisper_simple.py`)
- No pipeline orchestration

**5. Module Organization**
- All modules flat in `src/`
- No clear functional grouping

---

## Completed: New Configuration Files

### ✅ [config/training.yaml](../config/training.yaml)

**Key Features:**
- **Strategy Selection**: Choose between LoRA, full fine-tuning, or freeze-encoder
  ```yaml
  strategy:
    method: "lora"  # Options: "lora", "full", "freeze_encoder"
  ```

- **LoRA Configuration**: Complete control with presets
  ```yaml
  lora:
    rank: 32
    alpha: 64
    dropout: 0.05
    target_modules: ["q_proj", "v_proj"]

    presets:
      minimal: {rank: 8, alpha: 16}    # 4GB GPU
      small: {rank: 16, alpha: 32}     # 5GB GPU
      medium: {rank: 32, alpha: 64}    # 6GB GPU (RTX 3050)
      large: {rank: 64, alpha: 128}    # 8-10GB GPU
  ```

- **Training Hyperparameters**: All parameters configurable
  ```yaml
  hyperparameters:
    batch_size: 2
    gradient_accumulation_steps: 8
    learning_rate: 1.0e-5
    num_epochs: 5
    warmup_steps: 100
  ```

- **Dataset Management**
  ```yaml
  dataset:
    splits: {train: 0.8, validation: 0.1, test: 0.1}
    filters:
      min_duration: 1.0
      max_duration: 30.0
      max_wer: null
  ```

### ✅ [config/experiments.yaml](../config/experiments.yaml)

**Key Features:**
- **Pre-defined Experiments**: Ready-to-run configurations
  - `baseline_lora`: Matches your successful Dec 29 run (LoRA rank 32)
  - `full_finetune`: No LoRA, freeze encoder only
  - `full_model`: Complete model fine-tuning (requires 24GB+ GPU)
  - `quick_test`: Fast iteration with minimal LoRA
  - `lora_large`: High-capacity LoRA (rank 64)
  - `high_lr`: Learning rate exploration
  - `extended_training`: More epochs

- **Active Experiment Selection**
  ```yaml
  active_experiment: "baseline_lora"
  ```

- **Experiment Overrides**: Each experiment overrides training.yaml
  ```yaml
  experiments:
    baseline_lora:
      strategy: {method: "lora"}
      hyperparameters:
        batch_size: 2
        learning_rate: 1.0e-5
      lora:
        rank: 32
        alpha: 64
  ```

- **Hyperparameter Sweeps**: Automated grid/random search
  ```yaml
  sweep:
    enabled: false
    method: "grid"
    parameters:
      "strategy.method": ["lora", "freeze_encoder"]
      "lora.rank": [16, 32, 64]
      "hyperparameters.learning_rate": [5.0e-6, 1.0e-5, 5.0e-5]
  ```

- **Dataset Variations**: Support for multiple datasets
  ```yaml
  datasets:
    gpt_test: {segments: 79, words: 1045}  # Current
    full_interviews: {...}  # Future
    mixed: {...}  # Combined
  ```

---

## Proposed Modular Structure

### New Directory Organization

```
src/
├── __init__.py
├── config.py                    # Config loader (ENHANCED)
│
├── core/                        # Core utilities (NEW)
│   ├── __init__.py
│   ├── utils.py                 # General utilities
│   ├── torch_utils.py           # PyTorch compatibility
│   └── device.py                # Device detection
│
├── preprocessing/               # Audio & data preprocessing (NEW)
│   ├── __init__.py
│   ├── audio.py                 # Audio processing
│   ├── text.py                  # Text normalization
│   └── alignment.py             # Forced alignment wrapper
│
├── data/                        # Dataset management (NEW)
│   ├── __init__.py
│   ├── prepare_training.py      # Create aligned training data
│   ├── prepare_hf_dataset.py    # Convert to HF format
│   └── data_loader.py           # Custom data loaders
│
├── training/                    # Fine-tuning & training (NEW)
│   ├── __init__.py
│   ├── finetune.py              # Main fine-tuning (CONFIG-DRIVEN)
│   ├── trainer.py               # Trainer setup
│   ├── callbacks.py             # Training callbacks
│   └── lora.py                  # LoRA configuration
│
├── evaluation/                  # Metrics & evaluation (NEW)
│   ├── __init__.py
│   ├── metrics.py               # Metrics calculation
│   ├── evaluate.py              # Model evaluation
│   └── compare.py               # Model comparison
│
├── transcription/               # Transcription pipelines (NEW)
│   ├── __init__.py
│   ├── whisperx_pipeline.py     # WhisperX transcription
│   ├── whisper_pipeline.py      # Whisper transcription
│   └── build_transcript.py      # Transcript formatting
│
└── scripts/                     # High-level orchestration (NEW)
    ├── __init__.py
    ├── run_baseline.py          # Baseline test
    ├── run_training.py          # Training orchestration (NEW)
    ├── run_evaluation.py        # Evaluation orchestration (NEW)
    └── run_experiment.py        # Experiment runner (NEW)
```

---

## Implementation Phases

### Phase 1: Configuration Enhancement ⚡ **CRITICAL**
**Priority**: HIGHEST
**Effort**: 2-3 hours

- [x] Create `config/training.yaml` ✅ **DONE**
- [x] Create `config/experiments.yaml` ✅ **DONE**
- [ ] Update [config.py](../src/config.py) to load new configs
  - Add `TrainingConfig`, `LoRAConfig`, `ExperimentConfig` dataclasses
  - Add parsing methods
  - Add experiment loader with override logic
- [ ] Add configuration validation
- [ ] Test config loading

**Deliverable**: Fully config-driven system with no hardcoded parameters

---

### Phase 2: Training Module Refactor ⚡ **CRITICAL**
**Priority**: HIGHEST
**Effort**: 3-4 hours

- [ ] Create `src/training/` directory
- [ ] Extract LoRA logic → `src/training/lora.py`
  ```python
  def get_lora_config(config, preset=None):
      """Get LoRA config from training.yaml or preset"""
      if preset:
          return config.training.lora.presets[preset]
      return config.training.lora

  def apply_lora(model, lora_config):
      """Apply LoRA based on config"""
  ```

- [ ] Extract trainer setup → `src/training/trainer.py`
  ```python
  def setup_trainer(model, processor, dataset, config, experiment=None):
      """Create Seq2SeqTrainer from config"""
  ```

- [ ] Extract callbacks → `src/training/callbacks.py`
  - Move `MemoryMonitorCallback`
  - Add `WERTrackingCallback`

- [ ] Refactor [finetune_whisper.py](../src/finetune_whisper.py) → `src/training/finetune.py`
  - Replace ALL hardcoded params with `config.training.*`
  - Add strategy selector (LoRA vs full vs freeze-encoder)
  - Support experiment loading
  - Remove duplicate `finetune_whisper_simple.py`

- [ ] Create `src/scripts/run_training.py` (main entry point)
  ```python
  # Load config
  config = Config()
  experiment = config.get_experiment("baseline_lora")

  # Apply experiment overrides
  training_config = config.training.merge(experiment)

  # Run training with selected strategy
  if training_config.strategy.method == "lora":
      finetune_with_lora(config, training_config)
  elif training_config.strategy.method == "full":
      finetune_full_model(config, training_config)
  ```

**Deliverable**: Config-driven training with LoRA on/off switch

---

### Phase 3: Dataset Management ⚡ **HIGH PRIORITY**
**Priority**: HIGH
**Effort**: 2-3 hours

- [ ] Create `src/data/` directory
- [ ] Refactor [create_training_data.py](../src/create_training_data.py) → `src/data/prepare_training.py`
  - Use `config.dataset.*` for paths and filters
  - Support multiple dataset sources
- [ ] Consolidate [prepare_hf_dataset.py](../src/prepare_hf_dataset.py) and `prepare_hf_dataset_simple.py`
  - Single unified version
  - Config-driven splitting
- [ ] Create `src/data/data_loader.py`
  - Custom dataset loaders
  - Dataset validation utilities

**Deliverable**: Clean dataset preparation pipeline

---

### Phase 4: Evaluation & Comparison
**Priority**: MEDIUM
**Effort**: 2-3 hours

- [ ] Create `src/evaluation/` directory
- [ ] Move [metrics.py](../src/metrics.py) → `src/evaluation/metrics.py`
- [ ] Refactor [evaluate_finetuned.py](../src/evaluate_finetuned.py) → `src/evaluation/evaluate.py`
  - Config-driven evaluation
  - Support multiple models
- [ ] Create `src/evaluation/compare.py`
  - Compare LoRA vs full fine-tuning
  - Compare different LoRA ranks
  - Compare baseline vs fine-tuned
- [ ] Create `src/scripts/run_evaluation.py`
  - Run comparison groups from `experiments.yaml`

**Deliverable**: Automated model comparison

---

### Phase 5: Experiment Orchestration
**Priority**: MEDIUM
**Effort**: 2-3 hours

- [ ] Create `src/scripts/run_experiment.py`
  - Load experiment by name
  - Apply config overrides
  - Run training
  - Run evaluation
  - Save results
- [ ] Add sweep support
  - Grid search over parameters
  - Random search
  - Results tracking
- [ ] Add experiment tracking
  - Log all experiment configs
  - Track WER across experiments
  - Generate comparison reports

**Deliverable**: One-command experiment execution

---

### Phase 6: Code Cleanup
**Priority**: LOW
**Effort**: 2-3 hours

- [ ] Create `src/core/` utilities
- [ ] Create `src/preprocessing/` modules
- [ ] Create `src/transcription/` pipelines
- [ ] Move root scripts to `src/scripts/`
- [ ] Update all imports
- [ ] Add `__init__.py` files

**Deliverable**: Clean, modular codebase

---

## Next Steps After Reorganization

### Immediate Testing

1. **Validate Configuration**
   ```bash
   python -c "from src.config import Config; c = Config(); print(c.training.strategy.method)"
   ```

2. **Run Quick Test Experiment**
   ```bash
   python src/scripts/run_training.py --experiment quick_test
   ```

3. **Compare LoRA vs Full**
   ```bash
   python src/scripts/run_training.py --experiment baseline_lora
   python src/scripts/run_training.py --experiment full_finetune
   python src/scripts/run_evaluation.py --compare baseline full_finetune
   ```

### Systematic Experimentation

4. **LoRA Rank Comparison**
   - Run: minimal (r=8), small (r=16), medium (r=32), large (r=64)
   - Compare: WER, training time, GPU memory

5. **Learning Rate Sweep**
   - Test: [5e-6, 1e-5, 5e-5, 1e-4]
   - Find optimal LR for your data

6. **Strategy Comparison**
   - LoRA vs freeze-encoder vs full (if GPU allows)
   - Quality vs speed vs memory tradeoffs

### Real Data Analysis

7. **Full Interview Dataset**
   - Prepare complete interview transcripts
   - Create train/val/test splits
   - Run baseline evaluation
   - Fine-tune and compare

8. **Error Analysis**
   - WER breakdown by error type
   - Speaker-specific performance
   - Temporal patterns
   - Common error words

9. **Production Optimization**
   - Best LoRA configuration
   - Optimal hyperparameters
   - Model selection (large-v2 vs large-v3)

---

## Key Benefits

### ✅ Zero Hardcoding
- All parameters in config files
- Easy to modify and track
- Version control friendly

### ✅ LoRA Optional
- Switch between LoRA and full fine-tuning
- Compare different strategies
- Choose based on GPU memory

### ✅ Easy Iteration
- Change `active_experiment` in YAML
- No code changes needed
- Rapid experimentation

### ✅ Reproducibility
- Config files = experiment definition
- Git tracks all parameters
- Easy to replicate results

### ✅ Modular & Maintainable
- Clear separation of concerns
- Easy to find and update code
- Scalable architecture

### ✅ Ready for Deep Analysis
- Systematic testing framework
- Automated comparisons
- Production-ready pipelines

---

## Migration Effort

| Phase | Priority | Effort | Dependencies |
|-------|----------|--------|--------------|
| 1. Configuration | CRITICAL | 2-3h | None |
| 2. Training | CRITICAL | 3-4h | Phase 1 |
| 3. Dataset | HIGH | 2-3h | Phase 1 |
| 4. Evaluation | MEDIUM | 2-3h | Phase 2 |
| 5. Orchestration | MEDIUM | 2-3h | Phase 2, 4 |
| 6. Cleanup | LOW | 2-3h | All |

**Total**: ~14-20 hours
**Critical Path**: Phases 1-3 (~7-10 hours) gets you to iteration-ready state

---

## Recommended Execution

### Fast Track (Phases 1-3 only)
**Goal**: Config-driven, iteration-ready ASAP
**Time**: ~7-10 hours
**Result**: Can test LoRA vs full, different hyperparameters, dataset variations

### Complete (All Phases)
**Goal**: Production-ready, fully modular codebase
**Time**: ~14-20 hours
**Result**: Clean architecture, automated experimentation, deep analysis ready

---

## Example Usage After Reorganization

### Running Different Experiments

```bash
# Quick test (1 epoch, minimal LoRA)
python src/scripts/run_training.py --experiment quick_test

# Baseline (matches Dec 29 results)
python src/scripts/run_training.py --experiment baseline_lora

# Full fine-tuning (no LoRA)
python src/scripts/run_training.py --experiment full_finetune

# High capacity LoRA
python src/scripts/run_training.py --experiment lora_large

# Learning rate exploration
python src/scripts/run_training.py --experiment high_lr
```

### Hyperparameter Sweep

```bash
# Run grid search
python src/scripts/run_experiment.py --sweep grid \
  --params "lora.rank=[16,32,64]" \
  --params "hyperparameters.learning_rate=[1e-5,5e-5]"
```

### Model Comparison

```bash
# Compare baseline vs full vs lora_large
python src/scripts/run_evaluation.py \
  --compare baseline_lora full_finetune lora_large \
  --output comparison_report.md
```

### Custom Experiment

Edit `config/experiments.yaml`:
```yaml
my_custom_experiment:
  name: "custom-test"
  strategy: {method: "lora"}
  hyperparameters:
    num_epochs: 10
    learning_rate: 2.0e-5
  lora:
    rank: 48
    alpha: 96
```

Then run:
```bash
python src/scripts/run_training.py --experiment my_custom_experiment
```

---

## Configuration Examples

### Switching LoRA On/Off

**LoRA Fine-tuning (6GB GPU)**
```yaml
# config/training.yaml or experiments.yaml override
strategy:
  method: "lora"

lora:
  rank: 32
  alpha: 64
```

**Full Decoder Fine-tuning (8-12GB GPU)**
```yaml
strategy:
  method: "freeze_encoder"  # No LoRA, freeze encoder only

freezing:
  freeze_encoder: true
  freeze_decoder: false
```

**Complete Model Fine-tuning (24GB+ GPU)**
```yaml
strategy:
  method: "full"  # Train everything

freezing:
  freeze_encoder: false
  freeze_decoder: false
```

### Dataset Switching

```yaml
# Use test dataset
active_dataset: "gpt_test"

# Switch to full interviews when ready
active_dataset: "full_interviews"

# Use mixed dataset
active_dataset: "mixed"
```

---

## Files Modified/Created

### Created ✅
- [config/training.yaml](../config/training.yaml) ✅
- [config/experiments.yaml](../config/experiments.yaml) ✅
- [Notes/REORGANIZATION_PLAN.md](../Notes/REORGANIZATION_PLAN.md) ✅ (this file)

### To Modify
- [src/config.py](../src/config.py) - Add training/experiment configs
- [src/finetune_whisper.py](../src/finetune_whisper.py) - Refactor to use configs

### To Create
- `src/training/finetune.py` - Config-driven training
- `src/training/lora.py` - LoRA utilities
- `src/training/trainer.py` - Trainer setup
- `src/scripts/run_training.py` - Main training script
- `src/scripts/run_experiment.py` - Experiment orchestration

### To Move/Consolidate
- `finetune_whisper_simple.py` → Delete (merge into main)
- `prepare_hf_dataset_simple.py` → Delete (merge into main)
- Root scripts → `src/scripts/`

---

## Questions & Decisions

### Q: Should we implement all phases or just critical path?
**Recommendation**: Start with Phases 1-3 (critical path) to get iteration-ready ASAP. Add Phases 4-6 as needed.

### Q: Keep both LoRA and full fine-tuning code paths?
**Recommendation**: Yes, use strategy selector to choose at runtime based on config.

### Q: How to handle dataset variations?
**Recommendation**: Use `active_dataset` in `experiments.yaml`, support multiple dataset configs.

### Q: Backward compatibility with existing scripts?
**Recommendation**: Keep old scripts functional during migration, deprecate gradually.

---

**Status**: Configuration files created, ready for Phase 1 implementation
**Next Action**: Update [src/config.py](../src/config.py) to load training.yaml and experiments.yaml
**Owner**: Awaiting approval to proceed with implementation
