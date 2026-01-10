# Quick Start Guide - Config-Driven Experimentation

**Created**: January 7, 2026
**Purpose**: Fast reference for running experiments after reorganization

---

## Current Status

### ✅ Completed
- **Configuration Files Created**:
  - [config/training.yaml](../config/training.yaml) - All training hyperparameters, LoRA settings
  - [config/experiments.yaml](../config/experiments.yaml) - Pre-defined experiment configurations

### ⏳ Pending (Phase 1)
- Update [src/config.py](../src/config.py) to load new configs
- Refactor training scripts to use configs
- Create orchestration scripts

---

## Configuration Files Overview

### [config/training.yaml](../config/training.yaml)

**Controls ALL training parameters:**

```yaml
# Choose training method
strategy:
  method: "lora"  # Options: "lora", "full", "freeze_encoder"

# LoRA settings (when method = "lora")
lora:
  rank: 32
  alpha: 64
  dropout: 0.05
  target_modules: ["q_proj", "v_proj"]

# Training hyperparameters
hyperparameters:
  batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-5
  num_epochs: 5
```

**LoRA Presets** (for quick switching):
- `minimal`: rank=8, alpha=16 (4GB GPU, fast)
- `small`: rank=16, alpha=32 (5GB GPU)
- `medium`: rank=32, alpha=64 (6GB GPU - RTX 3050) ⭐ **RECOMMENDED**
- `large`: rank=64, alpha=128 (8-10GB GPU, best quality)

---

### [config/experiments.yaml](../config/experiments.yaml)

**Pre-defined experiments** (override training.yaml):

| Experiment | Strategy | LoRA Rank | LR | Epochs | Use Case |
|------------|----------|-----------|-----|--------|----------|
| `baseline_lora` | LoRA | 32 | 1e-5 | 5 | Matches Dec 29 success ⭐ |
| `full_finetune` | Freeze encoder | N/A | 5e-6 | 3 | No LoRA comparison |
| `quick_test` | LoRA | 8 | 1e-5 | 1 | Fast iteration |
| `lora_large` | LoRA | 64 | 1e-5 | 5 | High capacity |
| `high_lr` | LoRA | 32 | 5e-5 | 5 | LR exploration |
| `extended_training` | LoRA | 32 | 1e-5 | 10 | More epochs |

**Active experiment** (which to run by default):
```yaml
active_experiment: "baseline_lora"
```

---

## How to Run Experiments

### After Phase 1 Implementation

**Option 1: Use active experiment**
```bash
python src/scripts/run_training.py
# Runs whatever is set as active_experiment
```

**Option 2: Specify experiment**
```bash
python src/scripts/run_training.py --experiment quick_test
python src/scripts/run_training.py --experiment baseline_lora
python src/scripts/run_training.py --experiment full_finetune
```

**Option 3: Override parameters**
```bash
python src/scripts/run_training.py \
  --experiment baseline_lora \
  --override "hyperparameters.learning_rate=5e-5" \
  --override "hyperparameters.num_epochs=10"
```

---

## Common Experimentation Workflows

### 1. Test LoRA vs Full Fine-tuning

**Goal**: Compare parameter-efficient LoRA vs full decoder fine-tuning

```bash
# Run LoRA version (6GB GPU)
python src/scripts/run_training.py --experiment baseline_lora

# Run full decoder version (needs more VRAM)
python src/scripts/run_training.py --experiment full_finetune

# Compare results
python src/scripts/run_evaluation.py --compare baseline_lora full_finetune
```

**Expected outcome**: LoRA should be faster, use less memory, comparable WER

---

### 2. Find Optimal LoRA Rank

**Goal**: Balance quality vs speed/memory

```bash
# Test different ranks
python src/scripts/run_training.py --experiment quick_test    # rank=8
python src/scripts/run_training.py --experiment baseline_lora # rank=32
python src/scripts/run_training.py --experiment lora_large    # rank=64

# Compare
python src/scripts/run_evaluation.py --compare quick_test baseline_lora lora_large
```

**Expected outcome**: Higher rank = better WER but slower/more memory

---

### 3. Learning Rate Sweep

**Goal**: Find optimal learning rate

**Manual approach** (edit experiments.yaml):
```yaml
lr_test_1:
  strategy: {method: "lora"}
  hyperparameters: {learning_rate: 5.0e-6, num_epochs: 5}

lr_test_2:
  strategy: {method: "lora"}
  hyperparameters: {learning_rate: 1.0e-5, num_epochs: 5}

lr_test_3:
  strategy: {method: "lora"}
  hyperparameters: {learning_rate: 5.0e-5, num_epochs: 5}
```

Then run:
```bash
python src/scripts/run_training.py --experiment lr_test_1
python src/scripts/run_training.py --experiment lr_test_2
python src/scripts/run_training.py --experiment lr_test_3
```

**Automated approach** (after Phase 5):
```bash
python src/scripts/run_experiment.py --sweep grid \
  --param "hyperparameters.learning_rate=[5e-6,1e-5,5e-5,1e-4]"
```

---

### 4. Epoch Exploration

**Goal**: Avoid under/overfitting

```bash
# Short training
python src/scripts/run_training.py --experiment baseline_lora \
  --override "hyperparameters.num_epochs=3"

# Standard
python src/scripts/run_training.py --experiment baseline_lora  # 5 epochs

# Extended
python src/scripts/run_training.py --experiment extended_training  # 10 epochs
```

**Watch for**: Validation WER should improve then plateau (not increase = overfitting)

---

### 5. Test on New Dataset

**Goal**: Evaluate on full interview transcripts

**Step 1**: Prepare dataset
```bash
python src/data/prepare_training.py \
  --audio-dir training_data/full_interviews/audio \
  --transcript-dir training_data/full_interviews/transcripts \
  --output prepared_datasets/whisper_finetuning_full
```

**Step 2**: Update experiments.yaml
```yaml
active_dataset: "full_interviews"

datasets:
  full_interviews:
    input_dir: "prepared_datasets/whisper_finetuning_full"
    enabled: true
```

**Step 3**: Run training
```bash
python src/scripts/run_training.py --experiment baseline_lora
```

---

## Configuration Quick Reference

### Switching Training Method

**Use LoRA** (parameter-efficient, 6GB GPU):
```yaml
strategy:
  method: "lora"
lora:
  rank: 32
  alpha: 64
```

**Use Full Decoder** (no LoRA, 8-12GB GPU):
```yaml
strategy:
  method: "freeze_encoder"
freezing:
  freeze_encoder: true
  freeze_decoder: false
```

**Use Full Model** (everything, 24GB+ GPU):
```yaml
strategy:
  method: "full"
freezing:
  freeze_encoder: false
  freeze_decoder: false
```

---

### Adjusting for GPU Memory

**6GB GPU (RTX 3050)** - Current setup:
```yaml
hyperparameters:
  batch_size: 2
  gradient_accumulation_steps: 8  # Effective batch = 16
lora:
  rank: 32
```

**8GB GPU**:
```yaml
hyperparameters:
  batch_size: 4
  gradient_accumulation_steps: 4
lora:
  rank: 64
```

**4GB GPU** (minimal):
```yaml
hyperparameters:
  batch_size: 1
  gradient_accumulation_steps: 16
lora:
  rank: 8
```

**12GB+ GPU** (high performance):
```yaml
hyperparameters:
  batch_size: 8
  gradient_accumulation_steps: 2
lora:
  rank: 64
```

---

### Creating Custom Experiments

**Edit [config/experiments.yaml](../config/experiments.yaml):**

```yaml
my_custom_test:
  name: "my-custom-experiment"
  description: "Testing custom settings"
  enabled: true

  strategy:
    method: "lora"

  hyperparameters:
    batch_size: 2
    learning_rate: 2.0e-5  # Custom LR
    num_epochs: 7  # Custom epochs

  lora:
    rank: 48  # Custom rank
    alpha: 96
    target_modules: ["q_proj", "v_proj", "k_proj"]  # More modules

  dataset:
    output_dir: "models/whisper-finetuned-custom"
```

Then run:
```bash
python src/scripts/run_training.py --experiment my_custom_test
```

---

## Monitoring & Evaluation

### During Training

**TensorBoard** (after Phase 2):
```bash
tensorboard --logdir models/whisper-finetuned/logs
# Open http://localhost:6006
```

**Watch for**:
- Training loss decreasing
- Validation WER decreasing (should plateau, not increase)
- GPU memory usage (logged every 10 steps)

### After Training

**Evaluate on test set**:
```bash
python src/scripts/run_evaluation.py --model models/whisper-finetuned/final_model
```

**Compare to baseline**:
```bash
python src/scripts/run_evaluation.py \
  --baseline whisperx-large-v2 \
  --finetuned models/whisper-finetuned/final_model \
  --output comparison_report.md
```

---

## Dataset Management

### Current Dataset (GPT Test)
- **Location**: `prepared_datasets/whisper_finetuning`
- **Stats**: 79 segments, 1045 words, 383.5s audio
- **Config**: Set in `experiments.yaml` as `active_dataset: "gpt_test"`

### Adding New Datasets

**Step 1**: Prepare aligned training data
```bash
python src/data/prepare_training.py \
  --config-dataset "my_new_dataset" \
  --audio-dir path/to/audio \
  --transcript-dir path/to/transcripts
```

**Step 2**: Convert to HuggingFace format
```bash
python src/data/prepare_hf_dataset.py \
  --input training_data/aligned/my_new_dataset \
  --output datasets/my_new_dataset \
  --train-split 0.8 \
  --val-split 0.1
```

**Step 3**: Add to experiments.yaml
```yaml
datasets:
  my_new_dataset:
    name: "my_new_dataset"
    input_dir: "datasets/my_new_dataset"
    enabled: true

active_dataset: "my_new_dataset"
```

---

## Hyperparameter Sweep (Phase 5)

### Grid Search

**Edit [config/experiments.yaml](../config/experiments.yaml)**:
```yaml
sweep:
  enabled: true
  method: "grid"

  parameters:
    "lora.rank": [16, 32, 64]
    "hyperparameters.learning_rate": [1e-5, 5e-5]
    "hyperparameters.num_epochs": [3, 5, 10]
```

**Run**:
```bash
python src/scripts/run_experiment.py --sweep
# Runs 3 × 2 × 3 = 18 experiments automatically
```

### Random Search

```yaml
sweep:
  enabled: true
  method: "random"
  max_runs: 10  # Only run 10 random combinations

  parameters:
    "lora.rank": [8, 16, 32, 64]
    "hyperparameters.learning_rate": [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
```

---

## Troubleshooting

### Out of Memory

**Solution**: Reduce batch size or LoRA rank
```yaml
hyperparameters:
  batch_size: 1  # Reduce from 2
  gradient_accumulation_steps: 16  # Double to maintain effective batch

lora:
  rank: 16  # Reduce from 32
```

### Training Too Slow

**Solution**: Use minimal LoRA or reduce epochs
```yaml
# Use quick_test experiment
active_experiment: "quick_test"
```

### WER Not Improving

**Solutions**:
1. Lower learning rate
2. More epochs
3. Higher LoRA rank
4. Check dataset quality

---

## File Locations Reference

| File | Purpose |
|------|---------|
| [config/training.yaml](../config/training.yaml) | All training parameters |
| [config/experiments.yaml](../config/experiments.yaml) | Experiment definitions |
| [config/paths.yaml](../config/paths.yaml) | Directory paths |
| [config/models.yaml](../config/models.yaml) | Model selection |
| [config/processing.yaml](../config/processing.yaml) | Audio/text processing |
| [src/config.py](../src/config.py) | Config loader |
| `src/scripts/run_training.py` | Main training script (to create) |
| `src/scripts/run_evaluation.py` | Evaluation script (to create) |
| `src/scripts/run_experiment.py` | Sweep orchestrator (to create) |

---

## Next Steps

### For You (Now)
1. Review [REORGANIZATION_PLAN.md](../Notes/REORGANIZATION_PLAN.md)
2. Approve approach
3. Decide: Fast track (Phases 1-3) or complete reorganization?

### For Implementation (After Approval)
1. **Phase 1**: Update [src/config.py](../src/config.py) to load training.yaml & experiments.yaml
2. **Phase 2**: Refactor [src/finetune_whisper.py](../src/finetune_whisper.py) to use configs
3. **Phase 3**: Create orchestration scripts
4. **Test**: Run quick_test experiment to validate
5. **Experiment**: Test LoRA vs full, different ranks, learning rates

---

**Status**: Configuration ready, implementation pending approval
**Contact**: See [REORGANIZATION_PLAN.md](../Notes/REORGANIZATION_PLAN.md) for detailed plan
