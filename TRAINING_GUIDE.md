# Whisper Fine-Tuning Guide - Config-Driven Iteration

**Updated**: January 7, 2026
**Status**: Ready for training on interview transcripts

---

## Quick Start

### 1. Prepare Your Interview Dataset

```bash
# Convert your interview transcripts to HuggingFace format
python src/data/prepare_hf_dataset.py \
  --input training_data/aligned \
  --output prepared_datasets/whisper_finetuning \
  --train-split 0.8 \
  --val-split 0.1 \
  --test-split 0.1
```

### 2. List Available Experiments

```bash
python src/scripts/list_experiments.py
```

### 3. Run Training

```bash
# Run active experiment (baseline_lora by default)
python src/scripts/run_training.py

# Or run a specific experiment
python src/scripts/run_training.py --experiment quick_test
python src/scripts/run_training.py --experiment baseline_lora
python src/scripts/run_training.py --experiment full_finetune
```

---

## Configuration System

All training parameters are controlled via YAML files - **NO hardcoded values**.

### Configuration Files

| File | Purpose |
|------|---------|
| [config/training.yaml](config/training.yaml) | Training hyperparameters, LoRA settings, optimization |
| [config/experiments.yaml](config/experiments.yaml) | Pre-defined experiments, sweeps, comparisons |
| [config/models.yaml](config/models.yaml) | Model selection (WhisperX, Whisper) |
| [config/paths.yaml](config/paths.yaml) | Directories and file paths |
| [config/processing.yaml](config/processing.yaml) | Audio/text processing settings |

---

## Available Experiments

### Pre-Configured Experiments

#### 1. **baseline_lora** ⭐ RECOMMENDED (Active)
- **Strategy**: LoRA parameter-efficient fine-tuning
- **LoRA Rank**: 32, Alpha: 64
- **Learning Rate**: 1e-5
- **Epochs**: 5
- **GPU**: 6GB (RTX 3050)
- **Use Case**: Matches your successful Dec 29 run (25.01% WER)

```bash
python src/scripts/run_training.py --experiment baseline_lora
```

#### 2. **full_finetune**
- **Strategy**: Full decoder fine-tuning (NO LoRA)
- **Learning Rate**: 5e-6 (lower for stability)
- **Epochs**: 3
- **GPU**: 8-12GB required
- **Use Case**: Compare LoRA vs full fine-tuning quality

```bash
python src/scripts/run_training.py --experiment full_finetune
```

#### 3. **quick_test**
- **Strategy**: LoRA with minimal settings
- **LoRA Rank**: 8 (ultra-low memory)
- **Epochs**: 1
- **GPU**: 4-5GB
- **Use Case**: Fast iteration during development

```bash
python src/scripts/run_training.py --experiment quick_test
```

#### 4. **lora_large**
- **Strategy**: High-capacity LoRA
- **LoRA Rank**: 64, Alpha: 128
- **Target Modules**: Extended (q_proj, v_proj, k_proj, out_proj)
- **GPU**: 8-10GB
- **Use Case**: Maximum quality, more VRAM available

```bash
python src/scripts/run_training.py --experiment lora_large
```

#### 5. **high_lr**
- **Strategy**: LoRA with higher learning rate
- **Learning Rate**: 5e-5 (5x baseline)
- **Use Case**: Learning rate exploration

```bash
python src/scripts/run_training.py --experiment high_lr
```

#### 6. **extended_training**
- **Strategy**: LoRA with more epochs
- **Epochs**: 10 (vs 5 baseline)
- **Use Case**: Check if more training helps

```bash
python src/scripts/run_training.py --experiment extended_training
```

---

## Training Workflow for Interview Data

### Step 1: Prepare Your Interview Transcripts

**Required Format**: Audio files + aligned transcripts (CSV or JSON)

```bash
# If you have raw audio + transcripts, create aligned data first
python src/create_training_data.py \
  --audio-dir path/to/interview_audio \
  --transcript-dir path/to/transcripts \
  --output training_data/aligned/interviews

# Convert to HuggingFace format
python src/prepare_hf_dataset.py \
  --input training_data/aligned/interviews \
  --output prepared_datasets/whisper_finetuning_interviews \
  --train-split 0.8 \
  --val-split 0.1 \
  --test-split 0.1
```

### Step 2: Update Dataset Config

Edit [config/experiments.yaml](config/experiments.yaml):

```yaml
# Add your dataset
datasets:
  interview_data:
    name: "full_interview_dataset"
    description: "Complete interview transcripts"
    input_dir: "prepared_datasets/whisper_finetuning_interviews"
    enabled: true

# Set as active
active_dataset: "interview_data"
```

### Step 3: Choose Experiment Strategy

**For first run, use baseline_lora** (proven configuration):

```bash
python src/scripts/run_training.py --experiment baseline_lora
```

**Monitor training**:
```bash
tensorboard --logdir models/whisper-finetuned-lora-r32/logs
```

### Step 4: Evaluate Results

```bash
# After training completes, evaluate on test set
python src/evaluate_finetuned.py \
  --model models/whisper-finetuned-lora-r32/final_model \
  --test-data prepared_datasets/whisper_finetuning_interviews/test
```

### Step 5: Iterate on Parameters

Based on results, try different configurations:

**If WER is good but want faster training**:
```bash
python src/scripts/run_training.py --experiment quick_test
```

**If want better quality (have more VRAM)**:
```bash
python src/scripts/run_training.py --experiment lora_large
```

**If want to compare vs full fine-tuning**:
```bash
python src/scripts/run_training.py --experiment full_finetune
```

---

## LoRA vs Full Fine-Tuning

### When to Use LoRA

✅ **Use LoRA if:**
- GPU has 6-8GB VRAM (RTX 3050, RTX 2060)
- Want fast iteration
- Parameter-efficient tuning is sufficient
- Proven baseline (your Dec 29 success with rank 32)

### When to Use Full Fine-Tuning

✅ **Use Full Fine-Tuning if:**
- GPU has 12GB+ VRAM
- Want absolute maximum quality
- LoRA results are insufficient
- Willing to train slower

### Switch Between Modes

**Edit [config/experiments.yaml](config/experiments.yaml)**:

```yaml
my_experiment:
  strategy:
    method: "lora"  # Options: "lora", "freeze_encoder", "full"
```

- `lora`: LoRA parameter-efficient (6GB GPU)
- `freeze_encoder`: Full decoder, frozen encoder (10-12GB GPU)
- `full`: Train everything (24GB+ GPU)

---

## Parameter Tuning Guide

### LoRA Rank (If Using LoRA)

**Trade-off**: Quality vs Speed/Memory

| Rank | Alpha | Memory | Speed | Quality | Use Case |
|------|-------|--------|-------|---------|----------|
| 8 | 16 | 4-5GB | Fastest | Good | Quick tests |
| 16 | 32 | 5-6GB | Fast | Better | Constrained GPU |
| 32 | 64 | 6GB | Balanced | Very Good | **Recommended** |
| 64 | 128 | 8-10GB | Slow | Best | Maximum quality |

**Change in [config/training.yaml](config/training.yaml)**:
```yaml
lora:
  rank: 32  # Change to 16, 32, 64
  alpha: 64  # Typically 2 * rank
```

### Learning Rate

**Trade-off**: Convergence speed vs stability

| Learning Rate | Behavior | Use Case |
|---------------|----------|----------|
| 1e-6 | Very safe, slow | Unstable training |
| 5e-6 | Safe, moderate | Full fine-tuning |
| 1e-5 | Balanced | **LoRA default** |
| 5e-5 | Fast, risky | Quick convergence |
| 1e-4 | Very fast, unstable | Experimentation |

**Change in [config/training.yaml](config/training.yaml)**:
```yaml
hyperparameters:
  learning_rate: 1.0e-5  # Change as needed
```

### Number of Epochs

**Trade-off**: Training time vs overfitting

| Epochs | Use Case |
|--------|----------|
| 1 | Quick test |
| 3 | Small dataset or full fine-tuning |
| 5 | **Balanced (recommended)** |
| 10 | Large dataset, check for overfitting |

**Watch validation WER** - should decrease then plateau (not increase)

**Change in [config/training.yaml](config/training.yaml)**:
```yaml
hyperparameters:
  num_epochs: 5  # Change as needed
```

### Batch Size & Gradient Accumulation

**For 6GB GPU (RTX 3050)**:
```yaml
hyperparameters:
  batch_size: 2
  gradient_accumulation_steps: 8
  # Effective batch size = 2 * 8 = 16
```

**For 8GB GPU**:
```yaml
hyperparameters:
  batch_size: 4
  gradient_accumulation_steps: 4
  # Effective batch size = 4 * 4 = 16
```

**For 4GB GPU**:
```yaml
hyperparameters:
  batch_size: 1
  gradient_accumulation_steps: 16
  # Effective batch size = 1 * 16 = 16
```

---

## Custom Experiments

### Create Your Own Experiment

Edit [config/experiments.yaml](config/experiments.yaml):

```yaml
experiments:
  my_custom_experiment:
    name: "my-custom-test"
    description: "Testing custom settings for my interview data"
    enabled: true

    strategy:
      method: "lora"

    hyperparameters:
      batch_size: 2
      gradient_accumulation_steps: 8
      learning_rate: 2.0e-5  # Custom LR
      num_epochs: 7  # Custom epochs

    lora:
      rank: 48  # Custom rank
      alpha: 96

    dataset:
      output_dir: "models/whisper-finetuned-custom"
```

Then run:
```bash
python src/scripts/run_training.py --experiment my_custom_experiment
```

---

## Systematic Experimentation

### Recommended Iteration Process

#### Phase 1: Baseline
1. Run `baseline_lora` on your interview data
2. Measure WER on test set
3. Note GPU memory usage and training time

#### Phase 2: LoRA Rank Comparison
```bash
# Test different ranks
python src/scripts/run_training.py --experiment quick_test    # rank 8
python src/scripts/run_training.py --experiment baseline_lora # rank 32
python src/scripts/run_training.py --experiment lora_large    # rank 64

# Compare WER, memory, speed
```

#### Phase 3: Learning Rate Sweep

Create experiments in [config/experiments.yaml](config/experiments.yaml):
```yaml
lr_1e5:
  hyperparameters: {learning_rate: 1e-5}

lr_5e5:
  hyperparameters: {learning_rate: 5e-5}

lr_1e4:
  hyperparameters: {learning_rate: 1e-4}
```

Run each and compare.

#### Phase 4: Strategy Comparison (If GPU Allows)
```bash
# LoRA
python src/scripts/run_training.py --experiment baseline_lora

# Full fine-tuning
python src/scripts/run_training.py --experiment full_finetune

# Compare quality difference
```

#### Phase 5: Final Selection

Choose best configuration based on:
- **WER on test set** (primary metric)
- Training time
- GPU memory usage
- Inference speed (if important)

---

## Output & Evaluation

### Training Outputs

After training completes:
```
models/whisper-finetuned-<experiment>/
├── final_model/              # Final fine-tuned model
│   ├── config.json
│   ├── preprocessor_config.json
│   ├── pytorch_model.bin     # Model weights
│   └── ...
├── logs/                     # TensorBoard logs
├── checkpoint-*/             # Training checkpoints (best 3 kept)
└── ...
```

### Monitor Training

```bash
tensorboard --logdir models/whisper-finetuned-<experiment>/logs
# Open http://localhost:6006

# Watch for:
# - Training loss decreasing
# - Validation WER decreasing (should plateau, not increase)
# - GPU memory usage stable
```

### Evaluate on Test Set

```bash
python src/evaluate_finetuned.py \
  --model models/whisper-finetuned-lora-r32/final_model \
  --test-audio Audio_Local_tests/audio_files \
  --ground-truth Audio_Local_tests/transcription_test
```

### Compare Models

```bash
# Compare baseline vs fine-tuned
python src/evaluate_finetuned.py \
  --baseline openai/whisper-large-v2 \
  --finetuned models/whisper-finetuned-lora-r32/final_model \
  --test-data prepared_datasets/whisper_finetuning_interviews/test
```

---

## Final Production Pipeline

Once you've selected the best configuration:

### 1. Update Active Experiment

Edit [config/experiments.yaml](config/experiments.yaml):
```yaml
active_experiment: "my_best_config"  # Your chosen experiment
```

### 2. Train Final Model

```bash
python src/scripts/run_training.py  # Uses active experiment
```

### 3. Transcribe New Audio

```bash
# Use fine-tuned model for production transcription
python src/transcribe_with_finetuned.py \
  --model models/whisper-finetuned-<best>/final_model \
  --audio path/to/new/audio \
  --output transcripts/final_output
```

---

## Troubleshooting

### Out of Memory (OOM)

**Solution 1**: Reduce batch size
```yaml
hyperparameters:
  batch_size: 1  # Reduce from 2
  gradient_accumulation_steps: 16  # Double to maintain effective batch
```

**Solution 2**: Use smaller LoRA rank
```yaml
lora:
  rank: 16  # Reduce from 32
  alpha: 32
```

**Solution 3**: Use quick_test experiment
```bash
python src/scripts/run_training.py --experiment quick_test
```

### Training Too Slow

**Solution**: Use minimal LoRA or fewer epochs
```bash
python src/scripts/run_training.py --experiment quick_test  # 1 epoch, rank 8
```

### WER Not Improving

**Solutions**:
1. Lower learning rate (try 5e-6)
2. More epochs (try 10)
3. Higher LoRA rank (try 64)
4. Check dataset quality
5. Try full fine-tuning

### Validation WER Increasing (Overfitting)

**Solutions**:
1. Reduce epochs
2. Increase regularization (LoRA dropout)
3. Add more training data
4. Use early stopping

---

## File Structure

```
Audio_Transcription_Deidentification/
├── config/                       # All configuration files
│   ├── training.yaml            # Training hyperparameters, LoRA
│   ├── experiments.yaml         # Experiment definitions
│   ├── models.yaml              # Model selection
│   ├── paths.yaml               # File paths
│   └── processing.yaml          # Audio/text processing
│
├── src/
│   ├── training/                # Training modules
│   │   ├── finetune.py         # Main config-driven training
│   │   ├── lora.py             # LoRA utilities
│   │   └── callbacks.py        # Training callbacks
│   │
│   ├── scripts/                 # High-level scripts
│   │   ├── run_training.py     # Main training script
│   │   └── list_experiments.py # List available experiments
│   │
│   ├── data/                    # Dataset preparation
│   ├── evaluation/              # Evaluation utilities
│   └── config.py                # Configuration loader
│
├── datasets/                     # HuggingFace datasets
├── models/                       # Fine-tuned models
├── training_data/                # Aligned training data
└── Notes/                        # Documentation
    ├── REORGANIZATION_PLAN.md
    └── QUICK_START_GUIDE.md
```

---

## Next Steps

### Immediate (Today)
1. ✅ Configuration system ready
2. ✅ Experiments defined
3. ➡️ **Prepare your interview transcript dataset**
4. ➡️ **Run baseline_lora experiment**
5. ➡️ **Evaluate WER on test set**

### Short-term (This Week)
6. Test different LoRA ranks (8, 16, 32, 64)
7. Test learning rates (1e-5, 5e-5)
8. Compare LoRA vs full fine-tuning (if GPU allows)
9. Select best configuration

### Medium-term (Next Week)
10. Train final model with best config
11. Transcribe production audio files
12. Analyze error patterns
13. Iterate if needed

---

## Support & Documentation

- **Detailed Plan**: [Notes/REORGANIZATION_PLAN.md](Notes/REORGANIZATION_PLAN.md)
- **Quick Reference**: [Notes/QUICK_START_GUIDE.md](Notes/QUICK_START_GUIDE.md)
- **Training Config**: [config/training.yaml](config/training.yaml)
- **Experiments**: [config/experiments.yaml](config/experiments.yaml)

---

**Status**: ✅ Workspace ready for iteration and fine-tuning
**Last Updated**: January 7, 2026
