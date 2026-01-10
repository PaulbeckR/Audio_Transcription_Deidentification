# Whisper Fine-Tuning Guide

Complete guide for fine-tuning Whisper on interview transcription data using LoRA (Low-Rank Adaptation).

**Created:** 2025-12-28
**Hardware:** NVIDIA RTX 3050 (6GB VRAM)
**Base Model:** openai/whisper-large-v2

---

## Overview

This guide covers the complete pipeline for fine-tuning Whisper on domain-specific audio data (clinical/therapeutic interviews). The process uses:

- **WhisperX** for forced alignment (creates timestamped training data)
- **Hugging Face Transformers** for fine-tuning
- **LoRA** (Parameter-Efficient Fine-Tuning) for 6GB GPU compatibility
- **Training data**: GPT_test interviews (with plans to add more)

---

## Prerequisites

### 1. Install Dependencies

```bash
# Activate your WhisperX environment
.venv_whisperx\Scripts\activate  # Windows
# source .venv_whisperx/bin/activate  # Linux/Mac

# Install fine-tuning dependencies
pip install transformers datasets accelerate peft evaluate jiwer tensorboard
```

### 2. Verify GPU

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
VRAM: 6.0 GB
```

---

## Pipeline Steps

### Step 1: Create Aligned Training Data

Convert ground truth transcripts (no timestamps) into timestamped training data using WhisperX forced alignment.

**Script:** `src/create_training_data.py`

```bash
python src/create_training_data.py
```

**What it does:**
1. Loads audio file (e.g., `gpt_test.m4a`)
2. Loads ground truth transcript (e.g., `gpt_test_truth.csv`)
3. Uses WhisperX alignment model to create word-level timestamps
4. Validates alignment quality
5. Saves aligned data as JSON

**Output:**
```
training_data/
├── gpt_test_aligned.json
└── gpt_test_metadata.json
```

**Alignment Quality Checks:**
- ✓ All text successfully aligned
- ✓ Word-level timestamps generated
- ✓ Confidence scores reported
- ⚠️ Review low-confidence alignments

### Step 2: Prepare Hugging Face Dataset

Convert aligned JSON files into Hugging Face Dataset format.

**Script:** `src/prepare_hf_dataset.py`

```bash
python src/prepare_hf_dataset.py
```

**What it does:**
1. Loads all `*_aligned.json` files from `training_data/`
2. Extracts segments and creates dataset dictionary
3. Creates Hugging Face Dataset with Audio feature
4. Splits into train/validation (if dataset is large enough)
5. Saves to disk

**Output:**
```
prepared_datasets/whisper_finetuning/
├── train/
│   ├── data-00000-of-00001.arrow
│   ├── dataset_info.json
│   └── state.json
├── validation/  (if split)
└── dataset_dict.json
```

**Dataset Format:**
```python
{
    'audio': Audio(sampling_rate=16000),
    'sentence': 'Transcription text...'
}
```

### Step 3: Fine-Tune Whisper with LoRA

Train the model using parameter-efficient fine-tuning.

**Script:** `src/finetune_whisper.py`

```bash
python src/finetune_whisper.py
```

**What it does:**
1. Loads base Whisper model (`openai/whisper-large-v2`)
2. Freezes encoder (only fine-tune decoder)
3. Applies LoRA adapters (reduces trainable params from 1500M → ~10M)
4. Trains on prepared dataset
5. Saves checkpoints and final model

**Training Configuration:**
```python
BATCH_SIZE = 2  # Per GPU
GRADIENT_ACCUMULATION = 8  # Effective batch size = 16
LEARNING_RATE = 1e-5
NUM_EPOCHS = 5
```

**Memory Optimization:**
- Mixed precision (FP16)
- Gradient accumulation
- LoRA (rank=32)
- Encoder frozen
- Small batch size

**Output:**
```
models/whisper-finetuned-interviews/
├── final_model/
│   ├── pytorch_model.bin
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   ├── config.json
│   └── preprocessor_config.json
├── checkpoint-100/
├── checkpoint-200/
└── logs/
    └── events.out.tfevents...
```

**Monitor Training:**
```bash
tensorboard --logdir models/whisper-finetuned-interviews/logs
```

Open browser: `http://localhost:6006`

---

## Training Details

### LoRA Configuration

```python
{
    "r": 32,              # LoRA rank (higher = more capacity, more memory)
    "lora_alpha": 64,     # LoRA scaling factor
    "target_modules": ["q_proj", "v_proj"],  # Which attention layers to adapt
    "lora_dropout": 0.05, # Dropout for regularization
    "bias": "none"
}
```

**What is LoRA?**
- Low-Rank Adaptation: trains small adapter matrices instead of full model
- Reduces trainable parameters by ~99%
- Maintains 95-100% of full fine-tuning performance
- Critical for 6GB GPU constraint

### Training Metrics

**Tracked during training:**
- Training loss
- Validation WER (if validation set exists)
- GPU memory usage
- Learning rate schedule

**Expected GPU Usage:**
- Model loading: ~5.5 GB
- Training: ~5.8-6.0 GB
- Peak: Should stay under 6GB with current config

**If OOM (Out of Memory):**
1. Reduce `BATCH_SIZE` to 1
2. Increase `GRADIENT_ACCUMULATION` to 16
3. Reduce LoRA rank to 16
4. Use `model.gradient_checkpointing_enable()`

---

## Evaluation

### Step 4: Test Fine-Tuned Model

Compare fine-tuned model to baseline on test sets.

**Test Sets:**
1. **GPT_test** (training data) - Check for overfitting
2. **GPT_test2** (new speakers, same domain) - Generalization test
3. **hamlet_test** (different domain) - Catastrophic forgetting check

**Create evaluation script:** `src/evaluate_finetuned.py`

```python
from transformers import pipeline

# Load fine-tuned model
pipe = pipeline(
    "automatic-speech-recognition",
    model="models/whisper-finetuned-interviews/final_model",
    device="cuda"
)

# Transcribe
result = pipe("Audio_Local_tests/audio_files/gpt_test2.m4a")
print(result["text"])
```

**Compare WER:**
```python
from src.metrics import calculate_wer

baseline_wer = 0.1001  # 10.01% (from your baseline)
finetuned_wer = calculate_wer(truth, hypothesis)

improvement = (baseline_wer - finetuned_wer) / baseline_wer * 100
print(f"WER improvement: {improvement:.1f}%")
```

**Success Criteria:**
- ✅ **GPT_test2 WER improvement ≥ 5%** (e.g., 10% → 9.5%)
- ✅ GPT_test WER lower (but not 0% - would indicate overfitting)
- ✅ hamlet_test WER within 2% of baseline (no catastrophic forgetting)

---

## Adding More Training Data

### Current Data
- `gpt_test.m4a` (~3.3 MB, ~6 minutes)
- More files can be added to increase training data

### Adding New Files

1. **Place audio file:**
   ```
   Audio_Local_tests/audio_files/new_interview.m4a
   ```

2. **Create ground truth CSV:**
   ```
   Audio_Local_tests/transcription_test/new_interview_truth.csv
   ```

   Format:
   ```csv
   Speaker,Transcription
   SPEAKER_00,"Welcome to the interview..."
   SPEAKER_01,"Thank you for having me."
   ```

3. **Update `create_training_data.py`:**
   ```python
   training_files = [
       {'audio': audio_dir / "gpt_test.m4a",
        'truth': truth_dir / "gpt_test_truth.csv"},
       {'audio': audio_dir / "new_interview.m4a",
        'truth': truth_dir / "new_interview_truth.csv"},  # Add this
   ]
   ```

4. **Re-run pipeline:**
   ```bash
   python src/create_training_data.py
   python src/prepare_hf_dataset.py
   python src/finetune_whisper.py
   ```

---

## Troubleshooting

### Issue: Alignment fails

**Error:** `Alignment produced no results`

**Solutions:**
1. Check audio quality (clear speech, minimal noise)
2. Verify transcript matches audio
3. Try shorter audio segments
4. Check language code is correct (`en`)

### Issue: OOM during training

**Error:** `CUDA out of memory`

**Solutions:**
```python
# In src/finetune_whisper.py
BATCH_SIZE = 1  # Reduce from 2
GRADIENT_ACCUMULATION = 16  # Increase from 8
```

Or reduce LoRA rank:
```python
lora_config = {
    "r": 16,  # Reduce from 32
    ...
}
```

### Issue: Poor fine-tuning results

**Symptoms:** No improvement or worse WER

**Possible causes:**
1. **Insufficient data** - Need more training examples
2. **Overfitting** - Reduce epochs, add dropout
3. **Domain mismatch** - Training data not representative
4. **Model saturation** - Base model already optimal

**Solutions:**
1. Collect more training data (aim for 1+ hours)
2. Reduce `NUM_EPOCHS` to 3
3. Increase `lora_dropout` to 0.1
4. Try data augmentation (speed, noise)

### Issue: Model doesn't load after training

**Error:** `Model not found` or `Adapter not found`

**Solutions:**
1. Check model saved to correct path
2. Load with PEFT:
   ```python
   from peft import PeftModel
   from transformers import WhisperForConditionalGeneration

   base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
   model = PeftModel.from_pretrained(base_model, "models/whisper-finetuned-interviews/final_model")
   ```

---

## File Structure

```
Audio_Transcription_Deidentification/
│
├── Audio_Local_tests/
│   ├── audio_files/
│   │   ├── gpt_test.m4a
│   │   └── gpt_test2.m4a
│   └── transcription_test/
│       ├── gpt_test_truth.csv
│       └── gpt_test2_truth.csv
│
├── training_data/              # Created by Step 1
│   ├── gpt_test_aligned.json
│   └── gpt_test_metadata.json
│
├── datasets/                   # Created by Step 2
│   └── whisper_finetuning/
│       ├── train/
│       └── dataset_dict.json
│
├── models/                     # Created by Step 3
│   └── whisper-finetuned-interviews/
│       ├── final_model/
│       │   ├── pytorch_model.bin
│       │   ├── adapter_model.bin
│       │   └── config.json
│       ├── checkpoint-100/
│       └── logs/
│
└── src/
    ├── create_training_data.py  # Step 1
    ├── prepare_hf_dataset.py    # Step 2
    ├── finetune_whisper.py      # Step 3
    └── evaluate_finetuned.py    # Step 4 (create this)
```

---

## Next Steps

1. **Run baseline tests** if not done:
   ```bash
   python run_baseline_test.py
   python simple_word_errors.py
   ```

2. **Create aligned training data:**
   ```bash
   python src/create_training_data.py
   ```

3. **Prepare dataset:**
   ```bash
   python src/prepare_hf_dataset.py
   ```

4. **Start fine-tuning:**
   ```bash
   python src/finetune_whisper.py
   ```

5. **Evaluate results** - Compare to baseline

6. **Iterate:**
   - Add more training data
   - Adjust hyperparameters
   - Try different LoRA configs

---

## References

- **Hugging Face Whisper Fine-tuning:** https://huggingface.co/blog/fine-tune-whisper
- **LoRA Paper:** https://arxiv.org/abs/2106.09685
- **PEFT Library:** https://github.com/huggingface/peft
- **WhisperX:** https://github.com/m-bain/whisperX

---

**Document Status:** Active - Ready for Implementation
**Last Updated:** 2025-12-28
