"""
Fine-tune Whisper model using LoRA (Low-Rank Adaptation).

This script fine-tunes OpenAI's Whisper model on domain-specific data
using parameter-efficient fine-tuning (LoRA) to reduce GPU memory requirements.

Optimized for: NVIDIA RTX 3050 (6GB VRAM)

Usage:
    python src/finetune_whisper.py

Requirements:
    pip install transformers datasets accelerate peft evaluate jiwer
"""

import sys
from pathlib import Path
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config

config = Config()
PROJECT_PATH = Path(__file__).parent.parent

# Add FFmpeg to PATH
FFMPEG_PATH = config.get_ffmpeg_path()
if FFMPEG_PATH and FFMPEG_PATH not in os.environ.get("PATH", ""):
    os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ.get("PATH", "")

print("Importing dependencies...")
try:
    import torch
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
        TrainerCallback
    )
    from datasets import load_from_disk, DatasetDict
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import evaluate
    import numpy as np
except ImportError as e:
    print(f"\nERROR: Missing dependency: {e}")
    print("\nInstall required packages:")
    print("  pip install transformers datasets accelerate peft evaluate jiwer")
    sys.exit(1)


class MemoryMonitorCallback(TrainerCallback):
    """Monitor and log GPU memory usage during training."""

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available() and state.global_step % 10 == 0:
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            print(f"  [Step {state.global_step}] GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def load_dataset(dataset_dir: Path):
    """Load prepared Hugging Face dataset."""
    print(f"\nLoading dataset from: {dataset_dir}")

    train_path = dataset_dir / "train"
    val_path = dataset_dir / "validation"

    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")

    train_dataset = load_from_disk(str(train_path))
    print(f"  Train examples: {len(train_dataset)}")

    # Load validation dataset if exists
    if val_path.exists():
        val_dataset = load_from_disk(str(val_path))
        print(f"  Validation examples: {len(val_dataset)}")
        return DatasetDict({"train": train_dataset, "validation": val_dataset})
    else:
        print("  No validation set found - using train set only")
        return DatasetDict({"train": train_dataset})


def prepare_dataset(batch, processor):
    """
    Prepare batch for training.

    Process audio and transcriptions for Whisper model.
    """
    # Load and process audio
    audio = batch["audio"]

    # Compute input features
    input_features = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # Encode target text
    labels = processor.tokenizer(batch["sentence"]).input_ids

    batch["input_features"] = input_features
    batch["labels"] = labels

    return batch


def setup_model_and_processor(model_name: str = "openai/whisper-large-v2", device: str = "cuda"):
    """
    Load Whisper model and processor.

    Args:
        model_name: Hugging Face model identifier
        device: 'cuda' or 'cpu'
    """
    print(f"\nLoading model: {model_name}")
    print(f"  Device: {device}")

    # Load processor (tokenizer + feature extractor)
    processor = WhisperProcessor.from_pretrained(model_name)

    # Load model
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )

    # Freeze encoder (only fine-tune decoder)
    model.freeze_encoder()
    print("  OK: Encoder frozen (decoder only fine-tuning)")

    print(f"  OK: Model loaded")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")

    return model, processor


def apply_lora(model, lora_config: dict = None):
    """
    Apply LoRA (Low-Rank Adaptation) to model for parameter-efficient fine-tuning.

    This dramatically reduces memory requirements and training time.
    """
    print("\nApplying LoRA...")

    if lora_config is None:
        lora_config = {
            "r": 32,  # LoRA rank
            "lora_alpha": 64,  # LoRA scaling
            "target_modules": ["q_proj", "v_proj"],  # Which layers to adapt
            "lora_dropout": 0.05,
            "bias": "none",
        }

    config = LoraConfig(**lora_config)
    model = get_peft_model(model, config)

    print(f"  OK: LoRA applied (rank={lora_config['r']})")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"  Trainable %: {100 * sum(p.numel() for p in model.parameters() if p.requires_grad) / sum(p.numel() for p in model.parameters()):.2f}%")

    return model


def compute_metrics(pred, processor):
    """Compute WER metric during evaluation."""
    wer_metric = evaluate.load("wer")

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def train(
    model,
    processor,
    dataset_dict,
    output_dir: Path,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    num_epochs: int = 3,
    gradient_accumulation_steps: int = 4
):
    """
    Fine-tune Whisper model.

    Args:
        model: Whisper model (with LoRA applied)
        processor: Whisper processor
        dataset_dict: DatasetDict with train (and optionally validation) datasets
        output_dir: Where to save checkpoints and final model
        batch_size: Per-device batch size
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        gradient_accumulation_steps: Accumulate gradients over N steps (simulates larger batch)
    """
    print("\n" + "=" * 80)
    print("STARTING FINE-TUNING")
    print("=" * 80)

    # Prepare datasets
    print("\nPreparing datasets...")
    dataset_dict = dataset_dict.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=dataset_dict["train"].column_names,
        desc="Preparing dataset"
    )

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        warmup_steps=100,
        fp16=torch.cuda.is_available(),  # Mixed precision for GPU
        evaluation_strategy="epoch" if "validation" in dataset_dict else "no",
        save_strategy="epoch",
        save_total_limit=3,  # Keep only 3 best checkpoints
        load_best_model_at_end=True if "validation" in dataset_dict else False,
        metric_for_best_model="wer" if "validation" in dataset_dict else None,
        greater_is_better=False,  # Lower WER is better
        logging_steps=10,
        logging_dir=str(output_dir / "logs"),
        report_to=["tensorboard"],  # Use tensorboard for logging
        push_to_hub=False,
        predict_with_generate=True,
        generation_max_length=225,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    print("\nTraining configuration:")
    print(f"  Batch size (per device): {batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Mixed precision (fp16): {training_args.fp16}")
    print(f"  Output dir: {output_dir}")

    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict.get("validation"),
        tokenizer=processor.feature_extractor,
        compute_metrics=lambda pred: compute_metrics(pred, processor) if training_args.predict_with_generate else {},
        callbacks=[MemoryMonitorCallback()]
    )

    # Train
    print("\nStarting training...")
    print("  (Monitor progress with: tensorboard --logdir " + str(output_dir / "logs") + ")")

    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(str(output_dir / "final_model"))
    processor.save_pretrained(str(output_dir / "final_model"))

    print(f"  OK: Model saved to: {output_dir / 'final_model'}")

    return trainer


if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("WHISPER FINE-TUNING WITH LoRA")
    print("#" * 80)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Configuration
    MODEL_NAME = config.models.whisperx.model_name  # e.g., "openai/whisper-large-v2"
    DEVICE = config.get_device()

    DATASET_DIR = PROJECT_PATH / "datasets" / "whisper_finetuning"
    OUTPUT_DIR = PROJECT_PATH / "models" / "whisper-finetuned-interviews"

    # Hyperparameters (adjust for your GPU)
    BATCH_SIZE = 2  # Small batch for 6GB GPU
    GRADIENT_ACCUMULATION = 8  # Effective batch size = 2 * 8 = 16
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 5

    print(f"\nConfiguration:")
    print(f"  Base model: {MODEL_NAME}")
    print(f"  Device: {DEVICE}")
    print(f"  Dataset: {DATASET_DIR}")
    print(f"  Output: {OUTPUT_DIR}")

    # Check dataset exists
    if not DATASET_DIR.exists():
        print(f" Dataset not found: {DATASET_DIR}")
        print("   Run src/prepare_hf_dataset.py first to create the dataset.")
        sys.exit(1)

    try:
        # Load dataset
        dataset_dict = load_dataset(DATASET_DIR)

        # Load model and processor
        model, processor = setup_model_and_processor(MODEL_NAME, DEVICE)

        # Apply LoRA
        model = apply_lora(model)

        # Fine-tune
        trainer = train(
            model=model,
            processor=processor,
            dataset_dict=dataset_dict,
            output_dir=OUTPUT_DIR,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            num_epochs=NUM_EPOCHS,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION
        )

        # Summary
        print("\n" + "=" * 80)
        print("FINE-TUNING COMPLETE")
        print("=" * 80)
        print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nFine-tuned model saved to: {OUTPUT_DIR / 'final_model'}")
        print("\nNext steps:")
        print("  1. Evaluate on test set: python src/evaluate_finetuned.py")
        print("  2. View training logs: tensorboard --logdir " + str(OUTPUT_DIR / "logs"))
        print("  3. Compare to baseline performance")

    except Exception as e:
        print(f" Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)