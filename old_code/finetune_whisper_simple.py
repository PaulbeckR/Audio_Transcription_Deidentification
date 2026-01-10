"""
Fine-tune Whisper using LoRA (Simple version without HF Audio feature).

This version avoids audio backend conflicts by loading audio directly
using whisperx's audio loading instead of HuggingFace's Audio feature.

Optimized for: NVIDIA RTX 3050 (6GB VRAM)

Usage:
    python src/finetune_whisper_simple.py
"""

import sys
from pathlib import Path
import os
import json
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
    import numpy as np
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
        TrainerCallback
    )
    from peft import LoraConfig, get_peft_model
    import evaluate
    import whisperx
except ImportError as e:
    print(f"\nERROR: Missing dependency: {e}")
    print("\nInstall required packages:")
    print("  pip install transformers accelerate peft evaluate jiwer")
    sys.exit(1)


class MemoryMonitorCallback(TrainerCallback):
    """Monitor and log GPU memory usage during training."""

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available() and state.global_step % 10 == 0:
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            print(f"  [Step {state.global_step}] GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


class SimpleWhisperDataset(torch.utils.data.Dataset):
    """Simple dataset that loads audio on-the-fly."""

    def __init__(self, examples, processor):
        self.examples = examples
        self.processor = processor

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Load audio using whisperx (same as WhisperX uses)
        audio_path = example['audio_path']
        audio = whisperx.load_audio(audio_path)

        # Process audio with Whisper feature extractor
        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=16000
        ).input_features[0]

        # Tokenize text
        labels = self.processor.tokenizer(example['text']).input_ids

        return {
            'input_features': torch.tensor(input_features),
            'labels': torch.tensor(labels)
        }


def load_simple_dataset(dataset_file: Path):
    """Load simple JSON dataset."""
    print(f"\nLoading dataset from: {dataset_file}")

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_file}")

    with open(dataset_file, 'r', encoding='utf-8') as f:
        examples = json.load(f)

    print(f"  Examples: {len(examples)}")
    return examples


def setup_model_and_processor(model_name: str = "openai/whisper-large-v2", device: str = "cuda"):
    """Load Whisper model and processor."""
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
    """Apply LoRA for parameter-efficient fine-tuning."""
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


def data_collator(features):
    """Custom data collator for batching."""
    # Pad input features
    input_features = [f['input_features'] for f in features]
    max_input_length = max(f.shape[0] for f in input_features)

    batch_input_features = []
    for f in input_features:
        if f.shape[0] < max_input_length:
            # Pad with zeros
            padding = torch.zeros(max_input_length - f.shape[0], f.shape[1])
            f = torch.cat([f, padding], dim=0)
        batch_input_features.append(f)

    # Pad labels
    labels = [f['labels'] for f in features]
    max_label_length = max(len(l) for l in labels)

    batch_labels = []
    for l in labels:
        if len(l) < max_label_length:
            # Pad with -100 (ignored in loss)
            padding = torch.full((max_label_length - len(l),), -100, dtype=torch.long)
            l = torch.cat([l, padding], dim=0)
        batch_labels.append(l)

    return {
        'input_features': torch.stack(batch_input_features),
        'labels': torch.stack(batch_labels)
    }


def train(
    model,
    processor,
    train_examples,
    output_dir: Path,
    batch_size: int = 2,
    learning_rate: float = 1e-5,
    num_epochs: int = 5,
    gradient_accumulation_steps: int = 8
):
    """Fine-tune Whisper model."""
    print("\n" + "=" * 80)
    print("STARTING FINE-TUNING")
    print("=" * 80)

    # Create dataset
    print("\nCreating training dataset...")
    train_dataset = SimpleWhisperDataset(train_examples, processor)
    print(f"  Training examples: {len(train_dataset)}")

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        warmup_steps=50,
        fp16=torch.cuda.is_available(),
        eval_strategy="no",  # No validation set (renamed from evaluation_strategy)
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=10,
        logging_dir=str(output_dir / "logs"),
        report_to=["tensorboard"],
        push_to_hub=False,
        remove_unused_columns=False,
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
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
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
    print("WHISPER FINE-TUNING WITH LoRA (Simple Version)")
    print("#" * 80)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Configuration
    MODEL_NAME = config.models.whisperx.model_name  # e.g., "openai/whisper-large-v2"
    DEVICE = config.get_device()

    DATASET_FILE = PROJECT_PATH / "datasets" / "whisper_finetuning_simple" / "training_data.json"
    OUTPUT_DIR = PROJECT_PATH / "models" / "whisper-finetuned-interviews"

    # Hyperparameters (optimized for 6GB GPU)
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION = 8  # Effective batch size = 16
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 5

    print(f"\nConfiguration:")
    print(f"  Base model: {MODEL_NAME}")
    print(f"  Device: {DEVICE}")
    print(f"  Dataset: {DATASET_FILE}")
    print(f"  Output: {OUTPUT_DIR}")

    # Check dataset exists
    if not DATASET_FILE.exists():
        print(f"\nERROR: Dataset not found: {DATASET_FILE}")
        print("   Run src/prepare_hf_dataset_simple.py first to create the dataset.")
        sys.exit(1)

    try:
        # Load dataset
        train_examples = load_simple_dataset(DATASET_FILE)

        # Load model and processor
        model, processor = setup_model_and_processor(MODEL_NAME, DEVICE)

        # Apply LoRA
        model = apply_lora(model)

        # Fine-tune
        trainer = train(
            model=model,
            processor=processor,
            train_examples=train_examples,
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
        print("  1. Evaluate on test set")
        print("  2. View training logs: tensorboard --logdir " + str(OUTPUT_DIR / "logs"))
        print("  3. Compare to baseline performance")

    except Exception as e:
        print(f"\nERROR during fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)