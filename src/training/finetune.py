"""
Config-driven Whisper fine-tuning with optional LoRA.

This module provides the main fine-tuning logic that reads ALL parameters
from configuration files. Supports:
- LoRA fine-tuning (parameter-efficient)
- Full decoder fine-tuning (freeze encoder only)
- Full model fine-tuning (train everything)

Usage:
    from src.config import Config
    from src.training.finetune import finetune_whisper

    config = Config()
    experiment = config.get_experiment("baseline_lora")
    finetune_whisper(config, experiment_name="baseline_lora")
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from .lora import apply_lora, should_use_lora
from .callbacks import MemoryMonitorCallback, WERTrackingCallback, ExperimentTrackingCallback
from .dataset import (
    WhisperAudioDataset,
    WhisperDataCollator,
    load_simple_dataset_dir
)


def _import_evaluate():
    """
    Lazy import of evaluate library.

    Only imports when actually needed for metric computation.

    Returns:
        evaluate module

    Raises:
        ImportError: If evaluate is not installed, with helpful message
    """
    try:
        import evaluate
        return evaluate
    except ImportError as e:
        raise ImportError(
            "The 'evaluate' library is required for WER metric computation.\n"
            "Install with: pip install -r requirements_finetuning.txt\n"
            f"Original error: {e}"
        )


def setup_model_and_processor(config, experiment_overrides: Optional[Dict] = None):
    """
    Load Whisper model and processor from config.

    Args:
        config: Config object
        experiment_overrides: Optional experiment overrides

    Returns:
        Tuple of (model, processor)
    """
    model_name = config.models.whisperx.model_name
    device = config.get_device()

    print(f"\nLoading model: {model_name}")
    print(f"  Device: {device}")

    # Load processor
    processor = WhisperProcessor.from_pretrained(model_name)

    # Load model
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )

    # Apply freezing based on strategy
    strategy = experiment_overrides.get('strategy', {}).get('method') if experiment_overrides else None
    if strategy is None:
        strategy = config.training.strategy.method

    freezing_config = experiment_overrides.get('freezing', config.training.freezing) if experiment_overrides else config.training.freezing

    # Freeze encoder if requested
    if freezing_config.get('freeze_encoder', True):
        model.freeze_encoder()
        print("  OK: Encoder frozen")

    # Freeze decoder if requested (unusual but supported)
    if freezing_config.get('freeze_decoder', False):
        for param in model.model.decoder.parameters():
            param.requires_grad = False
        print("  OK: Decoder frozen")

    print(f"  OK: Model loaded")
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Parameters: {total_params:.1f}M total, {trainable_params:.1f}M trainable")

    return model, processor


def merge_experiment_overrides(config, experiment_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Merge experiment overrides with base training config.

    Args:
        config: Config object
        experiment_name: Name of experiment (or None for active)

    Returns:
        Merged configuration dictionary
    """
    if experiment_name:
        exp_overrides = config.get_experiment(experiment_name)
    else:
        exp_overrides = {}

    # Start with base config values
    merged = {
        'strategy': {'method': config.training.strategy.method},
        'hyperparameters': {
            'batch_size': config.training.hyperparameters.batch_size,
            'gradient_accumulation_steps': config.training.hyperparameters.gradient_accumulation_steps,
            'learning_rate': config.training.hyperparameters.learning_rate,
            'num_epochs': config.training.hyperparameters.num_epochs,
            'warmup_steps': config.training.hyperparameters.warmup_steps,
        },
        'lora': {
            'rank': config.training.lora.rank,
            'alpha': config.training.lora.alpha,
            'dropout': config.training.lora.dropout,
            'target_modules': config.training.lora.target_modules,
        },
        'dataset': {
            'output_dir': config.training.dataset.output_dir,
        },
        'freezing': config.training.freezing,
    }

    # Apply experiment overrides
    for key, value in exp_overrides.items():
        if key in merged and isinstance(value, dict) and isinstance(merged[key], dict):
            merged[key].update(value)
        elif key not in ['name', 'description', 'enabled']:
            merged[key] = value

    return merged


def compute_metrics(pred, processor):
    """
    Compute WER metric during evaluation.

    Args:
        pred: Predictions from model
        processor: Whisper processor

    Returns:
        Dictionary with WER metric
    """
    # Lazy import: only import evaluate when computing metrics
    evaluate = _import_evaluate()
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


class SafeWhisperTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 1) Pull labels out explicitly
        labels = inputs.pop("labels")

        # 2) Remove any stray keys that Whisper doesn't accept
        inputs.pop("input_ids", None)
        inputs.pop("attention_mask", None)

        # Optional: print once to confirm (remove after debugging)
        # print("FINAL INPUTS TO MODEL:", inputs.keys())

        # 3) Call the model directly (bypasses Seq2SeqTrainer's internal assumptions)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

def finetune_whisper(config, experiment_name: Optional[str] = None, preset: Optional[str] = None):
    """
    Fine-tune Whisper model using configuration.

    Args:
        config: Config object with all settings
        experiment_name: Name of experiment to run (uses active if None)
        preset: Optional LoRA preset to use (overrides config)

    Returns:
        Trained model and trainer

    Example:
        config = Config()
        model, trainer = finetune_whisper(config, experiment_name="baseline_lora")
    """
    print("\n" + "#" * 80)
    print("WHISPER FINE-TUNING (Config-Driven)")
    print("#" * 80)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Merge experiment overrides
    exp_config = merge_experiment_overrides(config, experiment_name)

    # Get experiment info
    if experiment_name:
        exp_def = config.experiments.experiments.get(experiment_name)
        exp_display_name = exp_def.name if exp_def else experiment_name
        exp_description = exp_def.description if exp_def else ""
    else:
        experiment_name = config.experiments.active_experiment
        exp_def = config.experiments.experiments[experiment_name]
        exp_display_name = exp_def.name
        exp_description = exp_def.description

    print(f"\nExperiment: {exp_display_name}")
    if exp_description:
        print(f"Description: {exp_description}")

    # Configuration
    strategy = exp_config['strategy']['method']
    hyperparams = exp_config['hyperparameters']
    lora_config = exp_config.get('lora', {})

    # Dataset paths
    project_root = Path(config.paths.project_root).resolve()
    dataset_dir = project_root / config.training.dataset.input_dir
    output_dir = project_root / exp_config['dataset']['output_dir']

    print(f"\nConfiguration:")
    print(f"  Strategy: {strategy}")
    print(f"  Dataset: {dataset_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Batch size: {hyperparams['batch_size']}")
    print(f"  Gradient accumulation: {hyperparams['gradient_accumulation_steps']}")
    print(f"  Learning rate: {hyperparams['learning_rate']}")
    print(f"  Epochs: {hyperparams['num_epochs']}")

    if strategy == "lora":
        print(f"  LoRA rank: {lora_config['rank']}")
        print(f"  LoRA alpha: {lora_config['alpha']}")
        if preset:
            print(f"  Using preset: {preset}")

    # Check dataset exists
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_dir}\n"
            "Run src/data/prepare_hf_dataset.py first to create the dataset."
        )

    # Load dataset examples (simple JSON format, no HF Audio feature)
    dataset_examples = load_simple_dataset_dir(dataset_dir)

    # Load model and processor
    model, processor = setup_model_and_processor(config, exp_config)

    # Apply LoRA if strategy requires it
    if strategy == "lora":
        model = apply_lora(model, config, preset=preset)

    # Create custom PyTorch datasets (loads audio on-the-fly, no torchcodec)
    print("\nCreating custom datasets...")
    train_dataset = WhisperAudioDataset(
        dataset_examples['train'],
        processor,
        max_length=config.training.dataset.filters.get('max_duration', 30)
    )
    print(f"  Train dataset: {len(train_dataset)} examples")

    eval_dataset = None
    if 'validation' in dataset_examples:
        eval_dataset = WhisperAudioDataset(
            dataset_examples['validation'],
            processor,
            max_length=config.training.dataset.filters.get('max_duration', 30)
        )
        print(f"  Validation dataset: {len(eval_dataset)} examples")

    # Create custom data collator (handles batching and padding)
    data_collator = WhisperDataCollator(processor)

    # Training arguments from config
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=hyperparams['batch_size'],
        per_device_eval_batch_size=hyperparams['batch_size'],
        gradient_accumulation_steps=hyperparams['gradient_accumulation_steps'],
        learning_rate=hyperparams['learning_rate'],
        num_train_epochs=hyperparams['num_epochs'],
        warmup_steps=hyperparams.get('warmup_steps', 100),
        fp16=config.training.mixed_precision.get('fp16', True) and torch.cuda.is_available(),
        eval_strategy=config.training.evaluation.get('strategy', 'epoch'),
        save_strategy=config.training.checkpoints.get('save_strategy', 'epoch'),
        save_total_limit=config.training.checkpoints.get('save_total_limit', 3),
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model=config.training.checkpoints.get('metric_for_best_model', 'wer'),       
        greater_is_better=config.training.checkpoints.get('greater_is_better', False),
        logging_steps=config.training.logging.get('steps', 10),
        logging_dir=str(output_dir / config.training.logging.get('dir', 'logs')),
        report_to=config.training.logging.get('report_to', ['tensorboard']),
        push_to_hub=False,
        predict_with_generate=True, #To get decoder outputs during evaluation
        generation_max_length=config.training.evaluation.get('generation_max_length', 225),
        remove_unused_columns=True,
        label_names=["labels"],
        seed=config.training.seed,
    )

    print("\nTraining configuration:")
    print(f"  Effective batch size: {hyperparams['batch_size'] * hyperparams['gradient_accumulation_steps']}")
    print(f"  Mixed precision (fp16): {training_args.fp16}")
    print(f"  Evaluation strategy: {training_args.eval_strategy}")
    print(f"  Save strategy: {training_args.save_strategy}")

    # Create callbacks
    callbacks = [
        MemoryMonitorCallback(log_interval=config.training.logging.get('steps', 10)),
        ExperimentTrackingCallback(exp_display_name, exp_config),
    ]

    if eval_dataset is not None:
        callbacks.append(WERTrackingCallback())

    # Create trainer with custom datasets and data collator
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=lambda pred: compute_metrics(pred, processor) if training_args.predict_with_generate else {},
        callbacks=callbacks
    )

    # Train
    print("\nStarting training...")
    print(f"  Monitor with: tensorboard --logdir {output_dir / 'logs'}")
    print("TRAINER CLASS:", type(trainer))
    print("TRAINER MODULE:", type(trainer).__module__)
    
    trainer.train()

    # Save final model
    print("\nSaving final model...")
    final_model_path = output_dir / "final_model"
    trainer.save_model(str(final_model_path))
    processor.save_pretrained(str(final_model_path))

    print(f"  OK: Model saved to: {final_model_path}")

    # Summary
    print("\n" + "=" * 80)
    print("FINE-TUNING COMPLETE")
    print("=" * 80)
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nExperiment: {exp_display_name}")
    print(f"Model saved: {final_model_path}")
    print("\nNext steps:")
    print("  1. Evaluate on test set")
    print("  2. Compare to baseline")
    print(f"  3. View logs: tensorboard --logdir {output_dir / 'logs'}")

    return model, trainer
