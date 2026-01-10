"""
LoRA (Low-Rank Adaptation) utilities for parameter-efficient fine-tuning.

Provides functions to configure and apply LoRA based on config files.
"""

from typing import Dict, Any, Optional


def _import_peft():
    """
    Lazy import of peft library.

    Only imports when actually needed, allowing the module to load
    without requiring peft/transformers to be installed.

    Returns:
        Tuple of (LoraConfig, get_peft_model) classes

    Raises:
        ImportError: If peft is not installed, with helpful message
    """
    try:
        from peft import LoraConfig, get_peft_model
        return LoraConfig, get_peft_model
    except ImportError as e:
        raise ImportError(
            "The 'peft' library is required for LoRA fine-tuning.\n"
            "Install with: pip install -r requirements_finetuning.txt\n"
            f"Original error: {e}"
        )


def should_use_lora(config) -> bool:
    """
    Determine if LoRA should be used based on training strategy.

    Args:
        config: Config object with training settings

    Returns:
        True if LoRA should be applied
    """
    return config.training.strategy.method == "lora"


def get_lora_config(config, preset: Optional[str] = None) -> Dict[str, Any]:
    """
    Get LoRA configuration from config file or preset.

    Args:
        config: Config object with LoRA settings
        preset: Optional preset name ('minimal', 'small', 'medium', 'large')
                If None, uses config values directly

    Returns:
        Dictionary with LoRA configuration parameters

    Example:
        # Use config values
        lora_config = get_lora_config(config)

        # Use preset
        lora_config = get_lora_config(config, preset='medium')
    """
    if preset:
        if preset not in config.training.lora.presets:
            raise ValueError(
                f"LoRA preset '{preset}' not found. "
                f"Available: {list(config.training.lora.presets.keys())}"
            )

        preset_config = config.training.lora.presets[preset]
        return {
            "r": preset_config['rank'],
            "lora_alpha": preset_config['alpha'],
            "lora_dropout": preset_config.get('dropout', 0.05),
            "target_modules": preset_config.get(
                'target_modules',
                config.training.lora.target_modules
            ),
            "bias": config.training.lora.bias,
           # "task_type": config.training.lora.task_type,
        }
    else:
        return {
            "r": config.training.lora.rank,
            "lora_alpha": config.training.lora.alpha,
            "lora_dropout": config.training.lora.dropout,
            "target_modules": config.training.lora.target_modules,
            "bias": config.training.lora.bias,
           # "task_type": config.training.lora.task_type,
        }


def apply_lora(model, config, preset: Optional[str] = None):
    """
    Apply LoRA to model for parameter-efficient fine-tuning.

    Args:
        model: Whisper model to apply LoRA to
        config: Config object with LoRA settings
        preset: Optional LoRA preset to use

    Returns:
        Model with LoRA applied

    Example:
        model, processor = setup_model_and_processor(config)
        model = apply_lora(model, config)  # Use config values
        # OR
        model = apply_lora(model, config, preset='medium')  # Use preset
    """
    print("\nApplying LoRA...")

    # Lazy import: only import peft when actually applying LoRA
    LoraConfig, get_peft_model = _import_peft()

    lora_config_dict = get_lora_config(config, preset)

    # Create PEFT LoraConfig
    lora_config = LoraConfig(**lora_config_dict)

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)

    # Print statistics
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / total_params

    print(f"  OK: LoRA applied (rank={lora_config_dict['r']})")
    print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
    print(f"  Total parameters: {total_params / 1e6:.2f}M")
    print(f"  Trainable %: {trainable_pct:.2f}%")

    if preset:
        print(f"  Using preset: {preset}")

    return model


def print_lora_config(config, preset: Optional[str] = None):
    """
    Print LoRA configuration for debugging.

    Args:
        config: Config object
        preset: Optional preset name
    """
    lora_config = get_lora_config(config, preset)

    print("\nLoRA Configuration:")
    print(f"  Rank (r): {lora_config['r']}")
    print(f"  Alpha: {lora_config['lora_alpha']}")
    print(f"  Dropout: {lora_config['lora_dropout']}")
    print(f"  Target modules: {lora_config['target_modules']}")
    print(f"  Bias: {lora_config['bias']}")
   # print(f"  Task type: {lora_config['task_type']}")
    if preset:
        print(f"  Preset: {preset}")
