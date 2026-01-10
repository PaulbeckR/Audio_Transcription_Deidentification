"""
Training module for Whisper fine-tuning.

Provides utilities for:
- LoRA configuration and application
- Trainer setup
- Training callbacks
- Config-driven fine-tuning
"""

from .lora import get_lora_config, apply_lora, should_use_lora

__all__ = [
    'get_lora_config',
    'apply_lora',
    'should_use_lora',
]
