"""
Core utilities for audio transcription and deidentification.

Provides:
- General utilities (device detection, file validation, etc.)
- PyTorch compatibility utilities
- Audio processing helpers
"""

from .utils import (
    get_device,
    get_gpu_info,
    validate_audio_file,
    ensure_directory,
    format_time,
    clear_temp_folder,
    convert_to_wav,
    get_audio_duration,
)

from .torch_utils import (
    patch_torch_load,
    restore_torch_load,
)

__all__ = [
    # utils.py
    'get_device',
    'get_gpu_info',
    'validate_audio_file',
    'ensure_directory',
    'format_time',
    'clear_temp_folder',
    'convert_to_wav',
    'get_audio_duration',
    # torch_utils.py
    'patch_torch_load',
    'restore_torch_load',
]
