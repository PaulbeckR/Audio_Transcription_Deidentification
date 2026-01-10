"""
PyTorch utilities for compatibility with WhisperX and HuggingFace models.

This module provides patches and utilities to ensure compatibility between
different PyTorch versions and model loading requirements.
"""

import torch


def patch_torch_load():
    """
    Monkey-patch torch.load to force weights_only=False for WhisperX/Pyannote compatibility.

    This is safe since we trust Hugging Face model sources.
    Should be called before any model loading operations.

    Background:
    - PyTorch 2.6+ defaults to weights_only=True for security
    - WhisperX and Pyannote models require weights_only=False
    - This patch ensures compatibility without modifying model code
    """
    # Store reference to original torch.load
    if not hasattr(torch, '_original_load'):
        torch._original_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        """Wrapper for torch.load that forces weights_only=False for compatibility"""
        # Force weights_only=False if not explicitly set
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return torch._original_load(*args, **kwargs)

    # Replace torch.load globally BEFORE any model loading
    torch.load = _patched_torch_load


def restore_torch_load():
    """
    Restore original torch.load behavior.

    Use this if you need to revert the patch (usually not necessary).
    """
    if hasattr(torch, '_original_load'):
        torch.load = torch._original_load
        delattr(torch, '_original_load')
