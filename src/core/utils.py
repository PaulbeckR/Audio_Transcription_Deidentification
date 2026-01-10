"""
Utility functions for audio transcription project.

Provides helper functions for:
- Device detection (CPU/GPU)
- Path validation
- Common operations
"""

import torch
import os
from pathlib import Path
from typing import Optional, Tuple
from pydub import AudioSegment

def convert_to_wav(input, output):
    try:
        audio = AudioSegment.from_file(input)
        audio.export(output, format='wav')
    except Exception as e:
        print("error during converstion: {e}")


def get_device(prefer_gpu: bool = True, verbose: bool = True) -> torch.device:
    """
    Detect and return the best available compute device.

    Args:
        prefer_gpu: If True, use GPU when available. If False, force CPU.
        verbose: If True, print device information.

    Returns:
        torch.device object (either 'cuda' or 'cpu')

    Example:
        >>> device = get_device()
        GPU detected: NVIDIA GeForce RTX 3080
        Using device: cuda
        >>> model.to(device)
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU detected: {gpu_name}")
            print(f"GPU memory: {gpu_memory:.2f} GB")
            print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        if verbose:
            if not torch.cuda.is_available():
                print("GPU not available, using CPU")
            else:
                print("GPU available but CPU requested")
            print(f"Using device: {device}")

    return device


def get_gpu_info() -> Optional[dict]:
    """
    Get detailed GPU information if available.

    Returns:
        Dictionary with GPU specs or None if no GPU available
    """
    if not torch.cuda.is_available():
        return None

    return {
        'name': torch.cuda.get_device_name(0),
        'total_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
        'cuda_version': torch.version.cuda,
        'device_count': torch.cuda.device_count(),
    }


def validate_audio_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate that an audio file exists and has a supported format.

    Args:
        file_path: Path to audio file

    Returns:
        Tuple of (is_valid, message)
    """
    supported_formats = ['.wav', '.m4a', '.mp3', '.flac', '.ogg']

    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"

    ext = Path(file_path).suffix.lower()
    if ext not in supported_formats:
        return False, f"Unsupported format: {ext}. Supported: {supported_formats}"

    return True, "Valid audio file"


def ensure_directory(directory: str) -> str:
    """
    Ensure a directory exists, create if it doesn't.

    Args:
        directory: Path to directory

    Returns:
        Absolute path to directory
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    return str(Path(directory).absolute())


def get_audio_duration(file_path: str) -> Optional[float]:
    """
    Get duration of audio file in seconds.

    Args:
        file_path: Path to audio file

    Returns:
        Duration in seconds or None if error
    """
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000.0  # Convert ms to seconds
    except Exception as e:
        print(f"Error getting audio duration: {e}")
        return None


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string like "2m 30s" or "1h 15m 30s"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


def clear_temp_folder(folder_path: str, extensions: Optional[list] = None):
    """
    Clear temporary files from a folder.

    Args:
        folder_path: Path to folder to clear
        extensions: List of file extensions to delete (e.g., ['.wav', '.tmp'])
                   If None, deletes all files
    """
    if not os.path.exists(folder_path):
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            if extensions is None:
                os.remove(file_path)
            else:
                if any(filename.endswith(ext) for ext in extensions):
                    os.remove(file_path)


# Example usage
if __name__ == "__main__":
    print("=== Device Detection ===")
    device = get_device()

    print("\n=== GPU Information ===")
    gpu_info = get_gpu_info()
    if gpu_info:
        print(f"Name: {gpu_info['name']}")
        print(f"Memory: {gpu_info['total_memory_gb']:.2f} GB")
        print(f"CUDA Version: {gpu_info['cuda_version']}")
    else:
        print("No GPU available")

    print("\n=== Utility Functions Test ===")
    print(f"Format 150s: {format_time(150)}")
    print(f"Format 3750s: {format_time(3750)}")
