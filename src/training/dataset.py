"""
Custom PyTorch Dataset for Whisper fine-tuning without HuggingFace Audio feature.

This avoids the torchcodec dependency introduced in datasets>=4.0.0 by loading
audio on-the-fly using whisperx, similar to the successful approach from
finetune_whisper_simple.py (Dec 29, 2025).
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import json


class WhisperAudioDataset(Dataset):
    """
    Custom dataset that loads audio on-the-fly without HuggingFace Audio feature.

    This avoids the torchcodec dependency and provides more control over audio loading.
    Uses whisperx.load_audio() for consistency with alignment pipeline.

    Args:
        examples: List of dicts with 'audio_path' and 'text' keys
        processor: Whisper processor (tokenizer + feature extractor)
        max_length: Maximum audio length in seconds (default: 30)
    """

    def __init__(self, examples, processor, max_length=30):
        self.examples = examples
        self.processor = processor
        self.max_length = max_length
        self.sampling_rate = 16000  # WhisperX standard

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Lazy import whisperx (only when actually loading audio)
        try:
            import whisperx
        except ImportError as e:
            raise ImportError(
                "The 'whisperx' library is required for audio loading.\n"
                "Install with: pip install -r requirements_finetuning.txt\n"
                f"Original error: {e}"
            )

        # Load audio using whisperx (same as alignment pipeline)
        audio_path = example['audio_path']
        audio = whisperx.load_audio(audio_path)

        # Trim to max length if needed (30 seconds = 480,000 samples at 16kHz)
        max_samples = int(self.max_length * self.sampling_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # Process audio with Whisper feature extractor (returns numpy array)
        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=self.sampling_rate
        ).input_features[0]

        # Tokenize text
        labels = self.processor.tokenizer(example['text']).input_ids

        # Convert to tensors (matching old working code)
        return {
            'input_features': torch.tensor(input_features),
            'labels': torch.tensor(labels)
        }


class WhisperDataCollator:
    """
    Custom data collator for batching Whisper training data.

    Handles padding of input_features and labels to create uniform batches.

    Args:
        processor: Whisper processor
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        """Custom data collator - matches old working code exactly."""
        # Extract features
        input_features = [f['input_features'] for f in features]
        labels = [f['labels'] for f in features]

        # Pad input features if needed
        max_input_length = max(f.shape[0] for f in input_features)

        batch_input_features = []
        for f in input_features:
            if f.shape[0] < max_input_length:
                # Pad with zeros
                padding = torch.zeros(max_input_length - f.shape[0], f.shape[1])
                f = torch.cat([f, padding], dim=0)
            batch_input_features.append(f)

        # Pad labels
        max_label_length = max(len(l) for l in labels)

        batch_labels = []
        for l in labels:
            if len(l) < max_label_length:
                # Pad with -100 (ignored in loss)
                padding = torch.full((max_label_length - len(l),), -100, dtype=torch.long)
                l = torch.cat([l, padding], dim=0)
            batch_labels.append(l)
        #print("Batch input features shape:", torch.stack(batch_input_features).shape   )
        #print("Batch input features example:", torch.stack(batch_input_features))
        batch = {
            'input_features': torch.stack(batch_input_features),
            'labels': torch.stack(batch_labels)
        }

        #print("COLLATOR RETURN KEYS:", batch.keys())
        return batch


def load_json_dataset(json_path: Path):
    """
    Load dataset from JSON file containing audio paths and text.

    Expected JSON format:
    [
        {"audio_path": "path/to/audio1.wav", "text": "transcription 1"},
        {"audio_path": "path/to/audio2.wav", "text": "transcription 2"},
        ...
    ]

    Args:
        json_path: Path to JSON file

    Returns:
        List of example dicts
    """
    print(f"\nLoading dataset from: {json_path}")

    if not json_path.exists():
        raise FileNotFoundError(f"Dataset not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        examples = json.load(f)

    print(f"  Loaded {len(examples)} examples")

    # Validate format
    if examples and isinstance(examples, list):
        required_keys = {'audio_path', 'text'}
        if not all(required_keys.issubset(ex.keys()) for ex in examples):
            raise ValueError(f"Each example must have keys: {required_keys}")

    return examples


def load_simple_dataset_dir(dataset_dir: Path):
    """
    Load train and optional validation datasets from directory.

    Expected structure:
        dataset_dir/
            train.json
            validation.json (optional)

    Args:
        dataset_dir: Path to dataset directory

    Returns:
        Dict with 'train' and optional 'validation' example lists
    """
    train_path = dataset_dir / "train.json"
    val_path = dataset_dir / "validation.json"

    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")

    train_examples = load_json_dataset(train_path)

    dataset = {'train': train_examples}

    if val_path.exists():
        val_examples = load_json_dataset(val_path)
        dataset['validation'] = val_examples
    else:
        print("  No validation dataset found")

    return dataset
