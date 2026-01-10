# Training Data Directory

This directory stores raw training data before preprocessing.

## Structure

```
training_data/
├── gpt_test_aligned.json       # Aligned transcription data (in git)
├── gpt_test_metadata.json      # Metadata (in git)
├── *.json                      # Other JSON datasets (gitignored)
├── *.arrow                     # Arrow files (gitignored)
└── .gitkeep
```

## Important Notes

- **Large data files are NOT tracked in git**
- Small metadata and example files may be included
- Raw audio files should be in separate audio directories
- This contains text transcriptions and metadata

## Data Format

Training data is typically JSON with:
```json
{
  "audio_path": "path/to/audio.wav",
  "text": "transcription text",
  "speaker": "speaker_id",
  "duration": 10.5
}
```

## Creating Training Data

Use `src/data/create_training_data.py` to generate training datasets from:
- Raw transcriptions
- Audio files
- Speaker metadata

## Workflow

1. **Raw Data** → `training_data/` (this directory)
2. **Preprocessing** → `src/data/prepare_hf_dataset.py`
3. **Prepared Data** → `prepared_datasets/`
4. **Training** → Use with `src/training/finetune.py`
