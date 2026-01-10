# Models Directory

This directory stores fine-tuned Whisper models and their checkpoints.

## Structure

Models are organized by training run:
```
models/
├── whisper-finetuned-[name]/
│   ├── checkpoint-[N]/      # Training checkpoints (gitignored)
│   ├── final_model/          # Final trained model (gitignored)
│   └── logs/                 # TensorBoard logs (gitignored)
└── .gitkeep
```

## Important Notes

- **Model files are NOT tracked in git** - They are too large (100MB+)
- Only this README and .gitkeep are version controlled
- Store models locally or in cloud storage (S3, GCS, etc.)
- Model configurations are in `config/models.yaml`

## Model File Types (Not in Git)

- `*.pt` - PyTorch checkpoint files (optimizer state, scheduler, etc.)
- `*.safetensors` - Model weights in SafeTensors format
- `*.bin` - Binary model files
- `*.json` - Model configuration files
- `vocab.json`, `merges.txt` - Tokenizer vocabulary files
- TensorBoard event files

## Reproducibility

To reproduce training:
1. Use the configuration files in `config/`
2. Run training scripts in `src/training/`
3. Models will be saved to this directory automatically
4. Keep local backups of important model checkpoints
