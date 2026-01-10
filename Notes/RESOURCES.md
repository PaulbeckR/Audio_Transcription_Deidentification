# Project Resources & References

## Official Documentation

### Speech Recognition & Transcription

**Whisper (OpenAI)**
- GitHub: https://github.com/openai/whisper
- Paper: https://arxiv.org/abs/2212.04356
- Model Card: https://github.com/openai/whisper/blob/main/model-card.md
- Installation: `pip install openai-whisper`
- Models: tiny, base, small, medium, large, large-v2, large-v3
- License: MIT

**WhisperX**
- GitHub: https://github.com/m-bain/whisperX
- Paper: https://arxiv.org/abs/2303.00747
- Installation: `pip install whisperx`
- Features: Whisper + Forced Alignment + Speaker Diarization
- License: BSD-2-Clause

**Faster Whisper** (Future consideration)
- GitHub: https://github.com/guillaumekln/faster-whisper
- Description: Optimized Whisper implementation (CTranslate2)
- Installation: `pip install faster-whisper`
- Benefit: 4x faster, lower memory usage

### Speaker Diarization

**Pyannote Audio**
- GitHub: https://github.com/pyannote/pyannote-audio
- Documentation: https://pyannote.github.io/
- Paper: https://arxiv.org/abs/2104.04045
- Installation: `pip install pyannote.audio`
- Version: 3.3.2 (current)
- HuggingFace: Requires auth token for models (get from https://huggingface.co/settings/tokens)
- License: MIT

**Pyannote Metrics**
- GitHub: https://github.com/pyannote/pyannote-metrics
- Documentation: https://pyannote.github.io/pyannote-metrics/
- Installation: `pip install pyannote.metrics`
- Features: DER, JER, speaker confusion matrix

### Forced Alignment

**Montreal Forced Aligner (MFA)**
- GitHub: https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner
- Documentation: https://montreal-forced-aligner.readthedocs.io/
- Installation: Binary download or conda
- Use Case: High-quality phoneme-level alignment
- License: MIT

**Gentle** (Alternative)
- GitHub: https://github.com/lowerquality/gentle
- Description: Kaldi-based forced aligner
- Installation: Docker or from source
- Use Case: Simpler alignment tasks

### Evaluation Metrics

**jiwer** (WER/CER)
- GitHub: https://github.com/jitsi/jiwer
- Documentation: https://jitsi.github.io/jiwer/
- Installation: `pip install jiwer`
- Features: WER, CER, MER, WIL
- License: Apache 2.0

**evaluate** (HuggingFace)
- GitHub: https://github.com/huggingface/evaluate
- Documentation: https://huggingface.co/docs/evaluate/
- Installation: `pip install evaluate`
- Features: Wide range of NLP metrics including WER
- License: Apache 2.0

---

## Audio Processing Libraries

**pydub**
- GitHub: https://github.com/jiaaro/pydub
- Documentation: http://pydub.com/
- Installation: `pip install pydub`
- Requires: ffmpeg
- Use: Audio manipulation, format conversion

**ffmpeg**
- Website: https://ffmpeg.org/
- Installation (Windows): Download binary or `pip install ffmpeg-python`
- Use: Audio/video processing backend

**librosa**
- GitHub: https://github.com/librosa/librosa
- Documentation: https://librosa.org/
- Installation: `pip install librosa`
- Use: Audio analysis, feature extraction

**soundfile**
- GitHub: https://github.com/bastibe/python-soundfile
- Installation: `pip install soundfile`
- Use: Reading/writing audio files

---

## GPU & Deep Learning

**PyTorch**
- Website: https://pytorch.org/
- Installation: https://pytorch.org/get-started/locally/
- CUDA Version: Check with `nvidia-smi`
- Current Install: `torch==2.4.1`
- Verify GPU: `torch.cuda.is_available()`

**CUDA Toolkit**
- Download: https://developer.nvidia.com/cuda-downloads
- Documentation: https://docs.nvidia.com/cuda/
- Check Version: `nvidia-smi` or `nvcc --version`

**cuDNN** (NVIDIA)
- Download: https://developer.nvidia.com/cudnn
- Documentation: https://docs.nvidia.com/deeplearning/cudnn/
- Note: Usually installed with PyTorch

---

## Alternative Frameworks (Future Exploration)

**NVIDIA NeMo**
- GitHub: https://github.com/NVIDIA/NeMo
- Documentation: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/
- Installation: `pip install nemo_toolkit[asr]`
- Features: GPU-optimized ASR, speaker diarization
- Use Case: Maximum GPU performance

**Kaldi**
- Website: https://kaldi-asr.org/
- GitHub: https://github.com/kaldi-asr/kaldi
- Installation: Compile from source
- Use Case: Research, highly customizable
- Note: Steep learning curve

**SpeechBrain**
- GitHub: https://github.com/speechbrain/speechbrain
- Documentation: https://speechbrain.github.io/
- Installation: `pip install speechbrain`
- Features: End-to-end speech processing
- Use Case: Research and experimentation

**ESPnet**
- GitHub: https://github.com/espnet/espnet
- Documentation: https://espnet.github.io/espnet/
- Installation: `pip install espnet`
- Features: End-to-end speech processing toolkit

---

## Academic Papers & Research

### Speech Recognition

1. **Robust Speech Recognition via Large-Scale Weak Supervision** (Whisper)
   - Authors: Radford et al., OpenAI
   - Year: 2022
   - Link: https://arxiv.org/abs/2212.04356
   - Key Insight: Large-scale weakly supervised training enables robust ASR

2. **WhisperX: Time-Accurate Speech Transcription**
   - Authors: Bain et al.
   - Year: 2023
   - Link: https://arxiv.org/abs/2303.00747
   - Key Insight: Forced alignment improves timestamp accuracy

### Speaker Diarization

3. **Pyannote.audio: Neural Building Blocks for Speaker Diarization**
   - Authors: Bredin et al.
   - Year: 2020
   - Link: https://arxiv.org/abs/2104.04045
   - Key Insight: End-to-end neural diarization pipeline

4. **Recent Methods in Speaker Segmentation Tasks**
   - Book Chapter: Springer
   - Year: 2024
   - Link: https://link.springer.com/chapter/10.1007/978-3-031-70259-4_21
   - Key Insight: Survey of modern segmentation algorithms
   - Notes: Referenced in Methods_Algorithms.md

### Fine-Tuning & Adaptation

5. **Fine-Tuning Whisper for Multilingual ASR**
   - Multiple sources, HuggingFace blog
   - Link: https://huggingface.co/blog/fine-tune-whisper
   - Key Insight: Domain adaptation significantly improves accuracy

---

## Tutorials & Guides

### Whisper Fine-Tuning
- HuggingFace Guide: https://huggingface.co/blog/fine-tune-whisper
- Google Colab Notebooks: https://github.com/openai/whisper/discussions
- Reddit r/MachineLearning: Active community discussions

### Pyannote Usage
- Official Tutorials: https://github.com/pyannote/pyannote-audio/tree/develop/tutorials
- Speaker Diarization: https://huggingface.co/pyannote/speaker-diarization
- Custom Training: https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/training.md

### WhisperX
- Quick Start: https://github.com/m-bain/whisperX#quick-start
- Alignment Guide: See repository README
- Integration Examples: Repository issues/discussions

---

## Datasets (For Reference)

**Note**: We use proprietary interview data, but these are useful references:

**LibriSpeech**
- Link: http://www.openslr.org/12
- Description: 1000 hours of English speech
- Use: ASR benchmarking

**Common Voice** (Mozilla)
- Link: https://commonvoice.mozilla.org/
- Description: Multilingual crowd-sourced speech
- Use: Diverse accent/speaker data

**VoxCeleb**
- Link: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/
- Description: Speaker recognition dataset
- Use: Diarization benchmarking

---

## Tools & Utilities

### System Monitoring

**psutil**
- GitHub: https://github.com/giampaolo/psutil
- Documentation: https://psutil.readthedocs.io/
- Installation: `pip install psutil`
- Use: CPU, memory, disk monitoring

**nvidia-smi**
- Included with NVIDIA drivers
- Command: `nvidia-smi` or `watch -n 1 nvidia-smi`
- Use: GPU monitoring

**GPUtil** (Python wrapper)
- GitHub: https://github.com/anderskm/gputil
- Installation: `pip install gputil`
- Use: GPU metrics in Python

### Configuration Management

**PyYAML**
- Website: https://pyyaml.org/
- Installation: `pip install pyyaml`
- Use: Configuration files

**python-dotenv**
- GitHub: https://github.com/theskumar/python-dotenv
- Installation: `pip install python-dotenv`
- Use: Environment variables

### Data Processing

**pandas**
- Documentation: https://pandas.pydata.org/
- Already installed: `pandas==2.2.3`
- Use: Data manipulation, CSV/Excel handling

**openpyxl**
- GitHub: https://foss.heptapod.net/openpyxl/openpyxl
- Already installed (in one_and_done.py)
- Use: Excel file handling

---

## Community Resources

### Forums & Discussion

**Reddit**
- r/MachineLearning: https://www.reddit.com/r/MachineLearning/
- r/LanguageTechnology: https://www.reddit.com/r/LanguageTechnology/
- r/speechrecognition: https://www.reddit.com/r/speechrecognition/

**Stack Overflow**
- Tag: [whisper]
- Tag: [speech-recognition]
- Tag: [speaker-diarization]

**GitHub Discussions**
- Whisper: https://github.com/openai/whisper/discussions
- Pyannote: https://github.com/pyannote/pyannote-audio/discussions
- WhisperX: https://github.com/m-bain/whisperX/discussions

### Discord/Slack
- HuggingFace Discord: https://discord.gg/hugging-face
- Pyannote: Community via GitHub

---

## Installation Commands Reference

### Week 1 Installations

```bash
# WER calculation
pip install jiwer

# WhisperX (includes dependencies)
pip install whisperx

# Metrics (if not already installed)
pip install pyannote.metrics

# System monitoring
pip install psutil

# Configuration (for Week 2)
pip install pyyaml
```

### Verify GPU Setup

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### Check Package Versions

```bash
pip list | grep -E "whisper|torch|pyannote|jiwer"
```

---

## Troubleshooting Resources

### Common Issues

**CUDA Out of Memory**
- Solution: Reduce batch size, use smaller model
- Reference: https://pytorch.org/docs/stable/notes/cuda.html

**Pyannote Authentication**
- Solution: Get HuggingFace token
- Reference: https://huggingface.co/pyannote/speaker-diarization

**FFmpeg Not Found**
- Windows: Download from https://ffmpeg.org/download.html
- Add to PATH or specify location
- Reference: https://github.com/jiaaro/pydub#getting-ffmpeg-set-up

**WhisperX Installation Fails**
- Check CUDA compatibility
- Install dependencies separately
- Reference: https://github.com/m-bain/whisperX/issues

---

## Fine-Tuning & Training Resources

**HuggingFace Training**
- **Transformers**: https://huggingface.co/docs/transformers/
- **Seq2SeqTrainer**: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainer
- **PEFT (LoRA)**: https://huggingface.co/docs/peft/
- **Datasets**: https://huggingface.co/docs/datasets/
- **Evaluate**: https://huggingface.co/docs/evaluate/

**LoRA (Low-Rank Adaptation)**
- Paper: https://arxiv.org/abs/2106.09685
- GitHub: https://github.com/microsoft/LoRA
- Implementation: PEFT library
- Use Case: Parameter-efficient fine-tuning (reduces trainable params by 10,000x)

**TensorBoard**
- Documentation: https://www.tensorflow.org/tensorboard
- PyTorch Integration: https://pytorch.org/docs/stable/tensorboard.html
- Installation: `pip install tensorboard`
- Use: Training visualization, metric tracking

**Accelerate** (HuggingFace)
- GitHub: https://github.com/huggingface/accelerate
- Documentation: https://huggingface.co/docs/accelerate/
- Installation: `pip install accelerate`
- Features: Multi-GPU, mixed precision, gradient accumulation

---

## Internal Project Documentation

### Main Documentation (Project Root)
- **[README.md](../README.md)** - Complete project overview and quick start
- **[WORKSPACE_READY.md](../WORKSPACE_READY.md)** - Workspace status and setup verification
- **[TRAINING_GUIDE.md](../TRAINING_GUIDE.md)** - Complete fine-tuning workflow
- **[DATA_OVERVIEW.md](../DATA_OVERVIEW.md)** - Dataset information and statistics
- **[PREPARE_YOUR_DATA.md](../PREPARE_YOUR_DATA.md)** - Data preparation guide

### Configuration Documentation
- **[config/README.md](../config/README.md)** - Configuration file reference
- **[config/paths.yaml](../config/paths.yaml)** - File paths and directories
- **[config/training.yaml](../config/training.yaml)** - Training hyperparameters
- **[config/experiments.yaml](../config/experiments.yaml)** - Experiment definitions

### Setup & Installation (Notes/Setup/)
- **[INSTALLATION_GUIDE.md](Setup/INSTALLATION_GUIDE.md)** - Detailed installation instructions
- **[GPU_SETUP.md](Setup/GPU_SETUP.md)** - GPU configuration and setup
- **[DUAL_ENVIRONMENT_SETUP.md](Setup/DUAL_ENVIRONMENT_SETUP.md)** - Managing multiple Python environments
- **[WHISPERX_INSTALLATION_ISSUE.md](Setup/WHISPERX_INSTALLATION_ISSUE.md)** - WhisperX installation troubleshooting

### Week 1: Model Comparisons (Notes/Week1-ModelComparisons/)
- **[WEEK1_FINAL_REPORT.md](Week1-ModelComparisons/WEEK1_FINAL_REPORT.md)** - Complete Week 1 results
- **[GPU_BASELINE_RESULTS.md](Week1-ModelComparisons/GPU_BASELINE_RESULTS.md)** - GPU baseline performance
- **[BASELINE_CPU_METRICS.md](Week1-ModelComparisons/BASELINE_CPU_METRICS.md)** - CPU baseline performance
- **[WHISPERX_EVALUATION.md](Week1-ModelComparisons/WHISPERX_EVALUATION.md)** - WhisperX evaluation results
- **[MODEL_COMPARISON.md](Week1-ModelComparisons/MODEL_COMPARISON.md)** - Model comparison analysis
- **[TORCH_UTILS_MODULE.md](Week1-ModelComparisons/TORCH_UTILS_MODULE.md)** - PyTorch utilities documentation

### Week 2: Fine-Tuning & Reorganization (Notes/Week2-FinetuneAndReorg/)
- **[WEEK2_PLAN.md](Week2-FinetuneAndReorg/WEEK2_PLAN.md)** - Week 2 objectives
- **[FINETUNING_RESULTS.md](Week2-FinetuneAndReorg/FINETUNING_RESULTS.md)** - Fine-tuning experiment results
- **[FINETUNING_GUIDE.md](Week2-FinetuneAndReorg/FINETUNING_GUIDE.md)** - Fine-tuning process documentation
- **[REORGANIZATION_COMPLETE.md](Week2-FinetuneAndReorg/REORGANIZATION_COMPLETE.md)** - Codebase reorganization summary
- **[REORGANIZATION_PLAN.md](Week2-FinetuneAndReorg/REORGANIZATION_PLAN.md)** - Reorganization planning
- **[CODEBASE_REORG_PLAN.md](Week2-FinetuneAndReorg/CODEBASE_REORG_PLAN.md)** - Detailed reorganization plan
- **[PRETRAINING_EXPERIMENT_PLAN.md](Week2-FinetuneAndReorg/PRETRAINING_EXPERIMENT_PLAN.md)** - Experiment planning

### Troubleshooting (Notes/Troubleshooting/)
- **[TORCHCODEC_SOLUTION.md](Troubleshooting/TORCHCODEC_SOLUTION.md)** - Torchcodec dependency resolution
- **[CODEBASE_ISSUES.md](Troubleshooting/CODEBASE_ISSUES.md)** - Known issues and fixes

### Project Planning & Status (Notes/)
- **[PROJECT_PLAN.md](PROJECT_PLAN.md)** - Overall project roadmap
- **[CURRENT_STATUS.md](CURRENT_STATUS.md)** - Current project status
- **[TODO.md](TODO.md)** - Task list and priorities
- **[QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)** - Quick start for new users
- **[Methods_Algorithms.md](Methods_Algorithms.md)** - Technical background and algorithms
- **[Ideas_notes.md](Ideas_notes.md)** - Research notes and discoveries
- **[File_Structure.md](File_Structure.md)** - Codebase organization
- **[REAL_DATA_INTEGRATION.md](REAL_DATA_INTEGRATION.md)** - Integrating production data

---

## Updates & Maintenance

**Last Updated**: 2026-01-09

**Project Status**:
- **Version**: 1.4.25 (Post-Reorganization)
- **Python**: 3.10+
- **PyTorch**: 2.4.1
- **CUDA**: 12.1 (if using GPU)
- **Phase**: Fine-tuning & Optimization (Week 2-3)

**Recent Changes**:
- ✅ Codebase reorganization completed (Dec 2025 - Jan 2026)
- ✅ Custom dataset pipeline implemented (avoids torchcodec dependency)
- ✅ LoRA fine-tuning pipeline ready
- ✅ TensorBoard integration configured
- ✅ Comprehensive documentation added

**Version Check Schedule**: Monthly
- Update requirements.txt with latest stable versions
- Check for breaking changes in dependencies
- Review new features in Whisper, Pyannote, WhisperX

**Known Issues**:
- `predict_with_generate=True` flag may cause input_ids error (under investigation)
- See [Notes/Troubleshooting/CODEBASE_ISSUES.md](Troubleshooting/CODEBASE_ISSUES.md) for details

**Deprecated/Legacy**:
- Old codebase moved to `old_code/` directory
- `finetune_whisper_simple.py` (Dec 29, 2025) - kept as reference for working implementation
