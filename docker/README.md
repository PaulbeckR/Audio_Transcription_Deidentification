# Docker Environments for Audio Transcription

Isolated Docker environments for running the audio transcription pipeline with proper GPU support and dependency separation.

**Last Updated:** 2026-01-09

---

## Overview

This project provides three specialized Docker environments, each optimized for different workflows. Using Docker ensures consistent environments, prevents package conflicts, and provides reliable GPU support across different systems.

---

## Available Environments

### 1. Main Pipeline (`transcription` service)
**Purpose:** Baseline transcription with Whisper + Pyannote

**Details:**
- **PyTorch:** 2.4.1+cu121 (tested and verified)
- **Base Image:** `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- **Container:** `transcription-main`
- **Use for:** Production transcription, baseline tests, evaluation

### 2. WhisperX (`whisperx` service)
**Purpose:** Word-level alignment and training data preparation

**Details:**
- **PyTorch:** 2.4.1+cu121 (WhisperX compatible)
- **Base Image:** `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- **Container:** `transcription-whisperx`
- **Use for:** Forced alignment, creating training datasets

### 3. Fine-Tuning (`finetuning` service) ✨ NEW
**Purpose:** Whisper fine-tuning with LoRA

**Details:**
- **PyTorch:** 2.4.1+cu121
- **Base Image:** `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04` (includes CUDA dev tools)
- **Container:** `transcription-finetuning`
- **Ports:** 6006 (TensorBoard)
- **Use for:** Model training, hyperparameter optimization, TensorBoard monitoring

---

## Prerequisites

### Windows (WSL2)

1. **Install Docker Desktop**
   - Download from: https://www.docker.com/products/docker-desktop/
   - Enable WSL2 backend during installation
   - Ensure Docker is running

2. **Install NVIDIA Container Toolkit (in WSL2 Ubuntu)**
   ```bash
   # Get distribution info
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

   # Add NVIDIA GPG key
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
     sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

   # Add NVIDIA repository
   curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

   # Install toolkit
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

3. **Verify GPU Access**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

### Linux

1. **Install Docker Engine**
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER
   # Log out and back in for group changes to take effect
   ```

2. **Install NVIDIA Container Toolkit** (same as Windows WSL2 above)

---

## Setup

### 1. Configure Environment Variables

```bash
# Copy example environment file
cp docker/.env.example docker/.env

# Edit with your HuggingFace token
nano docker/.env
```

**.env file:**
```bash
PYANNOTE_TOKEN=hf_your_actual_token_here
```

Get your token from: https://huggingface.co/settings/tokens

### 2. Build Containers

```bash
# Navigate to project root
cd Audio_Transcription_Deidentification

# Build all containers (takes 10-15 minutes first time)
docker-compose -f docker/docker-compose.yml build

# Or build individually
docker-compose -f docker/docker-compose.yml build transcription
docker-compose -f docker/docker-compose.yml build whisperx
docker-compose -f docker/docker-compose.yml build finetuning
```

---

## Usage

### Starting Containers

```bash
# Main transcription environment
docker-compose -f docker/docker-compose.yml run --rm transcription

# WhisperX environment
docker-compose -f docker/docker-compose.yml run --rm whisperx

# Fine-tuning environment
docker-compose -f docker/docker-compose.yml run --rm finetuning
```

### Verification Commands (inside container)

```bash
# Check GPU availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Check CUDA version
python -c "import torch; print(f'CUDA Version: {torch.version.cuda}')"

# Check GPU details
nvidia-smi

# Check PyTorch version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

---

## Workflows

### Workflow 1: Run Baseline Transcription

```bash
# Start main container
docker-compose -f docker/docker-compose.yml run --rm transcription

# Inside container - run baseline test
python run_baseline_test.py

# Exit container
exit
```

### Workflow 2: Create Training Data

```bash
# Start WhisperX container
docker-compose -f docker/docker-compose.yml run --rm whisperx

# Create aligned training data
python src/data/create_training_data.py \
  --audio-dir Audio_Local_tests/audio_files \
  --transcript-dir Audio_Local_tests/transcription_test \
  --output training_data/aligned/test

# Prepare dataset for training
python src/data/prepare_hf_dataset.py \
  --input training_data/aligned/test \
  --output prepared_datasets/test_dataset \
  --train-split 0.9

exit
```

### Workflow 3: Fine-Tune Whisper

```bash
# Start fine-tuning container
docker-compose -f docker/docker-compose.yml run --rm finetuning

# List available experiments
python src/scripts/run_training.py --list

# Run quick test (1 epoch)
python src/scripts/run_training.py --experiment quick_test

# Run full baseline training (5 epochs)
python src/scripts/run_training.py --experiment baseline_lora

exit
```

### Workflow 4: Monitor Training with TensorBoard

```bash
# Start container with TensorBoard port exposed
docker-compose -f docker/docker-compose.yml run --rm finetuning bash

# Inside container - start TensorBoard in background
tensorboard --logdir models/whisper-finetuned --host 0.0.0.0 --port 6006 &

# Start training
python src/scripts/run_training.py --experiment baseline_lora

# On host machine, open browser to:
# http://localhost:6006
```

### Workflow 5: One-Command Execution

```bash
# Run without entering container shell
docker-compose -f docker/docker-compose.yml run --rm transcription \
  python run_baseline_test.py

# Run training from host
docker-compose -f docker/docker-compose.yml run --rm finetuning \
  python src/scripts/run_training.py --experiment quick_test
```

---

## Volume Mounts

All containers mount project directories to persist data:

| Container | Mounted Volumes |
|-----------|----------------|
| **Main** | Project root, Audio_Local_tests, models, .cache |
| **WhisperX** | Project root, Audio_Local_tests, training_data, .cache |
| **Fine-Tuning** | Project root, prepared_datasets, models, .cache |

**All files created in containers persist on your host filesystem.**

---

## GPU Memory Management

### Monitor GPU Usage

```bash
# From host machine (in another terminal)
watch -n 1 nvidia-smi

# Inside container
nvidia-smi -l 1  # Updates every 1 second
```

### If Out-of-Memory Errors Occur

**Option 1:** Clear GPU cache in your Python script
```python
import torch
torch.cuda.empty_cache()

# Check memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

**Option 2:** Reduce batch size in `config/training.yaml`
```yaml
hyperparameters:
  batch_size: 1  # Reduce from 2
  gradient_accumulation_steps: 16  # Double to maintain effective batch size
```

---

## Advantages of Docker Approach

| Benefit | Description |
|---------|-------------|
| ✅ **Environment Isolation** | Main, WhisperX, and fine-tuning can't conflict |
| ✅ **Reproducibility** | Exact environment captured in version-controlled Dockerfiles |
| ✅ **GPU Support** | All containers access GPU independently |
| ✅ **Easy Cleanup** | Remove containers without affecting host system |
| ✅ **Portability** | Works on Windows (WSL2), Linux, and cloud platforms |
| ✅ **Dependency Management** | No virtual environment confusion or package conflicts |
| ✅ **Consistent Python** | Same Python 3.10 across all environments |
| ✅ **Team Collaboration** | Everyone uses identical environments |

---

## Troubleshooting

### GPU Not Detected

```bash
# Test NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If fails, check Docker daemon config
cat /etc/docker/daemon.json
# Should contain: {"default-runtime": "nvidia"}

# Reinstall NVIDIA Container Toolkit (see Prerequisites)
```

### Container Build Fails

```bash
# Clear Docker build cache
docker-compose -f docker/docker-compose.yml build --no-cache

# Rebuild specific service only
docker-compose -f docker/docker-compose.yml build --no-cache finetuning

# Check Docker disk space
docker system df

# WARNING: Remove all unused images (frees space)
docker system prune -a
```

### Package Import Errors

```bash
# Inside container, verify installation
pip list | grep transformers
pip list | grep whisperx
pip list | grep torch

# Manually install if missing
pip install transformers whisperx peft
```

### Permission Errors on Created Files

```bash
# On host, fix ownership of files created by container
sudo chown -R $USER:$USER prepared_datasets/
sudo chown -R $USER:$USER models/
sudo chown -R $USER:$USER training_data/
```

### TensorBoard Not Accessible

```bash
# Ensure TensorBoard is running on correct host
tensorboard --logdir models/whisper-finetuned --host 0.0.0.0 --port 6006

# Verify port forwarding
docker ps
# Should show: 0.0.0.0:6006->6006/tcp

# Check if port is blocked by firewall
curl http://localhost:6006
```

### Container Can't Find Project Files

```bash
# Verify you're running from project root
pwd  # Should end with Audio_Transcription_Deidentification

# Check volume mounts are correct
docker inspect transcription-finetuning | grep Mounts -A 20
```

---

## Environment Comparison

| Feature | Main | WhisperX | Fine-Tuning |
|---------|------|----------|-------------|
| **Python** | 3.10 | 3.10 | 3.10 |
| **PyTorch** | 2.4.1 | 2.4.1 | 2.4.1 |
| **CUDA** | 12.1 (runtime) | 12.1 (runtime) | 12.1 (devel) |
| **Whisper** | ✅ | ✅ | ✅ |
| **WhisperX** | ❌ | ✅ | ✅ |
| **Pyannote** | ✅ | ✅ | ✅ |
| **Transformers** | ✅ | ✅ | ✅ |
| **PEFT (LoRA)** | ❌ | ❌ | ✅ |
| **Accelerate** | ❌ | ❌ | ✅ |
| **TensorBoard** | ❌ | ❌ | ✅ |
| **Evaluate (WER)** | ❌ | ❌ | ✅ |
| **Datasets** | ❌ | ❌ | ✅ |
| **Primary Use** | Baseline | Data Prep | Training |

---

## Best Practices

### 1. Use the Right Container

- **Testing baseline model** → `transcription`
- **Creating training data** → `whisperx`
- **Fine-tuning model** → `finetuning`

### 2. Cache Management

HuggingFace models are cached in `.cache/huggingface/` and persist across container restarts. First download is slow, subsequent runs are fast.

### 3. Resource Monitoring

Always monitor GPU usage in a separate terminal:
```bash
watch -n 1 nvidia-smi
```

### 4. Regular Cleanup

```bash
# Stop all running containers
docker-compose -f docker/docker-compose.yml down

# Remove unused images (frees disk space)
docker image prune

# Remove build cache
docker builder prune
```

### 5. Development Workflow

For iterative development, keep the container running:
```bash
# Start with shell
docker-compose -f docker/docker-compose.yml run --rm finetuning bash

# Run multiple commands without restarting container
python src/scripts/run_training.py --list
python src/scripts/run_training.py --experiment quick_test
tensorboard --logdir models/whisper-finetuned

# Exit when done
exit
```

---

## File Structure

```
docker/
├── Dockerfile.main         # Main pipeline (baseline transcription)
├── Dockerfile.whisperx     # WhisperX (alignment, data preparation)
├── Dockerfile.finetuning   # Fine-tuning (LoRA, TensorBoard)
├── docker-compose.yml      # Service orchestration
├── .env.example            # Environment variable template
└── README.md              # This file
```

---

## Next Steps

1. ✅ Install Docker Desktop + NVIDIA Container Toolkit
2. ✅ Create `.env` file with Pyannote token
3. ✅ Build all three containers
4. ✅ Test GPU access in each container
5. ✅ Run baseline test in main container
6. ✅ Create training data in WhisperX container
7. ✅ Fine-tune model in fine-tuning container
8. ✅ Monitor training with TensorBoard

---

## Additional Resources

- **Project README:** [../README.md](../README.md)
- **Training Guide:** [../TRAINING_GUIDE.md](../TRAINING_GUIDE.md)
- **Data Preparation:** [../PREPARE_YOUR_DATA.md](../PREPARE_YOUR_DATA.md)
- **Configuration Reference:** [../config/README.md](../config/README.md)
- **Setup Guides:** [../Notes/Setup/](../Notes/Setup/)
- **Troubleshooting:** [../Notes/Troubleshooting/](../Notes/Troubleshooting/)

**External Links:**
- Docker Documentation: https://docs.docker.com/
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/
- Docker Compose: https://docs.docker.com/compose/

---

**Questions or Issues?**

Check the troubleshooting section above, or see:
- [../Notes/Troubleshooting/CODEBASE_ISSUES.md](../Notes/Troubleshooting/CODEBASE_ISSUES.md)
- [../Notes/RESOURCES.md](../Notes/RESOURCES.md)
