# Docker Environments for Audio Transcription

This directory contains Docker configurations for running the audio transcription pipeline in isolated environments with proper GPU support.

---

## Available Environments

### 1. Main Pipeline (`transcription` service)
- **Purpose:** Run Whisper + Pyannote pipeline with verified CUDA PyTorch 2.5.1
- **Use for:** Production transcription, baseline tests, fine-tuning
- **PyTorch:** 2.5.1+cu121 (verified GPU support)

### 2. WhisperX (`whisperx` service)
- **Purpose:** Run WhisperX for advanced alignment and forced alignment
- **Use for:** Word-level timestamp alignment, WhisperX testing
- **PyTorch:** Latest compatible version (may be 2.8.0+ CPU or CUDA when available)

---

## Prerequisites

### Windows

1. **Install Docker Desktop for Windows**
   - Download from: https://www.docker.com/products/docker-desktop/
   - Enable WSL2 backend during installation

2. **Install NVIDIA Container Toolkit**
   ```powershell
   # In WSL2 Ubuntu terminal
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

3. **Verify GPU Access in Docker**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

---

## Usage

### Build Containers

```bash
# Navigate to project root
cd c:\Users\rpaul\Documents\GitHub\Audio_Transcription_Deidentification

# Build both containers
docker-compose -f docker/docker-compose.yml build

# Or build individually
docker-compose -f docker/docker-compose.yml build transcription
docker-compose -f docker/docker-compose.yml build whisperx
```

### Run Containers

```bash
# Start main transcription environment
docker-compose -f docker/docker-compose.yml run --rm transcription

# Start WhisperX environment
docker-compose -f docker/docker-compose.yml run --rm whisperx

# Or use interactive shell
docker-compose -f docker/docker-compose.yml run --rm transcription bash
```

### Inside Container

Once inside the container, you can run scripts:

```bash
# Verify GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run baseline test
python run_baseline_test.py

# Test WhisperX (in whisperx container)
python test_whisperx.py
```

---

## Example Workflows

### Workflow 1: Run GPU Baseline Test

```bash
# From Windows PowerShell/CMD
cd c:\Users\rpaul\Documents\GitHub\Audio_Transcription_Deidentification

# Run in main container
docker-compose -f docker/docker-compose.yml run --rm transcription python run_baseline_test.py
```

### Workflow 2: Test WhisperX Alignment

```bash
# Run in WhisperX container
docker-compose -f docker/docker-compose.yml run --rm whisperx python test_whisperx.py
```

### Workflow 3: Interactive Development

```bash
# Start container with shell
docker-compose -f docker/docker-compose.yml run --rm transcription bash

# Inside container, run multiple commands
python test_gpu.py
python run_baseline_test.py
python calculate_wer_simple.py
exit
```

---

## Volume Mounts

Both containers mount:
- **Project root** → `/workspace`
- **Audio files** → `/workspace/Audio_Local_tests`

Any files created in the container will appear in your Windows filesystem.

---

## GPU Memory Management

If you encounter out-of-memory errors:

```python
# Add to your scripts
import torch
torch.cuda.empty_cache()

# Or reduce batch size in configs
```

---

## Troubleshooting

### GPU Not Detected

```bash
# Check if NVIDIA runtime is available
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If this fails, reinstall NVIDIA Container Toolkit in WSL2
```

### Container Build Fails

```bash
# Clear Docker build cache
docker-compose -f docker/docker-compose.yml build --no-cache

# Or rebuild specific service
docker-compose -f docker/docker-compose.yml build --no-cache transcription
```

### Package Conflicts

Each container has its own isolated environment, so package conflicts are impossible between containers. If packages conflict within a single container, update the respective Dockerfile.

---

## Advantages of Docker Approach

1. **Environment Isolation:** Main pipeline and WhisperX can't conflict
2. **Reproducibility:** Exact environment captured in Dockerfiles
3. **GPU Support:** Both containers can access GPU independently
4. **Easy Cleanup:** Remove containers without affecting host system
5. **Version Control:** Dockerfiles can be versioned in Git
6. **Portability:** Works on Windows (WSL2), Linux, and cloud platforms

---

## File Structure

```
docker/
├── Dockerfile.main         # Main pipeline (PyTorch 2.5.1+cu121)
├── Dockerfile.whisperx     # WhisperX (latest compatible PyTorch)
├── docker-compose.yml      # Service definitions
└── README.md              # This file
```

---

## Next Steps

1. Install Docker Desktop and NVIDIA Container Toolkit
2. Build both containers
3. Test GPU access in each container
4. Run baseline tests in main container
5. Test WhisperX in whisperx container
6. Compare results and decide on approach
