# RunPod Training Guide for EdgeMoE

Complete guide to train your router and LoRA experts on RunPod GPUs.

## Overview

**What we're training:**
1. **Router** (~2-3 hours on A40): Domain classifier on 87K samples
2. **3 LoRA Experts** (~4-6 hours each on A40): Code, General, Research adapters

**Total time:** ~14-20 hours of GPU time
**Estimated cost:** $15-30 (using A40 at ~$0.79/hour)

---

## Prerequisites

### 1. Create RunPod Account
- Go to [runpod.io](https://www.runpod.io)
- Sign up and add credits ($20-30 recommended)

### 2. Prepare Your Data Locally
Your data is already ready in `data/processed/`:
```
data/processed/
├── code/
│   ├── train.jsonl (31,418 samples)
│   ├── val.jsonl (6,732 samples)
│   └── test.jsonl (6,734 samples)
├── general/
│   ├── train.jsonl (35,000 samples)
│   ├── val.jsonl (7,500 samples)
│   └── test.jsonl (7,500 samples)
└── research/
    ├── train.jsonl (21,000 samples)
    ├── val.jsonl (4,500 samples)
    └── test.jsonl (4,500 samples)
```

---

## Step 1: Create RunPod Pod

### 1.1 Choose Template
- Go to **Pods** → **Deploy**
- Select **PyTorch 2.4 (CUDA 12.4)** template
  - Or use: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`

### 1.2 Select GPU
**Recommended options:**
- **Budget**: RTX 4090 (~$0.44/hr) - Good for testing
- **Recommended**: A40 (~$0.79/hr) - Best price/performance
- **Fast**: A100 (~$1.89/hr) - Fastest training

### 1.3 Configure Pod
- **Container Disk**: 50 GB (for models)
- **Volume Disk**: 100 GB (for data - persistent)
- **Expose HTTP**: Enable (for Jupyter)
- **Expose TCP**: 22 (for SSH)

### 1.4 Deploy
- Click **Deploy On-Demand**
- Wait ~1-2 minutes for pod to start

---

## Step 2: Upload Data to RunPod

### Method A: Using RunPod Web Terminal (Easiest)

1. **Connect to Pod**
   - Click **Connect** → **Start Web Terminal**

2. **Create directories**
   ```bash
   cd /workspace
   mkdir -p edgemoe/data/processed
   mkdir -p edgemoe/runpod
   mkdir -p edgemoe/src
   ```

3. **Upload via File Upload** (if small files)
   - Use web terminal's upload button
   - Upload each JSONL file to `/workspace/edgemoe/data/processed/`

### Method B: Using GitHub (Recommended - Fastest)

1. **Push your data to GitHub** (from local machine):
   ```bash
   # In your edge_v2 directory
   git init
   git add data/processed runpod src
   git commit -m "Add training data and scripts"
   git push origin main
   ```

2. **Clone on RunPod**:
   ```bash
   cd /workspace
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git edgemoe
   cd edgemoe
   ```

### Method C: Using runpodctl (For Large Files)

1. **Install runpodctl locally**:
   ```bash
   # On your local Windows machine
   winget install runpod.runpodctl
   ```

2. **Configure API key**:
   - Get API key from RunPod → Settings → API Keys
   ```bash
   runpodctl config --apiKey YOUR_API_KEY
   ```

3. **Upload data**:
   ```bash
   runpodctl send data/processed/ YOUR_POD_ID:/workspace/edgemoe/data/
   ```

---

## Step 3: Setup Environment on RunPod

**Connect to pod terminal** and run:

```bash
cd /workspace/edgemoe

# Install dependencies
pip install transformers>=4.55.0 peft datasets tqdm psutil

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

**Expected output:**
```
GPU: NVIDIA A40
CUDA: True
```

---

## Step 4: Train Router (2-3 hours)

```bash
cd /workspace/edgemoe

# Train router on 87K samples
python runpod/train_router_runpod.py \
    --data_dir ./data/processed \
    --output_dir ./models/router_real \
    --epochs 5 \
    --batch_size 32 \
    --lr 1e-4

# Expected output:
# Epoch 1/5: Train Loss: 1.095, Train Acc: 35%, Val Acc: 40%
# Epoch 2/5: Train Loss: 0.985, Train Acc: 48%, Val Acc: 55%
# Epoch 3/5: Train Loss: 0.875, Train Acc: 62%, Val Acc: 68%
# Epoch 4/5: Train Loss: 0.798, Train Acc: 71%, Val Acc: 75%
# Epoch 5/5: Train Loss: 0.745, Train Acc: 77%, Val Acc: 80%
# ✓ Target accuracy achieved: 80% >= 60%
```

**Auto-Checkpointing:**
- ✅ **Saves every 500 steps** (~10-15 min intervals)
- ✅ **Saves at end of each epoch**
- ✅ **Auto-resumes from latest checkpoint** if training is interrupted

**Checkpoint files saved to:**
- `/workspace/edgemoe/models/router_real/latest_checkpoint.pt` (auto-resume point)
- `/workspace/edgemoe/models/router_real/best_router.pt` (best validation loss)
- `/workspace/edgemoe/models/router_real/training_history.json`

**If training crashes:** Just re-run the same command - it will automatically resume from the last checkpoint!

---

## Step 5: Train LoRA Experts (4-6 hours each)

### Expert 0: Code (Python)
```bash
python runpod/train_expert_runpod.py \
    --expert_id 0 \
    --domain code \
    --data_dir ./data/processed/code \
    --output_dir ./models/experts/expert_0_code \
    --epochs 3 \
    --batch_size 4 \
    --lr 2e-4
```

### Expert 1: General (Wikipedia)
```bash
python runpod/train_expert_runpod.py \
    --expert_id 1 \
    --domain general \
    --data_dir ./data/processed/general \
    --output_dir ./models/experts/expert_1_general \
    --epochs 3 \
    --batch_size 4 \
    --lr 2e-4
```

### Expert 2: Research (ArXiv)
```bash
python runpod/train_expert_runpod.py \
    --expert_id 2 \
    --domain research \
    --data_dir ./data/processed/research \
    --output_dir ./models/experts/expert_2_research \
    --epochs 3 \
    --batch_size 4 \
    --lr 2e-4
```

**Auto-Checkpointing for Experts:**
- ✅ **Saves every 300 steps** (~10-15 min intervals)
- ✅ **Keeps last 3 checkpoints** (to save disk space)
- ✅ **Auto-resumes if interrupted** (built into HuggingFace Trainer)

**Training progress:**
Each expert trains for 3 epochs:
- Epoch 1: Loss ~3.5 → 2.8
- Epoch 2: Loss ~2.8 → 2.4
- Epoch 3: Loss ~2.4 → 2.1

**Output:** LoRA adapters (~30-50MB each) saved to `models/experts/`

---

## Step 6: Download Trained Models

### Method A: Using runpodctl (Easiest)

```bash
# From your local machine
runpodctl receive YOUR_POD_ID:/workspace/edgemoe/models/ ./models_from_runpod/
```

### Method B: Using rsync/scp

```bash
# Get pod SSH details from RunPod console
rsync -avz -e "ssh -p PORT" root@POD_IP:/workspace/edgemoe/models/ ./models_from_runpod/
```

### Method C: Upload to cloud storage (from pod)

```bash
# On RunPod pod
cd /workspace/edgemoe/models

# Zip models
tar -czf trained_models.tar.gz router_real/ experts/

# Upload to your preferred service
# Option 1: Dropbox, Google Drive (via rclone)
# Option 2: AWS S3, Hugging Face Hub
# Then download from your local machine
```

---

## Cost Optimization Tips

### 1. Use Spot Instances
- Save 50-70% by using "Spot" pricing
- Risk: Pod may be interrupted (rare)
- Mitigation: Scripts auto-save checkpoints

### 2. Train in Stages
- Train router first (2-3 hrs)
- Download checkpoint
- Train experts one at a time
- Stop pod between sessions

### 3. Use Cheaper GPUs for Testing
- Test scripts on RTX 4090 first
- Switch to A40/A100 for full training

### 4. Monitor Training
- Use Weights & Biases (wandb) for remote monitoring
- Stop training early if metrics plateau

---

## Troubleshooting

### Issue: "Out of CUDA memory"
**Solution:** Reduce batch size
```bash
# Instead of --batch_size 32, try:
--batch_size 16  # or 8
```

### Issue: "Connection lost" or "Training interrupted"
**Solution:** Training auto-resumes from last checkpoint!
```bash
# Router training: Just re-run the same command
python runpod/train_router_runpod.py --data_dir ./data/processed --output_dir ./models/router_real --epochs 5 --batch_size 32
# It will automatically load latest_checkpoint.pt and continue

# Expert training: Automatically resumes from checkpoint
python runpod/train_expert_runpod.py --expert_id 0 --domain code --data_dir ./data/processed/code --output_dir ./models/experts/expert_0_code --epochs 3
# HuggingFace Trainer auto-detects checkpoints in output_dir

# Check saved checkpoints:
ls -lh models/router_real/*.pt
ls -lh models/experts/expert_0_code/checkpoint-*/
```

**Note:** With checkpoints every 300-500 steps, you'll never lose more than 10-15 minutes of progress!

### Issue: "Slow upload speed"
**Solution:** Use GitHub or cloud storage
- Upload data to GitHub first
- Clone directly on RunPod (faster)

### Issue: "Pod terminated unexpectedly"
**Solution:** Enable auto-save
- Checkpoints saved every epoch
- Resume training from last checkpoint

---

## Training Timeline

**Parallel approach** (recommended):
- **Day 1**: Train router (3 hrs) + Expert 0 (5 hrs) = 8 hrs
- **Day 2**: Expert 1 (5 hrs) + Expert 2 (5 hrs) = 10 hrs
- **Total**: ~18 GPU hours (~$14 on A40)

**Sequential approach** (safer):
- Train each model separately
- Download after each completes
- Total time same, but more control

---

## What You'll Get

After training completes:

```
models/
├── router_real/
│   ├── best_router.pt (router weights)
│   ├── training_history.json
│   └── latest_checkpoint.pt
└── experts/
    ├── expert_0_code/
    │   ├── adapter_model.bin (30-50 MB)
    │   ├── adapter_config.json
    │   └── expert_config.json
    ├── expert_1_general/
    │   └── (same structure)
    └── expert_2_research/
        └── (same structure)
```

**Total size:** ~200-300 MB (easy to download)

---

## Next Steps After Training

1. **Download models to local machine**
2. **Test router accuracy** on test set
3. **Test expert generation quality**
4. **Integrate into LFM2-MoE system**
5. **Week 7-8**: Implement cache manager

---

## References

- [RunPod PyTorch Guide](https://www.runpod.io/articles/guides/pytorch-2-4-cuda-12-4)
- [RunPod File Transfer](https://docs.runpod.io/pods/storage/transfer-files)
- [RunPod Network Volumes](https://docs.runpod.io/storage/network-volumes)
- [How to Transfer Data](https://blog.runpod.io/how-do-i-transfer-data-into-my-pod/)

---

## Quick Reference Commands

```bash
# Check GPU
nvidia-smi

# Check training progress
tail -f models/router_real/training_history.json

# Monitor GPU usage during training
watch -n 1 nvidia-smi

# Check disk space
df -h /workspace

# Zip models for download
tar -czf models.tar.gz models/
```

---

**Ready to start!** Follow steps 1-6 and you'll have trained models in ~18 hours of GPU time.
