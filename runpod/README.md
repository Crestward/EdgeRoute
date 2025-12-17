# RunPod Training Scripts

Scripts optimized for training EdgeMoE models on RunPod GPUs.

## Files

- **RUNPOD_GUIDE.md** - Complete step-by-step guide
- **train_router_runpod.py** - Router training script
- **train_expert_runpod.py** - LoRA expert training script
- **setup_runpod.sh** - Environment setup script

## Quick Start

See **RUNPOD_GUIDE.md** for full instructions.

### 1. Setup Pod
```bash
cd /workspace/edgemoe
bash runpod/setup_runpod.sh
```

### 2. Train Router
```bash
python runpod/train_router_runpod.py \
    --data_dir ./data/processed \
    --output_dir ./models/router_real \
    --epochs 5 \
    --batch_size 32
```

### 3. Train Experts
```bash
# Code Expert
python runpod/train_expert_runpod.py \
    --expert_id 0 --domain code \
    --data_dir ./data/processed/code \
    --output_dir ./models/experts/expert_0_code \
    --epochs 3

# General Expert
python runpod/train_expert_runpod.py \
    --expert_id 1 --domain general \
    --data_dir ./data/processed/general \
    --output_dir ./models/experts/expert_1_general \
    --epochs 3

# Research Expert
python runpod/train_expert_runpod.py \
    --expert_id 2 --domain research \
    --data_dir ./data/processed/research \
    --output_dir ./models/experts/expert_2_research \
    --epochs 3
```

## Auto-Checkpointing

**Built-in crash protection:**
- ✅ Router: Saves every 500 steps (~10-15 min)
- ✅ Experts: Save every 300 steps (~10-15 min)
- ✅ Auto-resumes if training interrupted
- ✅ Never lose more than 15 minutes of progress

**If pod crashes:** Just re-run the same command - training continues automatically!

## Estimated Training Times

On **NVIDIA A40** (~$0.79/hour):
- Router: 2-3 hours
- Each Expert: 4-6 hours
- **Total: ~18 hours (~$14)**

## Download Models

```bash
# From local machine
runpodctl receive YOUR_POD_ID:/workspace/edgemoe/models/ ./models_from_runpod/
```

See RUNPOD_GUIDE.md for detailed instructions.
