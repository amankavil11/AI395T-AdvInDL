# Training Guide for 12GB GPU

## Optimized Settings for Your Hardware

With a 12GB GPU, you need to be careful about memory usage. Here are the best settings:

---

## VLM Training (Memory-Efficient)

### Recommended Settings

```bash
python -m homework.finetune train \
    --num_train_epochs 2.0 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_workers 4 \
    --learning_rate 5e-4
```

**What this does:**
- Batch size 1 (minimal memory per step)
- Gradient accumulation 8 (effective batch size = 8)
- 2 epochs (good balance of time vs accuracy)
- Gradient checkpointing is already enabled (saves ~50% memory)

### If You Get Out of Memory

Try even more conservative settings:

```bash
python -m homework.finetune train \
    --num_train_epochs 2.0 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_workers 2 \
    --learning_rate 5e-4
```

### Monitor GPU Memory

While training, in another terminal:

```bash
# Watch GPU memory usage
watch -n 1 nvidia-smi

# Or for one-time check
nvidia-smi
```

---

## CLIP Training

CLIP uses less memory, so you can use larger batches:

### Recommended Settings

```bash
python -m homework.clip train \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size 256 \
    --gradient_accumulation_steps 4 \
    --num_workers 4
```

### If Out of Memory

Reduce batch size:

```bash
python -m homework.clip train \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size 128 \
    --gradient_accumulation_steps 8 \
    --num_workers 4
```

---

## Memory Optimization Tips

### 1. Clear GPU Cache Before Training

```python
# At start of training script or before
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### 2. Use Mixed Precision (Already Enabled)

The code already uses `bf16` when available, which saves memory.

### 3. Close Other GPU Processes

Before training:
```bash
# Check what's using GPU
nvidia-smi

# Kill other processes if needed
# (Be careful not to kill system processes!)
```

### 4. Reduce Sequence Length (If Possible)

The model will pad sequences. Shorter sequences = less memory.

---

## Training Time Estimates (12GB GPU)

### VLM Training
- **1 epoch**: ~1-2 hours (with batch_size=1)
- **2 epochs**: ~2-4 hours
- **3 epochs**: ~3-6 hours

### CLIP Training
- **1 epoch**: ~30-60 minutes (with batch_size=256)
- **2 epochs**: ~1-2 hours

---

## Step-by-Step Training Plan

### Step 1: Train VLM (Priority)

```bash
# Start training - this will take a few hours
python -m homework.finetune train \
    --num_train_epochs 2.0 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_workers 4
```

**Monitor progress:**
- Check GPU memory: `nvidia-smi`
- Training logs will show progress
- Saves checkpoints every 50 steps

### Step 2: Test VLM

```bash
# After training completes
python -m homework.finetune test homework/vlm_sft

# Setup for grader
cd homework && ln -sf vlm_sft vlm_model && cd ..

# Check grade
python3 -m grader homework -v
```

### Step 3: Train CLIP

```bash
# Once VLM is done
python -m homework.clip train \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size 256 \
    --gradient_accumulation_steps 4
```

### Step 4: Test CLIP

```bash
python -m homework.clip test homework/clip

# Setup for grader
cd homework && ln -sf clip clip_model && cd ..

# Check full grade
python3 -m grader homework -v
```

---

## Troubleshooting

### Out of Memory Error

**Solution 1**: Reduce batch size further
```bash
--per_device_train_batch_size 1
--gradient_accumulation_steps 16  # Increase this to compensate
```

**Solution 2**: Reduce number of workers
```bash
--num_workers 1  # Instead of 4
```

**Solution 3**: Clear GPU cache
```python
import torch
torch.cuda.empty_cache()
```

### Training is Too Slow

**Acceptable trade-offs:**
- Training will be slower with small batch sizes
- That's normal for limited GPU memory
- 2-4 hours for VLM is reasonable

**Don't increase batch size** if it causes OOM errors - that will crash training.

### Want Faster Training?

If you have access to:
- **Cloud GPU** (Colab, AWS, etc.): Use larger GPUs for faster training
- **Multiple GPUs**: Not supported by this code, would need modifications
- **Longer training**: Just let it run overnight

---

## Recommended Strategy

1. **Train VLM first** (more important for points)
   - Use: batch_size=1, grad_accum=8, epochs=2
   - Time: ~2-4 hours
   - Target: 70% accuracy

2. **If VLM is good**, train CLIP
   - Use: batch_size=256, grad_accum=4, epochs=1
   - Time: ~30-60 minutes

3. **If running out of time**, focus on VLM
   - 50 points from VLM is better than 0 points from both

---

## Quick Command Reference

### VLM (Memory-Efficient)
```bash
python -m homework.finetune train \
    --num_train_epochs 2.0 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_workers 4
```

### CLIP (Faster)
```bash
python -m homework.clip train \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size 256 \
    --gradient_accumulation_steps 4
```

### Check GPU Memory
```bash
nvidia-smi
```

---

## Expected Results

With proper training on 12GB GPU:
- **VLM**: Should reach 60-70% accuracy with 2 epochs
- **CLIP**: Should reach 70%+ accuracy with 1 epoch
- **Total**: ~85-100 points (full credit + extra credit possible)

Good luck with your training!

