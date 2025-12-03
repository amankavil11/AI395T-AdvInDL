# Fix Out-of-Memory Errors

## Quick Fix: Use Memory-Efficient Training

If you're getting "CUDA out of memory" or similar errors, use these commands:

### For VLM Training (Memory-Efficient)

```bash
python -m homework.finetune train \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_workers 4
```

This uses:
- Batch size 1 (minimal memory per step)
- Gradient accumulation 8 (effective batch size = 8)
- Fewer workers (less memory overhead)

### Even More Memory-Efficient (If Still Failing)

```bash
python -m homework.finetune train \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_workers 2
```

---

## Understanding the Issue

The error happens because:
- Vision-language models are large
- Images take up memory
- Default batch size (8) might be too large for your GPU/device

---

## Solutions (From Easiest to Most Drastic)

### Solution 1: Reduce Batch Size (Recommended)

```bash
# Start with batch size 1
python -m homework.finetune train \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_workers 4
```

**What this does:**
- Processes 1 sample at a time (saves memory)
- Accumulates gradients over 8 steps (maintains training quality)
- Effective batch size = 1 × 8 = 8 (same as default)

### Solution 2: Enable Gradient Checkpointing

Gradient checkpointing is now enabled by default (I just added it). It trades compute for memory.

### Solution 3: Reduce Number of Workers

```bash
python -m homework.finetune train \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_workers 2
```

### Solution 4: Train on CPU (Slow but Works)

If GPU memory is really limited, you can force CPU:

```python
# This is slower but won't run out of memory
# The code will automatically use CPU if GPU is unavailable
```

Actually, if you're on Apple Silicon, MPS might have memory limits. You can check available memory.

---

## Recommended Training Commands

### Minimal Memory (Start Here)

```bash
python -m homework.finetune train \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_workers 2 \
    --num_train_epochs 1.0
```

### Moderate Memory (If Above Works)

```bash
python -m homework.finetune train \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_workers 4 \
    --num_train_epochs 1.0
```

### Default (If You Have Enough Memory)

```bash
python -m homework.finetune train
```

---

## For CLIP Training

CLIP uses larger batch sizes by default. If you get OOM errors:

```bash
python -m homework.clip train \
    --per_device_train_batch_size 512 \
    --gradient_accumulation_steps 2
```

Or even smaller:

```bash
python -m homework.clip train \
    --per_device_train_batch_size 256 \
    --gradient_accumulation_steps 4
```

---

## Check Your Memory Usage

### Check Available GPU Memory (CUDA)

```bash
nvidia-smi
```

### Check Memory Usage (MPS/Apple Silicon)

The error message will tell you how much memory is available.

---

## Tips

1. **Start small** - Use batch size 1 and increase if it works
2. **Gradient accumulation** - Maintains effective batch size while reducing memory
3. **Gradient checkpointing** - Already enabled, saves ~50% memory
4. **Clear cache** - If you've run other models, restart Python/clear cache:
   ```python
   import torch
   torch.cuda.empty_cache()  # For CUDA
   ```

---

## What If Nothing Works?

1. **Reduce dataset size** - Train on a subset first:
   ```python
   # Edit data.py or pass max_samples
   train_dataset = VQADataset("train", max_samples=1000)
   ```

2. **Use CPU** - Much slower but unlimited memory:
   - CPU training will be very slow (hours)
   - But it will work if GPU memory is the issue

3. **Train on smaller subset** - Generate less data or filter it

---

## Expected Training Times

With memory-efficient settings (batch_size=1):
- **MPS (Apple Silicon)**: ~1-2 hours per epoch
- **CUDA (NVIDIA GPU)**: ~30-60 minutes per epoch  
- **CPU**: ~6-12 hours per epoch

The training will be slower with smaller batches, but it will complete!

