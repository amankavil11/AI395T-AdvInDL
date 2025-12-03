# Speed Up Training - 2% in 20 Minutes is Too Slow!

## Problem

At 2% per 20 minutes, training will take:
- **1 epoch**: ~17 hours
- **2 epochs**: ~33 hours

This is way too slow! Let's fix it.

---

## Quick Solutions

### Solution 1: Train on Less Data (Fastest Fix)

You likely generated too many QA pairs. Limit training to a subset:

```bash
# Train on first 5000 samples only (much faster)
python -c "
from homework.data import VQADataset
import json
dataset = VQADataset('train', max_samples=5000)
print(f'Limited dataset size: {len(dataset)}')
"
```

**Better approach**: Modify the training to limit data:

Create a quick script or modify training command. Actually, let's check your data size first.

### Solution 2: Use Smaller Epochs First

Instead of 2 full epochs, try fewer:

```bash
# Start with 0.5 epochs (faster, still should improve)
python -m homework.finetune train \
    --num_train_epochs 0.5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_workers 4
```

**Time estimate**: 
- 0.5 epochs: ~8.5 hours
- Check accuracy, if good, stop
- If not, train another 0.5 epochs

### Solution 3: Reduce Dataset Size

If you have too many QA pairs, limit them:

**Option A**: Delete some QA pairs (temporary)
```bash
# Backup first
cp data/train/balanced_qa_pairs.json data/train/balanced_qa_pairs.json.backup

# Keep only first 10,000 entries
python -c "
import json
with open('data/train/balanced_qa_pairs.json') as f:
    data = json.load(f)
print(f'Original: {len(data)} pairs')
data = data[:10000]
print(f'Limited: {len(data)} pairs')
with open('data/train/balanced_qa_pairs.json', 'w') as f:
    json.dump(data, f, indent=2)
"
```

**Option B**: Modify data loading to limit samples

---

## Check Your Dataset Size

First, see how much data you have:

```bash
# Count QA pairs
python -c "
import json
import glob
files = glob.glob('data/train/*_qa_pairs.json')
total = sum(len(json.load(open(f))) for f in files)
print(f'Total QA pairs: {total:,}')
print(f'At current speed (2% per 20 min):')
print(f'  1 epoch: ~{total * 20 / 2 / 60:.1f} hours')
print(f'  2 epochs: ~{total * 20 / 2 / 60 * 2:.1f} hours')
"
```

---

## Recommended Action Plan

### Step 1: Check Dataset Size

```bash
python -c "
import json
import glob
files = glob.glob('data/train/*_qa_pairs.json')
total = sum(len(json.load(open(f))) for f in files)
print(f'Total QA pairs: {total:,}')
"
```

### Step 2: If You Have > 20,000 Pairs

Limit to 10,000-15,000 for faster training:

```bash
# Limit to 10,000 pairs
python -c "
import json
import glob

files = glob.glob('data/train/*_qa_pairs.json')
all_pairs = []
for f in files:
    with open(f) as file:
        all_pairs.extend(json.load(file))

print(f'Original: {len(all_pairs)} pairs')

# Limit to 10,000
all_pairs = all_pairs[:10000]

# Save back
for f in files:
    with open(f, 'w') as file:
        json.dump(all_pairs[:len(json.load(open(f))) if f == files[0] else 0], file, indent=2)

# Actually, better to overwrite the main file
if files:
    with open(files[0], 'w') as file:
        json.dump(all_pairs, file, indent=2)
        print(f'Saved {len(all_pairs)} pairs to {files[0]}')
"
```

**Better approach** - Use a script:

```python
# limit_data.py
import json
import glob

files = glob.glob('data/train/*_qa_pairs.json')
all_pairs = []
for f in files:
    with open(f) as file:
        all_pairs.extend(json.load(file))

print(f'Original: {len(all_pairs)} pairs')
all_pairs = all_pairs[:10000]  # Limit to 10k
print(f'Limited: {len(all_pairs)} pairs')

# Overwrite first file
if files:
    with open(files[0], 'w') as file:
        json.dump(all_pairs, file, indent=2)
    print(f'Saved to {files[0]}')
```

### Step 3: Restart Training with Limited Data

After limiting data, restart training:

```bash
# Should be much faster now
python -m homework.finetune train \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_workers 4
```

**Expected time with 10k samples**:
- Should be 5-10x faster
- 1 epoch: ~2-3 hours instead of 17 hours

---

## Alternative: Train Incrementally

Instead of 2 epochs at once, do multiple shorter runs:

### Step 1: Train 0.2 epochs, test
```bash
python -m homework.finetune train --num_train_epochs 0.2 ...
python -m homework.finetune test homework/vlm_sft
```

### Step 2: If accuracy < 70%, train another 0.2 epochs
```bash
python -m homework.finetune train --num_train_epochs 0.2 ...
```

### Step 3: Repeat until accuracy is good

This lets you stop early if you reach target accuracy.

---

## Why Is It So Slow?

Possible reasons:
1. **Huge dataset** - You generated too many QA pairs (check this first!)
2. **CPU instead of GPU** - Check if GPU is being used
3. **Batch size 1** - Very small batches are slower
4. **Gradient checkpointing** - Saves memory but slows down

---

## Quick Check Commands

```bash
# 1. Check dataset size
python -c "import json; import glob; files=glob.glob('data/train/*_qa_pairs.json'); print('QA pairs:', sum(len(json.load(open(f))) for f in files))"

# 2. Check if GPU is being used (should see CUDA or MPS)
python -c "import torch; print('Device:', 'cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')"

# 3. Check training progress - how many steps total?
# Look at the training output for "Total steps: X"
```

---

## Recommended Immediate Action

1. **Stop current training** (Ctrl+C)
2. **Check dataset size** (use command above)
3. **If > 15,000 pairs, limit to 10,000**
4. **Restart training with 1 epoch first**
5. **Test accuracy, then decide if you need more**

This should get you from 33 hours down to ~3-5 hours!

