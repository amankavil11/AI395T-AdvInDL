# Tune Hyperparameters to Improve Accuracy (Keep 0.05 Epochs)

Since the assignment requires 0.05 epochs, here's how to improve accuracy by tuning other hyperparameters.

## Current Settings (32.5% accuracy)

- `num_train_epochs`: 0.05 (fixed - can't change)
- `per_device_train_batch_size`: 8
- `gradient_accumulation_steps`: 4
- `learning_rate`: 5e-4
- `lora_r`: 8
- `lora_alpha`: 32
- `lora_dropout`: 0.0

## Tunable Hyperparameters

### 1. Learning Rate (Most Important!)

Try different learning rates:

```bash
# Option A: Higher LR (faster learning)
python -m homework.finetune train \
    --num_train_epochs 0.05 \
    --learning_rate 1e-3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4

# Option B: Lower LR (more stable)
python -m homework.finetune train \
    --num_train_epochs 0.05 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4

# Option C: Try middle ground
python -m homework.finetune train \
    --num_train_epochs 0.05 \
    --learning_rate 8e-4 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4
```

**Recommended**: Start with **1e-3** (higher) - with only 0.05 epochs, you need faster learning.

### 2. LoRA Rank (More Model Capacity)

Increase LoRA rank for more trainable parameters:

```bash
# Higher rank = more capacity, more memory
python -m homework.finetune train \
    --num_train_epochs 0.05 \
    --learning_rate 1e-3 \
    --lora_r 16 \
    --lora_alpha 64 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8
```

**Trade-off**: Higher rank uses more memory. Adjust batch size if needed.

### 3. Effective Batch Size

Larger effective batch size = more stable gradients:

```bash
# Increase effective batch size (8 * 8 = 64)
python -m homework.finetune train \
    --num_train_epochs 0.05 \
    --learning_rate 1e-3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8
```

Or if memory is tight:

```bash
# Smaller per-device, but same effective size
python -m homework.finetune train \
    --num_train_epochs 0.05 \
    --learning_rate 1e-3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16
```

### 4. LoRA Alpha

Scale with rank: `alpha = 2 * rank` is common:

```bash
python -m homework.finetune train \
    --num_train_epochs 0.05 \
    --learning_rate 1e-3 \
    --lora_r 16 \
    --lora_alpha 32 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8
```

## Recommended Configurations

### Configuration 1: Aggressive Learning (Try First)

```bash
python -m homework.finetune train \
    --num_train_epochs 0.05 \
    --learning_rate 1e-3 \
    --lora_r 16 \
    --lora_alpha 32 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8
```

**Rationale**: Higher LR + more capacity + stable gradients

### Configuration 2: High Capacity

```bash
python -m homework.finetune train \
    --num_train_epochs 0.05 \
    --learning_rate 8e-4 \
    --lora_r 32 \
    --lora_alpha 64 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16
```

**Warning**: Uses more memory, might OOM on 12GB GPU

### Configuration 3: Balanced (Memory-Efficient)

```bash
python -m homework.finetune train \
    --num_train_epochs 0.05 \
    --learning_rate 1e-3 \
    --lora_r 8 \
    --lora_alpha 32 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32
```

**Rationale**: Same capacity, but very stable gradients

## Best Strategy: Improve Your Data First!

**The #1 thing you can do**: Generate **more and better** QA pairs!

With only 0.05 epochs, the model can only learn so much. But if you have:
- **10,000+ diverse QA pairs** (vs 1,000)
- **High quality questions** covering all 5 types
- **Balanced dataset** (all question types represented)

You'll get much better accuracy!

### Check Your Data Quality

```bash
# Count QA pairs
python -c "import json; import glob; files=glob.glob('data/train/*_qa_pairs.json'); total=sum(len(json.load(open(f))) for f in files); print(f'Total: {total:,} QA pairs')"

# Check distribution of question types
python -c "
import json
import glob
from collections import Counter

files = glob.glob('data/train/*_qa_pairs.json')
all_questions = []
for f in files:
    with open(f) as file:
        data = json.load(file)
        all_questions.extend([q['question'] for q in data])

# Count question types
types = Counter()
for q in all_questions:
    if 'What kart is the ego car' in q:
        types['ego_car'] += 1
    elif 'How many karts are there' in q:
        types['total_karts'] += 1
    elif 'What track is this' in q:
        types['track'] += 1
    elif 'left or right' in q or 'front of or behind' in q or 'Where is' in q:
        types['position'] += 1
    elif 'How many karts are to the' in q or 'How many karts are in' in q:
        types['counting'] += 1

print('Question type distribution:')
for t, count in types.items():
    print(f'  {t}: {count}')
"
```

## Hyperparameter Search Strategy

### Step 1: Test Different Learning Rates

```bash
# Test 3 LR values
for lr in 1e-4 5e-4 1e-3; do
    python -m homework.finetune train \
        --num_train_epochs 0.05 \
        --learning_rate $lr \
        --output_dir vlm_lr_${lr} \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 4
    
    python -m homework.finetune test homework/vlm_lr_${lr}
done
```

### Step 2: Pick Best LR, Then Tune LoRA Rank

```bash
# Assuming 1e-3 was best
for rank in 8 16 32; do
    python -m homework.finetune train \
        --num_train_epochs 0.05 \
        --learning_rate 1e-3 \
        --lora_r $rank \
        --lora_alpha $((rank * 2)) \
        --output_dir vlm_rank_${rank} \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 8
    
    python -m homework.finetune test homework/vlm_rank_${rank}
done
```

### Step 3: Final Best Configuration

Use the best hyperparameters from your tests.

## Expected Improvements

With current settings (32.5% accuracy):
- **Better LR**: +5-10% accuracy
- **Higher LoRA rank**: +3-8% accuracy  
- **Better data (10k+ pairs)**: +10-20% accuracy
- **Combined**: Should reach 50-70% accuracy

## Quick Start Command

Try this first (most likely to improve):

```bash
python -m homework.finetune train \
    --num_train_epochs 0.05 \
    --learning_rate 1e-3 \
    --lora_r 16 \
    --lora_alpha 32 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8
```

Then test:
```bash
python -m homework.finetune test homework/vlm_sft
```

If accuracy improves, you're on the right track!

