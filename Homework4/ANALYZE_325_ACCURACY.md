# Analyzing 32.5% Accuracy

## What 32.5% Means

The grader tests on **128 samples**:
- 32.5% accuracy = **41.6 out of 128** correct
- Rounded: **~42 out of 128** correct

This is suspiciously low and suggests the model is barely learning!

---

## Why 32.5% is Suspicious

### Possible Explanations:

1. **Model is barely learning**
   - With only 0.05 epochs, model might be learning very little
   - Could be close to random guessing for some question types

2. **Pattern Matching Failure**
   - Model might be memorizing some patterns but not generalizing
   - Getting some questions right by chance

3. **Training Issue**
   - Learning rate might be too low/high
   - Model capacity (LoRA rank) might be insufficient
   - Not enough training data

---

## Will It Be Exactly 32.5% Again?

**Very unlikely!** With different hyperparameters, you should see:

### Better Case:
- **40-50%**: Model is learning something
- **50-60%**: Model is learning well
- **60-70%**: Good performance
- **70%+**: Excellent (full credit!)

### Worse Case:
- **20-30%**: Something is wrong, model is regressing
- **Below 20%**: Training is broken

### Exactly 32.5% Again?
- **Extremely unlikely** unless you use the exact same hyperparameters
- Different LR, LoRA rank, batch size will change the result

---

## What This Tells Us

### Current Status (32.5%):
- Model is learning **something** (above random ~20% for 5-choice questions)
- But it's **not learning enough** to be effective
- With only 0.05 epochs, this might be expected with suboptimal hyperparameters

### Expected with Better Hyperparameters:

```
Learning Rate    LoRA Rank    Expected Accuracy
-------------------------------------------------
5e-4 (current)   8 (current)  32.5% (current)
1e-3              8            35-45%
1e-3              16           40-50%
1e-3              32           45-55%
8e-4              16           40-50%
```

---

## How to Verify Training is Working

### Check 1: Loss Should Decrease
During training, watch for:
- Loss should go **down** over time
- If loss plateaus or increases, something is wrong

### Check 2: Accuracy Should Vary
Run the same training twice with same hyperparameters:
- Should get **similar** but not **identical** results
- If exactly the same, model might not be training

### Check 3: Different Hyperparameters = Different Results
Try these and compare:

```bash
# Test 1: Higher LR
python -m homework.finetune train \
    --num_train_epochs 0.05 \
    --learning_rate 1e-3 \
    --output_dir vlm_test1
python -m homework.finetune test homework/vlm_test1

# Test 2: Higher LoRA rank
python -m homework.finetune train \
    --num_train_epochs 0.05 \
    --learning_rate 1e-3 \
    --lora_r 16 \
    --output_dir vlm_test2
python -m homework.finetune test homework/vlm_test2
```

If both give ~32.5%, something is fundamentally wrong.

---

## Most Likely Issue

Given 0.05 epochs and current hyperparameters:

1. **Too little training time** - Only 5% of one epoch
2. **Learning rate might be suboptimal** - 5e-4 could be too conservative
3. **Insufficient model capacity** - LoRA rank 8 might be too small
4. **Not enough diverse training data** - Quality matters more than quantity with such short training

---

## Recommendation

Try this configuration (should give different result):

```bash
python -m homework.finetune train \
    --num_train_epochs 0.05 \
    --learning_rate 1e-3 \
    --lora_r 16 \
    --lora_alpha 32 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --output_dir vlm_optimized
```

Then test:
```bash
python -m homework.finetune test homework/vlm_optimized
```

**Expected**: Should get **different accuracy** (hopefully 40-55%)

If you still get ~32.5%, then:
- Check your training data quality
- Verify training is actually happening (loss decreasing)
- Check if model is loading correctly

---

## Bottom Line

**32.5% is likely:**
- Real result from minimal training (0.05 epochs)
- Model learning something but not enough
- With better hyperparameters, you should see **different** (hopefully better) results

**Try new hyperparameters and compare** - that's the best way to know if training is working!

