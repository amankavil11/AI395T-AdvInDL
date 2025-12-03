# How to Improve Your Score

## Current Status

✅ **Grader is working!**
- VLM Accuracy: **32.5%** → **23/50 points**
- Need **70% accuracy** for full 50 points

## Quick Fixes to Improve Score

### 1. Train for More Epochs (Most Important!)

The default is only **0.05 epochs** (5% of one epoch), which is way too short!

```bash
# Train for 1 full epoch
python -m homework.finetune train \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_workers 2
```

Or even more:

```bash
# Train for 2-3 epochs (better accuracy, longer time)
python -m homework.finetune train \
    --num_train_epochs 2.0 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8
```

**Expected time:**
- 1 epoch: ~1-2 hours (MPS)
- 2-3 epochs: ~2-4 hours (MPS)

### 2. Check Your Training Data

Make sure you generated lots of QA pairs:

```bash
python -c "import json; data=json.load(open('data/train/balanced_qa_pairs.json')); print(f'QA pairs: {len(data)}')"
```

**You should have:**
- At least 5,000+ QA pairs (ideally 10,000+)
- If you have fewer, regenerate with more data

### 3. Verify Data Quality

Check that your QA pairs are correct:

```bash
# Preview some QA pairs
python -m homework.generate_qa check --info_file data/valid/00000_info.json --view_index 0
```

Make sure questions and answers look reasonable.

### 4. Training Parameters

If you have enough memory, increase batch size (faster training):

```bash
# If you have more GPU memory
python -m homework.finetune train \
    --num_train_epochs 2.0 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4
```

---

## Recommended Training Command

For best results, use this:

```bash
python -m homework.finetune train \
    --num_train_epochs 2.0 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_workers 2 \
    --learning_rate 5e-4
```

This will:
- Train for 2 full epochs
- Use memory-efficient batch size
- Take ~2-4 hours on MPS

---

## After Training

1. **Check accuracy**:
   ```bash
   python -m homework.finetune test homework/vlm_sft
   ```

2. **Run grader again**:
   ```bash
   # Make sure path is correct
   cd homework && ln -sf vlm_sft vlm_model && cd ..
   python3 -m grader homework -v
   ```

---

## Expected Progress

- **0.05 epochs (current)**: 32.5% → 23 points
- **1 epoch**: Expect 40-50% → ~30-35 points
- **2 epochs**: Expect 55-65% → ~40-45 points
- **3+ epochs**: Should reach 70%+ → 50 points

**Goal: 70% accuracy = 50 points**

---

## Checklist

- [ ] Generate QA pairs: `python -m homework.generate_qa generate_all`
- [ ] Check QA pair count (should be 5,000+)
- [ ] Train for 1-2 epochs (not just 0.05!)
- [ ] Test accuracy after training
- [ ] Run grader: `python3 -m grader homework -v`
- [ ] Make sure model path is correct (symlink vlm_sft → vlm_model)

---

## Time Estimate

- **Data generation**: ~30 minutes
- **Training (1 epoch)**: ~1-2 hours (MPS)
- **Training (2 epochs)**: ~2-4 hours (MPS)
- **Testing**: ~5 minutes

**Total for full training**: ~3-5 hours

---

## Troubleshooting

**Still low accuracy after 2 epochs?**
- Check that you have enough training data (5,000+ pairs)
- Verify QA pairs are correct (use check command)
- Try 3-4 epochs
- Check if your implementation is correct

**Out of memory?**
- Use batch size 1
- Increase gradient accumulation steps
- See FIX_MEMORY.md for details

**Model not found?**
- Create symlink: `cd homework && ln -sf vlm_sft vlm_model`
- Or copy directory: `cp -r homework/vlm_sft homework/vlm_model`

