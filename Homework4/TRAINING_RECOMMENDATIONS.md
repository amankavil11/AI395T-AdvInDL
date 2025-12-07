# VLM Training Optimization Recommendations

## Current Performance
- **Accuracy**: 42.5% (30/50 points)
- **Target**: 70% (50/50 points)
- **Constraint**: Must use 0.05 epochs (cannot change)
- **Size Limit**: 50MB total submission (currently ~40MB)

## ⚠️ IMPORTANT: Size Constraint

**Current situation**: 40MB total (both VLM + CLIP models + code)
**Limit**: 50MB maximum
**Headroom**: ~10MB

**⚠️ Doubling LoRA rank (16→32) roughly doubles model size**, which could push you over 50MB!

## Recommended Training Commands (Size-Safe)

### Option 1: Higher Learning Rate Only (SAFEST - No Size Increase)
```bash
python -m homework.finetune train \
    --learning_rate 1e-3 \
    --warmup_steps 10 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --output_dir vlm_model_v2
```
**Size impact**: None (only changes training, not model architecture)

### Option 2: Higher LR + Moderate LoRA Increase (SAFE - ~2-3MB increase)
```bash
python -m homework.finetune train \
    --learning_rate 1.5e-3 \
    --lora_r 20 \
    --lora_alpha 40 \
    --warmup_steps 5 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --output_dir vlm_model_v3
```
**Size impact**: ~2-3MB (25% increase in LoRA params, not 2x)

### Option 3: Maximum LR + Slightly Larger LoRA (RISKY - ~5-6MB increase)
```bash
python -m homework.finetune train \
    --learning_rate 2e-3 \
    --lora_r 24 \
    --lora_alpha 48 \
    --warmup_steps 5 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --output_dir vlm_model_v4
```
**Size impact**: ~5-6MB (50% increase in LoRA params)

### Option 4: Very High LR + Keep LoRA Same (SAFE - No Size Increase)
```bash
python -m homework.finetune train \
    --learning_rate 2.5e-3 \
    --warmup_steps 5 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --output_dir vlm_model_v5
```
**Size impact**: None

## Key Changes Explained

1. **Learning Rate**: Increased from `5e-4` to `1e-3` or higher
   - With only 0.05 epochs, need faster learning
   - Higher LR helps model learn more in fewer steps

2. **LoRA Rank (r)**: Keep at `16` or increase moderately to `20-24`
   - **DO NOT double to 32** - this will likely exceed 50MB limit
   - Moderate increase (16→20) adds ~2-3MB, which is safe
   - More trainable parameters = more capacity, but size is constrained

3. **LoRA Alpha**: Increased proportionally (typically 2x rank)
   - `r=32` → `alpha=64`
   - `r=64` → `alpha=128`
   - Controls scaling of LoRA weights

4. **Warmup Steps**: Reduced from `100` to `5-10`
   - With only ~143 total steps, 100 warmup is too much
   - Need more steps at full learning rate

5. **Batch Size**: Keep same or adjust based on memory
   - Current: `8 * 8 = 64` effective batch size
   - Can try increasing if memory allows

## Testing After Training

```bash
# Test the new model
python -m homework.finetune test vlm_model_v2

# Or use full grader
python3 -m grader homework -v
```

## Additional Tips

1. **Monitor Training Loss**: Watch TensorBoard to see if loss is decreasing
   ```bash
   tensorboard --logdir homework/vlm_model_v2/tensorboard
   ```

2. **Check Data Quality**: Verify QA pairs are diverse and correct
   ```bash
   python -m homework.generate_qa check --info_file data/valid/00000_info.json --view_index 0
   ```

3. **Try Different Seeds**: If results vary, try training multiple times with different seeds

4. **Learning Rate Schedule**: Current is "cosine" which is good, but with 0.05 epochs, a linear schedule might work better (would require code change)

## Checking Model Size Before Submission

```bash
# Check size of your models
du -sh homework/vlm_model*
du -sh homework/clip*

# Check total size of homework directory (excluding checkpoints)
du -sh homework/

# Test bundle size
python3 bundle.py homework test_utid
# This will show the actual zip file size
```

## Expected Improvements

- **Option 1** (LR only): Should get you to ~50-55% accuracy
- **Option 2** (LR + moderate LoRA): Should get you to ~55-65% accuracy  
- **Option 3** (High LR + larger LoRA): Should get you to ~60-70% accuracy
- **Option 4** (Very high LR): Should get you to ~55-65% accuracy

**Recommendation**: Start with **Option 1** (safest, no size increase). If that doesn't get you to 70%, try **Option 2** or **Option 4**.

## Troubleshooting

If accuracy doesn't improve:
1. Check that training loss is decreasing (should go from ~7.5 to <0.5)
2. Verify you're using the correct checkpoint path when testing
3. Make sure QA data generation completed successfully
4. Try progressively higher learning rates (Option 1 → Option 4)
5. If still not at 70%, try Option 2 or 3 with moderate LoRA increase

## Size Management Tips

1. **Delete old checkpoints**: Only keep your best model
   ```bash
   rm -rf homework/vlm_model  # Keep only vlm_model_v2 (or your best)
   rm -rf homework/clip  # Keep only your best CLIP model
   ```

2. **Check before submitting**: Always test bundle size
   ```bash
   python3 bundle.py homework test_utid
   ```

3. **If over 50MB**: 
   - Remove any unnecessary files
   - Delete tensorboard logs
   - Keep only one checkpoint per model
   - Consider reducing LoRA rank if you increased it

