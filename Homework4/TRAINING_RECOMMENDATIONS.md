# VLM Training Optimization Recommendations

## Current Performance
- **Accuracy**: 42.5% (30/50 points)
- **Target**: 70% (50/50 points)
- **Constraint**: Must use 0.05 epochs (cannot change)

## Recommended Training Command

### Option 1: Higher Learning Rate + Larger LoRA (Recommended First Try)
```bash
python -m homework.finetune train \
    --learning_rate 1e-3 \
    --lora_r 32 \
    --lora_alpha 64 \
    --warmup_steps 10 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --output_dir vlm_model_v2
```

### Option 2: Even Higher Learning Rate (If Option 1 doesn't work)
```bash
python -m homework.finetune train \
    --learning_rate 1.5e-3 \
    --lora_r 32 \
    --lora_alpha 64 \
    --warmup_steps 5 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --output_dir vlm_model_v3
```

### Option 3: Maximum Learning Rate (If Options 1-2 don't work)
```bash
python -m homework.finetune train \
    --learning_rate 2e-3 \
    --lora_r 64 \
    --lora_alpha 128 \
    --warmup_steps 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --output_dir vlm_model_v4
```

## Key Changes Explained

1. **Learning Rate**: Increased from `5e-4` to `1e-3` or higher
   - With only 0.05 epochs, need faster learning
   - Higher LR helps model learn more in fewer steps

2. **LoRA Rank (r)**: Increased from `16` to `32` or `64`
   - More trainable parameters = more capacity to learn
   - Trade-off: slightly more memory, but better performance

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

## Expected Improvements

- **Option 1**: Should get you to ~50-60% accuracy
- **Option 2**: Should get you to ~60-70% accuracy  
- **Option 3**: Should get you above 70% if Options 1-2 don't work

## Troubleshooting

If accuracy doesn't improve:
1. Check that training loss is decreasing (should go from ~7.5 to <0.5)
2. Verify you're using the correct checkpoint path when testing
3. Make sure QA data generation completed successfully
4. Try Option 2 or 3 with even higher learning rates

