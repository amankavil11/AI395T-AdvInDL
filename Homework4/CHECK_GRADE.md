# How to Check Your Grade After Training

## Quick Commands

### After Training VLM Model

```bash
# Quick accuracy check (prints percentage like 0.72 = 72%)
python -m homework.finetune test homework/vlm_sft

# Full grader (shows points breakdown)
python3 -m grader homework -v
```

**What you'll see:**
- Quick test: Just prints accuracy (e.g., `0.72` = 72%)
- Full grader: Shows points like `45 / 50` for VLM

### After Training CLIP Model

```bash
# Quick accuracy check (prints percentage)
python -m homework.clip test homework/clip

# Full grader (shows points for BOTH models)
python3 -m grader homework -v
```

**What you'll see:**
- Quick test: Just prints accuracy (e.g., `Accuracy: 0.75`)
- Full grader: Shows points for both VLM and CLIP

---

## Understanding Your Score

### Accuracy → Points Conversion

- **70% accuracy = 50 points** (full credit for that part)
- **0-70% = linear scaling** (35% = 25 points, 50% = ~36 points, etc.)
- **0% = 0 points**

### Example Outputs

**Quick Test Output:**
```
0.72
```
This means 72% accuracy → You'd get ~51 points (above the 70% threshold for full 50 points)

**Full Grader Output:**
```
VLM Model Grader
  - Test the answer accuracy        [ 50 / 50 ]
CLIP Model Grader  
  - Test the CLIP accuracy          [ 45 / 50 ]
Total                                95 / 100
```

---

## Important Notes

1. **Replace checkpoint path** if you used a different `output_dir`:
   - If you used `--output_dir my_vlm`, use: `python -m homework.finetune test homework/my_vlm`
   - If you used `--output_dir my_clip`, use: `python -m homework.clip test homework/my_clip`

2. **Full grader requires both models** - It will test both VLM and CLIP even if you only trained one

3. **The grader uses the same test set** as the online grader, so your score should match

---

## Workflow Example

```bash
# Step 1: Train VLM
python -m homework.finetune train

# Step 2: Check VLM grade
python -m homework.finetune test homework/vlm_sft

# Step 3: Train CLIP  
python -m homework.clip train

# Step 4: Check CLIP grade
python -m homework.clip test homework/clip

# Step 5: Check overall grade (both models)
python3 -m grader homework -v
```

---

## Troubleshooting

**Error: "No such file or directory"**
- Make sure the checkpoint path is correct
- Default paths: `homework/vlm_sft` and `homework/clip`
- Check what was created: `ls homework/`

**Error: "Model not found"**
- Make sure training completed successfully
- Check for files: `ls homework/vlm_sft/adapter_model.safetensors`

**Low accuracy?**
- Try training for more epochs
- Check that data generation worked (lots of QA pairs/captions)
- Verify your implementations are correct

