# Quick Progress Check Guide

## 🚀 Quick Answer

**GPU/CPU**: You have **MPS (Apple Silicon GPU)** available! You don't need a separate GPU - CPU works but is slower. Your MPS will make training much faster.

**Check Progress**: After each training step, use these commands to see your score.

---

## 📊 Checking Progress After Each Step

### Step-by-Step Progress Checks

#### 1. After Generating QA Pairs
```bash
# Check how many QA pairs were generated
python -c "import json; data=json.load(open('data/train/balanced_qa_pairs.json')); print(f'Generated {len(data)} QA pairs')"

# Preview a sample
python -m homework.generate_qa check --info_file data/valid/00000_info.json --view_index 0
```

#### 2. After Generating Captions
```bash
# Check how many captions were generated
python -c "import json; data=json.load(open('data/train/balanced_captions.json')); print(f'Generated {len(data)} caption entries')"

# Preview a sample
python -m homework.generate_captions check --info_file data/valid/00000_info.json --view_index 0
```

#### 3. After Training VLM
```bash
# Quick accuracy check (prints percentage)
python -m homework.finetune test homework/vlm_sft

# Full grader check (shows points)
python3 -m grader homework -v
```

#### 4. After Training CLIP
```bash
# Quick accuracy check (prints percentage)
python -m homework.clip test homework/clip

# Full grader check (shows points for both models)
python3 -m grader homework -v
```

---

## 🎯 Understanding Your Score

### Accuracy Targets
- **70% accuracy = 50 points** (full credit for VLM or CLIP)
- **0-70% accuracy = linear scaling** (e.g., 35% = 25 points)
- **80-85% accuracy = 5 bonus points** (extra credit)

### Full Grader Output
The grader shows:
```
VLM Model Grader
  - Test the answer accuracy        [ 45 / 50 ]
CLIP Model Grader
  - Test the CLIP accuracy          [ 50 / 50 ]
Total                                95 / 100
```

---

## 🔍 Quick Diagnostic Commands

### Check Your Device
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}'); print(f'Will use: {\"cuda\" if torch.cuda.is_available() else \"mps\" if hasattr(torch.backends, \"mps\") and torch.backends.mps.is_available() else \"cpu\"}')"
```

### Check Generated Data Files
```bash
# Count QA pairs
ls -lh data/train/*_qa_pairs.json 2>/dev/null | wc -l
python -c "import json; import glob; files=glob.glob('data/train/*_qa_pairs.json'); print(f'QA files: {len(files)}'); [print(f'  {f}: {len(json.load(open(f)))} pairs') for f in files]"

# Count captions
ls -lh data/train/*_captions.json 2>/dev/null | wc -l
python -c "import json; import glob; files=glob.glob('data/train/*_captions.json'); print(f'Caption files: {len(files)}'); [print(f'  {f}: {len(json.load(open(f)))} entries') for f in files]"
```

### Check Trained Models
```bash
# Check if VLM model exists
ls -lh homework/vlm_sft/adapter_model.safetensors 2>/dev/null && echo "VLM model exists" || echo "VLM model missing"

# Check if CLIP model exists
ls -lh homework/clip/adapter_model.safetensors 2>/dev/null && ls -lh homework/clip/additional_weights.pt 2>/dev/null && echo "CLIP model exists" || echo "CLIP model missing"
```

---

## 📈 Expected Progress Milestones

1. ✅ **Data Generation** (~30 min on CPU, ~5 min on MPS)
   - QA pairs: Expect thousands to tens of thousands
   - Captions: Expect thousands to tens of thousands

2. ✅ **VLM Training** (~1-3 hours on CPU, ~20-40 min on MPS)
   - Look for checkpoints in `homework/vlm_sft/`
   - Test accuracy should be > 70% for full points

3. ✅ **CLIP Training** (~1-3 hours on CPU, ~20-40 min on MPS)
   - Look for checkpoints in `homework/clip/`
   - Test accuracy should be > 70% for full points

---

## 🆘 Troubleshooting

### "Model not found" errors
- Make sure you've trained the model first
- Check that checkpoint path is correct
- Verify files exist: `ls homework/vlm_sft/` or `ls homework/clip/`

### "No data files" errors
- Make sure you've generated QA pairs and captions first
- Check files exist: `ls data/train/*.json`

### Low accuracy scores
- Generate more training data
- Train for more epochs
- Check that data generation is working correctly

### Grader errors
- Make sure you're in the project root directory
- Check that all required files exist
- Try with verbose flag: `python3 -m grader homework -v`

---

## 💡 Pro Tips

1. **Check after data generation** - Make sure you have lots of QA pairs/captions before training
2. **Check after training** - Use quick test commands to see if accuracy is improving
3. **Use full grader before submission** - Make sure everything works with the official grader
4. **Save checkpoints** - The training saves checkpoints automatically, so you can resume if needed

