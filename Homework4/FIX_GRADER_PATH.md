# Fix Grader Model Path Issue

## Problem

The grader expects models at:
- `homework/vlm_model` (for VLM)
- `homework/clip_model` (for CLIP)

But training saves to:
- `homework/vlm_sft` (default VLM output)
- `homework/clip` (default CLIP output)

## Quick Fix Options

### Option 1: Create Symbolic Links (Recommended)

```bash
# For VLM model
cd homework
ln -s vlm_sft vlm_model

# For CLIP model  
ln -s clip clip_model
```

Then run the grader:
```bash
python3 -m grader homework -v
```

### Option 2: Copy/Rename Directories

```bash
# For VLM model
cd homework
cp -r vlm_sft vlm_model

# For CLIP model
cp -r clip clip_model
```

### Option 3: Train to the Expected Directory Names

Train with the correct output directory names:

```bash
# Train VLM to the expected location
python -m homework.finetune train --output_dir vlm_model

# Train CLIP to the expected location
python -m homework.clip train --output_dir clip_model
```

---

## What Files Need to Exist?

The grader expects these files:

**For VLM:**
- `homework/vlm_model/adapter_config.json`
- `homework/vlm_model/adapter_model.safetensors`

**For CLIP:**
- `homework/clip_model/adapter_config.json`
- `homework/clip_model/adapter_model.safetensors`
- `homework/clip_model/additional_weights.pt`

---

## Verify Your Model Files

Check if your models exist:

```bash
# Check VLM model
ls -lh homework/vlm_sft/
# Should see adapter_config.json and adapter_model.safetensors

# Check CLIP model
ls -lh homework/clip/
# Should see adapter_config.json, adapter_model.safetensors, and additional_weights.pt
```

---

## Recommended Workflow

1. **Train your models** (to default locations):
   ```bash
   python -m homework.finetune train
   python -m homework.clip train
   ```

2. **Create symlinks** for the grader:
   ```bash
   cd homework
   ln -s vlm_sft vlm_model
   ln -s clip clip_model
   ```

3. **Run the grader**:
   ```bash
   python3 -m grader homework -v
   ```

This way you keep your original training directories and also have the paths the grader expects.

