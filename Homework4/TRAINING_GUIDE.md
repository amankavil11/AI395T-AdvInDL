# Training and Testing Guide

This guide walks you through the complete process of generating data, training models, and testing them.

## Step 1: Generate Training Data

### Generate QA Pairs for VLM Training

First, generate question-answer pairs from the training data:

```bash
python -m homework.generate_qa generate_all --data_dir data/train --output_file data/train/balanced_qa_pairs.json
```

This will:
- Process all `*_info.json` files in `data/train/`
- Generate QA pairs for all views in each file
- Save the results to `data/train/balanced_qa_pairs.json`

**Optional: Preview QA pairs** before generating all:
```bash
python -m homework.generate_qa check --info_file data/valid/00000_info.json --view_index 0
```

### Generate Captions for CLIP Training

Next, generate image-caption pairs for CLIP training:

```bash
python -m homework.generate_captions generate_all --data_dir data/train --output_file data/train/balanced_captions.json
```

This will:
- Process all `*_info.json` files in `data/train/`
- Generate captions for all views in each file
- Save the results to `data/train/balanced_captions.json`

**Optional: Preview captions** before generating all:
```bash
python -m homework.generate_captions check --info_file data/valid/00000_info.json --view_index 0
```

---

## Step 2: Train the VLM Model

Train the Vision-Language Model on the generated QA pairs:

```bash
python -m homework.finetune train
```

This uses default parameters. You can customize training:

```bash
python -m homework.finetune train \
    --train_dataset_name train \
    --output_dir vlm_sft \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-4
```

The model will be saved to `homework/vlm_sft/` (or your specified `output_dir`).

**Checkpoints are saved** during training. Look for:
- `adapter_config.json`
- `adapter_model.safetensors`

---

## Step 3: Test the VLM Model

Test your trained VLM model on the validation set:

```bash
python -m homework.finetune test homework/vlm_sft
```

Replace `homework/vlm_sft` with your actual checkpoint path if you used a different `output_dir`.

The test command will:
- Load your trained model
- Evaluate on `valid_grader` dataset
- Print the accuracy

**Note**: According to the README, you need 70% accuracy to get full points (50pts) for Part 1.

---

## Step 4: Train the CLIP Model

Train the CLIP model on the generated captions:

```bash
python -m homework.clip train
```

This uses default parameters. You can customize training:

```bash
python -m homework.clip train \
    --output_dir clip \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size 1024 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-4
```

The model will be saved to `homework/clip/` (or your specified `output_dir`).

**Checkpoints are saved** during training. Look for:
- `adapter_config.json`
- `adapter_model.safetensors`
- `additional_weights.pt` (contains projection layer weights)

---

## Step 5: Test the CLIP Model

Test your trained CLIP model on the validation set:

```bash
python -m homework.clip test homework/clip
```

Replace `homework/clip` with your actual checkpoint path if you used a different `output_dir`.

The test command will:
- Load your trained model
- Evaluate on `valid_grader` dataset (multi-choice QA)
- Print the accuracy

**Note**: According to the README, you need 70% accuracy to get full points (50pts) for Part 2.

---

## Quick Reference

### Data Generation Commands

```bash
# Generate QA pairs
python -m homework.generate_qa generate_all

# Generate captions
python -m homework.generate_captions generate_all

# Preview QA pairs (optional)
python -m homework.generate_qa check --info_file data/valid/00000_info.json --view_index 0

# Preview captions (optional)
python -m homework.generate_captions check --info_file data/valid/00000_info.json --view_index 0
```

### Training Commands

```bash
# Train VLM
python -m homework.finetune train

# Train CLIP
python -m homework.clip train
```

### Testing Commands

```bash
# Test VLM (replace with your checkpoint path)
python -m homework.finetune test homework/vlm_sft

# Test CLIP (replace with your checkpoint path)
python -m homework.clip test homework/clip
```

---

## GPU vs CPU

### Device Detection
The code automatically detects and uses the best available device:
- **CUDA** (NVIDIA GPU) - fastest, if available
- **MPS** (Apple Silicon GPU) - fast, if available (you have this!)
- **CPU** - slowest, but works

You **don't need a GPU** - CPU will work but will be much slower. With your Apple Silicon (MPS), training will be faster than CPU.

### Check Your Device
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)"
```

### Performance Expectations
- **MPS (Apple Silicon)**: ~5-10x faster than CPU
- **CUDA (NVIDIA GPU)**: ~10-50x faster than CPU  
- **CPU**: Works but slow - expect hours of training

---

## Checking Your Progress/Score

### Method 1: Quick Accuracy Check (After Training)
After training each model, you can quickly check accuracy:

```bash
# Check VLM accuracy
python -m homework.finetune test homework/vlm_sft

# Check CLIP accuracy  
python -m homework.clip test homework/clip
```

These will print the accuracy percentage. Remember:
- **70% accuracy = 50 points** (full credit for that part)
- Accuracy scales linearly down to 0% = 0 points
- 80-85% accuracy = extra credit (5 points)

### Method 2: Full Local Grader (Most Accurate)
Use the grader to get your full score breakdown:

```bash
# Grade your homework directory
python3 -m grader homework

# Or with verbose output
python3 -m grader homework -v
```

This will:
- Test both VLM and CLIP models
- Give you point breakdowns
- Show exactly what's working and what's not
- Match what the online grader will do

### Method 3: Test Individual Components
You can test specific parts:

```bash
# Test data generation (check a sample)
python -m homework.generate_qa check --info_file data/valid/00000_info.json --view_index 0
python -m homework.generate_captions check --info_file data/valid/00000_info.json --view_index 0

# Check how many QA pairs/captions were generated
python -c "import json; data=json.load(open('data/train/balanced_qa_pairs.json')); print(f'QA pairs: {len(data)}')"
python -c "import json; data=json.load(open('data/train/balanced_captions.json')); print(f'Captions: {len(data)}')"
```

---

## Troubleshooting

### Issue: Out of Memory
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps` (to maintain effective batch size)
- Reduce `num_workers`

### Issue: Training is too slow
- Increase `num_workers` (if you have CPU cores available)
- Use a GPU if available (CUDA or MPS for Apple Silicon)

### Issue: Model accuracy is low
- Generate more training data (you might be missing some info files)
- Train for more epochs
- Adjust learning rate
- Check that your data generation functions are working correctly

### Issue: File not found errors
- Make sure you've generated the QA pairs and captions JSON files first
- Check that files are in the correct `data/train/` directory
- Verify the JSON files follow the expected format (check demo files)

---

## Expected File Structure

After data generation, you should have:
```
data/train/
  ├── balanced_qa_pairs.json      (generated by generate_qa)
  ├── balanced_captions.json      (generated by generate_captions)
  ├── 00000_info.json
  ├── 00001_info.json
  ├── ...
  └── [image files]
```

After training, you should have:
```
homework/
  ├── vlm_sft/                    (or your VLM output_dir)
  │   ├── adapter_config.json
  │   └── adapter_model.safetensors
  └── clip/                       (or your CLIP output_dir)
      ├── adapter_config.json
      ├── adapter_model.safetensors
      └── additional_weights.pt
```

---

## Grading Targets

- **Part 1 (VLM)**: Need 70% accuracy for full 50 points
- **Part 2 (CLIP)**: Need 70% accuracy for full 50 points
- **Extra Credit**: 85% accuracy gets 5 bonus points (linearly from 80%)

Good luck with your training!

