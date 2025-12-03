#!/usr/bin/env python3
"""
Limit training dataset size for faster training.
Run this if you have too many QA pairs.
"""
import json
import glob
from pathlib import Path

def limit_qa_pairs(max_pairs=10000):
    """Limit QA pairs to specified number."""
    files = glob.glob('data/train/*_qa_pairs.json')
    
    if not files:
        print("No QA pair files found!")
        return
    
    # Load all pairs
    all_pairs = []
    for f in files:
        with open(f) as file:
            data = json.load(file)
            all_pairs.extend(data)
    
    original_count = len(all_pairs)
    print(f"Original: {original_count:,} QA pairs")
    
    if original_count <= max_pairs:
        print(f"Already at or below {max_pairs:,} pairs. No need to limit.")
        return
    
    # Limit
    limited_pairs = all_pairs[:max_pairs]
    print(f"Limited to: {len(limited_pairs):,} QA pairs")
    
    # Save to first file (overwrite)
    main_file = files[0]
    with open(main_file, 'w') as f:
        json.dump(limited_pairs, f, indent=2)
    
    print(f"Saved {len(limited_pairs):,} pairs to {main_file}")
    print(f"\nEstimated training time:")
    print(f"  Before: ~{original_count * 0.01 / 60:.1f} hours per epoch")
    print(f"  After:  ~{len(limited_pairs) * 0.01 / 60:.1f} hours per epoch")

if __name__ == "__main__":
    import sys
    max_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    limit_qa_pairs(max_pairs)

