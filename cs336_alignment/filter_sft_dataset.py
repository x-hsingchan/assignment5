#!/usr/bin/env python3
"""
Filter SFT Dataset for Correct Answers Only

This script:
1. Loads the full SFT dataset
2. Evaluates each example using r1_zero_reward_fn
3. Keeps only examples that produce correct answers
4. Saves the filtered dataset
5. Reports statistics
"""

import json
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports


from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def filter_sft_dataset(
    input_path: Path,
    output_path: Path,
    verbose: bool = True
) -> dict:
    """
    Filter SFT dataset to keep only correct examples.
    
    Args:
        input_path: Path to original sft.jsonl
        output_path: Path to save filtered dataset
        verbose: Print detailed statistics
        
    Returns:
        Dictionary with filtering statistics
    """
    
    print("=" * 80)
    print("SFT Dataset Filtering for Correct Answers")
    print("=" * 80)
    print()
    
    # Load original dataset
    print(f"Loading original dataset from: {input_path}")
    original_examples = []
    with open(input_path, "r") as f:
        for line in f:
            original_examples.append(json.loads(line))
    
    print(f"Original dataset size: {len(original_examples)} examples")
    print()
    
    # Filter examples
    print("Filtering examples...")
    filtered_examples = []
    stats = {
        "total": len(original_examples),
        "correct": 0,
        "format_correct": 0,
        "format_incorrect": 0,
        "answer_correct_given_format": 0,
        "answer_incorrect_given_format": 0,
    }
    
    for example in tqdm(original_examples, desc="Evaluating examples"):
        response = example["response"]
        ground_truth = example["ground_truth"]
        
        # Evaluate using the grader
        reward_metrics = r1_zero_reward_fn(response, ground_truth)
        
        # Track statistics
        if reward_metrics["format_reward"] == 1.0:
            stats["format_correct"] += 1
            
            if reward_metrics["answer_reward"] == 1.0:
                stats["answer_correct_given_format"] += 1
                stats["correct"] += 1
                # Keep this example
                filtered_examples.append(example)
            else:
                stats["answer_incorrect_given_format"] += 1
        else:
            stats["format_incorrect"] += 1
    
    print()
    print("=" * 80)
    print("Filtering Results")
    print("=" * 80)
    print(f"Original dataset size:              {stats['total']:,}")
    print(f"Filtered dataset size:              {stats['correct']:,}")
    print(f"Reduction:                          {stats['total'] - stats['correct']:,} examples removed")
    print(f"Retention rate:                     {100 * stats['correct'] / stats['total']:.2f}%")
    print()
    print(f"Format correct:                     {stats['format_correct']:,} ({100 * stats['format_correct'] / stats['total']:.2f}%)")
    print(f"Format incorrect:                   {stats['format_incorrect']:,} ({100 * stats['format_incorrect'] / stats['total']:.2f}%)")
    print()
    print(f"Answer correct (given format OK):   {stats['answer_correct_given_format']:,} ({100 * stats['answer_correct_given_format'] / stats['format_correct']:.2f}%)")
    print(f"Answer incorrect (given format OK): {stats['answer_incorrect_given_format']:,} ({100 * stats['answer_incorrect_given_format'] / stats['format_correct']:.2f}%)")
    print("=" * 80)
    print()
    
    # Save filtered dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving filtered dataset to: {output_path}")
    
    with open(output_path, "w") as f:
        for example in filtered_examples:
            f.write(json.dumps(example) + "\n")
    
    print(f"✓ Filtered dataset saved successfully!")
    print()
    
    # Show sample examples
    if verbose and len(filtered_examples) > 0:
        print("=" * 80)
        print("Sample Filtered Examples (first 3)")
        print("=" * 80)
        for i, example in enumerate(filtered_examples[:3]):
            print(f"\nExample {i+1}:")
            print(f"Prompt: {example['prompt'][:100]}...")
            print(f"Response: {example['response'][:150]}...")
            print(f"Ground Truth: {example['ground_truth']}")
            print("-" * 80)
    
    return stats


def main():
    """Main function to filter the SFT dataset."""
    
    # Define paths
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data" / "MATH"
    
    input_path = DATA_DIR / "sft.jsonl"
    output_path = DATA_DIR / "sft_filtered.jsonl"
    
    # Check input file exists
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        print("Please ensure the SFT dataset is in the correct location.")
        return 1
    
    # Filter dataset
    stats = filter_sft_dataset(input_path, output_path, verbose=True)
    
    # Save statistics
    stats_path = DATA_DIR / "filtering_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to: {stats_path}")
    print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"✓ Filtered dataset created: {output_path}")
    print(f"✓ Dataset size: {stats['correct']:,} examples ({100 * stats['correct'] / stats['total']:.2f}% of original)")
    print(f"✓ Ready for training!")
    print()
    print("Next steps:")
    print(f"  1. Run SFT on filtered dataset:")
    print(f"     python sft.py --train_data {output_path} --wandb_run_name sft_filtered_full")
    print()
    print(f"  2. Compare with original full dataset results")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
