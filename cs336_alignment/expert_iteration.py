"""
Expert Iteration (EI) Script for MATH

This script implements Algorithm 2: Expert Iteration.
It alternates between:
1. Generating reasoning traces using the current policy (on vLLM).
2. Filtering for correct answers using the reward function.
3. Fine-tuning the policy on the new correct traces (SFT).

Hardware:
- cuda:0: Policy Model (Training)
- cuda:1: vLLM Instance (Inference/Rollout)
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import wandb
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import DataLoader
from vllm import LLM, SamplingParams

# Import components from existing modules
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils import (
    get_response_log_probs,
    sft_microbatch_train_step,
)
from cs336_alignment.sft import (
    init_vllm,
    load_policy_into_vllm_instance,
    SFTDataset,
    ValidationDataset,
    collate_fn_sft,
    evaluate_on_validation,
    PROMPT_TEMPLATE,
    MODEL_PATH,
    DATA_DIR,
    VAL_DATA_PATH,
    SAVE_DIR,  # Imported as per request
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TRAIN_DATA_PATH = DATA_DIR / "MATH" / "train.jsonl"
GENERATED_DATA_DIR = DATA_DIR / "generated_data"  # Corrected path


def extract_answer_content(text: str) -> str:
    """Helper to ensure we extract clean content for re-training."""
    if not text.strip().startswith("<think>"):
        return "<think>" + text
    return text


def sample_questions(data_path: Path, n_samples: Optional[int] = None) -> List[Dict]:
    """Load questions from the source dataset."""
    questions = []
    with open(data_path, "r") as f:
        for line in f:
            questions.append(json.loads(line))
    
    if n_samples and n_samples < len(questions):
        random.shuffle(questions)
        questions = questions[:n_samples]
    return questions


def generate_and_filter_data(
    llm: LLM,
    policy_model: PreTrainedModel,
    questions: List[Dict],
    args,
    step_idx: int,
    retry_idx: int = 0,
) -> Path:
    """
    Step 3-7 of Algorithm 2: Sample, Evaluate, Filter.
    Returns path to the new temporary dataset.
    """
    print(f"\n[Step {step_idx} (Try {retry_idx+1})] Generating rollouts for {len(questions)} questions...")
    
    # 1. Sync weights (Only needed on first try of a step, but safe to redo)
    # Note: In a retry loop, policy hasn't changed, but checking this is fast.
    print(f"[Step {step_idx}] Syncing weights to vLLM...")
    load_policy_into_vllm_instance(policy_model, llm)
    
    # 2. Prepare Prompts
    prompts = [PROMPT_TEMPLATE.format(question=q["problem"]) for q in questions]
    
    # 3. vLLM Sampling Params
    # Use retry_idx to alter seed if we are retrying
    current_seed = args.seed + (step_idx * 100) + retry_idx
    
    sampling_params = SamplingParams(
        temperature=args.sampling_temperature,
        max_tokens=args.sampling_max_tokens,
        min_tokens=args.sampling_min_tokens, # Use argument
        n=args.n_rollouts,
        stop=["</answer>"],
        seed=current_seed,
    )
    
    # 4. Generate
    outputs = llm.generate(prompts, sampling_params)
    
    new_data_buffer = []
    stats = {"total": 0, "correct": 0, "format_error": 0}
    
    print(f"[Step {step_idx}] Filtering outputs...")
    
    for i, output in enumerate(outputs):
        question_data = questions[i]
        ground_truth = question_data["answer"]
        
        for sample in output.outputs:
            stats["total"] += 1
            generated_text = sample.text
            
            # Close the tag if vLLM stopped exactly at </answer>
            if "</answer>" not in generated_text:
                generated_text += "</answer>"
                
            # Prepend <think> if missing (for consistent grading)
            full_response_for_grading = "<think>" + generated_text if not generated_text.strip().startswith("<think>") else generated_text
            
            reward_metrics = r1_zero_reward_fn(full_response_for_grading, ground_truth)
            
            if reward_metrics["reward"] == 1.0:
                stats["correct"] += 1
                
                new_data_buffer.append({
                    "prompt": prompts[i],
                    "response": extract_answer_content(generated_text),
                    "original_question": question_data
                })
            elif reward_metrics["format_reward"] == 0.0:
                stats["format_error"] += 1

    # 5. Save to temporary file
    GENERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = GENERATED_DATA_DIR / f"ei_step_{step_idx}_G{args.n_rollouts}.jsonl"
    
    with open(output_path, "w") as f:
        for item in new_data_buffer:
            f.write(json.dumps(item) + "\n")
            
    print(f"[Step {step_idx}] Generation stats: {stats}")
    print(f"[Step {step_idx}] Correct/Total: {stats['correct']}/{stats['total']} ({stats['correct']/stats['total']:.2%})")
    print(f"[Step {step_idx}] Saved {len(new_data_buffer)} training examples to {output_path}")
    
    if wandb.run is not None:
        wandb.log({
            "ei/generation_acc": stats['correct'] / (stats['total'] + 1e-9),
            "ei/num_new_samples": len(new_data_buffer),
            "ei_step": step_idx
        })
        
    return output_path


def train_step_with_entropy(
    model: PreTrainedModel,
    tokenizer,
    train_dataset: SFTDataset,
    args,
    step_idx: int
):
    """
    Step 8: Custom SFT Training Loop that logs Entropy.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Calculate steps
    total_steps = (len(train_dataset) // args.batch_size) * args.epochs_per_step
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_sft(batch, tokenizer),
    )
    
    model.train()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        
    print(f"[Step {step_idx}] Starting SFT on {len(train_dataset)} examples for {args.epochs_per_step} epochs...")
    
    global_micro_step = 0
    
    for epoch in range(args.epochs_per_step):
        pbar = tqdm(train_loader, desc=f"EI Step {step_idx} - Epoch {epoch+1}")
        
        epoch_entropy = []
        epoch_loss = []

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(args.device)
            labels = batch["labels"].to(args.device)
            response_mask = batch["response_mask"].to(args.device)
            
            # --- Forward Pass with Entropy ---
            log_probs_output = get_response_log_probs(
                model, input_ids, labels, return_token_entropy=True
            )
            policy_log_probs = log_probs_output["log_probs"]
            token_entropies = log_probs_output["token_entropy"]
            
            # Calculate metrics
            num_response_tokens = response_mask.sum().item()
            
            # Compute average entropy for *response tokens only*
            avg_entropy = (token_entropies * response_mask).sum() / (num_response_tokens + 1e-9)
            epoch_entropy.append(avg_entropy.item())
            
            # Standard SFT Loss
            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                normalize_constant=num_response_tokens
            )
            
            epoch_loss.append(metadata["loss"].item())
            
            # Backward
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip 1.0 per prompt
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_micro_step += 1
                
                if wandb.run is not None and global_micro_step % 10 == 0:
                    wandb.log({
                        "train/loss": metadata["loss"].item(),
                        "train/entropy": avg_entropy.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "ei_step": step_idx
                    })
            
            pbar.set_postfix({
                "loss": f"{metadata['loss'].item():.4f}", 
                "ent": f"{avg_entropy.item():.4f}"
            })

    print(f"[Step {step_idx}] Finished Training. Avg Loss: {np.mean(epoch_loss):.4f}, Avg Entropy: {np.mean(epoch_entropy):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Expert Iteration for Math")
    
    # Experiment Variations
    # Reasonable defaults: G=4, full data
    parser.add_argument("--n_rollouts", type=int, default=4, help="G: Number of rollouts per question")
    parser.add_argument("--epochs_per_step", type=int, default=1, help="SFT epochs per EI step")
    parser.add_argument("--n_ei_steps", type=int, default=5, help="Total EI iterations")
    
    # Data & Sampling
    parser.add_argument("--num_train_samples", type=int, default=None, help="Subset of questions to use")
    parser.add_argument("--num_eval_samples",type=int,default=5000,help="Number of validation samples to evaluate on")
    parser.add_argument("--sampling_temperature", type=float, default=0.7)
    parser.add_argument("--sampling_max_tokens", type=int, default=1024)
    parser.add_argument("--sampling_min_tokens", type=int, default=4, help="Min tokens for generation")
    
    # Training Hyperparams
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5) # Usually lower for EI
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    
    # System
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy_device", type=str, default="cuda:0")
    parser.add_argument("--vllm_device", type=str, default="cuda:1")
    parser.add_argument("--save_dir", type=str, default=str(SAVE_DIR), help="Path to save checkpoints")
    parser.add_argument("--wandb_project", type=str, default="expert-iteration-math")
    
    args = parser.parse_args()
    args.device = args.policy_device
    
    # Init Random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Init WandB
    run_name = f"EI_G{args.n_rollouts}_E{args.epochs_per_step}"
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
    
    # 1. Load Resources
    print("Loading Policy Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    policy_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        device_map=args.policy_device
    )
    
    print("Initializing vLLM...")
    llm = init_vllm(str(MODEL_PATH), args.vllm_device, args.seed)
    
    print("Loading Datasets...")
    # Source questions (D)
    questions = sample_questions(TRAIN_DATA_PATH, args.num_train_samples)
    # Validation set for evaluation curves
    val_dataset = ValidationDataset(VAL_DATA_PATH)
    
    # 2. Initial Evaluation (Baseline)
    print("Running Initial Evaluation...")
    policy_model.eval()
    load_policy_into_vllm_instance(policy_model, llm)
    metrics, _ = evaluate_on_validation(llm, tokenizer, val_dataset, args.num_eval_samples)
    print(f"Baseline Accuracy: {metrics['eval/accuracy']:.4f}")
    wandb.log({**metrics, "ei_step": 0})
    
    # 3. Expert Iteration Loop
    for step in range(1, args.n_ei_steps + 1):
        print(f"\n{'='*40}\nStarting Expert Iteration Step {step}/{args.n_ei_steps}\n{'='*40}")
        
        # --- A. Generate & Filter (Rollout) with RETRY Logic ---
        policy_model.eval()
        retry_count = 0
        max_retries = 5
        train_dataset = None
        
        while retry_count < max_retries:
            # Generate
            new_data_path = generate_and_filter_data(
                llm, policy_model, questions, args, step, retry_idx=retry_count
            )
            
            # Load and check length using SFTDataset
            # This handles the "primitive" line counting issue naturally
            try:
                potential_dataset = SFTDataset(new_data_path)
                if len(potential_dataset) > 0:
                    train_dataset = potential_dataset
                    print(f"[Step {step}] Successfully generated {len(train_dataset)} correct samples.")
                    break
                else:
                    print(f"[Warning] Generated 0 correct samples on try {retry_count+1}. Retrying...")
            except Exception as e:
                print(f"[Error] Failed to load generated data: {e}. Retrying...")
                
            retry_count += 1
        
        # If we failed after max_retries, we might have to skip or exit
        if train_dataset is None or len(train_dataset) == 0:
            print(f"[Critical] Failed to generate any correct samples after {max_retries} attempts. Skipping Step {step}.")
            continue
            
        # --- B. Train (SFT) ---
        # train_dataset is already instantiated
        train_step_with_entropy(policy_model, tokenizer, train_dataset, args, step)
        
        # --- C. Evaluate ---
        print(f"[Step {step}] Validating...")
        policy_model.eval()
        load_policy_into_vllm_instance(policy_model, llm)
        
        metrics, logs = evaluate_on_validation(
            llm, 
            tokenizer, 
            val_dataset, 
            args.num_eval_samples # Sample for speed, or use full for final
        )
        
        print(f"Step {step} Validation Accuracy: {metrics['eval/accuracy']:.4f}")
        wandb.log({**metrics, "ei_step": step})
        
        # Log Table Samples
        if wandb.run is not None:
            table = wandb.Table(columns=["Prompt", "Response", "GT", "Reward"])
            for log in logs[:5]:
                table.add_data(log["prompt"][:100], log["response"], str(log["ground_truth"]), log["total_reward"])
            wandb.log({f"eval/samples_step_{step}": table})

    # 4. Save Final Model
    save_path = Path(args.save_dir) / run_name
    print(f"Saving final model to {save_path}...")
    policy_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    wandb.finish()

if __name__ == "__main__":
    main()