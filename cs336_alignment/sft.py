"""
Supervised Fine-Tuning (SFT) Script for Qwen 2.5 Math 1.5B

This script implements robust SFT training with:
- Dynamic dataset sub-sampling
- Multi-GPU orchestration (Policy on cuda:0, vLLM on cuda:1)
- Comprehensive WandB logging
- Periodic validation evaluation on MATH dataset
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

import numpy as np
import torch
import wandb
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

# Import local utilities
from cs336_alignment.utils import (
    get_response_log_probs,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    get_linear_schedule_with_warmup,
)

# vLLM Imports
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

# -----------------------------------------------------------------------------
# Path Configuration
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
MODEL_NAME = "Qwen2.5-Math-1.5B"
MODEL_PATH = MODEL_DIR / MODEL_NAME
SFT_DATA_PATH = DATA_DIR / "MATH" / "sft.jsonl"
VAL_DATA_PATH = DATA_DIR / "MATH" / "validation.jsonl"
SAVE_DIR = BASE_DIR / "checkpoints"

PROMPT_TEMPLATE = (
    "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
    "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
    "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, "
    "respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n"
    "User: {question}\n"
    "Assistant: <think>"
)


# -----------------------------------------------------------------------------
# vLLM Helper Functions
# -----------------------------------------------------------------------------
def init_vllm(
    model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85
) -> LLM:
    """
    Initialize vLLM inference engine on a specific device.

    Args:
        model_id: Path to the model
        device: Device to place vLLM (e.g., 'cuda:1')
        seed: Random seed for reproducibility
        gpu_memory_utilization: Fraction of GPU memory to use

    Returns:
        Initialized vLLM instance
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL to enable single-GPU vLLM placement
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Load the current policy weights into the vLLM instance.

    Args:
        policy: The policy model
        llm: The vLLM instance
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


# -----------------------------------------------------------------------------
# Data Handling
# -----------------------------------------------------------------------------
class SFTDataset(Dataset):
    """Dataset for SFT training."""

    def __init__(self, data_path: Path, num_samples: Optional[int] = None):
        """
        Initialize the dataset.

        Args:
            data_path: Path to the JSONL data file
            num_samples: Number of samples to load (None = load all)
        """
        self.examples = []
        with open(data_path, "r") as f:
            for line in f:
                self.examples.append(json.loads(line))

        # Sub-sample if requested
        if num_samples is not None and num_samples < len(self.examples):
            random.shuffle(self.examples)
            self.examples = self.examples[:num_samples]

        print(f"Loaded {len(self.examples)} training examples from {data_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.examples[idx]


class ValidationDataset(Dataset):
    """Dataset for validation evaluation."""

    def __init__(self, data_path: Path):
        """
        Initialize the validation dataset.

        Args:
            data_path: Path to the validation JSONL file
        """
        self.examples = []
        with open(data_path, "r") as f:
            for line in f:
                self.examples.append(json.loads(line))
        print(f"Loaded {len(self.examples)} validation examples from {data_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.examples[idx]


def collate_fn_sft(batch: List[Dict], tokenizer) -> Dict[str, torch.Tensor]:
    """
    Collate function for SFT training.

    Args:
        batch: List of examples from the dataset
        tokenizer: Tokenizer to use

    Returns:
        Dictionary with tokenized inputs, labels, and response masks
    """
    prompts = [item["prompt"] for item in batch]
    responses = [item["response"] for item in batch]

    # Use the utility function to tokenize
    tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)

    return tokenized


# -----------------------------------------------------------------------------
# Evaluation Logic
# -----------------------------------------------------------------------------
def evaluate_on_validation(
    llm: LLM,
    tokenizer,
    val_dataset: ValidationDataset,
    num_eval_samples: int = 100,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> Dict[str, float]:
    """
    Evaluate the model on the validation set using vLLM.

    Args:
        llm: vLLM instance with loaded policy weights
        tokenizer: Tokenizer
        val_dataset: Validation dataset
        num_eval_samples: Number of samples to evaluate on
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Dictionary of evaluation metrics
    """
    # Sample validation examples
    eval_indices = random.sample(
        range(len(val_dataset)), min(num_eval_samples, len(val_dataset))
    )

    prompts = []
    ground_truths = []
    for idx in eval_indices:
        example = val_dataset[idx]
        question = example["problem"]
        prompt = PROMPT_TEMPLATE.format(question=question)
        prompts.append(prompt)
        ground_truths.append(example["answer"])

    # Generate responses using vLLM
    sampling_params = SamplingParams(
        temperature=temperature, max_tokens=max_tokens, stop=["</answer>"]
    )

    outputs = llm.generate(prompts, sampling_params)

    # Evaluate responses
    correct_count = 0
    format_correct_count = 0
    total_reward = 0.0
    response_lengths = []

    eval_logs = []

    for i, output in enumerate(outputs):
        response_text = output.outputs[0].text
        # Add back the stop token for complete formatting
        if "</answer>" not in response_text:
            response_text += " </answer>"

        ground_truth = ground_truths[i]

        # Compute rewards
        reward_metrics = r1_zero_reward_fn(response_text, ground_truth)

        if reward_metrics["answer_reward"] == 1.0:
            correct_count += 1
        if reward_metrics["format_reward"] == 1.0:
            format_correct_count += 1

        total_reward += reward_metrics["reward"]
        response_lengths.append(len(tokenizer.encode(response_text)))

        eval_logs.append(
            {
                "prompt": prompts[i],
                "response": response_text,
                "ground_truth": ground_truth,
                "format_reward": reward_metrics["format_reward"],
                "answer_reward": reward_metrics["answer_reward"],
                "total_reward": reward_metrics["reward"],
            }
        )

    # Calculate metrics
    accuracy = correct_count / len(eval_indices)
    format_accuracy = format_correct_count / len(eval_indices)
    avg_reward = total_reward / len(eval_indices)
    avg_response_length = np.mean(response_lengths)

    metrics = {
        "eval/accuracy": accuracy,
        "eval/format_accuracy": format_accuracy,
        "eval/avg_reward": avg_reward,
        "eval/avg_response_length": avg_response_length,
    }

    return metrics, eval_logs


# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
def train_sft(
    model: PreTrainedModel,
    tokenizer,
    train_dataset: SFTDataset,
    val_dataset: ValidationDataset,
    llm: LLM,
    args,
) -> PreTrainedModel:
    """
    Main SFT training loop.

    Args:
        model: Policy model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        val_dataset: Validation dataset
        llm: vLLM instance for evaluation
        args: Training arguments

    Returns:
        Trained model
    """
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Setup learning rate scheduler
    total_steps = (len(train_dataset) // args.batch_size) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Setup dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_sft(batch, tokenizer),
    )

    # Training state
    global_step = 0
    eval_step = 0
    model.train()

    print(f"\n{'=' * 80}")
    print(f"Starting SFT Training")
    print(f"{'=' * 80}")
    print(f"Total examples: {len(train_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Total epochs: {args.num_epochs}")
    print(f"Total steps: {total_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"{'=' * 80}\n")

    # Training loop
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}", leave=True
        )

        for step, batch in enumerate(pbar):
            # Move batch to device
            input_ids = batch["input_ids"].to(args.device)
            labels = batch["labels"].to(args.device)
            response_mask = batch["response_mask"].to(args.device)

            # Compute log probabilities
            with torch.no_grad():
                log_probs_output = get_response_log_probs(
                    model, input_ids, labels, return_token_entropy=False
                )
            policy_log_probs = log_probs_output["log_probs"]

            # Compute number of response tokens for normalization
            num_response_tokens = response_mask.sum().item()

            # Training step
            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                normalize_constant=num_response_tokens,
            )

            epoch_loss += metadata["loss"].item()

            # Update progress bar
            pbar.set_postfix({"loss": f"{metadata['loss'].item():.4f}"})

            # Gradient accumulation
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Log to WandB
                if wandb.run is not None:
                    wandb.log(
                        {
                            "train/loss": metadata["loss"].item(),
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train_step": global_step,
                        }
                    )

                # Periodic evaluation
                if global_step % args.eval_every == 0:
                    print(f"\n{'=' * 80}")
                    print(f"Running evaluation at step {global_step}")
                    print(f"{'=' * 80}")

                    model.eval()

                    # Load current policy weights into vLLM
                    load_policy_into_vllm_instance(model, llm)

                    # Evaluate
                    eval_metrics, eval_logs = evaluate_on_validation(
                        llm,
                        tokenizer,
                        val_dataset,
                        num_eval_samples=args.num_eval_samples,
                        temperature=args.eval_temperature,
                        max_tokens=args.max_eval_tokens,
                    )

                    # Log evaluation metrics
                    print(f"\nEvaluation Results:")
                    print(f"  Accuracy: {eval_metrics['eval/accuracy']:.4f}")
                    print(
                        f"  Format Accuracy: {eval_metrics['eval/format_accuracy']:.4f}"
                    )
                    print(f"  Avg Reward: {eval_metrics['eval/avg_reward']:.4f}")
                    print(
                        f"  Avg Response Length: {eval_metrics['eval/avg_response_length']:.1f}"
                    )

                    if wandb.run is not None:
                        # Create WandB table for sample outputs
                        table = wandb.Table(
                            columns=[
                                "Prompt",
                                "Response",
                                "Ground Truth",
                                "Format Reward",
                                "Answer Reward",
                                "Total Reward",
                            ]
                        )

                        # Add first 10 examples to table
                        for log in eval_logs[:10]:
                            table.add_data(
                                log["prompt"][:200] + "...",  # Truncate for readability
                                log["response"],
                                str(log["ground_truth"]),
                                log["format_reward"],
                                log["answer_reward"],
                                log["total_reward"],
                            )

                        eval_metrics["eval/samples_table"] = table
                        eval_metrics["eval_step"] = eval_step

                        wandb.log(eval_metrics)

                    eval_step += 1
                    model.train()

                    print(f"{'=' * 80}\n")

        # End of epoch logging
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}\n")

    return model


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SFT Training for Qwen 2.5 Math 1.5B")

    # Data arguments
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of training samples to use (None = use all)",
    )
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=100,
        help="Number of validation samples to evaluate on",
    )

    # Model arguments
    parser.add_argument(
        "--model_path", type=str, default=str(MODEL_PATH), help="Path to base model"
    )

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-6, help="Learning rate"
    )
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping",
    )

    # Evaluation arguments
    parser.add_argument(
        "--eval_every", type=int, default=50, help="Evaluate every N steps"
    )
    parser.add_argument(
        "--eval_temperature", type=float, default=0.7, help="Temperature for evaluation"
    )
    parser.add_argument(
        "--max_eval_tokens", type=int, default=1024, help="Max tokens for evaluation"
    )

    # Device arguments
    parser.add_argument(
        "--policy_device", type=str, default="cuda:0", help="Device for policy model"
    )
    parser.add_argument(
        "--vllm_device", type=str, default="cuda:1", help="Device for vLLM"
    )

    # WandB arguments
    parser.add_argument(
        "--wandb_project", type=str, default="sft-qwen-math", help="WandB project name"
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None, help="WandB run name"
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Checkpoint saving
    parser.add_argument(
        "--save_model", action="store_true", help="Save model after training"
    )
    parser.add_argument(
        "--save_path", type=str, default=None, help="Path to save model"
    )

    args = parser.parse_args()

    # Set device
    args.device = args.policy_device

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Initialize WandB
    if not args.no_wandb:
        run_name = args.wandb_run_name
        if run_name is None:
            run_name = f"sft_samples{args.num_samples if args.num_samples else 'full'}_lr{args.learning_rate}_bs{args.batch_size * args.gradient_accumulation_steps}"

        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

        # Setup WandB metrics
        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*", step_metric="eval_step")

    print(f"\n{'=' * 80}")
    print(f"SFT Training Configuration")
    print(f"{'=' * 80}")
    for key, value in sorted(vars(args).items()):
        print(f"{key:30s}: {value}")
    print(f"{'=' * 80}\n")

    # Load tokenizer
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load policy model
    print(f"Loading policy model on {args.policy_device}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map=args.policy_device
    )
    model.train()

    # Initialize vLLM
    print(f"Initializing vLLM on {args.vllm_device}...")
    llm = init_vllm(args.model_path, args.vllm_device, args.seed)

    # Load datasets
    print(f"\nLoading datasets...")
    train_dataset = SFTDataset(SFT_DATA_PATH, num_samples=args.num_samples)
    val_dataset = ValidationDataset(VAL_DATA_PATH)

    # Train
    print(f"\nStarting training...\n")
    trained_model = train_sft(model, tokenizer, train_dataset, val_dataset, llm, args)

    # Final evaluation
    print(f"\n{'=' * 80}")
    print(f"Running final evaluation")
    print(f"{'=' * 80}")

    trained_model.eval()
    load_policy_into_vllm_instance(trained_model, llm)

    final_metrics, final_logs = evaluate_on_validation(
        llm,
        tokenizer,
        val_dataset,
        num_eval_samples=len(val_dataset),  # Evaluate on full validation set
        temperature=args.eval_temperature,
        max_tokens=args.max_eval_tokens,
    )

    print(f"\nFinal Evaluation Results:")
    print(f"  Accuracy: {final_metrics['eval/accuracy']:.4f}")
    print(f"  Format Accuracy: {final_metrics['eval/format_accuracy']:.4f}")
    print(f"  Avg Reward: {final_metrics['eval/avg_reward']:.4f}")
    print(f"  Avg Response Length: {final_metrics['eval/avg_response_length']:.1f}")

    if not args.no_wandb:
        final_metrics_log = {
            "final_eval/accuracy": final_metrics["eval/accuracy"],
            "final_eval/format_accuracy": final_metrics["eval/format_accuracy"],
            "final_eval/avg_reward": final_metrics["eval/avg_reward"],
            "final_eval/avg_response_length": final_metrics["eval/avg_response_length"],
        }
        wandb.log(final_metrics_log)

    # Save model
    if args.save_model:
        save_path = args.save_path
        if save_path is None:
            save_path = (
                SAVE_DIR
                / f"sft_samples{args.num_samples if args.num_samples else 'full'}"
            )
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving model to {save_path}...")
        trained_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model saved successfully!")

    if not args.no_wandb:
        wandb.finish()

    print(f"\n{'=' * 80}")
    print(f"Training completed!")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
