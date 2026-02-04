import torch
import copy
import torch.nn.functional as F
from transformers import PreTrainedModel


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    """
    Tokenize the prompt and output strings, and construct a mask that is 1 for 
    the response tokens and 0 for other tokens (prompt or padding).
    """
    # Ensure pad_token_id is set (common issue with some tokenizers like Llama/Qwen)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # List to store the processed sequences
    full_input_ids = []
    full_masks = []

    # 1. Iterate through the batch to tokenize and concat
    for prompt, output in zip(prompt_strs, output_strs):
        # Tokenize prompt and output separately
        # We generally avoid adding special tokens automatically here to control 
        # boundaries manually, but usually prompt gets BOS (if applicable) and output gets EOS.
        # For simplicity in this specific SFT context:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False) 
        output_ids = tokenizer.encode(output, add_special_tokens=False)

        # Combine: Prompt + Output + EOS
        # Note: Depending on the tokenizer, you might want to add BOS to prompt_ids.
        # Here we assume prompt_ids starts the sequence.
        ids = prompt_ids + output_ids
        
        # Calculate lengths
        prompt_len = len(prompt_ids)
        output_len = len(output_ids) 
        
        # Construct the mask for the FULL sequence (before shifting)
        # 0 for prompt, 1 for response
        # We want to predict the tokens IN the response. 
        # In causal LM, input[i] predicts label[i] (which is input[i+1]).
        # If we want to predict the first token of the response, we need the 
        # label at that position to be masked as 1.
        
        # Mask: 0s for prompt, 1s for output
        mask = [0] * prompt_len + [1] * output_len
        
        full_input_ids.append(torch.tensor(ids, dtype=torch.long))
        full_masks.append(torch.tensor(mask, dtype=torch.long))

    # 2. Pad the sequences
    # We use torch.nn.utils.rnn.pad_sequence to handle batch padding easily
    # batch_first=True returns (batch, seq_len)
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        full_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    
    # Pad masks with 0 (ignore padding in loss)
    masks_padded = torch.nn.utils.rnn.pad_sequence(
        full_masks, batch_first=True, padding_value=False
    )

    # 3. Create the Targets (Inputs vs Labels)
    # Standard Causal LM setup:
    # input_ids: [x0, x1, x2, ... xN-1]
    # labels:    [x1, x2, x3, ... xN]
    
    # Input to model (remove last token)
    input_ids = input_ids_padded[:, :-1]
    
    # Labels (remove first token - shifted left)
    labels = input_ids_padded[:, 1:]
    
    # Mask (remove first token)
    # Why? The mask corresponds to the LABELS.
    # If full sequence is [P0, P1, R0, R1]
    # mask is           [0,  0,  1,  1]
    # input is          [P0, P1, R0]
    # label is          [P1, R0, R1] (We want to predict R0 and R1)
    # sliced mask is    [0,  1,  1]
    # This correctly masks P1 (part of prompt, don't calculate loss) 
    # and highlights R0, R1 (response tokens).
    response_mask = masks_padded[:, 1:]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    # 1. 计算 Log Z (归一化常数的对数)
    # 对应公式里的: log Z
    # logsumexp 这个函数内部自动处理了数值稳定性（Max Trick），防止指数爆炸
    log_z = torch.logsumexp(logits, dim=-1)
    
    # 2. 计算概率 p
    # 我们还是需要 p 来计算期望值
    probs = torch.softmax(logits, dim=-1)
    
    # 3. 计算 Logits 的期望值
    # 对应公式里的: sum(p_i * x_i)
    # 也就是：概率 * 原始分数，然后求和
    expected_logits = torch.sum(probs * logits, dim=-1)
    
    # 4. 套用最终简化的公式
    # 对应公式: H(p) = log Z - sum(p * x)
    entropy = log_z - expected_logits
    
    return entropy


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Computes per-token conditional log-probabilities and optionally token entropies.
    
    Assumes 'input_ids' and 'labels' are already shifted/aligned:
    - input_ids: [x0, x1, ..., xN-1]
    - labels:    [x1, x2, ..., xN]
    
    Args:
        model: HuggingFace causal LM.
        input_ids: (batch_size, seq_len)
        labels: (batch_size, seq_len)
        return_token_entropy: If True, computes entropy of next-token distribution using compute_entropy.
    
    Returns:
        dict containing:
            "log_probs": (batch_size, seq_len)
            "token_entropy": (batch_size, seq_len) (optional)
    """
    
    # 1. Forward pass to get logits
    # input_ids shape: (batch_size, seq_len)
    # logits shape: (batch_size, seq_len, vocab_size)
    outputs = model(input_ids)
    logits = outputs.logits 

    # 2. Compute Log Probabilities
    # Since input_ids and labels are already aligned (shifted externally),
    # logits[i] is the prediction for the token at labels[i].
    log_probs_all = F.log_softmax(logits, dim=-1) 
    
    # 3. Gather the log-probs of the target tokens
    # gather dim must match: (batch_size, seq_len, 1)
    tgt_log_probs = log_probs_all.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    result = {"log_probs": tgt_log_probs}

    # 4. Optional: Compute Entropy using the provided function
    if return_token_entropy:
        # Calculate entropy for the distribution at each position
        entropy_vals = compute_entropy(logits)
        result["token_entropy"] = entropy_vals

    return result


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Sums over tensor elements and normalizes by a constant while respecting a boolean mask.
    
    Args:
        tensor: The tensor to sum and normalize.
        mask: Same shape as tensor; positions with 1 are included in the sum.
        normalize_constant: The constant to divide by for normalization.
        dim: The dimension to sum along before normalization. If None, sum over all dimensions.
        
    Returns:
        The normalized sum, where masked elements (mask == 0) don't contribute.
    """
    # 1. Apply the mask
    # We multiply the tensor by the mask. Elements corresponding to mask=0 become 0.
    # Note: This assumes mask acts as a multiplier (0s and 1s). 
    masked_tensor = tensor * mask

    # 2. Sum over the specified dimension(s)
    if dim is not None:
        summed_tensor = masked_tensor.sum(dim=dim)
    else:
        summed_tensor = masked_tensor.sum()

    # 3. Normalize
    return summed_tensor / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch for SFT.
    
    Args:
        policy_log_probs: (batch, seq_len) per-token log-probabilities.
        response_mask: (batch, seq_len) 1 for response tokens, 0 for others.
        gradient_accumulation_steps: Denominator for gradient scaling.
        normalize_constant: Denominator for loss normalization (e.g. token count).
        
    Returns:
        tuple containing:
        - loss: The microbatch loss (scaled for accumulation).
        - metadata: Dictionary containing original loss and statistics.
    """
    # 1. Compute the normalized sum of log probabilities
    # We use the helper to mask out prompt/padding and normalize.
    # dim=None ensures we get a scalar sum over the entire batch.
    
    normalized_log_probs = masked_normalize(
        tensor=policy_log_probs,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=-1
    )

    # 2. Calculate SFT Loss
    # SFT minimizes Negative Log-Likelihood (NLL).
    # Since we have log_probs, we just negate the result.
    # batch_size = policy_log_probs.shape[0]
    sft_loss = -normalized_log_probs.mean()

    # 3. Scale for Gradient Accumulation
    # We divide by accumulation steps BEFORE backward to average gradients correctly.
    scaled_loss = sft_loss / gradient_accumulation_steps

    # 4. Backward Pass
    scaled_loss.backward()

    # 5. Prepare Metadata
    # We usually log the *unscaled* loss to track actual model performance.
    # We use .detach() to prevent memory leaks in logging (breaking the graph).
    metadata = {
        "loss": sft_loss.detach(),
    }

    return scaled_loss, metadata