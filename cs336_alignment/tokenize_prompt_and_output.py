import torch
import copy

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