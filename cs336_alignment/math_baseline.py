import json
import pathlib
from typing import List, Callable, Dict, Any
from vllm import LLM, SamplingParams
import drgrpo_grader  # Importing the provided grader module

# --- Configuration ---
BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MATH_DATA_PATH = DATA_DIR / "MATH" / "validation.jsonl"
OUTPUT_FILE = BASE_DIR / "math_eval_results.jsonl"
# 对应 prepare_model.py 中的路径: BASE_DIR / "model" / "Qwen2.5-Math-1.5B"
MODEL_LOCAL_PATH = BASE_DIR / "model" / "Qwen2.5-Math-1.5B"
# vLLM 接受路径字符串，如果路径存在则加载本地模型，否则会尝试去 HF 下载
MODEL_NAME = str(MODEL_LOCAL_PATH)

# --- Prompt Template ---
# Note: The prompt ends with <think> to induce chain-of-thought
R1_ZERO_PROMPT_TEMPLATE = (
    "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
    "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
    "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, "
    "respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n"
    "User: {question}\n"
    "Assistant: <think>"
)

def load_math_data(file_path: pathlib.Path) -> List[Dict[str, Any]]:
    """Loads MATH validation examples from a JSONL file."""
    data = []
    print(f"Loading data from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        print(f"Loaded {len(data)} examples.")
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        print("Please ensure ./data/MATH/validation.jsonl exists.")
        exit(1)
    return data

def format_prompts(data: List[Dict[str, Any]]) -> List[str]:
    """Formats problems into the r1_zero prompt structure."""
    return [R1_ZERO_PROMPT_TEMPLATE.format(question=item['problem']) for item in data]

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
    output_path: pathlib.Path
) -> None:
    """
    Evaluate a language model on a list of prompts, compute evaluation metrics, 
    and serialize results to disk.
    """
    print("Starting generation...")
    # Generate outputs
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results = []

    # Counters for the specific categories requested
    stats = {
        "correct_formatted": 0,      # (1) Format=1, Answer=1
        "incorrect_formatted": 0,    # (2) Format=1, Answer=0
        "format_error": 0            # (3) Format=0, Answer=0
    }
    
    print("Calculating metrics...")
    for i, output in enumerate(outputs):
        prompt = output.prompt
        # vLLM returns the generated text. Since we prompted with "<think>", 
        # the model continues from there. We prepend "<think>" to the generation
        # to ensure the full xml structure is present for the parser if strict parsing is used,
        # though the grader handles extraction. 
        # However, the prompt provided in requirements ends with <think>, so the model output starts *inside* the think tag.
        # To make the response strictly valid XML for extraction (if needed by simple parsers), 
        # we construct the full response string effectively.
        
        generated_text = output.outputs[0].text
        full_response_for_grading = "<think>" + generated_text
        
        #这里ground_truth可以是str，在r1_zero_reward_fn中会被处理
        ground_truth = ground_truths[i]
        
        # Calculate Reward
        reward_dict = reward_fn(full_response_for_grading, ground_truth)
        
        f_reward = reward_dict.get("format_reward", 0.0)
        a_reward = reward_dict.get("answer_reward", 0.0)
        
        # Categorize results
        if f_reward == 1.0 and a_reward == 1.0:
            stats["correct_formatted"] += 1
        elif f_reward == 1.0 and a_reward == 0.0:
            stats["incorrect_formatted"] += 1
        else:
            # Typically format_reward 0 implies answer_reward 0 in this grader
            stats["format_error"] += 1

        # Prepare result entry
        result_entry = {
            "prompt": prompt,
            "ground_truth": ground_truth,
            "generated_text": generated_text,
            "full_response": full_response_for_grading,
            "metrics": reward_dict
        }
        results.append(result_entry)

    # Calculate aggregate metrics
    total_count = len(results)
    accuracy = stats["correct_formatted"] / total_count if total_count > 0 else 0.0

    print("-" * 40)
    print(f"Evaluation Complete. Total Examples: {total_count}")
    print(f"Accuracy: {accuracy:.2%}")
    print("-" * 40)
    print("Category Breakdown:")
    print(f"(1) Correct (Format=1, Answer=1): {stats['correct_formatted']}")
    print(f"(2) Incorrect (Format=1, Answer=0): {stats['incorrect_formatted']}")
    print(f"(3) Format Error (Format=0, Answer=0): {stats['format_error']}")
    print("-" * 40)

    # Serialize to disk
    print(f"Saving results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    print("Done.")

def main():
    # 1. Load Data
    math_data = load_math_data(MATH_DATA_PATH)
    
    # Extract Ground Truths (usually the 'solution' field in MATH dataset)
    # The grader expects the solution string.
    ground_truths = [item['solution'] for item in math_data]

    # 2. Format Prompts
    prompts = format_prompts(math_data)

    # 3. Initialize Model
    print(f"Initializing vLLM with model: {MODEL_NAME}")
    llm = LLM(model=MODEL_NAME)

    # Define Sampling Params
    # Temperature 1.0, top_p 1.0, max tokens 1024.
    # Stop when model outputs </answer> and include it.
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # 4 & 5. Generate, Evaluate, and Serialize
    # We pass the grading function from the uploaded file
    evaluate_vllm(
        vllm_model=llm,
        reward_fn=drgrpo_grader.r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
        output_path=OUTPUT_FILE
    )

if __name__ == "__main__":
    main()