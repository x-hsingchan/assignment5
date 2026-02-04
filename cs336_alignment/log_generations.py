import torch
import wandb
import numpy as np
from typing import List, Dict, Any, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer

# 引入提供的工具函数
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.get_response_log_probs import get_response_log_probs

def log_generations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    ground_truths: List[str],
    step: Optional[int] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    对给定的所有 prompt 生成回复，计算奖励与熵，并记录详细日志。
    此函数会处理传入列表中的每一个样本，不进行截断。

    Args:
        model: 用于生成的语言模型。
        tokenizer: 配套的分词器。
        prompts: 输入提示列表。
        ground_truths: 对应的真实答案列表。
        step: 当前训练步数（用于 WandB 日志）。
        device: 运行计算的设备。

    Returns:
        包含统计信息（平均长度、奖励、熵）和详细样本表的字典。
    """
    model.eval()
    
    # 存储结果容器
    logs = []
    total_len = 0
    correct_lens = []
    incorrect_lens = []
    
    print(f"正在对全部 {len(prompts)} 条样本进行生成与评估...")

    # 遍历每一条数据，不进行切片/子集选择
    for i in range(len(prompts)):
        prompt = prompts[i]
        gt = ground_truths[i]

        # 1. Tokenize 输入
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]

        # 2. 模型生成
        # 使用 sample 模式以便观察分布的熵
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,  # 根据数学题解答长度适当调整
                do_sample=True,      
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # 3. 解码生成的文本
        full_seq = outputs[0]
        # 提取回复部分的 token 和文本
        response_tokens = full_seq[prompt_len:]
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

        # 4. 计算奖励 (Format, Answer, Total)
        # 引用 drgrpo_grader.py 中的 r1_zero_reward_fn
        reward_metrics = r1_zero_reward_fn(response_text, gt)
        
        # 5. 计算 Token 熵
        # 引用 get_response_log_probs.py
        # 构造 input 和 labels 用于计算 log_probs
        # inputs: [x0, x1, ... xN-1]
        # labels: [x1, x2, ... xN]
        gen_input_ids = full_seq[:-1].unsqueeze(0)  
        gen_labels = full_seq[1:].unsqueeze(0)      

        with torch.no_grad():
            probs_out = get_response_log_probs(
                model, 
                gen_input_ids, 
                gen_labels, 
                return_token_entropy=True
            )
        
        # 提取熵值：
        # full_seq 的索引 p 处的 token，是由索引 p-1 处的输入预测的。
        # 我们只关心 response 部分的熵。
        # Response 从 prompt_len 开始，对应的预测位置是 prompt_len - 1。
        full_entropies = probs_out["token_entropy"][0]
        
        if len(response_tokens) > 0:
            # 截取属于生成的回复部分的熵
            response_entropies = full_entropies[prompt_len - 1 :]
            avg_entropy = response_entropies.mean().item()
        else:
            avg_entropy = 0.0

        # 6. 记录长度统计
        resp_len = len(response_tokens)
        total_len += resp_len
        
        # 根据答案奖励判断正确性 (阈值设为 1.0)
        if reward_metrics["answer_reward"] == 1.0:
            correct_lens.append(resp_len)
        else:
            incorrect_lens.append(resp_len)

        # 7. 记录单条日志
        logs.append({
            "prompt": prompt,
            "response": response_text,
            "ground_truth": gt,
            "format_reward": reward_metrics["format_reward"],
            "answer_reward": reward_metrics["answer_reward"],
            "total_reward": reward_metrics["reward"],
            "avg_entropy": avg_entropy,
            "response_length": resp_len
        })

    # 8. 聚合统计指标
    avg_len_all = total_len / len(logs) if logs else 0
    avg_len_correct = np.mean(correct_lens) if correct_lens else 0
    avg_len_incorrect = np.mean(incorrect_lens) if incorrect_lens else 0
    avg_entropy_all = np.mean([l["avg_entropy"] for l in logs]) if logs else 0
    avg_reward = np.mean([l["total_reward"] for l in logs]) if logs else 0

    stats = {
        "gen/avg_response_length": avg_len_all,
        "gen/avg_len_correct": avg_len_correct,
        "gen/avg_len_incorrect": avg_len_incorrect,
        "gen/avg_token_entropy": avg_entropy_all,
        "gen/avg_reward": avg_reward,
    }

    # 9. WandB 日志记录
    if wandb.run is not None:
        # 创建可视化表格
        table = wandb.Table(columns=[
            "Prompt", "Response", "Ground Truth", "Format Reward", 
            "Answer Reward", "Entropy", "Length"
        ])
        for log in logs:
            table.add_data(
                log["prompt"],  # 记录完整 prompt
                log["response"],
                str(log["ground_truth"]),
                log["format_reward"],
                log["answer_reward"],
                f"{log['avg_entropy']:.4f}",
                log["response_length"]
            )
        
        wandb_log_dict = stats.copy()
        wandb_log_dict["gen/samples_table"] = table
        
        if step is not None:
            wandb.log(wandb_log_dict, step=step)
        else:
            wandb.log(wandb_log_dict)

    model.train()  # 恢复训练模式
    return {"stats": stats, "logs": logs}