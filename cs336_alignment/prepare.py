import pandas as pd
from pathlib import Path
from huggingface_hub import snapshot_download

# 绝对基准路径锁定：确保生成在 cs336_alignment/data/MATH/ 下
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# 1. 统一的模型根目录: Project/model/
MODEL_DIR = BASE_DIR / "model"

# 2. 具体的模型版本目录: Project/model/Qwen2.5-Math-1.5B
MODEL_NAME = "Qwen2.5-Math-1.5B"
MODEL_PATH = MODEL_DIR / MODEL_NAME



def prepare_math_dataset_final_reliable():
    output_dir = Path(DATA_DIR) / "MATH"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义任务配置：(远程分集名, 本地文件名)
    tasks = [
        ("test", "validation.jsonl"),
        ("train", "train_raw.jsonl")
    ]
    
    subjects = ["algebra", "counting_and_probability", "geometry", 
                "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    
    base_url = "https://huggingface.co/datasets/EleutherAI/hendrycks_math/resolve/main"

    for remote_split, local_name in tasks:
        output_path = output_dir / local_name
        
        if output_path.exists():
            print(f"✅ 数据集已存在: {output_path}")
            continue

        all_dfs = []
        print(f"🚀 正在构建 {local_name} (源分集: {remote_split})...")

        try:
            for sub in subjects:
                # 拼接下载地址，注意 train 和 test 在 HF 上的路径结构通常一致
                url = f"{base_url}/{sub}/{remote_split}-00000-of-00001.parquet"
                df = pd.read_parquet(url)
                all_dfs.append(df)

            final_df = pd.concat(all_dfs, ignore_index=True)
            # 导出为 JSONL 格式
            final_df.to_json(output_path, orient='records', lines=True, force_ascii=False)
            
            print(f"✨ {local_name} 转换成功！总样本数: {len(final_df)}")

        except Exception as e:
            print(f"❌ 处理 {local_name} 时失败: {e}")

    print(f"📄 所有文件存放在: {output_dir}")

# --- 下载逻辑 ---
def download_model():
    # mkdir(parents=True) 会自动创建 model/ 文件夹
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {MODEL_NAME} to {MODEL_PATH}...")
    snapshot_download(
        repo_id=f"Qwen/{MODEL_NAME}",
        local_dir=MODEL_PATH,
        local_dir_use_symlinks=False,
        resume_download=True
    )


if __name__ == "__main__":
    #prepare_math_dataset_final_reliable()
    download_model()