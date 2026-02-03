import pandas as pd
from pathlib import Path

# 绝对基准路径锁定：确保生成在 cs336_alignment/data/MATH/ 下
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

def prepare_math_dataset_final_reliable():
    output_dir = DATA_DIR / "MATH"
    output_path = output_dir / "validation.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"✅ 数据集已存在: {output_path}")
        return

    subjects = ["algebra", "counting_and_probability", "geometry", 
                "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    
    # 绕过所有库探测逻辑，直接直连获取 Parquet 文件
    base_url = "https://huggingface.co/datasets/EleutherAI/hendrycks_math/resolve/main"
    
    all_dfs = []
    print(f"🚀 正在构建验证集 (绝对路径模式)...")

    try:
        for sub in subjects:
            # 显式路径加载，绝对不触发 ** 匹配逻辑
            url = f"{base_url}/{sub}/test-00000-of-00001.parquet"
            df = pd.read_parquet(url)
            all_dfs.append(df)

        final_df = pd.concat(all_dfs, ignore_index=True)
        # 导出为作业要求的 JSONL 格式
        final_df.to_json(output_path, orient='records', lines=True, force_ascii=False)
        
        print(f"✨ 转换成功！总样本数: {len(final_df)} (预期 ~5000)")
        print(f"📄 文件存放在: {output_path}")

    except Exception as e:
        print(f"❌ 流程失败: {e}")

if __name__ == "__main__":
    prepare_math_dataset_final_reliable()