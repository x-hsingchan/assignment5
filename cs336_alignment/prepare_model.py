import pathlib
from huggingface_hub import snapshot_download


# --- 路径解析 ---
BASE_DIR = pathlib.Path(__file__).resolve().parent

# 1. 统一的模型根目录: Project/model/
MODEL_DIR = BASE_DIR / "model"

# 2. 具体的模型版本目录: Project/model/Qwen2.5-Math-1.5B
MODEL_NAME = "Qwen2.5-Math-1.5B"
MODEL_PATH = MODEL_DIR / MODEL_NAME

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
    download_model()