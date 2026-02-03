sudo apt install fish
uv sync --no-install-package flash-attn
uv sync
uv run prepare_dataset.py
uv run prepare_model.py
