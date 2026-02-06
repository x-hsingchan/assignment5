sudo apt install fish -y
git clone https://github.com/x-hsingchan/assignment5.git
cd assignment5-alignment/
uv sync --no-install-package flash-attn
uv sync
uv run prepare.py
