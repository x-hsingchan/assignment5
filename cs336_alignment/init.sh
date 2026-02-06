git clone https://github.com/x-hsingchan/assignment5.git
cd assignment5
uv sync --no-install-package flash-attn
uv sync
cd cs336_alignment/
uv run prepare.py
uv run filter_sft_dataset.py
sudo apt install fish -y
fish