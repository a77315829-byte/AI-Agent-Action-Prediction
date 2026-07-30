$ErrorActionPreference = "Stop"
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements_llm.txt
python check_gpu.py
