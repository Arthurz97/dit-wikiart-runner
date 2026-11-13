open powershell

py "-V:Astral/CPython3.12.13" -m venv wikiart_env

.\wikiart_env\Scripts\Activate.ps1

$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

python --version

python -m pip install --upgrade pip

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install diffusers transformers huggingface_hub pillow numpy

pip install accelerate

python test_wikiart.py

deactivate