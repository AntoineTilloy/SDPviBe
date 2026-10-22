#!/usr/bin/env bash
# Run ON the rented instance. Recommended image: pytorch/pytorch:<ver>-cuda13.0-cudnn9-devel
# (Ubuntu: vast's ssh setup works; devel: gcc for Triton). Creates ~/venv on top of the image's
# torch if it has one, otherwise installs torch matching the driver's CUDA version.
set -euo pipefail
nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max --format=csv
CUDA=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1\2/p')
echo "driver supports CUDA ${CUDA}"
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    TV=$(python3 -c "import torch; print(torch.version.cuda.split('.')[0])")
    mkdir -p ~/venv/bin && echo "# image python already has torch" > ~/venv/bin/activate
    for i in 1 2 3; do   # throwaway container; retries: a flaky download once left cupy missing
        python3 -m pip install -q --break-system-packages --retries 10 --timeout 60 "cupy-cuda${TV}x" numpy scipy mpmath gmpy2 && break
    done
else
    python3 -m venv ~/venv 2>/dev/null || { apt-get update -qq && apt-get install -y -qq python3-venv && python3 -m venv ~/venv; }
    . ~/venv/bin/activate
    pip install -q --upgrade pip
    if [ "$CUDA" -ge 130 ]; then
        pip install -q torch --index-url https://download.pytorch.org/whl/cu130; pip install -q cupy-cuda13x
    else
        pip install -q torch --index-url https://download.pytorch.org/whl/cu128; pip install -q "cupy-cuda12x[ctk]"
    fi
    pip install -q numpy scipy mpmath
fi
python - <<'EOF'
import torch, triton, cupy
print("torch", torch.__version__, "triton", triton.__version__, "cupy", cupy.__version__)
print(torch.cuda.get_device_name(), torch.cuda.get_device_capability(),
      f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")
EOF
