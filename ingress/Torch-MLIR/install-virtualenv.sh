#!/usr/bin/env bash

# Install torch-mlir inside a virtual environment
echo "Preparing the virtual environment"
python3 -m venv torch-mlir-venv
source torch-mlir-venv/bin/activate
python -m pip install --upgrade pip wheel

# GPU support ("AMD", "NVIDIA", "Intel")
GPU_TYPE=$(lspci | grep VGA)
if [[ $GPU_TYPE == *"NVIDIA"* ]]; then
  # This is the default pytorch
  echo "Installing PyTorch for GPU: NVIDIA"
  pip3 install torch torchvision torchaudio transformers
  if [ $? != 0 ]; then
    exit 1
  fi
elif [[ $GPU_TYPE == *"AMD"* ]]; then
  # https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native_linux/install-pytorch.html
  echo "Installing PyTorch for GPU: AMD"
  mkdir -p .cache
  pushd .cache
  wget -nc https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/torch-2.6.0%2Brocm6.4.1.git1ded221d-cp312-cp312-linux_x86_64.whl
  wget -nc https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/torchvision-0.21.0%2Brocm6.4.1.git4040d51f-cp312-cp312-linux_x86_64.whl
  wget -nc https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/pytorch_triton_rocm-3.2.0%2Brocm6.4.1.git6da9e660-cp312-cp312-linux_x86_64.whl
  wget -nc https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/torchaudio-2.6.0%2Brocm6.4.1.gitd8831425-cp312-cp312-linux_x86_64.whl
  #pip uninstall torch torchvision pytorch-triton-rocm
  pip3 install \
    torch-2.6.0+rocm6.4.1.git1ded221d-cp312-cp312-linux_x86_64.whl \
    torchvision-0.21.0+rocm6.4.1.git4040d51f-cp312-cp312-linux_x86_64.whl \
    torchaudio-2.6.0+rocm6.4.1.gitd8831425-cp312-cp312-linux_x86_64.whl \
    pytorch_triton_rocm-3.2.0+rocm6.4.1.git6da9e660-cp312-cp312-linux_x86_64.whl
  if [ $? != 0 ]; then
    exit 1
  fi
  popd
elif [[ $GPU_TYPE == *"Intel"* ]]; then
  echo "Installing PyTorch for GPU: Intel"
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
  if [ $? != 0 ]; then
    exit 1
  fi
else
  echo "No GPU support detected, installing CPU version."
  exit 1
fi

echo "Installing torch-mlir"
pip3 install --pre torch-mlir \
  --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
  -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
if [ $? != 0 ]; then
  exit 1
fi

