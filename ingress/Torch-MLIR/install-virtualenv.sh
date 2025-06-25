#!/usr/bin/env bash

# Command line argument for device type
# Options are: "AMD", "NVIDIA", "Intel", "CPU"
# If no argument is provided, it will detect the device type using lspci
if [ $# -eq 1 ]; then
  DEVICE_TYPE=$1
else
  DEVICE_TYPE=$(lspci | grep VGA)
fi


# Install torch-mlir inside a virtual environment
echo "Preparing the virtual environment"
python3 -m venv torch-mlir-venv
source torch-mlir-venv/bin/activate
python -m pip install --upgrade pip wheel

# GPU support ("AMD", "NVIDIA", "Intel")
if [[ $DEVICE_TYPE == *"NVIDIA"* ]]; then
  # This is the default pytorch
  echo "Installing PyTorch for GPU: NVIDIA"
  pip3 install torch torchvision torchaudio transformers
  if [ $? != 0 ]; then
    exit 1
  fi
elif [[ $DEVICE_TYPE == *"AMD"* ]]; then
  # https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native_linux/install-pytorch.html
  echo "Installing PyTorch for GPU: AMD"
  pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4/
  if [ $? != 0 ]; then
    exit 1
  fi
elif [[ $DEVICE_TYPE == *"Intel"* ]]; then
  echo "Installing PyTorch for GPU: Intel"
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
  if [ $? != 0 ]; then
    exit 1
  fi
else
  echo "No GPU support detected, installing CPU version."
  pip install torch torchvision torchaudio transformers --extra-index-url https://download.pytorch.org/whl/cpu
  if [ $? != 0 ]; then
    exit 1
  fi
fi

echo "Installing torch-mlir"
# This only seems to work on Ubuntu
pip3 install --pre torch-mlir \
  --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
  -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
if [ $? != 0 ]; then
  exit 1
fi

