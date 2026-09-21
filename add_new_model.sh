#!/bin/bash

# Use MNIST_ML_ROOT if it is already defined; otherwise use the folder of this script
MNIST_ML_ROOT="${MNIST_ML_ROOT:-$(cd "$(dirname "$0")" && pwd)}"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <model_name>"
  exit 1
fi

dir=$(echo "$1" | tr a-z A-Z) # makes input all uppercase
model_name_lower=$(echo "$1" | tr A-Z a-z)

mkdir -p "$MNIST_ML_ROOT/$dir/include" "$MNIST_ML_ROOT/$dir/src"
touch "$MNIST_ML_ROOT/$dir/Makefile"
touch "$MNIST_ML_ROOT/$dir/include/$model_name_lower.hpp"
touch "$MNIST_ML_ROOT/$dir/src/$model_name_lower.cc"
