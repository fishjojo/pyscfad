#!/bin/bash
set -e  # Exit on error

# Exclude all CUDA versions dynamically
CUDA_LIBS=$(ldconfig -p | grep cuda | awk '{print "--exclude "$1}' | tr '\n' ' ')

# Run auditwheel repair with exclusions
auditwheel -v repair $CUDA_LIBS -w "$1" "$2"
