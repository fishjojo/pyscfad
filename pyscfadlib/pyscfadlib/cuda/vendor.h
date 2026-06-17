#ifndef PYSCFADLIB_CUDA_VENDOR_H_
#define PYSCFADLIB_CUDA_VENDOR_H_

// CUDA toolkit headers, resolved from the CUDA include directory (the CMake
// build adds CUDAToolkit_INCLUDE_DIRS). The former Bazel build referenced these
// through XLA's hermetic layout ("third_party/gpus/cuda/include/...").
#include <cusolverDn.h>
#include <cusolver_common.h>

#endif
