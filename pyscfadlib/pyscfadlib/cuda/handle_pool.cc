#include "pyscfadlib/cuda/handle_pool.h"

#include <mutex>
#include <stdexcept>
#include <string>

namespace pyscfad {
namespace cuda {

template <>
SolverHandlePool::Handle SolverHandlePool::Borrow(cudaStream_t stream) {
    SolverHandlePool* pool = Instance();
    std::lock_guard<std::mutex> lock(pool->mu_);

    cusolverDnHandle_t handle;
    auto& free_handles = pool->handles_[stream];
    if (free_handles.empty()) {
        cusolverStatus_t status = cusolverDnCreate(&handle);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            throw std::runtime_error(
                "cusolverDnCreate failed with status " +
                std::to_string(static_cast<int>(status)));
        }
    } else {
        handle = free_handles.back();
        free_handles.pop_back();
    }

    if (stream) {
        cusolverStatus_t status = cusolverDnSetStream(handle, stream);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            throw std::runtime_error(
                "cusolverDnSetStream failed with status " +
                std::to_string(static_cast<int>(status)));
        }
    }

    return Handle(pool, handle, stream);
}

} // namespace cuda
} // namespace pyscfad
