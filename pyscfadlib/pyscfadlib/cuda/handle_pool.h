#ifndef PYSCFADLIB_CUDA_HANDLE_POOL_H_
#define PYSCFADLIB_CUDA_HANDLE_POOL_H_

#include <map>
#include <mutex>
#include <vector>

#include "pyscfadlib/cuda/vendor.h"

namespace pyscfad {
namespace cuda {

// A thread-safe pool of cuSOLVER-style library handles, keyed by CUDA stream.
//
// Creating a handle (e.g. cusolverDnCreate) allocates internal host/device
// state and is comparatively expensive, so paying it on every FFI invocation as
// a create/destroy pair wastes time in hot loops -- notably the per-iteration
// generalized eigensolve of an SCF. This pool amortizes that cost: a handle is
// created on first use for a given stream and reused thereafter (it is never
// destroyed; the handful of cached handles live for the process lifetime, which
// also avoids tearing them down after the CUDA context is gone).
//
// A handle is only ever reused on the stream it was borrowed for, so the
// in-order execution of a single CUDA stream keeps reuse safe even while
// previously submitted (asynchronous) work is still in flight. Concurrent
// borrows on the same stream simply create additional handles on demand.
//
// Mirrors jaxlib's jaxlib/gpu/handle_pool.h.
template <typename HandleType, typename StreamType>
class HandlePool {
public:
    HandlePool() = default;

    // RAII wrapper around a borrowed handle; returns it to the pool when it
    // goes out of scope.
    class Handle {
    public:
        Handle() = default;
        ~Handle() {
            if (pool_) {
                pool_->Return(handle_, stream_);
            }
        }

        Handle(const Handle&) = delete;
        Handle& operator=(const Handle&) = delete;

        Handle(Handle&& other) noexcept
            : pool_(other.pool_), handle_(other.handle_), stream_(other.stream_) {
            other.pool_ = nullptr;
            other.handle_ = nullptr;
            other.stream_ = nullptr;
        }
        Handle& operator=(Handle&& other) noexcept {
            if (this != &other) {
                if (pool_) {
                    pool_->Return(handle_, stream_);
                }
                pool_ = other.pool_;
                handle_ = other.handle_;
                stream_ = other.stream_;
                other.pool_ = nullptr;
                other.handle_ = nullptr;
                other.stream_ = nullptr;
            }
            return *this;
        }

        HandleType get() const { return handle_; }

    private:
        friend class HandlePool<HandleType, StreamType>;
        Handle(HandlePool<HandleType, StreamType>* pool, HandleType handle, StreamType stream)
            : pool_(pool), handle_(handle), stream_(stream) {}

        HandlePool* pool_ = nullptr;
        HandleType handle_ = nullptr;
        StreamType stream_ = nullptr;
    };

    // Borrows a handle bound to `stream`, creating one if none are free. Defined
    // per handle type as an explicit specialization (handle creation is
    // library-specific); see handle_pool.cc.
    static Handle Borrow(StreamType stream);

private:
    // Leaked-on-purpose singleton (see class comment).
    static HandlePool* Instance() {
        static auto* pool = new HandlePool();
        return pool;
    }

    void Return(HandleType handle, StreamType stream) {
        std::lock_guard<std::mutex> lock(mu_);
        handles_[stream].push_back(handle);
    }

    std::mutex mu_;
    std::map<StreamType, std::vector<HandleType>> handles_;
};

using SolverHandlePool = HandlePool<cusolverDnHandle_t, cudaStream_t>;

template <>
SolverHandlePool::Handle SolverHandlePool::Borrow(cudaStream_t stream);

} // namespace cuda
} // namespace pyscfad

#endif // PYSCFADLIB_CUDA_HANDLE_POOL_H_
