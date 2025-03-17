#ifndef PYSCFADLIB_CUDA_SOLVER_KERNELS_H_
#define PYSCFADLIB_CUDA_SOLVER_KERNELS_H_

#include <stdexcept>
#include <string>
#include "xla/ffi/api/ffi.h"
#include "pyscfadlib/cuda/vendor.h"

namespace pyscfad {
namespace cuda {

XLA_FFI_DECLARE_HANDLER_SYMBOL(SygvdFfi);

namespace solver {

template <typename T>
struct RealType {
    using value = T;
};

template <>
struct RealType<cuComplex> {
    using value = float;
};

template <>
struct RealType<cuDoubleComplex> {
    using value = double;
};

#define CUSOLVER_EXPAND_DEFINITION(ReturnType, FunctionName)                    \
    template <typename T>                                                       \
    ReturnType FunctionName(                                                    \
            CUSOLVER_##FunctionName##_ARGS(T, typename RealType<T>::value)) {   \
        throw std::runtime_error(                                               \
            std::string(#FunctionName) +                                        \
            " not implemented for type " +                                      \
            std::string(typeid(T).name()));                                     \
    }                                                                           \
    template <>                                                                 \
    ReturnType FunctionName<float>(                                             \
        CUSOLVER_##FunctionName##_ARGS(float, float));                          \
    template <>                                                                 \
    ReturnType FunctionName<double>(                                            \
        CUSOLVER_##FunctionName##_ARGS(double, double));                        \
    template <>                                                                 \
    ReturnType FunctionName<cuComplex>(                                         \
        CUSOLVER_##FunctionName##_ARGS(cuComplex, float));                      \
    template <>                                                                 \
    ReturnType FunctionName<cuDoubleComplex>(                                   \
        CUSOLVER_##FunctionName##_ARGS(cuDoubleComplex, double))

#define CUSOLVER_SygvdBufferSize_ARGS(...)        \
    cusolverDnHandle_t handle,                    \
    cusolverEigType_t itype,                      \
    cusolverEigMode_t jobz,                       \
    cublasFillMode_t uplo,                        \
    int n
CUSOLVER_EXPAND_DEFINITION(int, SygvdBufferSize);
#undef CUSOLVER_SygvdBufferSize_ARGS

#define CUSOLVER_Sygvd_ARGS(Type, Real)           \
    cusolverDnHandle_t handle,                    \
    cusolverEigType_t itype,                      \
    cusolverEigMode_t jobz,                       \
    cublasFillMode_t uplo,                        \
    int n, Type *a, Type *b, Real *w,             \
    Type *work, int lwork, int *info
CUSOLVER_EXPAND_DEFINITION(void, Sygvd);
#undef CUSOLVER_Sygvd_ARGS

#undef CUSOLVER_EXPAND_DEFINITION

} // namespace solver
} // namespace cuda
} // namespace pyscfad

#endif
