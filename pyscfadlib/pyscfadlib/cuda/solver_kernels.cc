#include "xla/ffi/api/ffi.h"
#include "pyscfadlib/ffi_helpers.h"
#include "pyscfadlib/cuda/vendor.h"
#include "pyscfadlib/cuda/solver_kernels.h"

namespace pyscfad {
namespace cuda {
namespace solver {

#define CUDA_DEFINE_SYGVD(Type, Name)                                          \
    template <>                                                                \
    int SygvdBufferSize<Type>(cusolverDnHandle_t handle,                       \
                              cusolverEigType_t itype,                         \
                              cusolverEigMode_t jobz,                          \
                              cublasFillMode_t uplo,                           \
                              int n) {                                         \
        int lwork;                                                             \
        Name##_bufferSize(handle, itype, jobz, uplo, n,                        \
                          /*A=*/nullptr, /*lda=*/n,                            \
                          /*B=*/nullptr, /*ldb=*/n,                            \
                          /*W=*/nullptr, &lwork);                              \
        return lwork;                                                          \
    }                                                                          \
                                                                               \
    template <>                                                                \
    void Sygvd<Type>(cusolverDnHandle_t handle,                                \
                     cusolverEigType_t itype,                                  \
                     cusolverEigMode_t jobz,                                   \
                     cublasFillMode_t uplo,                                    \
                     int n, Type *a, Type *b, RealType<Type>::value *w,        \
                     Type *work, int lwork, int *info) {                       \
        Name(handle, itype, jobz, uplo, n, a, n, b, n, w, work, lwork, info);  \
    }

CUDA_DEFINE_SYGVD(float, cusolverDnSsygvd);
CUDA_DEFINE_SYGVD(double, cusolverDnDsygvd);
CUDA_DEFINE_SYGVD(cuComplex, cusolverDnChegvd);
CUDA_DEFINE_SYGVD(cuDoubleComplex, cusolverDnZhegvd);
#undef CUDA_DEFINE_SYGVD

} // namespace solver

namespace ffi = ::xla::ffi;

#define SOLVER_DISPATCH_IMPL(impl, ...)                 \
    switch (dtype) {                                    \
        case ffi::F32:                                  \
            return impl<float>(__VA_ARGS__);            \
        case ffi::F64:                                  \
            return impl<double>(__VA_ARGS__);           \
        case ffi::C64:                                  \
            return impl<cuComplex>(__VA_ARGS__);        \
        case ffi::C128:                                 \
            return impl<cuDoubleComplex>(__VA_ARGS__);  \
        default:                                        \
            break;                                      \
    }


template <typename T>
ffi::Error SygvdImpl(
        int64_t batch_count, int64_t a_cols,
        cudaStream_t stream, ffi::ScratchAllocator& scratch,
        ffi::AnyBuffer a, ffi::AnyBuffer b,
        ffi::Result<ffi::AnyBuffer> a_out, ffi::Result<ffi::AnyBuffer> b_out,
        ffi::Result<ffi::AnyBuffer> eigenvalues,
        ffi::Result<ffi::Buffer<ffi::S32>> info,
        int itype_int, bool lower)
{
    int n = MaybeCastNoOverflow<int>(a_cols);

    cusolverDnHandle_t handle = NULL;
    cusolverDnCreate(&handle);
    cusolverDnSetStream(handle, stream);

    auto *a_data = static_cast<T*>(a.untyped_data());
    auto *b_data = static_cast<T*>(b.untyped_data());
    auto *a_out_data = static_cast<T*>(a_out->untyped_data());
    auto *b_out_data = static_cast<T*>(b_out->untyped_data());
    auto *eigenvalues_data =
        static_cast<typename solver::RealType<T>::value*>(eigenvalues->untyped_data());
    auto *info_data = info->typed_data();

    if (a_data != a_out_data) {
        cudaMemcpyAsync(a_out_data, a_data, a.size_bytes(),
                        cudaMemcpyDeviceToDevice, stream);
    }

    if (b_data != b_out_data) {
        cudaMemcpyAsync(b_out_data, b_data, b.size_bytes(),
                        cudaMemcpyDeviceToDevice, stream);
    }

    cusolverEigType_t itype = static_cast<cusolverEigType_t>(itype_int);
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;

    int lwork = solver::SygvdBufferSize<T>(handle, itype, jobz, uplo, n);
    auto *work = AllocateWorkspace<T>(scratch, lwork, "sygvd");

    int64_t out_step = a_cols * a_cols;
    for (int64_t i = 0; i < batch_count; ++i) {
        solver::Sygvd<T>(handle, itype, jobz, uplo, n,
                         a_out_data, b_out_data, eigenvalues_data,
                         work, lwork, info_data);
        a_out_data += out_step;
        b_out_data += out_step;
        eigenvalues_data += n;
        ++info_data;
    }

    cusolverDnDestroy(handle);
    return ffi::Error::Success();
}

ffi::Error SygvdDispatch(
    cudaStream_t stream, ffi::ScratchAllocator scratch,
    ffi::AnyBuffer a, ffi::AnyBuffer b,
    ffi::Result<ffi::AnyBuffer> a_out, ffi::Result<ffi::AnyBuffer> b_out,
    ffi::Result<ffi::AnyBuffer> eigenvalues,
    ffi::Result<ffi::Buffer<ffi::S32>> info,
    int itype, bool lower)
{
    auto dtype = a.element_type();
    if (dtype != a_out->element_type() || dtype != b.element_type()) {
        return ffi::Error::InvalidArgument(
            "The inputs and outputs to sygvd must have the same element type");
    }

    auto [batch_count, a_rows, a_cols] = SplitBatch2D(a.dimensions());
    if (a_rows != a_cols) {
        return ffi::Error::InvalidArgument(
            "The input matrix to sygvd must be square.");
    }

    SOLVER_DISPATCH_IMPL(SygvdImpl, batch_count, a_cols, stream, scratch,
                         a, b, a_out, b_out, eigenvalues, info, itype, lower);

    return ffi::Error::InvalidArgument("Unsupported dtype in sygvd");
}

#undef SOLVER_DISPATCH_IMPL

XLA_FFI_DEFINE_HANDLER_SYMBOL(
  SygvdFfi, SygvdDispatch,
  ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<cudaStream_t>>()
      .Ctx<ffi::ScratchAllocator>()
      .Arg<ffi::AnyBuffer>(/*a*/)
      .Arg<ffi::AnyBuffer>(/*b*/)
      .Ret<ffi::AnyBuffer>(/*a_out*/)
      .Ret<ffi::AnyBuffer>(/*b_out*/)
      .Ret<ffi::AnyBuffer>(/*eigenvalues*/)
      .Ret<ffi::Buffer<ffi::S32>>(/*info*/)
      .Attr<int>("itype")
      .Attr<bool>("lower")
);

} // namespace cuda
} // namespace pyscfad
