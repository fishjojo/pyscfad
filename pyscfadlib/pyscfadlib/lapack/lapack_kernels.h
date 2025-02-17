#ifndef PYSCFADLIB_LAPACK_KERNELS_H
#define PYSCFADLIB_LAPACK_KERNELS_H

#include <complex>
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

extern "C" {
void ssygvd_(int* itype, char* jobz, char* uplo, int* n, float* a,
             int* lda, float* b, int* ldb, float* w, float* work, int* lwork,
             int* iwork, int* liwork, int* info);

void dsygvd_(int* itype, char* jobz, char* uplo, int* n, double* a,
             int* lda, double* b, int* ldb, double* w, double* work, int* lwork,
             int* iwork, int* liwork, int* info);

void chegvd_(int* itype, char* jobz, char* uplo, int* n, float _Complex* a,
             int* lda, float _Complex* b, int* ldb, float* w, float _Complex* work,
             int* lwork, float* rwork, int* lrwork, int* iwork, int* liwork,
             int* info);

void zhegvd_(int* itype, char* jobz, char* uplo, int* n, double _Complex* a,
             int* lda, double _Complex* b, int* ldb, double* w, double _Complex* work,
             int* lwork, double* rwork, int* lrwork, int* iwork, int* liwork,
             int* info);
}

namespace pyscfad {

template <typename KernelType>
void AssignKernelFn(void* func) {
    KernelType::fn = reinterpret_cast<typename KernelType::FnType*>(func);
}

template <typename KernelType>
void AssignKernelFn(typename KernelType::FnType* func) {
    KernelType::fn = func;
}

int SygvdWorkSize(int n);
int SygvdIworkSize(int n);

template <typename T>
struct RealSygvd {
    using FnType = void(int* itype, char* jobz, char* uplo, int* n, T* a,
                        int* lda, T* b, int* ldb, T* w, T* work, int* lwork,
                        int* iwork, int* liwork, int* info);
    static FnType* fn;
    static void Kernel(void* out, void** data);
};

int HegvdWorkSize(int n);
int HegvdRworkSize(int n);

template <typename T>
struct ComplexHegvd {
    using FnType = void(int* itype, char* jobz, char* uplo, int* n, T* a,
                        int* lda, T* b, int* ldb, typename T::value_type* w,
                        T* work, int* lwork, typename T::value_type* rwork, int* lrwork,
                        int* iwork, int* liwork, int* info);
    static FnType* fn;
    static void Kernel(void* out, void** data);
};

} //namespace pyscfad

// FFI functions
namespace pyscfad {

struct MatrixParams {
    enum class UpLo : char { kLower = 'L', kUpper = 'U' };
    enum class Transpose : char {
        kNoTrans = 'N',
        kTrans = 'T',
        kConjTrans = 'C'
    };
};

namespace eig {
enum class JobZ : char {
    kNoEigenvectors = 'N',
    kComputeEigenvectors = 'V',
};
}

} //namespace pyscfad

#define DEFINE_CHAR_ENUM_ATTR_DECODING(ATTR)                             \
  template <>                                                            \
  struct xla::ffi::AttrDecoding<ATTR> {                                  \
    using Type = ATTR;                                                   \
    static std::optional<Type> Decode(XLA_FFI_AttrType type, void* attr, \
                                      DiagnosticEngine& diagnostic);     \
  }

// XLA needs attributes to have deserialization method specified
DEFINE_CHAR_ENUM_ATTR_DECODING(pyscfad::MatrixParams::UpLo);
DEFINE_CHAR_ENUM_ATTR_DECODING(pyscfad::MatrixParams::Transpose);
DEFINE_CHAR_ENUM_ATTR_DECODING(pyscfad::eig::JobZ);

#undef DEFINE_CHAR_ENUM_ATTR_DECODING

namespace pyscfad {

inline constexpr auto LapackIntDtype = ::xla::ffi::DataType::S32;

template <::xla::ffi::DataType dtype>
struct EighReal {
    static_assert(!::xla::ffi::IsComplexType<dtype>());

    using ValueType = ::xla::ffi::NativeType<dtype>;
    using FnType = void(int* itype, char* jobz, char* uplo,
                        int* n, ValueType* a, int* lda, ValueType* b, int* ldb, ValueType* w,
                        ValueType* work, int* lwork, int* iwork, int* liwork, int* info);

    inline static FnType* fn = nullptr;

    static ::xla::ffi::Error Kernel(::xla::ffi::Buffer<dtype> a,
                                    ::xla::ffi::Buffer<dtype> b,
                                    ::xla::ffi::ResultBuffer<dtype> a_out,
                                    ::xla::ffi::ResultBuffer<dtype> b_out,
                                    ::xla::ffi::ResultBuffer<dtype> eigenvalues,
                                    ::xla::ffi::ResultBuffer<LapackIntDtype> info,
                                    int itype, eig::JobZ jobz, MatrixParams::UpLo uplo);
};

template <::xla::ffi::DataType dtype>
struct EighComplex {
    static_assert(::xla::ffi::IsComplexType<dtype>());

    using ValueType = ::xla::ffi::NativeType<dtype>;
    using RealType = ::xla::ffi::NativeType<::xla::ffi::ToReal(dtype)>;
    using FnType = void(int* itype, char* jobz, char* uplo,
                        int* n, ValueType* a, int* lda, ValueType* b, int* ldb, RealType* w,
                        ValueType* work, int* lwork, RealType* rwork, int* lrwork,
                        int* iwork, int* liwork, int* info);

    inline static FnType* fn = nullptr;

    static ::xla::ffi::Error Kernel(::xla::ffi::Buffer<dtype> a,
                                    ::xla::ffi::Buffer<dtype> b,
                                    ::xla::ffi::ResultBuffer<dtype> a_out,
                                    ::xla::ffi::ResultBuffer<dtype> b_out,
                                    ::xla::ffi::ResultBuffer<::xla::ffi::ToReal(dtype)> eigenvalues,
                                    ::xla::ffi::ResultBuffer<LapackIntDtype> info,
                                    int itype, eig::JobZ jobz, MatrixParams::UpLo uplo);
};

XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_ssygvd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_dsygvd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_chegvd_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(lapack_zhegvd_ffi);
} // namespace pyscfad

#endif
