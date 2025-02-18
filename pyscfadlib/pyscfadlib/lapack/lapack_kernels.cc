#include <cassert>
#include <cstdint>
#include <cstring>
#include "ffi_helpers.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "lapack/lapack_kernels.h"


namespace pyscfad {

int SygvdWorkSize(int n) {
    int size = 1 + 6 * n + 2 * n * n;
    return size;
}

int SygvdIworkSize(int n) {
    int size = 3 + 5 * n;
    return size;
}

template <typename T>
typename RealSygvd<T>::FnType* RealSygvd<T>::fn = nullptr;

template <typename T>
void RealSygvd<T>::Kernel(void* out_tuple, void** data) {
    int itype = *(reinterpret_cast<int32_t*>(data[0]));
    int32_t lower = *(reinterpret_cast<int32_t*>(data[1]));
    int batch_dim = *(reinterpret_cast<int32_t*>(data[2]));
    int n = *(reinterpret_cast<int32_t*>(data[3]));
    const T* a_in = reinterpret_cast<T*>(data[4]);
    const T* b_in = reinterpret_cast<T*>(data[5]);
    void** out = reinterpret_cast<void**>(out_tuple);
    T* a_out = reinterpret_cast<T*>(out[0]);
    T* w_out = reinterpret_cast<T*>(out[1]);
    int* info_out = reinterpret_cast<int*>(out[2]);
    T* b_out = reinterpret_cast<T*>(out[3]);
    T* work = reinterpret_cast<T*>(out[4]);
    int* iwork = reinterpret_cast<int*>(out[5]);
    if (a_out != a_in) {
        std::memcpy(a_out, a_in,
                    static_cast<int64_t>(n) * n * sizeof(T) * batch_dim);
    }
    if (b_out != b_in) {
        std::memcpy(b_out, b_in,
                    static_cast<int64_t>(n) * n * sizeof(T) * batch_dim);
    }

    char jobz = 'V';
    char uplo = lower ? 'L' : 'U';

    int lwork = SygvdWorkSize(n);
    int liwork = SygvdIworkSize(n);
    for (int i = 0; i < batch_dim; ++i) {
        fn(&itype, &jobz, &uplo, &n, a_out, &n, b_out, &n,
           w_out, work, &lwork, iwork, &liwork,
           info_out);
        a_out += static_cast<int64_t>(n) * n;
        b_out += static_cast<int64_t>(n) * n;
        w_out += n;
        ++info_out;
    }
}

int HegvdWorkSize(int n) {
    int size = 2 * n + n * n;
    return size;
}

int HegvdRworkSize(int n) {
    int size = 1 + 5 * n + 2 * n * n;
    return size;
}

template <typename T>
typename ComplexHegvd<T>::FnType* ComplexHegvd<T>::fn = nullptr;

template <typename T>
void ComplexHegvd<T>::Kernel(void* out_tuple, void** data) {
    int itype = *(reinterpret_cast<int32_t*>(data[0]));
    int32_t lower = *(reinterpret_cast<int32_t*>(data[1]));
    int batch_dim = *(reinterpret_cast<int32_t*>(data[2]));
    int n = *(reinterpret_cast<int32_t*>(data[3]));
    const T* a_in = reinterpret_cast<T*>(data[4]);
    const T* b_in = reinterpret_cast<T*>(data[5]);
    void** out = reinterpret_cast<void**>(out_tuple);
    T* a_out = reinterpret_cast<T*>(out[0]);
    typename T::value_type* w_out = reinterpret_cast<typename T::value_type*>(out[1]);
    int* info_out = reinterpret_cast<int*>(out[2]);
    T* b_out = reinterpret_cast<T*>(out[3]);
    T* work = reinterpret_cast<T*>(out[4]);
    typename T::value_type* rwork = reinterpret_cast<typename T::value_type*>(out[5]);
    int* iwork = reinterpret_cast<int*>(out[6]);
    if (a_out != a_in) {
        std::memcpy(a_out, a_in,
                    static_cast<int64_t>(n) * n * sizeof(T) * batch_dim);
    }
    if (b_out != b_in) {
        std::memcpy(b_out, b_in,
                    static_cast<int64_t>(n) * n * sizeof(T) * batch_dim);
    }

    char jobz = 'V';
    char uplo = lower ? 'L' : 'U';

    int lwork = HegvdWorkSize(n);
    int lrwork = HegvdRworkSize(n);
    int liwork = SygvdIworkSize(n);
    for (int i = 0; i < batch_dim; ++i) {
        fn(&itype, &jobz, &uplo, &n, a_out, &n, b_out, &n,
           w_out, work, &lwork, rwork, &lrwork, iwork, &liwork,
           info_out);
        a_out += static_cast<int64_t>(n) * n;
        b_out += static_cast<int64_t>(n) * n;
        w_out += n;
        ++info_out;
    }
}

template struct RealSygvd<float>;
template struct RealSygvd<double>;
template struct ComplexHegvd<std::complex<float>>;
template struct ComplexHegvd<std::complex<double>>;

} //namespace pyscfad


// FFI functions
namespace ffi = xla::ffi;

#define REGISTER_CHAR_ENUM_ATTR_DECODING(type)                                \
  std::optional<type> xla::ffi::AttrDecoding<type>::Decode(                   \
      XLA_FFI_AttrType attr_type, void* attr, DiagnosticEngine& diagnostic) { \
    if (attr_type != XLA_FFI_AttrType_SCALAR) [[unlikely]] {                  \
      return diagnostic.Emit("Wrong attribute type: expected ")               \
             << XLA_FFI_AttrType_SCALAR << " but got" << attr_type;           \
    }                                                                         \
    auto* scalar = reinterpret_cast<XLA_FFI_Scalar*>(attr);                   \
    if (scalar->dtype != XLA_FFI_DataType_U8) [[unlikely]] {                  \
      return diagnostic.Emit("Wrong scalar data type: expected ")             \
             << XLA_FFI_DataType_U8 << " but got " << scalar->dtype;          \
    }                                                                         \
    auto underlying =                                                         \
        *reinterpret_cast<std::underlying_type_t<type>*>(scalar->value);      \
    return static_cast<type>(underlying);                                     \
  }

REGISTER_CHAR_ENUM_ATTR_DECODING(pyscfad::MatrixParams::Transpose);
REGISTER_CHAR_ENUM_ATTR_DECODING(pyscfad::MatrixParams::UpLo);
REGISTER_CHAR_ENUM_ATTR_DECODING(pyscfad::eig::JobZ);

#undef REGISTER_CHAR_ENUM_ATTR_DECODING

namespace pyscfad {

template <ffi::DataType dtype>
void CopyIfDiffBuffer(ffi::Buffer<dtype> x, ffi::ResultBuffer<dtype> x_out) {
  if (x.typed_data() != x_out->typed_data()) {
    const auto x_size = x.element_count();
    std::copy_n(x.typed_data(), x_size, x_out->typed_data());
  }
}

template <ffi::DataType dtype>
ffi::Error EighReal<dtype>::Kernel(
    ffi::Buffer<dtype> a, ffi::Buffer<dtype> b,
    ffi::ResultBuffer<dtype> a_out, ffi::ResultBuffer<dtype> b_out,
    ffi::ResultBuffer<dtype> eigenvalues,
    ffi::ResultBuffer<LapackIntDtype> info,
    int itype, eig::JobZ jobz, MatrixParams::UpLo uplo)
{
    auto [batch_count, a_rows, a_cols] = SplitBatch2D(a.dimensions());
    assert(a_rows == a_cols);

    auto* a_out_data = a_out->typed_data();
    auto* b_out_data = b_out->typed_data();
    auto* eigenvalues_data = eigenvalues->typed_data();
    auto* info_data = info->typed_data();

    CopyIfDiffBuffer(a, a_out);
    CopyIfDiffBuffer(b, b_out);

    auto jobz_v = static_cast<char>(jobz);
    auto uplo_v = static_cast<char>(uplo);
    int n = MaybeCastNoOverflow<int>(a_cols);
    int lwork = SygvdWorkSize(n);
    int liwork = SygvdIworkSize(n);
    auto work = AllocateScratchMemory<dtype>(lwork);
    auto iwork = AllocateScratchMemory<LapackIntDtype>(liwork);

    const int64_t a_out_step{a_cols * a_cols};
    for (int64_t i = 0; i < batch_count; ++i) {
        fn(&itype, &jobz_v, &uplo_v,
           &n, a_out_data, &n, b_out_data, &n, eigenvalues_data,
           work.get(), &lwork, iwork.get(), &liwork,
           info_data);
        a_out_data += a_out_step;
        b_out_data += a_out_step;
        eigenvalues_data += a_cols;
        ++info_data;
    }
    return ffi::Error::Success();
}

template <ffi::DataType dtype>
ffi::Error EighComplex<dtype>::Kernel(
    ffi::Buffer<dtype> a, ffi::Buffer<dtype> b,
    ffi::ResultBuffer<dtype> a_out, ffi::ResultBuffer<dtype> b_out,
    ffi::ResultBuffer<ffi::ToReal(dtype)> eigenvalues,
    ffi::ResultBuffer<LapackIntDtype> info,
    int itype, eig::JobZ jobz, MatrixParams::UpLo uplo)
{
    auto [batch_count, a_rows, a_cols] = SplitBatch2D(a.dimensions());
    assert(a_rows == a_cols);

    auto* a_out_data = a_out->typed_data();
    auto* b_out_data = b_out->typed_data();
    auto* eigenvalues_data = eigenvalues->typed_data();
    auto* info_data = info->typed_data();

    CopyIfDiffBuffer(a, a_out);
    CopyIfDiffBuffer(b, b_out);

    auto jobz_v = static_cast<char>(jobz);
    auto uplo_v = static_cast<char>(uplo);
    int n = MaybeCastNoOverflow<int>(a_cols);
    int lwork = HegvdWorkSize(n);
    int lrwork = HegvdRworkSize(n);
    int liwork = SygvdIworkSize(n);
    auto work = AllocateScratchMemory<dtype>(lwork);
    auto rwork = AllocateScratchMemory<ffi::ToReal(dtype)>(lrwork);
    auto iwork = AllocateScratchMemory<LapackIntDtype>(liwork);

    const int64_t a_out_step{a_cols * a_cols};
    for (int64_t i = 0; i < batch_count; ++i) {
        fn(&itype, &jobz_v, &uplo_v,
           &n, a_out_data, &n, b_out_data, &n, eigenvalues_data,
           work.get(), &lwork, rwork.get(), &lrwork, iwork.get(), &liwork,
           info_data);
        a_out_data += a_out_step;
        b_out_data += a_out_step;
        eigenvalues_data += a_cols;
        ++info_data;
    }
    return ffi::Error::Success();
}

template struct EighReal<ffi::DataType::F32>;
template struct EighReal<ffi::DataType::F64>;
template struct EighComplex<ffi::DataType::C64>;
template struct EighComplex<ffi::DataType::C128>;

#define CPU_DEFINE_SYGVD(name, data_type)                        \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                 \
      name, EighReal<data_type>::Kernel,                         \
      ::xla::ffi::Ffi::Bind()                                    \
          .Arg<::xla::ffi::Buffer<data_type>>(/*a*/)             \
          .Arg<::xla::ffi::Buffer<data_type>>(/*b*/)             \
          .Ret<::xla::ffi::Buffer<data_type>>(/*a_out*/)         \
          .Ret<::xla::ffi::Buffer<data_type>>(/*b_out*/)         \
          .Ret<::xla::ffi::Buffer<data_type>>(/*eigenvalues*/)   \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/)     \
          .Attr<int>("itype") \
          .Attr<eig::JobZ>("jobz") \
          .Attr<MatrixParams::UpLo>("uplo"))

#define CPU_DEFINE_HEGVD(name, data_type)                          \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                   \
      name, EighComplex<data_type>::Kernel,                        \
      ::xla::ffi::Ffi::Bind()                                      \
          .Arg<::xla::ffi::Buffer<data_type>>(/*a*/)               \
          .Arg<::xla::ffi::Buffer<data_type>>(/*b*/)               \
          .Ret<::xla::ffi::Buffer<data_type>>(/*a_out*/)           \
          .Ret<::xla::ffi::Buffer<data_type>>(/*b_out*/)           \
          .Ret<::xla::ffi::Buffer<::xla::ffi::ToReal(data_type)>>( \
              /*eigenvalues*/)                                     \
          .Ret<::xla::ffi::Buffer<LapackIntDtype>>(/*info*/)       \
          .Attr<int>("itype") \
          .Attr<eig::JobZ>("jobz") \
          .Attr<MatrixParams::UpLo>("uplo"))

CPU_DEFINE_SYGVD(lapack_ssygvd_ffi, ::xla::ffi::DataType::F32);
CPU_DEFINE_SYGVD(lapack_dsygvd_ffi, ::xla::ffi::DataType::F64);
CPU_DEFINE_HEGVD(lapack_chegvd_ffi, ::xla::ffi::DataType::C64);
CPU_DEFINE_HEGVD(lapack_zhegvd_ffi, ::xla::ffi::DataType::C128);

#undef CPU_DEFINE_SYGVD
#undef CPU_DEFINE_HEGVD
} // namespace pyscfad

