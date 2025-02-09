#include <cstdint>
#include <cstring>
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

} // namespace pyscfad

