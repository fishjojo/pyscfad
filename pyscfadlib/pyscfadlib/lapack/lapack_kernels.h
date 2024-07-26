#ifndef PYSCFADLIB_LAPACK_KERNELS_H
#define PYSCFADLIB_LAPACK_KERNELS_H

#include <complex>

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

} // namespace pyscfad
#endif
