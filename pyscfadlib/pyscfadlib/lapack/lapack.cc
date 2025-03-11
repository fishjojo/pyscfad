#include <mutex>
#include "kernel_nanobind_helpers.h"
#include "lapack/lapack_kernels.h"

namespace pyscfad {

namespace nb = nanobind;

using ::xla::ffi::DataType;

void GetLapackKernelsFromBLAS() {
    static std::once_flag initialized;

    std::call_once(initialized, [&](){
        AssignKernelFn<RealSygvd<float>>(ssygvd_);
        AssignKernelFn<RealSygvd<double>>(dsygvd_);
        AssignKernelFn<ComplexHegvd<std::complex<float>>>(
            reinterpret_cast<ComplexHegvd<std::complex<float>>::FnType*>(chegvd_));
        AssignKernelFn<ComplexHegvd<std::complex<double>>>(
            reinterpret_cast<ComplexHegvd<std::complex<double>>::FnType*>(zhegvd_));

        AssignKernelFn<EighReal<DataType::F32>>(ssygvd_);
        AssignKernelFn<EighReal<DataType::F64>>(dsygvd_);
        AssignKernelFn<EighComplex<DataType::C64>>(
            reinterpret_cast<EighComplex<DataType::C64>::FnType*>(chegvd_));
        AssignKernelFn<EighComplex<DataType::C128>>(
            reinterpret_cast<EighComplex<DataType::C128>::FnType*>(zhegvd_));
    });
}

nb::dict Registrations() {
    nb::dict dict;

    dict["lapack_ssygvd"] = EncapsulateFunction(RealSygvd<float>::Kernel);
    dict["lapack_dsygvd"] = EncapsulateFunction(RealSygvd<double>::Kernel);
    dict["lapack_chegvd"] =
        EncapsulateFunction(ComplexHegvd<std::complex<float>>::Kernel);
    dict["lapack_zhegvd"] =
        EncapsulateFunction(ComplexHegvd<std::complex<double>>::Kernel);

    dict["lapack_ssygvd_ffi"] = EncapsulateFunction(lapack_ssygvd_ffi);
    dict["lapack_dsygvd_ffi"] = EncapsulateFunction(lapack_dsygvd_ffi);
    dict["lapack_chegvd_ffi"] = EncapsulateFunction(lapack_chegvd_ffi);
    dict["lapack_zhegvd_ffi"] = EncapsulateFunction(lapack_zhegvd_ffi);
    return dict;
}

NB_MODULE(pyscfad_lapack, m) {
    m.def("initialize", GetLapackKernelsFromBLAS);
    m.def("registrations", &Registrations);

    m.def("sygvd_work_size", &SygvdWorkSize, nb::arg("n"));
    m.def("sygvd_iwork_size", &SygvdIworkSize, nb::arg("n"));
    m.def("hegvd_work_size", &HegvdWorkSize, nb::arg("n"));
    m.def("hegvd_rwork_size", &HegvdRworkSize, nb::arg("n"));
}

} // namespace pyscfad
