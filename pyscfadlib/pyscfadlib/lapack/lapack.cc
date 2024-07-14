#include "lapack/kernel_nanobind_helpers.h"
#include "lapack/lapack_kernels.h"

namespace pyscfad {

namespace nb = nanobind;

void GetLapackKernelsFromBLAS() {
    static bool initialized = false;  // Protected by GIL
    if (initialized) return;

    AssignKernelFn<RealSygvd<float>>(ssygvd_);
    AssignKernelFn<RealSygvd<double>>(dsygvd_);
    AssignKernelFn<ComplexHegvd<std::complex<float>>>(
        reinterpret_cast<ComplexHegvd<std::complex<float>>::FnType*>(chegvd_));
    AssignKernelFn<ComplexHegvd<std::complex<double>>>(
        reinterpret_cast<ComplexHegvd<std::complex<double>>::FnType*>(zhegvd_));

    initialized = true;
}

nb::dict Registrations() {
    nb::dict dict;

    dict["lapack_ssygvd"] = EncapsulateFunction(RealSygvd<float>::Kernel);
    dict["lapack_dsygvd"] = EncapsulateFunction(RealSygvd<double>::Kernel);
    dict["lapack_chegvd"] =
        EncapsulateFunction(ComplexHegvd<std::complex<float>>::Kernel);
    dict["lapack_zhegvd"] =
        EncapsulateFunction(ComplexHegvd<std::complex<double>>::Kernel);

    return dict;
}

NB_MODULE(lapack_ad, m) {
    m.def("initialize", GetLapackKernelsFromBLAS);
    m.def("registrations", &Registrations);

    m.def("sygvd_work_size", &SygvdWorkSize, nb::arg("n"));
    m.def("sygvd_iwork_size", &SygvdIworkSize, nb::arg("n"));
    m.def("hegvd_work_size", &HegvdWorkSize, nb::arg("n"));
    m.def("hegvd_rwork_size", &HegvdRworkSize, nb::arg("n"));
}

} // namespace pyscfad
