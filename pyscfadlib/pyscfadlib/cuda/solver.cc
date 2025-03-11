#include "kernel_nanobind_helpers.h"
#include "cuda/solver_kernels.h"

namespace pyscfad {
namespace cuda {

namespace nb = nanobind;

nb::dict Registrations() {
    nb::dict dict;

    dict["cusolver_sygvd_ffi"] = EncapsulateFfiHandler(SygvdFfi);
    return dict;
}

NB_MODULE(pyscfad_cusolver, m) {
    m.def("registrations", &Registrations);
}

} // namespace cuda
} // namespace pyscfad
