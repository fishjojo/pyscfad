#include "pyscfadlib/kernel_nanobind_helpers.h"
#include "pyscfadlib/cuint/ovlp.h"
#include "pyscfadlib/cuint/dipole.h"
#include "pyscfadlib/cuint/quadrupole.h"

namespace pyscfad {
namespace cuint {

namespace nb = nanobind;

nb::dict Registrations() {
    nb::dict dict;

    dict["cuint_overlap_ffi"] = EncapsulateFfiHandler(OverlapFfi);
    dict["cuint_overlap_gradient_ffi"] = EncapsulateFfiHandler(OverlapGradientFfi);

    dict["cuint_dipole_ffi"] = EncapsulateFfiHandler(DipoleFfi);
    dict["cuint_dipole_gradient_ffi"] = EncapsulateFfiHandler(DipoleGradientFfi);

    dict["cuint_dipole_ffi"] = EncapsulateFfiHandler(QuadrupoleFfi);
    dict["cuint_dipole_gradient_ffi"] = EncapsulateFfiHandler(QuadrupoleGradientFfi);
    return dict;
}

NB_MODULE(_cuint, m) {
    m.def("registrations", &Registrations);
}

} // namespace cuint
} // namespace pyscfad
