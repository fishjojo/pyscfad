#include "pyscfadlib/kernel_nanobind_helpers.h"
#include "pyscfadlib/cuint/ovlp.h"
#include "pyscfadlib/cuint/dipole.h"
#include "pyscfadlib/cuint/quadrupole.h"
#include "pyscfadlib/cuint/latovlp.h"

namespace pyscfad {
namespace cuint {

namespace nb = nanobind;

nb::dict Registrations() {
    nb::dict dict;

    dict["cuint_overlap_ffi"] = EncapsulateFfiHandler(OverlapFfi);
    dict["cuint_overlap_gradient_ffi"] = EncapsulateFfiHandler(OverlapGradientFfi);

    dict["cuint_dipole_ffi"] = EncapsulateFfiHandler(DipoleFfi);
    dict["cuint_dipole_gradient_ffi"] = EncapsulateFfiHandler(DipoleGradientFfi);

    dict["cuint_quadrupole_ffi"] = EncapsulateFfiHandler(QuadrupoleFfi);
    dict["cuint_quadrupole_gradient_ffi"] = EncapsulateFfiHandler(QuadrupoleGradientFfi);

    dict["cuint_lat_overlap_ffi"] = EncapsulateFfiHandler(LatOverlapFfi);
    dict["cuint_lat_overlap_gradient_ffi"] = EncapsulateFfiHandler(LatOverlapGradientFfi);
    return dict;
}

NB_MODULE(_cuint, m) {
    m.def("registrations", &Registrations);
}

} // namespace cuint
} // namespace pyscfad
