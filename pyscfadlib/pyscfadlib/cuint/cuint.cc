#include "pyscfadlib/kernel_nanobind_helpers.h"
#include "pyscfadlib/cuint/ovlp.h"
#include "pyscfadlib/cuint/dipole.h"
#include "pyscfadlib/cuint/quadrupole.h"
#include "pyscfadlib/cuint/latovlp.h"

namespace pyscfad {
namespace cuint {

namespace nb = nanobind;

// Highest total derivative order (i_deriv + j_deriv) the linked cuint overlap
// kernels were compiled for; set by the plugin build (CUINT_MAX_DERIV). cuint's
// dispatch silently does nothing above its compiled cap, so callers must check
// this before requesting a derivative order.
#ifndef PYSCFAD_CUINT_MAX_DERIV
#define PYSCFAD_CUINT_MAX_DERIV 2
#endif

int MaxDeriv() { return PYSCFAD_CUINT_MAX_DERIV; }

nb::dict Registrations() {
    nb::dict dict;

    dict["cuint_overlap_ffi"] = EncapsulateFfiHandler(OverlapFfi);
    dict["cuint_overlap_gradient_ffi"] = EncapsulateFfiHandler(OverlapGradientFfi);
    dict["cuint_gen_overlap_ffi"] = EncapsulateFfiHandler(GenOverlapFfi);

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
    m.def("max_deriv", &MaxDeriv);
}

} // namespace cuint
} // namespace pyscfad
