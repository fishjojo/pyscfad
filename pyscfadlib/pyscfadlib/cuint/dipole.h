#ifndef PYSCFADLIB_CUINT_DIPOLE_H_
#define PYSCFADLIB_CUINT_DIPOLE_H_

#include "xla/ffi/api/ffi.h"

namespace pyscfad {
namespace cuint {

XLA_FFI_DECLARE_HANDLER_SYMBOL(DipoleFfi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(DipoleGradientFfi);

} // namespace cuint
} // namespace pyscfad

#endif
