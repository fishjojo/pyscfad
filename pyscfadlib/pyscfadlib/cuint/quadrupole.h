#ifndef PYSCFADLIB_CUINT_QUADRUPOLE_H_
#define PYSCFADLIB_CUINT_QUADRUPOLE_H_

#include "xla/ffi/api/ffi.h"

namespace pyscfad {
namespace cuint {

XLA_FFI_DECLARE_HANDLER_SYMBOL(QuadrupoleFfi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(QuadrupoleGradientFfi);

} // namespace cuint
} // namespace pyscfad

#endif
