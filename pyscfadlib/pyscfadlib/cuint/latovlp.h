#ifndef PYSCFADLIB_CUINT_LATOVLP_H_
#define PYSCFADLIB_CUINT_LATOVLP_H_

#include "xla/ffi/api/ffi.h"

namespace pyscfad {
namespace cuint {

XLA_FFI_DECLARE_HANDLER_SYMBOL(LatOverlapFfi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(LatOverlapGradientFfi);

} // namespace cuint
} // namespace pyscfad

#endif

