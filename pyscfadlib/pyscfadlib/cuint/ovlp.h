#ifndef PYSCFADLIB_CUINT_OVLP_H_
#define PYSCFADLIB_CUINT_OVLP_H_

#include "xla/ffi/api/ffi.h"

namespace pyscfad {
namespace cuint {

XLA_FFI_DECLARE_HANDLER_SYMBOL(OverlapFfi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(OverlapGradientFfi);

} // namespace cuint
} // namespace pyscfad

#endif
