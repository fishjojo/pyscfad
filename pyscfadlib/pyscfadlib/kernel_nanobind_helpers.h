#ifndef PYSCFADLIB_KERNEL_NANOBIND_HELPERS_H_
#define PYSCFADLIB_KERNEL_NANOBIND_HELPERS_H_

#include <bit>
#include <type_traits>
#include <nanobind/nanobind.h>
#include "xla/ffi/api/c_api.h"

namespace pyscfad {

template <typename T>
nanobind::capsule EncapsulateFunction(T* fn) {
    return nanobind::capsule(reinterpret_cast<void*>(fn),
                             "xla._CUSTOM_CALL_TARGET");
}

template <typename T>
nanobind::capsule EncapsulateFfiHandler(T* fn) {
    static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                  "Encapsulated function must be an XLA FFI handler");
    return nanobind::capsule(reinterpret_cast<void*>(fn));
}

} // namespace pyscfad

#endif
