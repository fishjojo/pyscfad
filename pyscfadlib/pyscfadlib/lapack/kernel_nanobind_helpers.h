#ifndef PYSCFADLIB_KERNEL_NANOBIND_HELPERS_H_
#define PYSCFADLIB_KERNEL_NANOBIND_HELPERS_H_

#include <nanobind/nanobind.h>

namespace pyscfad {

template <typename T>
nanobind::capsule EncapsulateFunction(T* fn) {
    return nanobind::capsule((void*) fn,
                             "xla._CUSTOM_CALL_TARGET");
}

} // namespace pyscfad

#endif
