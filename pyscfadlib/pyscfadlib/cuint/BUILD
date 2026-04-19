load(
    "//pyscfadlib:pyscfad.bzl",
    "nanobind_extension",
)

package(
    default_applicable_licenses = [],
    default_visibility = ["//pyscfadlib:internal"],
)

cc_library(
    name = "cuint_ovlp",
    srcs = ["//pyscfadlib/cuint:ovlp.cc"],
    hdrs = ["//pyscfadlib/cuint:ovlp.h"],
    deps = [
        "//pyscfadlib/cuda:cuda_vendor",
        "//pyscfadlib:ffi_helpers",
        "@local_config_cuda//cuda:cuda_headers",
        "@xla//xla/ffi/api:ffi",
        "@xla//xla/tsl/cuda:cudart",
        "@cuint//:cuint"
    ],
)

cc_library(
    name = "cuint_dipole",
    srcs = ["//pyscfadlib/cuint:dipole.cc"],
    hdrs = ["//pyscfadlib/cuint:dipole.h"],
    deps = [
        "//pyscfadlib/cuda:cuda_vendor",
        "//pyscfadlib:ffi_helpers",
        "@local_config_cuda//cuda:cuda_headers",
        "@xla//xla/ffi/api:ffi",
        "@xla//xla/tsl/cuda:cudart",
        "@cuint//:cuint"
    ],
)

cc_library(
    name = "cuint_quadrupole",
    srcs = ["//pyscfadlib/cuint:quadrupole.cc"],
    hdrs = ["//pyscfadlib/cuint:quadrupole.h"],
    deps = [
        "//pyscfadlib/cuda:cuda_vendor",
        "//pyscfadlib:ffi_helpers",
        "@local_config_cuda//cuda:cuda_headers",
        "@xla//xla/ffi/api:ffi",
        "@xla//xla/tsl/cuda:cudart",
        "@cuint//:cuint"
    ],
)

nanobind_extension(
    name = "_cuint",
    srcs = ["//pyscfadlib/cuint:cuint.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_cuint",
    deps = [
        ":cuint_ovlp",
        ":cuint_dipole",
        ":cuint_quadrupole",
        "//pyscfadlib:kernel_nanobind_helpers",
        "@local_config_cuda//cuda:cuda_headers",
        "@nanobind",
        "@xla//xla/tsl/cuda:cudart",
    ],
)
