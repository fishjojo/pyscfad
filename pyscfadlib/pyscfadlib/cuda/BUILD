load(
    "//pyscfadlib:pyscfad.bzl",
    "nanobind_extension",
)

package(
    default_applicable_licenses = [],
    default_visibility = ["//pyscfadlib:internal"],
)

cc_library(
    name = "cuda_vendor",
    hdrs = ["//pyscfadlib/cuda:vendor.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "cusolver_kernels",
    srcs = ["//pyscfadlib/cuda:solver_kernels.cc"],
    hdrs = ["//pyscfadlib/cuda:solver_kernels.h"],
    deps = [
        ":cuda_vendor",
        "//pyscfadlib:ffi_helpers",
        "@local_config_cuda//cuda:cuda_headers",
        "@xla//xla/ffi/api:ffi",
        "@xla//xla/tsl/cuda:cublas",
        "@xla//xla/tsl/cuda:cudart",
        "@xla//xla/tsl/cuda:cusolver",
    ],
)

nanobind_extension(
    name = "_solver",
    srcs = ["//pyscfadlib/cuda:solver.cc"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    module_name = "_solver",
    deps = [
        ":cuda_vendor",
        ":cusolver_kernels",
        "//pyscfadlib:kernel_nanobind_helpers",
        "@local_config_cuda//cuda:cuda_headers",
        "@nanobind",
        "@xla//xla/tsl/cuda:cublas",
        "@xla//xla/tsl/cuda:cudart",
        "@xla//xla/tsl/cuda:cusolver",
    ],
)
