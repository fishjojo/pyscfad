package_group(
    name = "internal",
    packages = [
        "//...",
    ],
)

exports_files(["version.py"])

cc_library(
    name = "ffi_helpers",
    hdrs = ["ffi_helpers.h"],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        "@xla//xla/ffi/api:c_api",
        "@xla//xla/ffi/api:ffi",
    ],
)

cc_library(
    name = "kernel_nanobind_helpers",
    hdrs = ["kernel_nanobind_helpers.h"],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        "@nanobind",
        "@xla//xla/ffi/api:c_api",
    ],
)
