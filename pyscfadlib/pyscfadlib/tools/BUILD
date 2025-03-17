load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load(
    "//pyscfadlib:pyscfad.bzl",
    "PLATFORM_TAGS_DICT",
    "pytype_strict_library",
)

package(default_visibility = ["//visibility:public"])

genrule(
    name = "platform_tags_py",
    srcs = [],
    outs = ["platform_tags.py"],
    cmd = "echo 'PLATFORM_TAGS_DICT = %s' > $@;" % PLATFORM_TAGS_DICT,
)

pytype_strict_library(
    name = "build_utils",
    srcs = [
        "build_utils.py",
        ":platform_tags_py",
    ],
)

py_binary(
    name = "build_cuda_kernels_wheel",
    srcs = ["build_cuda_kernels_wheel.py"],
    data = [
        "LICENSE.txt",
    ] + if_cuda([
        "//pyscfadlib:version.py",
        "//pyscfadlib/cuda:_solver",
        "//plugins/cuda:plugin_pyproject.toml",
        "//plugins/cuda:plugin_setup.py",
    ]),
    deps = [
        ":build_utils",
        "@bazel_tools//tools/python/runfiles",
        "@pypi_build//:pkg",
        "@pypi_setuptools//:pkg",
        "@pypi_wheel//:pkg",
    ],
)
