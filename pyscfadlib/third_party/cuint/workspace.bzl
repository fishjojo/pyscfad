load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

CUINT_COMMIT = "aa9f6ada7550cbe5f4f79eb22540066fa6a1aacc"
CUINT_SHA256 = "1b51db98edf532684a11784985a81092b7f215b41ab25ff5a2342f88870d3d94"

def repo():
    http_archive(
        name = "cuint",
        strip_prefix = "cuint-{commit}".format(commit=CUINT_COMMIT),
        url = "https://github.com/fishjojo/cuint/archive/{commit}.tar.gz".format(commit=CUINT_COMMIT),
        sha256 = CUINT_SHA256,
        build_file = "//third_party/cuint:cuint.bazel",
    )
