load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

CUINT_COMMIT = "efbdeb1d4380f49805aeafd686d936a048c7b7e5"
CUINT_SHA256 = "86930bfc5bb931f005d3bfed6f855d53390b42bb2ad2e35c8f9a263450632e1c"

def repo():
    http_archive(
        name = "cuint",
        strip_prefix = "cuint-{commit}".format(commit=CUINT_COMMIT),
        url = "https://github.com/fishjojo/cuint/archive/{commit}.tar.gz".format(commit=CUINT_COMMIT),
        sha256 = CUINT_SHA256,
        build_file = "//third_party/cuint:cuint.bazel",
    )
