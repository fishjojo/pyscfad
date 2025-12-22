load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

XLA_COMMIT = "913ae2eaa3cb88971003592a90959685a78c9e30"
XLA_SHA256 = "9fa67049a1b3db0ae25895dc7e6edb8bf05edaabe811484053636c0127090590"

def repo():
    http_archive(
        name = "xla",
        strip_prefix = "xla-{commit}".format(commit=XLA_COMMIT),
        url = "https://github.com/openxla/xla/archive/{commit}.tar.gz".format(commit=XLA_COMMIT),
        sha256 = XLA_SHA256,
    )
