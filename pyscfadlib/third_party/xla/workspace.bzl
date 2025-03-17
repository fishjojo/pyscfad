load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

XLA_COMMIT = "6e396aae2e534dc7fc5387e2aa8b1a3a8d79a3db"
XLA_SHA256 = "03e73863ff041c57d2fdefb4216c2774bb12faf70dd083dbe8e961ccb9ea42b2"

def repo():
    http_archive(
        name = "xla",
        strip_prefix = "xla-{commit}".format(commit=XLA_COMMIT),
        url = "https://github.com/openxla/xla/archive/{commit}.tar.gz".format(commit=XLA_COMMIT),
        sha256 = XLA_SHA256,
    )
