load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

XLA_COMMIT = "a7e58876b73dfa9bb6bc785f6981a332c133bb55"
XLA_SHA256 = "28808e845c5d409422c0762a02324348d598a57e4d5250d0ad4866ae3e1a531f"

def repo():
    http_archive(
        name = "xla",
        strip_prefix = "xla-{commit}".format(commit=XLA_COMMIT),
        url = "https://github.com/openxla/xla/archive/{commit}.tar.gz".format(commit=XLA_COMMIT),
        sha256 = XLA_SHA256,
    )
