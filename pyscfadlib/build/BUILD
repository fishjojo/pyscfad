load("@python_version_repo//:py_version.bzl", "REQUIREMENTS")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")

compile_pip_requirements(
    name = "requirements",
    srcs = ["requirements.in",],
    extra_args = [
        "--allow-unsafe",
        "--build-isolation",
        "--rebuild",
        "--index-url=https://us-python.pkg.dev/ml-oss-artifacts-published/pypi-mirror/simple",
    ],
    requirements_txt = REQUIREMENTS,
    generate_hashes = True,
)
