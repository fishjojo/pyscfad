#!/usr/bin/env python
"""Build the pyscfad-cuda{12,13}-plugin wheel with CMake (no Bazel).

Drives plugins/cuda/CMakeLists.txt to compile the _solver and _cuint nanobind
modules, assembles a small setuptools source tree, and runs ``python -m build``.
Replaces the former Bazel flow (build/build.py + tools/build_cuda_kernels_wheel.py).

Usage::

    python build_plugin.py --cuda-major 13                 # detect arch from toolkit
    python build_plugin.py --cuda-major 12 --cuda-arch "70-real;80-real;90-real"
    python build_plugin.py --cuda-arch 75-real             # fast local single-arch

The CUDA major version (default: detected from ``nvcc``) tags the wheel as
``pyscfad-cuda<major>-plugin`` and selects the matching ``nvidia-*-cu<major>``
runtime dependencies. Build against a CUDA toolkit whose major matches.
"""

import argparse
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent            # pyscfadlib/plugins/cuda
PYSCFADLIB_ROOT = HERE.parent.parent                      # pyscfadlib/ (outer dir)
VERSION_PY = PYSCFADLIB_ROOT / "pyscfadlib" / "version.py"


def detect_cuda_major():
    """Return the CUDA major version reported by nvcc, or None."""
    try:
        out = subprocess.check_output(["nvcc", "--version"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    m = re.search(r"release (\d+)\.", out)
    return m.group(1) if m else None


def run(cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cuda-major", default=None,
                    help="CUDA major version (12 or 13). Default: detect from nvcc.")
    ap.add_argument("--cuda-arch", default=None,
                    help="Override CMAKE_CUDA_ARCHITECTURES, e.g. '75-real;80-real'. "
                         "Default: up to sm_120 for the CUDA major.")
    ap.add_argument("--output-path", default=str(pathlib.Path.cwd() / "dist"),
                    help="Directory the wheel is written to (default: ./dist).")
    ap.add_argument("--build-dir", default=None,
                    help="CMake build directory (default: a temporary directory).")
    ap.add_argument("--jobs", default=str(os.cpu_count() or 4),
                    help="Parallel build jobs.")
    args = ap.parse_args()

    cuda_major = args.cuda_major or detect_cuda_major()
    if cuda_major is None:
        sys.exit("Could not detect the CUDA major version; pass --cuda-major.")
    package = f"pyscfad_cuda{cuda_major}_plugin"
    print(f"Building {package} (CUDA major {cuda_major})", flush=True)

    build_dir = pathlib.Path(args.build_dir
                             or tempfile.mkdtemp(prefix="pyscfad_cuda_build_"))
    src_tree = pathlib.Path(tempfile.mkdtemp(prefix="pyscfad_cuda_wheel_"))

    # 1. Configure, build, and install the .so directly into the package dir.
    configure = [
        "cmake", "-S", str(HERE), "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        # Pin CMake to *this* interpreter so it finds the nanobind/jax installed
        # here. Manylinux images ship several Pythons and CMake's FindPython
        # would otherwise pick an arbitrary (dependency-less) one.
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-DPYSCFAD_CUDA_MAJOR={cuda_major}",
        f"-DPYSCFAD_PLUGIN_INSTALL_DIR={package}",
    ]
    if args.cuda_arch:
        configure.append(f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}")
    run(configure)
    run(["cmake", "--build", str(build_dir), "-j", args.jobs])
    run(["cmake", "--install", str(build_dir), "--prefix", str(src_tree)])

    pkg_dir = src_tree / package
    if not list(pkg_dir.glob("_solver*.so")) or not list(pkg_dir.glob("_cuint*.so")):
        sys.exit(f"Expected _solver/_cuint modules under {pkg_dir}; build incomplete.")

    # 2. Assemble the setuptools source tree (mirrors the former wheel builder).
    shutil.copy(VERSION_PY, pkg_dir / "version.py")
    shutil.copy(HERE / "plugin_pyproject.toml", src_tree / "pyproject.toml")
    setup_text = (HERE / "plugin_setup.py").read_text()
    setup_text = setup_text.replace("cuda_version = 0  # placeholder",
                                    f"cuda_version = {cuda_major}")
    (src_tree / "setup.py").write_text(setup_text)

    # 3. Build the wheel. Use build isolation so setuptools/wheel are provisioned
    #    from plugin_pyproject.toml's requires (the only build backend deps); this
    #    keeps the driver self-contained in minimal environments.
    os.makedirs(args.output_path, exist_ok=True)
    run([sys.executable, "-m", "build", "--wheel", str(src_tree)])
    for wheel in sorted((src_tree / "dist").glob("*.whl")):
        dest = pathlib.Path(args.output_path) / wheel.name
        shutil.copy(wheel, dest)
        print(f"\nOutput wheel: {dest}\n  pip install '{dest}[with_cuda]'", flush=True)


if __name__ == "__main__":
    main()
