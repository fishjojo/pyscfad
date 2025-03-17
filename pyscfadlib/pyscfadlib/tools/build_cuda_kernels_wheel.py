# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Script that builds a jax-cuda12-plugin wheel for cuda kernels, intended to be
# run via bazel run as part of the jax cuda plugin build process.

# Most users should not run this script directly; use build.py instead.

import argparse
import functools
import os
import pathlib
import tempfile

from bazel_tools.tools.python.runfiles import runfiles
from pyscfadlib.tools import build_utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_path",
    default=None,
    required=True,
    help="Path to which the output wheel should be written. Required.",
)
parser.add_argument(
    "--pyscfadlib_git_hash",
    default="",
    required=True,
    help="Git hash. Empty if unknown. Optional.",
)
parser.add_argument(
    "--cpu", default=None, required=True, help="Target CPU architecture. Required."
)
parser.add_argument(
    "--platform_version",
    default=None,
    required=True,
    help="Target CUDA version. Required.",
)
parser.add_argument(
    "--editable",
    action="store_true",
    help="Create an 'editable' CUDA plugin build instead of a wheel.",
)
parser.add_argument(
    "--enable-cuda",
    default=False,
    help="Should we build with CUDA enabled?")
args = parser.parse_args()

r = runfiles.Create()
pyext = "so"


def write_setup_cfg(sources_path, cpu):
  tag = build_utils.platform_tag(cpu)
  with open(sources_path / "setup.cfg", "w") as f:
    f.write(f"""[metadata]
license_files = LICENSE.txt

[bdist_wheel]
plat_name={tag}
""")


def prepare_wheel_cuda(
    sources_path: pathlib.Path, *, cpu, cuda_version
):
  """Assembles a source tree for the cuda kernel wheel in `sources_path`."""
  copy_runfiles = functools.partial(build_utils.copy_file, runfiles=r)

  copy_runfiles(
      "__main__/plugins/cuda/plugin_pyproject.toml",
      dst_dir=sources_path,
      dst_filename="pyproject.toml",
  )
  copy_runfiles(
      "__main__/plugins/cuda/plugin_setup.py",
      dst_dir=sources_path,
      dst_filename="setup.py",
  )
  build_utils.update_setup_with_cuda_version(sources_path, cuda_version)
  write_setup_cfg(sources_path, cpu)

  plugin_dir = sources_path / f"pyscfad_cuda{cuda_version}_plugin"
  copy_runfiles(
      dst_dir=plugin_dir,
      src_files=[
          f"__main__/pyscfadlib/cuda/_solver.{pyext}",
          "__main__/pyscfadlib/version.py",
      ],
  )


# Build wheel for cuda kernels
tmpdir = tempfile.TemporaryDirectory(prefix="pyscfad_cuda_plugin")
sources_path = tmpdir.name
try:
  os.makedirs(args.output_path, exist_ok=True)
  if args.enable_cuda:
    prepare_wheel_cuda(
        pathlib.Path(sources_path), cpu=args.cpu, cuda_version=args.platform_version
    )
    package_name = f"pyscfad cuda{args.platform_version} plugin"
  if args.editable:
    build_utils.build_editable(sources_path, args.output_path, package_name)
  else:
    build_utils.build_wheel(
        sources_path,
        args.output_path,
        package_name,
        git_hash=args.pyscfadlib_git_hash,
    )
finally:
  tmpdir.cleanup()
