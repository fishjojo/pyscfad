cmake_minimum_required(VERSION 3.18)
project(pyscfad_cuda_plugin LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Builds the two CUDA nanobind extensions that make up the
# pyscfad-cuda{12,13}-plugin wheel:
#   _solver -> cuSOLVER-backed generalized eigensolver (sygvd/hegvd)
#   _cuint  -> custom GPU integral kernels (overlap/dipole/quadrupole)
#
# Replaces the former Bazel build (WORKSPACE + BUILD files). The wheel itself is
# assembled by plugins/cuda/build_plugin.py, which drives this CMake project.
# ---------------------------------------------------------------------------

# The C++ sources live in the sibling pyscfadlib package. Their #includes are
# written relative to that package's parent directory, e.g.
# "pyscfadlib/cuda/solver_kernels.h", so the include root is PYSCFADLIB_ROOT.
get_filename_component(PYSCFADLIB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule
)

# nanobind (same discovery as the core library's CMakeLists.txt).
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# XLA FFI headers (xla/ffi/api/ffi.h), discovered from the installed jax exactly
# as the core library does.
execute_process(
  COMMAND "${Python_EXECUTABLE}" "-c" "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
if(NOT XLA_DIR)
  message(FATAL_ERROR "Failed to find XLA include directory. Ensure JAX is installed.")
endif()
message(STATUS "XLA include directory: ${XLA_DIR}")

# CUDA toolkit. Provides imported targets CUDA::cudart / CUDA::cublas /
# CUDA::cusolver and the include dir, without enabling the CUDA language: the
# plugin's own translation units are host C++; device code lives in the external
# cuint static library, which is built by its own CMake (see below).
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA toolkit: ${CUDAToolkit_VERSION} (major ${CUDAToolkit_VERSION_MAJOR})")

# CUDA major version the wheel is tagged for. Defaults to the toolkit major so a
# plain `cmake` against a CUDA 13 toolkit produces a cuda13 plugin.
set(PYSCFAD_CUDA_MAJOR "${CUDAToolkit_VERSION_MAJOR}" CACHE STRING
    "CUDA major version the plugin targets (12 or 13).")
if(NOT "${PYSCFAD_CUDA_MAJOR}" STREQUAL "${CUDAToolkit_VERSION_MAJOR}")
  message(WARNING
    "PYSCFAD_CUDA_MAJOR=${PYSCFAD_CUDA_MAJOR} but the CUDA toolkit is major "
    "${CUDAToolkit_VERSION_MAJOR}; build against the matching toolkit.")
endif()

# Device architectures for the cuint kernels, controlled by the standard
# CMAKE_CUDA_ARCHITECTURES. When unset, default to a set covering up to sm_120
# (PTX) for the target CUDA major (CUDA 13 dropped sm_70 and below). Override
# with e.g. -DCMAKE_CUDA_ARCHITECTURES="75-real" for a fast single-arch build.
if(NOT CMAKE_CUDA_ARCHITECTURES)
  if(PYSCFAD_CUDA_MAJOR GREATER_EQUAL 13)
    # CUDA 13 has no sm_101 (Blackwell arches are sm_100/103/110/120).
    set(CMAKE_CUDA_ARCHITECTURES "75-real;80-real;90-real;100-real;110-real;120-virtual")
  else()
    set(CMAKE_CUDA_ARCHITECTURES "70-real;80-real;90-real;100-real;101-real;120-virtual")
  endif()
endif()
message(STATUS "cuint CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

# Where the built .so are placed/installed (the wheel package directory).
# build_plugin.py sets this to pyscfad_cuda<major>_plugin.
set(PYSCFAD_PLUGIN_INSTALL_DIR "." CACHE STRING
    "Install destination (relative to CMAKE_INSTALL_PREFIX) for the .so files.")

# ---------------------------------------------------------------------------
# External cuint library (custom GPU integral kernels), fetched from GitHub and
# built with its own CMake. The architecture list is forwarded via the standard
# CMAKE_CUDA_ARCHITECTURES (cuint honours it as of the pinned commit).
# ---------------------------------------------------------------------------
include(ExternalProject)
set(CUINT_COMMIT "8fb61a157357a8e030ce6766e5d507815d48fe8a")
set(CUINT_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/cuint")
set(CUINT_INSTALL_DIR "${CUINT_PREFIX}/install")
set(CUINT_LIBRARY "${CUINT_INSTALL_DIR}/lib/libcuint.a")

# ExternalProject splits ';'-lists into separate args, so pass the architecture
# list with an alternate separator and let ExternalProject restore the ';'.
string(REPLACE ";" "|" CUINT_ARCH "${CMAKE_CUDA_ARCHITECTURES}")
ExternalProject_Add(cuint_ext
  GIT_REPOSITORY https://github.com/fishjojo/cuint.git
  GIT_TAG ${CUINT_COMMIT}
  PREFIX ${CUINT_PREFIX}
  INSTALL_DIR ${CUINT_INSTALL_DIR}
  LIST_SEPARATOR "|"
  CMAKE_CACHE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=Release
    -DBUILD_SHARED_LIBS:BOOL=OFF
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_CUDA_ARCHITECTURES:STRING=${CUINT_ARCH}
    -DCMAKE_INSTALL_LIBDIR:PATH=lib
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
  BUILD_BYPRODUCTS ${CUINT_LIBRARY}
)

# ---------------------------------------------------------------------------
# _solver : cuSOLVER generalized eigensolver
# ---------------------------------------------------------------------------
nanobind_add_module(_solver STABLE_ABI
  ${PYSCFADLIB_ROOT}/pyscfadlib/cuda/solver.cc
  ${PYSCFADLIB_ROOT}/pyscfadlib/cuda/solver_kernels.cc
  ${PYSCFADLIB_ROOT}/pyscfadlib/cuda/handle_pool.cc
)
target_include_directories(_solver PRIVATE
  ${PYSCFADLIB_ROOT} ${XLA_DIR} ${CUDAToolkit_INCLUDE_DIRS})
target_compile_options(_solver PRIVATE -fno-strict-aliasing)
target_link_libraries(_solver PRIVATE CUDA::cusolver CUDA::cublas CUDA::cudart)

# Opt-in cuSOLVER FP32/FP64 emulation (handle_pool.cc, gated by PYSCFADLIB_CUSOLVER_*
# env vars). The math-mode/strategy API and FP32 BF16x9 emulation arrived in CUDA 13.0;
# FP64 fixed-point emulation in 13.2. Gate on the toolkit version (minor matters) so older
# toolkits and the cuda12 plugin build cleanly and treat the env vars as a no-op.
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
  target_compile_definitions(_solver PRIVATE PYSCFAD_HAVE_CUSOLVER_EMULATION)
endif()
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.2")
  target_compile_definitions(_solver PRIVATE PYSCFAD_HAVE_CUSOLVER_FP64_EMULATION)
endif()
message(STATUS "_solver: cuSOLVER emulation FP32>=13.0 FP64>=13.2 (toolkit ${CUDAToolkit_VERSION})")

# ---------------------------------------------------------------------------
# _cuint : custom GPU integral kernels (links the external cuint static lib)
# ---------------------------------------------------------------------------
nanobind_add_module(_cuint STABLE_ABI
  ${PYSCFADLIB_ROOT}/pyscfadlib/cuint/cuint.cc
  ${PYSCFADLIB_ROOT}/pyscfadlib/cuint/ovlp.cc
  ${PYSCFADLIB_ROOT}/pyscfadlib/cuint/dipole.cc
  ${PYSCFADLIB_ROOT}/pyscfadlib/cuint/quadrupole.cc
)
add_dependencies(_cuint cuint_ext)
target_include_directories(_cuint PRIVATE
  ${PYSCFADLIB_ROOT} ${XLA_DIR} ${CUDAToolkit_INCLUDE_DIRS} ${CUINT_INSTALL_DIR}/include)
target_compile_options(_cuint PRIVATE -fno-strict-aliasing)
target_link_libraries(_cuint PRIVATE ${CUINT_LIBRARY} CUDA::cudart)

install(TARGETS _solver _cuint LIBRARY DESTINATION ${PYSCFAD_PLUGIN_INSTALL_DIR})
