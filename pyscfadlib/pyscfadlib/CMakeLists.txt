cmake_minimum_required (VERSION 3.18)
project (pyscfadlib)

find_package(Python
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule
)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# FFI
execute_process(
  COMMAND "${Python_EXECUTABLE}"
          "-c" "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")

if(NOT XLA_DIR)
  message(FATAL_ERROR "Failed to find XLA include directory. Ensure JAX is installed and accessible.")
endif()

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_VERBOSE_MAKEFILE OFF)
if (CMAKE_COMPILER_IS_GNUCC)
  if(UNIX AND NOT APPLE AND NOT DEFINED ENV{LDFLAGS})
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed")
  endif()
endif()
set(CMAKE_C_FLAGS "-Wall ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-Wall ${CMAKE_CXX_FLAGS}")

#if (APPLE)
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -faligned-allocation")
#endif()

option(BUILD_MARCH_NATIVE "gcc flag -march=native" off)
if (BUILD_MARCH_NATIVE)
  include(CheckCCompilerFlag)
  CHECK_C_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
  if(COMPILER_SUPPORTS_MARCH_NATIVE)
    if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
      message("Add CFLAGS -march=native -unroll-aggressive")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -unroll-aggressive -ipo")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-math-errno")
    else()
      message("Add CFLAGS -march=native -ftree-vectorize")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -funroll-loops -ftree-vectorize")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-math-errno")
    endif()
  endif()
else()
  if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    include(CheckCCompilerFlag)
    CHECK_C_COMPILER_FLAG("-msse3" COMPILER_SUPPORTS_SSE3)
    if(COMPILER_SUPPORTS_SSE3)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse3")
    endif()
  endif()
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (NOT BLAS_LIBRARIES)
  find_package(BLAS)
endif()

if (NOT BLAS_LIBRARIES)
  message(FATAL_ERROR "A required library with BLAS API not found.")
else()
  message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
endif()

if (NOT LAPACK_LIBRARIES)
  find_package(LAPACK)
endif()

if (NOT LAPACK_LIBRARIES)
  message(FATAL_ERROR "A required library with LAPACK API not found.")
else()
  message(STATUS "LAPACK libraries: ${LAPACK_LIBRARIES}")
endif()

option(ENABLE_OPENMP "Compiling C extensions with openmp" ON)
set(OPENMP_C_PROPERTIES "")
if(ENABLE_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(HAVE_OPENMP 1)
    set(OPENMP_C_PROPERTIES OpenMP::OpenMP_C)
  endif()
endif()

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/include)
include_directories(${CMAKE_INSTALL_PREFIX}/include)
include_directories(${XLA_DIR})
link_directories(${PROJECT_SOURCE_DIR}/thirdparty/lib ${PROJECT_SOURCE_DIR}/thirdparty/lib64)
link_directories(${CMAKE_INSTALL_PREFIX}/lib ${CMAKE_INSTALL_PREFIX}/lib64)

configure_file(
  "${PROJECT_SOURCE_DIR}/config.h.in"
  "${PROJECT_SOURCE_DIR}/config.h")
include_directories("${PROJECT_BINARY_DIR}")

if (WIN32)
  #?
elseif (APPLE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/thirdparty/lib;@loader_path/thirdparty/lib64")
  set(CMAKE_BUILD_RPATH "@loader_path;@loader_path/thirdparty/lib;@loader_path/thirdparty/lib64")
else ()
  set(CMAKE_SKIP_BUILD_RPATH  TRUE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/thirdparty/lib:\$ORIGIN/thirdparty/lib64")
endif()

add_subdirectory(np_helper)
add_subdirectory(gto)
add_subdirectory(vjp)
add_subdirectory(lapack)

set(C_LINK_TEMPLATE "<CMAKE_C_COMPILER> <CMAKE_SHARED_LIBRARY_C_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")

include(ExternalProject)

option(BUILD_LIBCINT "Using libcint for analytical gaussian integral" ON)
option(WITH_F12 "Compling F12 integrals" OFF)

#FIXME here I have to rename libcint otherwise it conflicts with the one used by pyscf
set(patch_libcint ${PROJECT_SOURCE_DIR}/apply_patch.sh libcint.patch)

if(BUILD_LIBCINT)
  ExternalProject_Add(libcint
    GIT_REPOSITORY https://github.com/fishjojo/libcint.git
    GIT_TAG 0e4e4c74de4d92c211e68bc4f547ec999bf9012e
    PATCH_COMMAND ${patch_libcint}
    PREFIX ${PROJECT_BINARY_DIR}/thirdparty
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/thirdparty
    CMAKE_CACHE_ARGS
        -DWITH_F12:STRING=${WITH_F12}
        -DWITH_RANGE_COULOMB:STRING=1
        -DWITH_FORTRAN:STRING=0
        -DWITH_CINT2_INTERFACE:STRING=0
        -DMIN_EXPCUTOFF:STRING=20
        -DKEEP_GOING:STRING=1
        -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        -DCMAKE_INSTALL_LIBDIR:PATH=lib
        -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
        -DBLAS_LIBRARIES:STRING=${BLAS_LIBRARIES}
        -DCMAKE_C_CREATE_SHARED_LIBRARY:STRING=${C_LINK_TEMPLATE}
        -DBUILD_MARCH_NATIVE:STRING=${BUILD_MARCH_NATIVE}
  )
  add_dependencies(cgto_ad libcint)
endif()
