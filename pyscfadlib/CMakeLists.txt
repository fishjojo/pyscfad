cmake_minimum_required (VERSION 3.11)
project (pyscfadlib)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
endif()
set(CMAKE_VERBOSE_MAKEFILE OFF)
if (CMAKE_COMPILER_IS_GNUCC)
  if(UNIX AND NOT APPLE AND NOT DEFINED ENV{LDFLAGS})
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed")
  endif()
endif()
set(CMAKE_C_FLAGS "-Wall ${CMAKE_C_FLAGS}")

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
  set(CMAKE_SKIP_BUILD_RPATH  True)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH True)
  set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/thirdparty/lib:\$ORIGIN/thirdparty/lib64")
endif()

add_subdirectory(vjp)

set(C_LINK_TEMPLATE "<CMAKE_C_COMPILER> <CMAKE_SHARED_LIBRARY_C_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")

include(ExternalProject)

option(BUILD_LIBCINT "Using libcint for analytical gaussian integral" ON)
option(WITH_F12 "Compling F12 integrals" ON)

if(BUILD_LIBCINT)
  ExternalProject_Add(libcint
    GIT_REPOSITORY https://github.com/fishjojo/libcint.git
    GIT_TAG ad
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
        -DCMAKE_C_CREATE_SHARED_LIBRARY:STRING=${C_LINK_TEMPLATE}
        -DBUILD_MARCH_NATIVE:STRING=${BUILD_MARCH_NATIVE}
  )
  add_dependencies(cgto_vjp libcint)
endif()


option(BUILD_PYSCF_LIB "Building PySCF library" ON)

set(patch_pyscf ${PROJECT_SOURCE_DIR}/patch_pyscf.sh)

if(BUILD_PYSCF_LIB)
  ExternalProject_Add(pyscf
    GIT_REPOSITORY https://github.com/pyscf/pyscf.git
    GIT_TAG v2.3.0
    PATCH_COMMAND ${patch_pyscf}
    SOURCE_SUBDIR "pyscf/lib"
    PREFIX ${PROJECT_BINARY_DIR}/thirdparty
    INSTALL_DIR ${PROJECT_SOURCE_DIR}/thirdparty
    CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
            -DCMAKE_INSTALL_LIBDIR:PATH=lib
            -DCMAKE_INSTALL_INCLUDEDIR:PATH=include/pyscf
            -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
            -DBLA_VENDOR:STRING=${BLA_VENDOR}
            -DENABLE_LIBXC:STRING=OFF
            -DENABLE_XCFUN:STRING=OFF
            -DBUILD_LIBCINT:STRING=OFF
  )
  add_dependencies(cgto_vjp pyscf)
  add_dependencies(np_helper_vjp pyscf)
  add_dependencies(ao2mo_vjp pyscf)
  add_dependencies(cvhf_vjp pyscf)
  add_dependencies(cc_vjp pyscf)
endif()
