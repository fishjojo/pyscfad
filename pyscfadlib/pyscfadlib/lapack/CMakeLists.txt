nanobind_add_module(_lapack lapack.cc lapack_kernels.cc STABLE_ABI)
set_target_properties(_lapack PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)
target_link_libraries(_lapack PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
