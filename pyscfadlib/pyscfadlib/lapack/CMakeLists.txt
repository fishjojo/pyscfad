nanobind_add_module(lapack_ad lapack.cc lapack_kernels.cc)
set_target_properties(lapack_ad PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)
target_link_libraries(lapack_ad PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
