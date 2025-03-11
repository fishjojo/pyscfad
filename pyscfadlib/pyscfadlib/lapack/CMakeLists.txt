nanobind_add_module(pyscfad_lapack lapack.cc lapack_kernels.cc STABLE_ABI)
set_target_properties(pyscfad_lapack PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)
target_link_libraries(pyscfad_lapack PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
