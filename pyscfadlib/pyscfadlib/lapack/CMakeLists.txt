#add_library(lapack_kernels STATIC
#  lapack_kernels.cc
#)
#set_target_properties(lapack_kernels PROPERTIES
#  POSITION_INDEPENDENT_CODE ON
#  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
#)
#target_link_libraries(lapack_kernels ${BLAS_LIBRARIES})

nanobind_add_module(lapack_ad lapack.cc lapack_kernels.cc)
set_target_properties(lapack_ad PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)
target_link_libraries(lapack_ad PRIVATE ${BLAS_LIBRARIES})
