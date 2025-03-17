nanobind_add_module(pyscfad_cusolver solver.cc solver_kernels.cc STABLE_ABI)
set_target_properties(pyscfad_cusolver PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)
target_include_directories(pyscfad_cusolver PUBLIC ${CUDAToolkit_INCLUDE_DIRS} ${XLA_DIR})
target_link_libraries(pyscfad_cusolver PRIVATE CUDA::cusolver CUDA::cudart)
