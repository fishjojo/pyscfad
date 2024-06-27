#cgto
add_library(cgto_vjp SHARED
  gto/fill_int2c_vjp.c
  gto/fill_int2e_vjp.c
  gto/fill_nr_3c_vjp.c
)
set_target_properties(cgto_vjp PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)
target_link_libraries(cgto_vjp cgto_ad ${OPENMP_C_PROPERTIES})

#np_helper
add_library(np_helper_vjp SHARED
  np_helper/pack_tril_vjp.c
)
set_target_properties(np_helper_vjp PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)
target_link_libraries(np_helper_vjp ${OPENMP_C_PROPERTIES})

#ao2mo
add_library(ao2mo_vjp SHARED
  ao2mo/nr_ao2mo_vjp.c
)
set_target_properties(ao2mo_vjp PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)
target_link_libraries(ao2mo_vjp np_helper_ad vjp_util ${BLAS_LIBRARIES} ${OPENMP_C_PROPERTIES})

#vhf
add_library(cvhf_vjp SHARED
  vhf/nr_incore_vjp.c
  vhf/df_jk_vjp.c
)
set_target_properties(cvhf_vjp PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)
target_link_libraries(cvhf_vjp np_helper_ad vjp_util ${BLAS_LIBRARIES} ${OPENMP_C_PROPERTIES})

#cc
add_library(cc_vjp SHARED
  cc/ccsd_t.c
  cc/ccsd_t_vjp.c
  cc/lno_ccsd_t.c
  cc/lno_ccsd_t_vjp.c
)
set_target_properties(cc_vjp PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)
target_link_libraries(cc_vjp vjp_util ${BLAS_LIBRARIES} ${OPENMP_C_PROPERTIES})

#util
add_library(vjp_util SHARED
  util/omp_reduce.c
  util/pack_tril.c
)
set_target_properties(vjp_util PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)
target_link_libraries(vjp_util ${OPENMP_C_PROPERTIES})

