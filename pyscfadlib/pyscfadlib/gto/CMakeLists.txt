# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#FIXME here renaming pyscf libraries to avoid runtime conflict
add_library(cgto_ad SHARED
  fill_int2c.c fill_nr_3c.c fill_r_3c.c fill_int2e.c fill_r_4c.c
  ft_ao.c ft_ao_deriv.c fill_grids_int2c.c
  grid_ao_drv.c deriv1.c deriv2.c nr_ecp.c nr_ecp_deriv.c
  autocode/auto_eval1.c)

set_target_properties(cgto_ad PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

target_link_libraries(cgto_ad cintad np_helper_ad ${BLAS_LIBRARIES} ${OPENMP_C_PROPERTIES})

