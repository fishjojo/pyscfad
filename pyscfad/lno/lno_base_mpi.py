# Copyright 2023-2026 The PySCFAD Authors
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

from mpi4py import MPI
import numpy
from pyscfad.ops import stop_trace
from pyscfad.lno import lno_base
from pyscfad.lno.tools import autofrag, map_lo_to_frag

def partition_jobs(frag_lolist, frag_wghtlist):
    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    rank = comm.Get_rank()
    nfrag = len(frag_lolist)
    lolist = [frag_lolist[i] for i in range(nfrag) if i % nproc == rank]
    wghtlist = [frag_wghtlist[i] for i in range(nfrag) if i % nproc == rank]
    return lolist, wghtlist

class LNO(lno_base.LNO):
    def kernel(self,
               frag_lolist=None,
               frag_wghtlist=None,
               frag_atmlist=None,
               lo_type=None,
               no_type=None,
               frag_nonvlist=None,
               orbloc=None,
               lo_init_guess=None,
               lo_symmetry=False,
               lo_options=None,
               job_partition_list=None):
        if lo_type is None:
            lo_type = self.lo_type
        if no_type is None:
            no_type = self.no_type
        if orbloc is None:
            orbloc = self.get_lo(lo_type=lo_type, init_guess=lo_init_guess,
                                 symmetry=lo_symmetry,  options=lo_options)

        # LO assignment to fragments
        if frag_lolist is None:
            if frag_atmlist is None:
                #log.info('Grouping LOs by single-atom fragments')
                frag_atmlist = stop_trace(autofrag)(self.mol)
            else:
                #log.info('Grouping LOs by user input atom-based fragments')
                pass
            frag_lolist = stop_trace(map_lo_to_frag)(self.mol, orbloc, frag_atmlist,
                                                          verbose=self.verbose)
        elif frag_lolist == '1o':
            #log.info('Using single-LO fragment')
            frag_lolist = numpy.arange(orbloc.shape[1]).reshape(-1,1)
        else:
            #log.info('Using user input LO-fragment assignment')
            pass

        if job_partition_list is not None:
            tmp = []
            for i in job_partition_list:
                tmp.append(frag_lolist[i])
            frag_lolist = tmp

        nfrag = len(frag_lolist)
        if frag_wghtlist is None:
            frag_wghtlist = numpy.ones(nfrag)
        else:
            assert len(frag_wghtlist) == len(frag_lolist)

        frag_lolist_local, frag_wghtlist_local = partition_jobs(frag_lolist, frag_wghtlist)

        frag_res_local = lno_base.kernel(self, orbloc, frag_lolist_local,
                                         frag_nonvlist=frag_nonvlist)
        self._post_proc(frag_res_local, frag_wghtlist_local)
