'''LNO-CCSD(T) with MPI parallelization.

run with:
mpirun -n 2 python 11-mpi_lno_ccsd_t.py
'''
from mpi4py import MPI
import jax
import numpy
from pyscfad import gto, scf, mp
from pyscfad.cc import dfccsd
from pyscfad import config
from pyscfad.lno import MPI_LNOCCSD

config.update('pyscfad_moleintor_opt', True)
config.update('pyscfad_scf_implicit_diff', True)
config.update('pyscfad_ccsd_implicit_diff', True)

atom = 'water_dimer.xyz'
basis = 'ccpvdz'

mol = gto.Mole(atom=atom, basis=basis)
mol.verbose = 4
mol.build(trace_exp=False, trace_ctr_coeff=False)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
frozen = 2
thresh_occ = 1e-3
thresh_vir = 1e-4
def energy(mol):
    mf = scf.RHF(mol).density_fit()
    ehf = mf.kernel()

    mfcc = MPI_LNOCCSD(mf, frozen=frozen)
    mfcc.thresh_occ = thresh_occ
    mfcc.thresh_vir = thresh_vir
    mfcc.lo_type = 'iao'
    mfcc.ccsd_t = True
    mfcc.kernel(frag_lolist=None)

    if rank == 0:
        mmp = mp.dfmp2.MP2(mf, frozen=frozen)
        mmp.kernel(with_t2=False)
        ecc_pt2corrected = mfcc.e_corr_pt2corrected(mmp.e_corr)
        etot = ehf + ecc_pt2corrected
    else:
        etot = mfcc.e_corr - mfcc.e_corr_pt2
    return etot

e, jac = jax.value_and_grad(energy)(mol)
e = numpy.asarray(e)
grad = numpy.asarray(jac.coords)

if rank == 0:
    etot = numpy.zeros_like(e)
    grad_tot = numpy.zeros_like(grad)
else:
    etot = None
    grad_tot = None

comm.Reduce([e, MPI.DOUBLE], [etot, MPI.DOUBLE],
            op=MPI.SUM, root=0)

comm.Reduce([grad, MPI.DOUBLE], [grad_tot, MPI.DOUBLE],
            op=MPI.SUM, root=0)

if rank == 0:
    print(f'LNO-CCSD(T) energy: {etot}')
    print(f'LNO-CCSD(T) gradient:\n{grad_tot}')
