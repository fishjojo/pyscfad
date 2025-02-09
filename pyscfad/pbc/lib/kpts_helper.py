import numpy
from pyscf.pbc.lib.kpts_helper import KPT_DIFF_TOL
from pyscfad.ops import stop_grad

def is_zero(kpt):
    return abs(numpy.asarray(stop_grad(kpt))).sum() < KPT_DIFF_TOL
gamma_point = is_zero
