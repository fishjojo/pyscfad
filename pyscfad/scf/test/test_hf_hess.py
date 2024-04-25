import numpy
from jax import jacrev
from pyscfad import scf

def test_hessian(get_n2):
    mol = get_n2
    def ehf(mol):
        mf = scf.RHF(mol)
        e = mf.kernel()
        return e
    hess = jacrev(jacrev(ehf))(mol).coords.coords
    #analytic result
    hess0 = numpy.asarray(
                [[[[ 1.50164118e-03, 0., 0.],
                   [-1.50164118e-03, 0., 0.]],
                  [[0.,  1.50164118e-03, 0.],
                   [0., -1.50164118e-03, 0.]],
                  [[0., 0.,  1.86573234e+00],
                   [0., 0., -1.86573234e+00]]],
                 [[[-1.50164118e-03, 0., 0.],
                   [ 1.50164118e-03, 0., 0.]],
                  [[0., -1.50164118e-03, 0.],
                   [0.,  1.50164118e-03, 0.]],
                  [[0., 0., -1.86573234e+00],
                   [0., 0.,  1.86573234e+00]]]])
    assert(abs(hess - hess0).max()) < 1e-6
