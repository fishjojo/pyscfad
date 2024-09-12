import numpy
import scipy


def logm(A, real=False, **kwargs):
    '''
    Calculates the matrix logarithm ensuring that it is real when the matrix is normal.
    For a normal matrix, the Schur-decomposed matrix t is block-diagonal with blocks of 
    size 1 and 2, the blocks of size 1 are either positive numbers or they come in 
    pairs when they are negative while the blocks of size 2 always have the same value 
    along the diagonal and values with different signs but the same magnitude on the 
    off-diagonals. Since the block-diagonal matrix is similar to the original matrix, 
    the real logarithm can be calculated by determining the real logarithm of the 
    individual blocks and backtransforming with the Schur vectors
    '''
    if real:
        # perform the real Schur decomposition
        t, q = scipy.linalg.schur(A)

        norb = A.shape[0]
        idx = 0
        normalmatrix = True
        while idx < norb:
            # final block reached for an odd number of orbitals
            if (idx == norb - 1):
                # single positive block
                if t[idx,idx] > 0.:
                    t[idx,idx] = numpy.log(t[idx,idx])
                # single negative block (should not happen for normal matrix)
                else:
                    normalmatrix = False
                    break
            else:
                diag = numpy.isclose(t[idx,idx+1], 0.0) and numpy.isclose(t[idx+1,idx], 0.0)
                # single positive block
                if t[idx,idx] > 0. and diag:
                    t[idx,idx] = numpy.log(t[idx,idx])
                # pair of two negative blocks
                elif (
                    t[idx,idx] < 0.
                    and diag
                    and numpy.isclose(t[idx,idx],t[idx + 1,idx + 1])
                ):
                    log_lambda = numpy.log(-t[idx,idx])
                    t[idx:idx+2,idx:idx+2] = numpy.array(
                        [[log_lambda, numpy.pi], [-numpy.pi, log_lambda]]
                    )
                    idx += 1
                # antisymmetric 2x2 block
                elif (
                    numpy.isclose(t[idx,idx], t[idx + 1,idx + 1]) 
                    and numpy.isclose(t[idx + 1,idx],-t[idx,idx + 1])
                ):
                    log_comp = numpy.log(complex(t[idx,idx], t[idx,idx + 1]))
                    t[idx:idx+2,idx:idx+2] = numpy.array(
                        [
                            [numpy.real(log_comp), numpy.imag(log_comp)], 
                            [-numpy.imag(log_comp), numpy.real(log_comp)],
                        ],
                    )
                    idx += 1
                # should not happen for normal matrix
                else:
                    normalmatrix = False
            idx += 1

        if not normalmatrix:
            raise ValueError(
                'Real matrix logarithm can only be ensured for normal matrix'
            )

        return q @ t @ q.T
    else:
        return scipy.linalg.logm(A, **kwargs)
