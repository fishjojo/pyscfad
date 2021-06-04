import sys
import numpy
from pyscf.pbc.dft import numint as pyscf_numint
from pyscf.pbc.dft.gen_grid import make_mask, BLKSIZE
from pyscfad.lib import numpy as jnp
from pyscfad.lib import ops
from pyscfad.dft import numint
from pyscfad.dft.numint import _contract_rho, _dot_ao_dm

def eval_ao_kpts(cell, coords, kpts=None, deriv=0, relativity=0,
                 shls_slice=None, non0tab=None, out=None, verbose=None, **kwargs):
    if kpts is None:
        if 'kpt' in kwargs:
            sys.stderr.write('WARN: KNumInt.eval_ao function finds keyword '
                             'argument "kpt" and converts it to "kpts"\n')
            kpts = kwargs['kpt']
        else:
            kpts = numpy.zeros((1,3))
    kpts = numpy.reshape(kpts, (-1,3))

    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    if cell.cart:
        feval = 'GTOval_cart_deriv%d' % deriv
    else:
        feval = 'GTOval_sph_deriv%d' % deriv
    return cell.pbc_eval_gto(feval, coords, comp, kpts,
                             shls_slice=shls_slice, non0tab=non0tab, out=out)

def eval_rho(cell, ao, dm, non0tab=None, xctype='LDA', hermi=0, verbose=None):
    if xctype == 'LDA' or xctype == 'HF':
        ngrids, nao = ao.shape
    else:
        ngrids, nao = ao[0].shape

    if non0tab is None:
        non0tab = numpy.empty(((ngrids+BLKSIZE-1)//BLKSIZE, cell.nbas),
                              dtype=numpy.uint8)
        non0tab[:] = 0xff

    # complex orbitals or density matrix
    if jnp.iscomplexobj(ao) or jnp.iscomplexobj(dm):
        shls_slice = (0, cell.nbas)
        ao_loc = cell.ao_loc_nr()
        dm = jnp.asarray(dm, dtype=jnp.complex128)

# For GGA, function eval_rho returns   real(|\nabla i> D_ij <j| + |i> D_ij <\nabla j|)
#       = real(|\nabla i> D_ij <j| + |i> D_ij <\nabla j|)
#       = real(|\nabla i> D_ij <j| + conj(|\nabla j> conj(D_ij) < i|))
#       = real(|\nabla i> D_ij <j|) + real(|\nabla j> conj(D_ij) < i|)
#       = real(|\nabla i> [D_ij + (D^\dagger)_ij] <j|)
# symmetrization dm (D + D.conj().T) then /2 because the code below computes
#       2*real(|\nabla i> D_ij <j|)
        if not hermi:
            dm = (dm + dm.conj().T) * .5

        def dot_bra(bra, aodm):
            # rho = numpy.einsum('pi,pi->p', bra.conj(), aodm).real
            #:rho  = numpy.einsum('pi,pi->p', bra.real, aodm.real)
            #:rho += numpy.einsum('pi,pi->p', bra.imag, aodm.imag)
            #:return rho
            return _contract_rho(bra, aodm)

        if xctype == 'LDA' or xctype == 'HF':
            c0 = _dot_ao_dm(cell, ao, dm, non0tab, shls_slice, ao_loc)
            rho = dot_bra(ao, c0)
        elif xctype == 'GGA':
            rho = jnp.empty((4,ngrids))
            c0 = _dot_ao_dm(cell, ao[0], dm, non0tab, shls_slice, ao_loc)
            #rho[0] = dot_bra(ao[0], c0)
            rho = ops.index_update(rho, ops.index[0], dot_bra(ao[0], c0))
            for i in range(1, 4):
                #rho[i] = dot_bra(ao[i], c0) * 2
                rho = ops.index_update(rho, ops.index[i], dot_bra(ao[i], c0) * 2)
        else:
            # rho[4] = \nabla^2 rho, rho[5] = 1/2 |nabla f|^2
            rho = jnp.empty((6,ngrids))
            c0 = _dot_ao_dm(cell, ao[0], dm, non0tab, shls_slice, ao_loc)
            #rho[0] = dot_bra(ao[0], c0)
            rho = ops.index_update(rho, ops.index[0], dot_bra(ao[0], c0))
            #rho[5] = 0
            rho = ops.index_update(rho, ops.index[5], 0)
            for i in range(1, 4):
                #rho[i] = dot_bra(ao[i], c0) * 2  # *2 for +c.c.
                rho = ops.index_update(rho, ops.index[i], dot_bra(ao[i], c0) * 2)
                c1 = _dot_ao_dm(cell, ao[i], dm, non0tab, shls_slice, ao_loc)
                #rho[5] += dot_bra(ao[i], c1)
                rho = ops.index_add(rho, ops.index[5], dot_bra(ao[i], c1))
            XX, YY, ZZ = 4, 7, 9
            ao2 = ao[XX] + ao[YY] + ao[ZZ]
            #rho[4] = dot_bra(ao2, c0)
            rho = ops.index_update(rho, ops.index[4], dot_bra(ao2, c0))
            #rho[4] += rho[5]
            rho = ops.index_add(rho, ops.index[4], rho[5])
            #rho[4] *= 2 # *2 for +c.c.
            rho = ops.index_mul(rho, ops.index[4], 2)
            #rho[5] *= .5
            rho = ops.index_mul(rho, ops.index[5], .5)
    else:
        # real orbitals and real DM
        rho = numint.eval_rho(cell, ao, dm, non0tab, xctype, hermi, verbose)
    return rho

class KNumInt(numint.NumInt):
    def __init__(self, kpts=numpy.zeros((1,3))):
        numint.NumInt.__init__(self)
        self.kpts = numpy.reshape(kpts, (-1,3))

    def eval_ao(self, cell, coords, kpts=numpy.zeros((1,3)), deriv=0, relativity=0,
                shls_slice=None, non0tab=None, out=None, verbose=None, **kwargs):
        return eval_ao_kpts(cell, coords, kpts, deriv,
                            relativity, shls_slice, non0tab, out, verbose)

    def eval_rho(self, cell, ao_kpts, dm_kpts, non0tab=None, xctype='LDA',
                 hermi=0, verbose=None):
        nkpts = len(ao_kpts)
        rhoR = 0
        for k in range(nkpts):
            rhoR += eval_rho(cell, ao_kpts[k], dm_kpts[k], non0tab, xctype,
                             hermi, verbose)
        rhoR *= 1./nkpts
        return rhoR

    def _gen_rho_evaluator(self, cell, dms, hermi=0):
        if getattr(dms, 'mo_coeff', None) is not None:
            raise NotImplementedError
        else:
            if getattr(dms[0], "ndim", 0) == 2:
                dms = [jnp.stack(dms)]
            nao = dms[0].shape[-1]
            ndms = len(dms)

            def make_rho(idm, ao_kpts, non0tab, xctype):
                return self.eval_rho(cell, ao_kpts, dms[idm], non0tab, xctype,
                                     hermi=hermi)
        return make_rho, ndms, nao

    block_loop = pyscf_numint.KNumInt.block_loop
