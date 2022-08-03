import sys
import numpy
from pyscf.pbc.dft import numint as pyscf_numint
from pyscf.pbc.dft.gen_grid import make_mask, BLKSIZE
from pyscfad.lib import numpy as np
from pyscfad.lib import ops, stop_grad
from pyscfad.dft import numint
from pyscfad.dft.numint import eval_mat, _contract_rho, _dot_ao_dm

def nr_rks(ni, cell, grids, xc_code, dms, spin=0, relativity=0, hermi=0,
           kpts=None, kpts_band=None, max_memory=2000, verbose=None):
    if kpts is None:
        kpts = numpy.zeros((1,3))

    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(cell, dms, hermi)

    nelec = [0]*nset
    excsum = [0]*nset
    vmat = [0]*nset
    if xctype == 'LDA':
        ao_deriv = 0
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, kpts_band,
                                 max_memory):
            for i in range(nset):
                rho = make_rho(i, ao_k2, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, rho, spin=0,
                                      relativity=relativity, deriv=1)[:2]
                den = rho*weight
                nelec[i] += stop_grad(den).sum()
                excsum[i] += (den*exc).sum()
                vmat[i] += ni.eval_mat(cell, ao_k1, weight, rho, vxc,
                                       mask, xctype, 0, verbose)
    elif xctype == 'GGA':
        ao_deriv = 1
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, kpts_band,
                                 max_memory):
            for i in range(nset):
                rho = make_rho(i, ao_k2, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, rho, spin=0,
                                      relativity=relativity, deriv=1)[:2]
                den = rho[0]*weight
                nelec[i] += stop_grad(den).sum()
                excsum[i] += (den*exc).sum()
                vmat[i] += ni.eval_mat(cell, ao_k1, weight, rho, vxc,
                                       mask, xctype, 0, verbose)
    elif xctype == 'MGGA':
        if any(x in xc_code.upper() for x in ('CC06', 'CS', 'BR89', 'MK00')):
            raise NotImplementedError('laplacian in meta-GGA method')
        ao_deriv = 2
        for ao_k1, ao_k2, mask, weight, coords \
                in ni.block_loop(cell, grids, nao, ao_deriv, kpts, kpts_band,
                                 max_memory):
            for i in range(nset):
                rho = make_rho(i, ao_k2, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, rho, spin=0,
                                      relativity=relativity, deriv=1)[:2]
                den = rho[0]*weight
                nelec[i] += stop_grad(den).sum()
                excsum[i] += (den*exc).sum()
                vmat[i] += ni.eval_mat(cell, ao_k1, weight, rho, vxc,
                                       mask, xctype, 0, verbose)

    nelec = numpy.asarray(nelec)
    excsum = np.asarray(excsum)
    vmat = np.asarray(vmat)
    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
        vmat = vmat[0]
    return nelec, excsum, vmat

def eval_ao(cell, coords, kpt=numpy.zeros(3), deriv=0, relativity=0, shls_slice=None,
            non0tab=None, out=None, verbose=None):
    ao_kpts = eval_ao_kpts(cell, coords, numpy.reshape(kpt, (-1,3)), deriv,
                           relativity, shls_slice, non0tab, out, verbose)
    return ao_kpts[0]

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
    if xctype in ['LDA', 'HF']:
        ngrids, nao = ao.shape
    else:
        ngrids, nao = ao[0].shape

    if non0tab is None:
        non0tab = numpy.empty(((ngrids+BLKSIZE-1)//BLKSIZE, cell.nbas),
                              dtype=numpy.uint8)
        non0tab[:] = 0xff

    # complex orbitals or density matrix
    if np.iscomplexobj(ao) or np.iscomplexobj(dm):
        shls_slice = (0, cell.nbas)
        ao_loc = cell.ao_loc_nr()
        dm = np.asarray(dm, dtype=np.complex128)

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
            rho = np.empty((4,ngrids))
            c0 = _dot_ao_dm(cell, ao[0], dm, non0tab, shls_slice, ao_loc)
            #rho[0] = dot_bra(ao[0], c0)
            rho = ops.index_update(rho, ops.index[0], dot_bra(ao[0], c0))
            for i in range(1, 4):
                #rho[i] = dot_bra(ao[i], c0) * 2
                rho = ops.index_update(rho, ops.index[i], dot_bra(ao[i], c0) * 2)
        else:
            # rho[4] = \nabla^2 rho, rho[5] = 1/2 |nabla f|^2
            rho = np.empty((6,ngrids))
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

class NumInt(numint.NumInt):
    def nr_rks(self, cell, grids, xc_code, dms, hermi=0,
               kpt=numpy.zeros(3), kpts_band=None, max_memory=2000, verbose=None):
        if kpts_band is not None:
            # To compute Vxc on kpts_band, convert the NumInt object to KNumInt object.
            ni = KNumInt()
            ni.__dict__.update(self.__dict__)
            nao = dms.shape[-1]
            return ni.nr_rks(cell, grids, xc_code, dms.reshape(-1,1,nao,nao),
                             hermi, kpt.reshape(1,3), kpts_band, max_memory,
                             verbose)
        return nr_rks(self, cell, grids, xc_code, dms,
                      0, 0, hermi, kpt, kpts_band, max_memory, verbose)

    def eval_ao(self, cell, coords, kpt=numpy.zeros(3), deriv=0, relativity=0,
                shls_slice=None, non0tab=None, out=None, verbose=None):
        return eval_ao(cell, coords, kpt, deriv, relativity, shls_slice,
                       non0tab, out, verbose)

    def eval_mat(self, cell, ao, weight, rho, vxc,
                 non0tab=None, xctype='LDA', spin=0, verbose=None):
        # Guess whether ao is evaluated for kpts_band.  When xctype is LDA, ao on grids
        # should be a 2D array.  For other xc functional, ao should be a 3D array.
        if ao.ndim == 2 or (xctype != 'LDA' and ao.ndim == 3):
            mat = eval_mat(cell, ao, weight, rho, vxc, non0tab, xctype, spin, verbose)
        else:
            nkpts = len(ao)
            nao = ao[0].shape[-1]
            mat = np.empty((nkpts,nao,nao), dtype=np.complex128)
            for k in range(nkpts):
                mat[k] = eval_mat(cell, ao[k], weight, rho, vxc,
                                  non0tab, xctype, spin, verbose)
        return mat

    def block_loop(self, cell, grids, nao=None, deriv=0, kpt=numpy.zeros(3),
                   kpts_band=None, max_memory=2000, non0tab=None, blksize=None):
        '''Define this macro to loop over grids by blocks.
        '''
        # For UniformGrids, grids.coords does not indicate whehter grids are initialized
        if grids.non0tab is None:
            grids.build(with_non0tab=True)
        if nao is None:
            nao = cell.nao
        grids_coords = grids.coords
        grids_weights = grids.weights
        ngrids = grids_coords.shape[0]
        comp = (deriv+1)*(deriv+2)*(deriv+3)//6
# NOTE to index grids.non0tab, the blksize needs to be the integer multiplier of BLKSIZE
        if blksize is None:
            blksize = int(max_memory*1e6/(comp*2*nao*16*BLKSIZE))*BLKSIZE
            blksize = max(BLKSIZE, min(blksize, ngrids, BLKSIZE*1200))
        if non0tab is None:
            non0tab = grids.non0tab
        if non0tab is None:
            non0tab = numpy.empty(((ngrids+BLKSIZE-1)//BLKSIZE,cell.nbas),
                                  dtype=numpy.uint8)
            non0tab[:] = 0xff
        kpt = numpy.reshape(kpt, 3)
        if kpts_band is None:
            kpt1 = kpt2 = kpt
        else:
            kpt1 = kpts_band
            kpt2 = kpt

        for ip0 in range(0, ngrids, blksize):
            ip1 = min(ngrids, ip0+blksize)
            coords = grids_coords[ip0:ip1]
            weight = grids_weights[ip0:ip1]
            non0 = non0tab[ip0//BLKSIZE:]
            ao_k2 = self.eval_ao(cell, coords, kpt2, deriv=deriv, non0tab=non0)
            if abs(kpt1-kpt2).sum() < 1e-9:
                ao_k1 = ao_k2
            else:
                ao_k1 = self.eval_ao(cell, coords, kpt1, deriv=deriv)
            yield ao_k1, ao_k2, non0, weight, coords
            ao_k1 = ao_k2 = None

class KNumInt(numint.NumInt):
    def __init__(self, kpts=numpy.zeros((1,3))):
        numint.NumInt.__init__(self)
        self.kpts = kpts #numpy.reshape(kpts, (-1,3))

    def nr_rks(self, cell, grids, xc_code, dms, hermi=0, kpts=None, kpts_band=None,
               max_memory=2000, verbose=None, **kwargs):
        if kpts is None:
            if 'kpt' in kwargs:
                sys.stderr.write('WARN: KNumInt.nr_rks function finds keyword '
                                 'argument "kpt" and converts it to "kpts"\n')
                kpts = kwargs['kpt']
            else:
                kpts = self.kpts
        kpts = kpts.reshape(-1,3)

        return nr_rks(self, cell, grids, xc_code, dms, 0, 0,
                      hermi, kpts, kpts_band, max_memory, verbose)

    def eval_mat(self, cell, ao_kpts, weight, rho, vxc,
                 non0tab=None, xctype='LDA', spin=0, verbose=None):
        nkpts = len(ao_kpts)
        nao = ao_kpts[0].shape[-1]
        #dtype = np.result_type(*ao_kpts)
        #mat = np.empty((nkpts,nao,nao), dtype=dtype)
        mat = [0] * nkpts
        for k in range(nkpts):
            mat[k] = eval_mat(cell, ao_kpts[k], weight, rho, vxc,
                              non0tab, xctype, spin, verbose)
        return np.asarray(mat)

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
                dms = [np.stack(dms)]
            nao = dms[0].shape[-1]
            ndms = len(dms)

            def make_rho(idm, ao_kpts, non0tab, xctype):
                return self.eval_rho(cell, ao_kpts, dms[idm], non0tab, xctype,
                                     hermi=hermi)
        return make_rho, ndms, nao

    block_loop = pyscf_numint.KNumInt.block_loop
