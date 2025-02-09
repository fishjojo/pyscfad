import warnings
from functools import partial
import numpy
from pyscf.dft import numint
from pyscf.dft.gen_grid import BLKSIZE
from pyscfad import numpy as np
from pyscfad import ops
from pyscfad.ops import (
    stop_grad,
    jit,
    custom_jvp,
)
from pyscfad.dft import libxc

def eval_mat(mol, ao, weight, rho, vxc,
             non0tab=None, xctype='LDA', spin=0, verbose=None):
    xctype = xctype.upper()
    if xctype in ['LDA', 'HF']:
        ngrids, _ = ao.shape
    else:
        ngrids, _ = ao[0].shape

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.uint8)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    transpose_for_uks = False
    if xctype in ['LDA', 'HF']:
        if not getattr(vxc, 'ndim', None) == 2:
            vrho = vxc[0]
        else:
            vrho = vxc
        wv = .5 * weight * vrho
        mat = _dot_ao_ao(ao, ao, wv)
    else:
        vrho, vsigma = vxc[:2]
        if spin == 0:
            assert(vsigma is not None and rho.ndim==2)
            wv_rho = weight * vrho * .5
            wv_sigma = rho[1:4] * (weight * vsigma * 2)
            wv = np.concatenate((wv_rho.reshape(1,-1), wv_sigma))
        else:
            rho_a, rho_b = rho
            wv_rho = weight * vrho * .5
            try:
                wv_sigma  = rho_a[1:4] * (weight * vsigma[0] * 2)
                wv_sigma += rho_b[1:4] * (weight * vsigma[1])
            except ValueError:
                warnings.warn('Note the output of libxc.eval_xc cannot be '
                              'directly used in eval_mat.\nvsigma from eval_xc '
                              'should be restructured as '
                              '(vsigma[:,0],vsigma[:,1])\n')
                transpose_for_uks = True
                vsigma = vsigma.T
                wv_sigma  = rho_a[1:4] * (weight * vsigma[0] * 2)
                wv_sigma += rho_b[1:4] * (weight * vsigma[1])
            wv = np.concatenate((wv_rho.reshape(1,-1), wv_sigma))
        mat = _dot_ao_ao(ao[:4], ao[0], wv)

    # JCP 138, 244108 (2013); DOI:10.1063/1.4811270
    # JCP 112, 7002 (2000); DOI:10.1063/1.481298
    if xctype == 'MGGA':
        vlapl, vtau = vxc[2:]

        if vlapl is None:
            vlapl = 0
        else:
            if spin != 0:
                if transpose_for_uks:
                    vlapl = vlapl.T
                vlapl = vlapl[0]
            XX, YY, ZZ = 4, 7, 9
            ao2 = ao[XX] + ao[YY] + ao[ZZ]
            wv = .5 * weight * vlapl
            mat += _dot_ao_ao(ao2, ao[0], wv)

        if spin != 0:
            if transpose_for_uks:
                vtau = vtau.T
            vtau = vtau[0]
        wv = weight * (.25 * vtau + vlapl)
        mat += _dot_ao_ao(ao[1], ao[1], wv)
        mat += _dot_ao_ao(ao[2], ao[2], wv)
        mat += _dot_ao_ao(ao[3], ao[3], wv)

    return mat + mat.T.conj()

def nr_rks(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
           max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi)
    ao_loc = mol.ao_loc_nr()

    shls_slice = (0, mol.nbas)

    nelec  = [0,] * nset
    excsum = [0,] * nset
    vmat   = [0,] * nset
    aow = None
    if xctype == 'LDA':
        ao_deriv = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for idm in range(nset):
                rho = make_rho(idm, ao, mask, 'LDA')
                exc, vxc = ni.eval_xc(xc_code, rho, spin=0,
                                      relativity=relativity, deriv=1,
                                      verbose=verbose)[:2]
                den = rho * weight
                nelec[idm] += stop_grad(den).sum()
                excsum[idm] += np.dot(den, exc)
                wv = vxc[0] * weight * .5
                vmat[idm] += _dot_ao_ao(ao, ao, wv)
    elif xctype == 'GGA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for idm in range(nset):
                rho = make_rho(idm, ao, mask, 'GGA')
                exc, vxc = ni.eval_xc(xc_code, rho, spin=0,
                                      relativity=relativity, deriv=1,
                                      verbose=verbose)[:2]
                den = rho[0] * weight
                nelec[idm] += stop_grad(den).sum()
                excsum[idm] += np.dot(den, exc)
                wv = _rks_gga_wv0(rho, vxc, weight)
                vmat[idm] += _dot_ao_ao(ao, ao[0], wv)
    elif xctype == 'MGGA':
        if any(x in xc_code.upper() for x in ('CC06', 'CS', 'BR89', 'MK00')):
            raise NotImplementedError('laplacian in meta-GGA method')
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for idm in range(nset):
                rho = make_rho(idm, ao, mask, 'MGGA')
                exc, vxc = ni.eval_xc(xc_code, rho, spin=0,
                                      relativity=relativity, deriv=1,
                                      verbose=verbose)[:2]
                # pylint: disable=W0612
                vrho, vsigma, vlapl, vtau = vxc[:4]
                den = rho[0] * weight
                nelec[idm] += stop_grad(den).sum()
                excsum[idm] += np.dot(den, exc)

                wv = _rks_gga_wv0(rho, vxc, weight)
                vmat[idm] += _dot_ao_ao(ao[:4], ao[0], wv)
                # NOTE .5 * .5   First 0.5 for v+v.T symmetrization.
                # Second 0.5 is due to the Libxc convention tau = 1/2 \nabla\phi\dot\nabla\phi
                wv = .5 * .5 * weight * vtau
                vmat[idm] += _dot_ao_ao(ao[1], ao[1], wv)
                vmat[idm] += _dot_ao_ao(ao[2], ao[2], wv)
                vmat[idm] += _dot_ao_ao(ao[3], ao[3], wv)
    elif xctype == 'HF':
        pass
    else:
        raise NotImplementedError(f'numint.nr_rks for functional {xc_code}')

    for i in range(nset):
        vmat[i] = vmat[i] + vmat[i].conj().T
    nelec = numpy.asarray(nelec)
    excsum = np.asarray(excsum)
    vmat = np.asarray(vmat)
    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
        vmat = vmat[0]
    return nelec, excsum, vmat

def nr_uks(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
           max_memory=2000, verbose=None):

    xctype = ni._xc_type(xc_code)

    dma, dmb = _format_uks_dm(dms)
    nao      = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(mol, dma, hermi, False, grids)[:2]
    make_rhob       = ni._gen_rho_evaluator(mol, dmb, hermi, False, grids)[0]

    nelec = numpy.zeros((2,nset))
    excsum = [0] * nset
    vmat = [[0]*nset for _ in range(2)]
    aow    = None

    if xctype == 'LDA':
        ao_deriv = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for idm in range(nset):
                rho_a = make_rhoa(idm, ao, mask, 'LDA')
                rho_b = make_rhob(idm, ao, mask, 'LDA')

                exc, vxc = ni.eval_xc(xc_code, (rho_a, rho_b), spin=1,
                                      relativity=relativity, deriv=1,
                                      verbose=verbose)[:2]

                vrho = vxc[0]

                den            = rho_a * weight
                nelec[0][idm] += stop_grad(den).sum()
                excsum[idm]   += np.dot(den, exc)

                den            = rho_b * weight
                nelec[1][idm] += stop_grad(den).sum()
                excsum[idm]   += np.dot(den, exc)

                aow           = _scale_ao(ao, .5*weight*vrho[:,0])
                vmat[0][idm] += _dot_ao_ao(ao, aow)

                aow           = _scale_ao(ao, .5*weight*vrho[:,1])
                vmat[1][idm] += _dot_ao_ao(ao, aow)

    elif xctype == 'GGA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for idm in range(nset):
                rho_a = make_rhoa(idm, ao, mask, 'GGA')
                rho_b = make_rhob(idm, ao, mask, 'GGA')

                exc, vxc = ni.eval_xc(xc_code, (rho_a, rho_b), spin=1,
                                      relativity=relativity, deriv=1,
                                      verbose=verbose)[:2]

                den            = rho_a[0] * weight
                nelec[0][idm] += stop_grad(den).sum()
                excsum[idm]   += np.dot(den, exc)

                den            = rho_b[0] * weight
                nelec[1][idm] += stop_grad(den).sum()
                excsum[idm]   += np.dot(den, exc)

                wva, wvb      = _uks_gga_wv0((rho_a,rho_b), vxc, weight)

                aow           = _scale_ao(ao, wva)
                vmat[0][idm] += _dot_ao_ao(ao[0], aow)

                aow           = _scale_ao(ao, wvb)
                vmat[1][idm] += _dot_ao_ao(ao[0], aow)

    elif xctype == 'MGGA':
        if any(x in xc_code.upper() for x in ('CC06', 'CS', 'BR89', 'MK00')):
            raise NotImplementedError('laplacian in meta-GGA method')
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            for idm in range(nset):
                rho_a = make_rhoa(idm, ao, mask, 'MGGA')
                rho_b = make_rhob(idm, ao, mask, 'MGGA')

                exc, vxc = ni.eval_xc(xc_code, (rho_a, rho_b), spin=1,
                                      relativity=relativity, deriv=1,
                                      verbose=verbose)[:2]

                vrho, vsigma, vlapl, vtau = vxc[:4]

                den            = rho_a[0]*weight
                nelec[0][idm] += stop_grad(den).sum()
                excsum[idm]   += np.dot(den, exc)

                den            = rho_b[0]*weight
                nelec[1][idm] += stop_grad(den).sum()
                excsum[idm]   += np.dot(den, exc)

                wva, wvb      = _uks_gga_wv0((rho_a,rho_b), vxc, weight)

                aow           = _scale_ao(ao[:4], wva)
                vmat[0][idm] += _dot_ao_ao(ao[0], aow)

                aow           = _scale_ao(ao[:4], wvb)
                vmat[1][idm] += _dot_ao_ao(ao[0], aow)

                wv = (.25 * weight * vtau[:,0]).reshape(-1,1)
                vmat[0][idm] += _dot_ao_ao(ao[1], wv*ao[1])
                vmat[0][idm] += _dot_ao_ao(ao[2], wv*ao[2])
                vmat[0][idm] += _dot_ao_ao(ao[3], wv*ao[3])

                wv = (.25 * weight * vtau[:,1]).reshape(-1,1)
                vmat[1][idm] += _dot_ao_ao(ao[1], wv*ao[1])
                vmat[1][idm] += _dot_ao_ao(ao[2], wv*ao[2])
                vmat[1][idm] += _dot_ao_ao(ao[3], wv*ao[3])

    elif xctype == 'HF':
        pass
    else:
        raise NotImplementedError(f'numint.nr_uks for functional {xc_code}')

    for i in range(nset):
        vmat[0][i] = vmat[0][i] + vmat[0][i].conj().T
        vmat[1][i] = vmat[1][i] + vmat[1][i].conj().T

    if getattr(dma, 'ndim', None) == 2:
        excsum = excsum[0]
        nelec  = numpy.asarray([nelec[0], nelec[1]])
        vmat   = np.asarray([vmat[0][0], vmat[1][0]])

    return nelec, excsum, vmat

def _format_uks_dm(dms):
    if getattr(dms, 'ndim', None) == 2:  # RHF DM
        dma = dmb = dms * .5
    else:
        dma, dmb = dms
    return dma, dmb

def eval_rho(mol, ao, dm, non0tab=None, xctype='LDA', hermi=0,
             with_lapl=True, verbose=None):
    xctype = xctype.upper()

    if xctype == 'LDA' or xctype == 'HF':
        c0 = np.dot(ao, dm)
        rho = _contract_rho(ao, c0)
    elif xctype in ('GGA', 'NLC'):
        rho = _rks_gga_assemble_rho(ao, dm, hermi)
    else: # meta-GGA
        rho = _rks_mgga_assemble_rho(ao, dm, hermi, with_lapl)
    return rho

@partial(jit, static_argnames=['hermi'])
def _rks_gga_assemble_rho(ao, dm, hermi):
    rho = []
    fac = 2. if hermi==1 else 1.

    c0 = np.dot(ao[0], dm)
    rho.append(_contract_rho(ao[0], c0))
    for i in range(1, 4):
        rho.append(_contract_rho(ao[i], c0, fac))

    if hermi != 1:
        c1 = np.dot(ao[0], dm.conj().T)
        for i in range(1, 4):
            rho[i] += _contract_rho(c1, ao[i])

    rho = np.asarray(rho)
    return rho

@partial(jit, static_argnames=['hermi', 'with_lapl'])
def _rks_mgga_assemble_rho(ao, dm, hermi, with_lapl):
    ngrids = ao.shape[-2]
    if with_lapl:
        # rho[4] = \nabla^2 rho, rho[5] = 1/2 |nabla f|^2
        rho = np.empty((6,ngrids))
    else:
        rho = np.empty((5,ngrids))

    tau_idx = -1
    fac = 2. if hermi==1 else 1.

    c0 = np.dot(ao[0], dm)
    rho = ops.index_update(rho, ops.index[0], _contract_rho(ao[0], c0))
    rho = ops.index_update(rho, ops.index[tau_idx], 0)
    for i in range(1, 4):
        c1 = np.dot(ao[i], dm)
        rho = ops.index_add(rho, ops.index[tau_idx], _contract_rho(ao[i], c1))

        rho = ops.index_update(rho, ops.index[i], _contract_rho(ao[i], c0, fac))
        if hermi != 1:
            rho = ops.index_add(rho, ops.index[i], _contract_rho(c1, ao[0]))

    if with_lapl:
        assert ao.shape[0] > 4
        XX, YY, ZZ = 4, 7, 9
        ao2 = ao[XX] + ao[YY] + ao[ZZ]
        rho = ops.index_update(rho, ops.index[4], _contract_rho(ao2, c0))
        rho = ops.index_add(rho, ops.index[4], rho[5])
        if hermi==1:
            rho = ops.index_mul(rho, ops.index[4], 2.)
        else:
            c2 = np.dot(ao2, dm)
            rho = ops.index_add(rho, ops.index[4], _contract_rho(ao[0], c2))
            rho = ops.index_add(rho, ops.index[4], rho[5])

    rho = ops.index_mul(rho, ops.index[tau_idx], .5)
    return rho

def _scale_ao(ao, wv):
    '''aow = einsum('npi,np->pi', ao[:4], wv)
    '''
    if wv.ndim == 2:
        ao = ao.transpose(0,2,1)
    else:
        ngrids, nao = ao.shape
        ao = ao.T.reshape(1,nao,ngrids)
        wv = wv.reshape(1,ngrids)

    aow = np.einsum('nip,np->pi', ao, wv)
    return aow

@jit
def _dot_ao_ao(ao1, ao2, wv=None):
    '''(ao1*wv).T.dot(ao2)
    '''
    if wv is None:
        return np.dot(ao1.conj().T, ao2)
    else:
        ao1 = _scale_ao(ao1, wv)
        return np.dot(ao1.conj().T, ao2)

@jit
def _contract_rho(bra, ket, factor=1.0):
    bra = bra.T
    ket = ket.T

    rho  = np.einsum('ip,ip->p', bra.real, ket.real)
    rho += np.einsum('ip,ip->p', bra.imag, ket.imag)
    return rho * factor

@jit
def _rks_gga_wv0(rho, vxc, weight):
    vrho, vgamma = vxc[:2]
    wv_rho = weight * vrho * .5
    wv_sigma = (weight * vgamma * 2) * rho[1:4]
    wv = np.concatenate((wv_rho.reshape(1,-1), wv_sigma))
    return wv

@jit
def _uks_gga_wv0(rho, vxc, weight):
    rhoa, rhob   = rho
    vrho, vsigma = vxc[:2]

    wv_rho_a = weight * vrho[:,0] * .5
    wv_sigma_a = (weight * vsigma[:,0] * 2) * rhoa[1:4] + (weight * vsigma[:,1]) * rhob[1:4]
    wva = np.concatenate((wv_rho_a.reshape(1,-1), wv_sigma_a))

    wv_rho_b = weight * vrho[:,1] * .5
    wv_sigma_b = (weight * vsigma[:,2] * 2) * rhob[1:4] + (weight * vsigma[:,1]) * rhoa[1:4]
    wvb = np.concatenate((wv_rho_b.reshape(1,-1), wv_sigma_b))
    return wva, wvb

def nr_nlc_vxc(ni, mol, grids, xc_code, dm, relativity=0, hermi=1,
               max_memory=2000, verbose=None):
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm, hermi, False, grids)
    assert nset == 1

    ao_deriv = 1
    vvrho = []
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
        vvrho.append(make_rho(0, ao, mask, 'GGA'))
    rho = np.hstack(vvrho)

    exc = 0
    vxc = 0
    nlc_coefs = ni.nlc_coeff(xc_code)
    for nlc_pars, fac in nlc_coefs:
        e, v = _vv10nlc(rho, grids.coords, rho, grids.weights,
                        grids.coords, nlc_pars)
        exc += e * fac
        vxc += v * fac
    den = rho[0] * grids.weights
    nelec = stop_grad(den).sum()
    excsum = np.dot(den, exc)

    vmat = np.zeros((nao,nao))
    p1 = 0
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
        p0, p1 = p1, p1 + weight.size
        wv = _rks_gga_wv0(rho[:,p0:p1], vxc[:,p0:p1], weight)
        vmat += _dot_ao_ao(ao, ao[0], wv)

    vmat = vmat + vmat.T
    return nelec, excsum, vmat

@partial(custom_jvp, nondiff_argnums=(1,2,3,4,5,))
def _vv10nlc(rho, coords, vvrho, vvweight, vvcoords, nlc_pars):
    rho = ops.to_numpy(rho)
    vvrho = ops.to_numpy(rho)
    return numint._vv10nlc(rho, coords, vvrho, vvweight, vvcoords, nlc_pars)

@_vv10nlc.defjvp
def _vv10nlc_jvp(coords, vvrho, vvweight, vvcoords, nlc_pars,
                 primals, tangents):
    rho, = primals
    rho_t, = tangents

    exc, vxc = _vv10nlc(rho, coords, vvrho, vvweight, vvcoords, nlc_pars)

    exc_jvp = ((vxc[0] - exc) / rho[0] * rho_t[0]
               + vxc[1] / rho[0] * 2. * np.einsum('np,np->p', rho[1:4], rho_t[1:4]))
    # pylint: disable=W0511
    # FIXME jvp for vxc not implemented
    vxc_jvp = np.zeros_like(vxc)
    return (exc, vxc), (exc_jvp, vxc_jvp)

class NumInt(numint.NumInt):
    def _gen_rho_evaluator(self, mol, dms, hermi=0, with_lapl=True, grids=None):
        if getattr(dms, 'mo_coeff', None) is not None:
            # should be inside pyscf
            return super()._gen_rho_evaluator(mol, dms, hermi=hermi,
                                              with_lapl=with_lapl, grids=grids)

        if getattr(dms, 'ndim', None) == 2:
            dms = [dms]

        if hermi != 1 and dms[0].dtype == np.double:
            dms = [(dm + dm.T) * .5 for dm in dms]
            hermi = 1

        nao = dms[0].shape[0]
        ndms = len(dms)

        def make_rho(idm, ao, non0tab, xctype):
            return self.eval_rho(mol, ao, dms[idm], non0tab, xctype, hermi, with_lapl)

        return make_rho, ndms, nao

    def eval_xc(self, xc_code, rho, spin=0, relativity=0, deriv=1, omega=None,
                verbose=None):
        if omega is None:
            omega = self.omega
        return libxc.eval_xc(xc_code, rho, spin, relativity, deriv,
                             omega, verbose)

    eval_rho = staticmethod(eval_rho)
    nr_rks = nr_rks
    nr_uks = nr_uks
    nr_nlc_vxc = nr_nlc_vxc
