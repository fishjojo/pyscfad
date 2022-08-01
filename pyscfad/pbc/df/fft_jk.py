import numpy
from jax import vmap
from pyscf import lib as pyscf_lib
from pyscf.lib import logger
from pyscf.pbc.tools import get_coulG
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point
from pyscfad.lib import numpy as np
from pyscfad.lib import ops
from pyscfad.pbc import tools
from pyscfad.pbc.df.df_jk import _ewald_exxdiv_for_G0

def get_j_kpts(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), kpts_band=None, cell=None):
    if cell is None:
        cell = mydf.cell
    mesh = mydf.mesh

    ni = mydf._numint
    make_rho, nset, nao = ni._gen_rho_evaluator(cell, dm_kpts, hermi)
    dm_kpts = np.asarray(dm_kpts)
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    coulG = get_coulG(cell, mesh=mesh)
    ngrids = len(coulG)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)

    coords = mydf.grids.coords
    mask = mydf.grids.non0tab
    ao2_kpts = mydf._numint.eval_ao(cell, coords, kpts=kpts, non0tab=mask)
    if input_band is None:
        ao1_kpts = ao2_kpts
    else:
        ao1_kpts = mydf._numint.eval_ao(cell, coords, kpts=kpts_band, non0tab=mask)

    if hermi == 1 or gamma_point(kpts):
        #rhoR = np.zeros((nset,ngrids))
        #for ao_ks_etc, p0, p1 in mydf.aoR_loop(mydf.grids, kpts, cell=cell):
        #    ao_ks, mask = ao_ks_etc[0], ao_ks_etc[2]
        #    for i in range(nset):
        #        #rhoR[i,p0:p1] += make_rho(i, ao_ks, mask, 'LDA')
        #        rhoR = ops.index_add(rhoR, ops.index[i,p0:p1],
        #                             make_rho(i, ao_ks, mask, 'LDA'))
        #    ao = ao_ks = None

        rhoR = [make_rho(i, ao2_kpts, mask, 'LDA') for i in range(nset)]
        rhoR = np.asarray(rhoR, dtype=np.double)

        def _rhoR_to_vR_real(rhoR):
            rhoG = tools.fft(rhoR, mesh)
            vG = coulG * rhoG
            vR = tools.ifft(vG, mesh).real
            return vR
        vR = vmap(_rhoR_to_vR_real)(rhoR)

    else:  # vR may be complex if the underlying density is complex
        rhoR = np.zeros((nset,ngrids), dtype=np.complex128)
        #for ao_ks_etc, p0, p1 in mydf.aoR_loop(mydf.grids, kpts, cell=cell):
        #    ao_ks, mask = ao_ks_etc[0], ao_ks_etc[2]
        #    for i in range(nset):
        #        for k, ao in enumerate(ao_ks):
        #            ao_dm = np.dot(ao, dms[i,k])
        #            #rhoR[i,p0:p1] += np.einsum('xi,xi->x', ao_dm, ao.conj())
        #            rhoR = ops.index_add(rhoR, ops.index[i,p0:p1], 
        #                                 np.einsum('xi,xi->x', ao_dm, ao.conj()))
        for i in range(nset):
            for k, ao in enumerate(ao2_kpts):
                ao_dm = np.dot(ao, dms[i,k])
                rhoR = rhoR.at[i].add(np.einsum('xi,xi->x', ao_dm, ao.conj()))
        rhoR *= 1./nkpts

        def _rhoR_to_vR_complex(rhoR):
            rhoG = tools.fft(rhoR, mesh)
            vG = coulG * rhoG
            vR = tools.ifft(vG, mesh)
            return vR
        vR = vmap(_rhoR_to_vR_complex)(rhoR)

    #kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    #nband = len(kpts_band)
    weight = cell.vol / ngrids
    vR *= weight
    if gamma_point(kpts_band):
        vj_kpts = np.zeros((nset,nband,nao,nao))
    else:
        vj_kpts = np.zeros((nset,nband,nao,nao), dtype=np.complex128)

    #for ao_ks_etc, p0, p1 in mydf.aoR_loop(mydf.grids, kpts_band, cell=cell):
    #    ao_ks, mask = ao_ks_etc[0], ao_ks_etc[2]
    #    for i in range(nset):
    #        # ni.eval_mat can handle real vR only
    #        # vj_kpts[i] += ni.eval_mat(cell, ao_ks, 1., None, vR[i,p0:p1], mask, 'LDA')
    #        for k, ao in enumerate(ao_ks):
    #            aow = np.einsum('xi,x->xi', ao, vR[i,p0:p1])
    #            #vj_kpts[i,k] += lib.dot(ao.conj().T, aow)
    #            vj_kpts = ops.index_add(vj_kpts, ops.index[i,k],
    #                                    np.dot(ao.conj().T, aow))

    for i in range(nset):
        for k, ao in enumerate(ao1_kpts):
            aow = np.einsum('xi,x->xi', ao, vR[i])
            vj_kpts = vj_kpts.at[i,k].add(np.dot(ao.conj().T, aow))

    return _format_jks(vj_kpts, dm_kpts, input_band, kpts)

def get_k_kpts(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), kpts_band=None,
               exxdiv=None, cell=None):
    if cell is None:
        cell = mydf.cell
    mesh = mydf.mesh
    coords = cell.gen_uniform_grids(mesh)
    ngrids = coords.shape[0]

    if getattr(dm_kpts, 'mo_coeff', None) is not None:
        mo_coeff = dm_kpts.mo_coeff
        mo_occ   = dm_kpts.mo_occ
    else:
        mo_coeff = None

    kpts = np.asarray(kpts)
    dm_kpts = np.asarray(dm_kpts)
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    weight = 1./nkpts * (cell.vol/ngrids)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)

    if gamma_point(kpts_band) and gamma_point(kpts):
        vk_kpts = np.zeros((nset,nband,nao,nao), dtype=dms.dtype)
    else:
        vk_kpts = np.zeros((nset,nband,nao,nao), dtype=np.complex128)

    coords = mydf.grids.coords
    ao2_kpts = [np.asarray(ao.T)
                for ao in mydf._numint.eval_ao(cell, coords, kpts=kpts)]
    if input_band is None:
        ao1_kpts = ao2_kpts
    else:
        ao1_kpts = [np.asarray(ao.T)
                    for ao in mydf._numint.eval_ao(cell, coords, kpts=kpts_band)]
    if mo_coeff is not None and nset == 1:
        mo_coeff = [mo_coeff[k][:,occ>0] * numpy.sqrt(occ[occ>0])
                    for k, occ in enumerate(mo_occ)]
        ao2_kpts = [np.dot(mo_coeff[k].T, ao) for k, ao in enumerate(ao2_kpts)]

    mem_now = pyscf_lib.current_memory()[0]
    max_memory = mydf.max_memory - mem_now
    blksize = int(min(nao, max(1, (max_memory-mem_now)*1e6/16/4/ngrids/nao)))
    logger.debug1(mydf, 'fft_jk: get_k_kpts max_memory %s  blksize %d',
                  max_memory, blksize)
    #ao1_dtype = np.result_type(*ao1_kpts)
    #ao2_dtype = np.result_type(*ao2_kpts)
    vR_dm = np.empty((nset,nao,ngrids), dtype=vk_kpts.dtype)

    #t1 = (logger.process_clock(), logger.perf_counter())
    for k2, ao2T in enumerate(ao2_kpts):
        if ao2T.size == 0:
            continue

        kpt2 = kpts[k2]
        naoj = ao2T.shape[0]
        if mo_coeff is None or nset > 1:
            ao_dms = [np.dot(dms[i,k2], ao2T.conj()) for i in range(nset)]
        else:
            ao_dms = [ao2T.conj()]

        for k1, ao1T in enumerate(ao1_kpts):
            kpt1 = kpts_band[k1]

            # If we have an ewald exxdiv, we add the G=0 correction near the
            # end of the function to bypass any discretization errors
            # that arise from the FFT.
            if exxdiv == 'ewald' or exxdiv is None:
                coulG = get_coulG(cell, kpt2-kpt1, False, mydf, mesh)
            else:
                coulG = get_coulG(cell, kpt2-kpt1, exxdiv, mydf, mesh)
            if is_zero(kpt1-kpt2):
                expmikr = numpy.array(1.)
            else:
                expmikr = np.exp(-1j * np.dot(coords, kpt2-kpt1))

            for p0, p1 in pyscf_lib.prange(0, nao, blksize):
                rho1 = np.einsum('ig,jg->ijg', ao1T[p0:p1].conj()*expmikr, ao2T)
                vG = tools.fft(rho1.reshape(-1,ngrids), mesh)
                rho1 = None
                vG *= coulG
                vR = tools.ifft(vG, mesh).reshape(p1-p0,naoj,ngrids)
                vG = None
                if vR_dm.dtype == np.double:
                    vR = vR.real
                for i in range(nset):
                    #np.einsum('ijg,jg->ig', vR, ao_dms[i], out=vR_dm[i,p0:p1])
                    vR_dm = ops.index_update(vR_dm, ops.index[i,p0:p1],
                                             np.einsum('ijg,jg->ig', vR, ao_dms[i]))
                vR = None
            vR_dm *= expmikr.conj()

            for i in range(nset):
                #vk_kpts[i,k1] += weight * lib.dot(vR_dm[i], ao1T.T)
                vk_kpts = ops.index_add(vk_kpts, ops.index[i,k1],
                                        weight * np.dot(vR_dm[i], ao1T.T))
        #t1 = logger.timer_debug1(mydf, 'get_k_kpts: make_kpt (%d,*)'%k2, *t1)

    # Function _ewald_exxdiv_for_G0 to add back in the G=0 component to vk_kpts
    # Note in the _ewald_exxdiv_for_G0 implementation, the G=0 treatments are
    # different for 1D/2D and 3D systems.  The special treatments for 1D and 2D
    # can only be used with AFTDF/GDF/MDF method.  In the FFTDF method, 1D, 2D
    # and 3D should use the ewald probe charge correction.
    if exxdiv == 'ewald':
        vk_kpts = _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts_band=kpts_band)

    return _format_jks(vk_kpts, dm_kpts, input_band, kpts)

def get_jk(mydf, dm, hermi=1, kpt=numpy.zeros(3), kpts_band=None,
           with_j=True, with_k=True, exxdiv=None, cell=None):
    dm = np.asarray(dm)
    vj = vk = None
    if with_j:
        vj = get_j(mydf, dm, hermi, kpt, kpts_band, cell=cell)
    if with_k:
        vk = get_k(mydf, dm, hermi, kpt, kpts_band, exxdiv, cell=cell)
    return vj, vk

def get_j(mydf, dm, hermi=1, kpt=numpy.zeros(3), kpts_band=None, cell=None):
    dm = np.asarray(dm)
    nao = dm.shape[-1]
    dm_kpts = dm.reshape(-1,1,nao,nao)
    vj = get_j_kpts(mydf, dm_kpts, hermi, kpt.reshape(1,3), kpts_band, cell=cell)
    if kpts_band is None:
        vj = vj[:,0]
    if dm.ndim == 2:
        vj = vj[0]
    return vj

def get_k(mydf, dm, hermi=1, kpt=numpy.zeros(3), kpts_band=None, exxdiv=None, cell=None):
    dm = np.asarray(dm)
    nao = dm.shape[-1]
    dm_kpts = dm.reshape(-1,1,nao,nao)
    vk = get_k_kpts(mydf, dm_kpts, hermi, kpt.reshape(1,3), kpts_band, exxdiv, cell=cell)
    if kpts_band is None:
        vk = vk[:,0]
    if dm.ndim == 2:
        vk = vk[0]
    return vk
