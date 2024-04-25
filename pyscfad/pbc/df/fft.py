import numpy
from jax import numpy as np
from pyscf import __config__
from pyscf import lib
from pyscf.pbc.gto import Cell
from pyscf.pbc.df import fft as pyscf_fft
#from pyscfad import util
from pyscfad.pbc import tools
from pyscfad.pbc.lib.kpts_helper import gamma_point

def get_pp(mydf, kpts=None):
    from pyscf import gto
    from pyscf.pbc.gto import pseudo
    from pyscfad.gto.mole import Mole
    cell = mydf.cell
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = np.reshape(kpts, (-1,3))

    mesh = mydf.mesh
    SI = cell.get_SI()
    Gv = cell.get_Gv(mesh)
    vpplocG = pseudo.get_vlocG(cell, Gv)
    vpplocG = -np.einsum('ij,ij->j', SI, vpplocG)
    ngrids = len(vpplocG)

    # vpploc evaluated in real-space
    vpplocR = tools.ifft(vpplocG, mesh).real
    vpp = [0] * len(kpts_lst)
    for ao_ks_etc, p0, p1 in mydf.aoR_loop(mydf.grids, kpts_lst):
        ao_ks = ao_ks_etc[0]
        for k, ao in enumerate(ao_ks):
            vpp[k] += np.dot(ao.T.conj()*vpplocR[p0:p1], ao)
        ao = ao_ks = None

    # vppnonloc evaluated in reciprocal space
    fakemol = Mole()
    fakemol._atm = numpy.zeros((1,gto.ATM_SLOTS), dtype=numpy.int32)
    fakemol._bas = numpy.zeros((1,gto.BAS_SLOTS), dtype=numpy.int32)
    ptr = gto.PTR_ENV_START
    fakemol._env = numpy.zeros(ptr+10)
    fakemol._bas[0,gto.NPRIM_OF ] = 1
    fakemol._bas[0,gto.NCTR_OF  ] = 1
    fakemol._bas[0,gto.PTR_EXP  ] = ptr+3
    fakemol._bas[0,gto.PTR_COEFF] = ptr+4

    # buf for SPG_lmi upto l=0..3 and nl=3
    #buf = np.empty((48,ngrids), dtype=np.complex128)
    def vppnl_by_k(kpt):
        Gk = Gv + kpt
        #G_rad = np.linalg.norm(Gk, axis=1)
        #G_rad = np.where(G_rad>1e-16, G_rad, 0.)
        absG2 = np.einsum('gx,gx->g', Gk, Gk)
        G_rad = np.where(absG2>1e-16, np.sqrt(np.where(absG2>1e-16, absG2, 0.)), 0.)
        #aokG = ft_ao.ft_ao(cell, Gv, kpt=kpt) * (1/cell.vol)**.5
        # use numerical fft for now
        coords = mydf.grids.coords
        aoR = cell.pbc_eval_gto('PBCGTOval', coords, kpt=kpt)
        assert numpy.prod(mesh) == len(coords) == ngrids
        aokG = tools.fftk(aoR.T, mesh, np.exp(-1j*np.dot(coords, kpt))).T

        vppnl = 0
        for ia in range(cell.natm):
            symb = cell.atom_symbol(ia)
            if symb not in cell._pseudo:
                continue
            pp = cell._pseudo[symb]
            p1 = 0
            buf = []
            for l, proj in enumerate(pp[5:]):
                rl, nl, hl = proj
                if nl > 0:
                    fakemol._bas[0,gto.ANG_OF] = l
                    fakemol._env[ptr+3] = .5*rl**2
                    fakemol._env[ptr+4] = rl**(l+1.5)*numpy.pi**1.25
                    pYlm_part = fakemol.eval_gto('GTOval', Gk)

                    p0, p1 = p1, p1+nl*(l*2+1)
                    # pYlm is real, SI[ia] is complex
                    #pYlm = np.array((nl,l*2+1,ngrids), dtype=numpy.complex128, buffer=buf[p0:p1])
                    pYlm = []
                    for k in range(nl):
                        qkl = pseudo.pp._qli(G_rad*rl, l, k)
                        pYlm.append(pYlm_part.T * qkl)
                    buf.append(np.asarray(pYlm).reshape(-1,ngrids))

                    #:SPG_lmi = numpy.einsum('g,nmg->nmg', SI[ia].conj(), pYlm)
                    #:SPG_lm_aoG = numpy.einsum('nmg,gp->nmp', SPG_lmi, aokG)
                    #:tmp = numpy.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                    #:vppnl += numpy.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)

            buf = np.vstack(buf)
            if p1 > 0:
                #SPG_lmi = buf #buf[:p1]
                SPG_lmi = buf * SI[ia].conj()
                SPG_lm_aoGs = np.dot(SPG_lmi, aokG)
                p1 = 0
                for l, proj in enumerate(pp[5:]):
                    rl, nl, hl = proj
                    if nl > 0:
                        p0, p1 = p1, p1+nl*(l*2+1)
                        hl = numpy.asarray(hl)
                        SPG_lm_aoG = SPG_lm_aoGs[p0:p1].reshape(nl,l*2+1,-1)
                        tmp = np.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                        vppnl += np.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
        #return vppnl * (1./cell.vol)
        return vppnl * (1./ngrids**2)

    for k, kpt in enumerate(kpts_lst):
        vppnl = vppnl_by_k(kpt)
        if gamma_point(kpt):
            vpp[k] = vpp[k].real + vppnl.real
        else:
            vpp[k] += vppnl

    if kpts is None or numpy.shape(kpts) == (3,):
        vpp = vpp[0]
    return np.asarray(vpp)


#FIXME converting the class to Jax traceable type
# sometimes lose tracing of its attributes
#@util.pytree_node(['cell','kpts'])
class FFTDF(pyscf_fft.FFTDF):
    def __init__(self, cell, kpts=numpy.zeros((1,3))):#, **kwargs):
        from pyscf.pbc.dft import gen_grid
        from pyscfad.pbc.dft import numint
        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.max_memory = cell.max_memory

        self.kpts = kpts
        self.grids = gen_grid.UniformGrids(cell)

        self.blockdim = getattr(__config__, 'pbc_df_df_DF_blockdim', 240)

        self.exxdiv = None
        self._numint = numint.KNumInt()
        self._rsh_df = {}  # Range separated Coulomb DF objects
        self._keys = set(self.__dict__.keys())
        #self.__dict__.update(kwargs)

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        from pyscfad.pbc.df import fft_jk
        if omega is not None:  # J/K for RSH functionals
            raise NotImplementedError

        if kpts is None:
            if numpy.all(self.kpts == 0): # Gamma-point J/K by default
                kpts = numpy.zeros(3)
            else:
                kpts = self.kpts
        else:
            kpts = np.asarray(kpts)

        vj = vk = None
        if kpts.shape == (3,):
            vj, vk = fft_jk.get_jk(self, dm, hermi, kpts, kpts_band,
                                   with_j, with_k, exxdiv)
        else:
            if with_k:
                vk = fft_jk.get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv)
            if with_j:
                vj = fft_jk.get_j_kpts(self, dm, hermi, kpts, kpts_band)
        return vj, vk

    def aoR_loop(self, grids=None, kpts=None, deriv=0):
        if grids is None:
            grids = self.grids
            cell = self.cell
        else:
            cell = grids.cell

        # NOTE stop tracing cell through grids
        grids.cell = cell.view(Cell)
        if grids.non0tab is None:
            grids.build(with_non0tab=True)

        if kpts is None:
            kpts = self.kpts
        kpts = np.asarray(kpts)

        if (cell.dimension < 2 or
            (cell.dimension == 2 and cell.low_dim_ft_type == 'inf_vacuum')):
            raise RuntimeError('FFTDF method does not support low-dimension '
                               'PBC system.  DF, MDF or AFTDF methods should '
                               'be used.\nSee also examples/pbc/31-low_dimensional_pbc.py')

        max_memory = max(2000, self.max_memory-lib.current_memory()[0])
        ni = self._numint
        nao = cell.nao_nr()
        p1 = 0
        for ao_k1_etc in ni.block_loop(cell, grids, nao, deriv, kpts,
                                       max_memory=max_memory):
            coords = ao_k1_etc[4]
            p0, p1 = p1, p1 + coords.shape[0]
            yield ao_k1_etc, p0, p1

    get_pp = get_pp
