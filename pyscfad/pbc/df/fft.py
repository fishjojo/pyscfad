from typing import Optional, Any
import numpy
from pyscf import __config__
from pyscf import lib as pyscf_lib
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf.pbc.df import fft
from pyscfad import lib
from pyscfad.lib import numpy as jnp
from pyscfad.pbc import tools 
from pyscfad.pbc.gto import Cell

def get_pp(mydf, kpts=None, cell=None):
    '''Get the periodic pseudotential nuc-el AO matrix, with G=0 removed.
    '''
    from pyscf import gto
    from pyscf.pbc.gto import pseudo
    if cell is None:
        cell = mydf.cell
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    mesh = mydf.mesh
    SI = cell.get_SI()
    Gv = cell.get_Gv(mesh)
    vpplocG = pseudo.get_vlocG(cell, Gv)
    vpplocG = -jnp.einsum('ij,ij->j', SI, vpplocG)
    ngrids = len(vpplocG)

    # vpploc evaluated in real-space
    vpplocR = tools.ifft(vpplocG, mesh).real
    vpp = [0] * len(kpts_lst)
    for ao_ks_etc, p0, p1 in mydf.aoR_loop(mydf.grids, kpts_lst, cell=cell):
        ao_ks = ao_ks_etc[0]
        for k, ao in enumerate(ao_ks):
            vpp[k] += jnp.dot(ao.T.conj()*vpplocR[p0:p1], ao)
        ao = ao_ks = None

    # vppnonloc evaluated in reciprocal space
    fakemol = gto.Mole()
    fakemol._atm = numpy.zeros((1,gto.ATM_SLOTS), dtype=numpy.int32)
    fakemol._bas = numpy.zeros((1,gto.BAS_SLOTS), dtype=numpy.int32)
    ptr = gto.PTR_ENV_START
    fakemol._env = numpy.zeros(ptr+10)
    fakemol._bas[0,gto.NPRIM_OF ] = 1
    fakemol._bas[0,gto.NCTR_OF  ] = 1
    fakemol._bas[0,gto.PTR_EXP  ] = ptr+3
    fakemol._bas[0,gto.PTR_COEFF] = ptr+4

    # buf for SPG_lmi upto l=0..3 and nl=3
    buf = numpy.empty((48,ngrids), dtype=numpy.complex128)
    def vppnl_by_k(kpt):
        Gk = Gv + kpt
        G_rad = pyscf_lib.norm(Gk, axis=1)
        #aokG = ft_ao.ft_ao(cell, Gv, kpt=kpt) * (1/cell.vol)**.5
        # use numerical fft for now
        coords = mydf.grids.coords
        aoR = cell.pbc_eval_gto('GTOval', coords, kpt=kpt)
        assert numpy.prod(mesh) == len(coords) == ngrids
        aokG = tools.fftk(aoR.T, mesh, numpy.exp(-1j*numpy.dot(coords, kpt))).T

        vppnl = 0
        for ia in range(cell.natm):
            symb = cell.atom_symbol(ia)
            if symb not in cell._pseudo:
                continue
            pp = cell._pseudo[symb]
            p1 = 0
            for l, proj in enumerate(pp[5:]):
                rl, nl, hl = proj
                if nl > 0:
                    fakemol._bas[0,gto.ANG_OF] = l
                    fakemol._env[ptr+3] = .5*rl**2
                    fakemol._env[ptr+4] = rl**(l+1.5)*numpy.pi**1.25
                    pYlm_part = fakemol.eval_gto('GTOval', Gk)

                    p0, p1 = p1, p1+nl*(l*2+1)
                    # pYlm is real, SI[ia] is complex
                    pYlm = numpy.ndarray((nl,l*2+1,ngrids), dtype=numpy.complex128, buffer=buf[p0:p1])
                    for k in range(nl):
                        qkl = pseudo.pp._qli(G_rad*rl, l, k)
                        pYlm[k] = pYlm_part.T * qkl
                    #:SPG_lmi = numpy.einsum('g,nmg->nmg', SI[ia].conj(), pYlm)
                    #:SPG_lm_aoG = numpy.einsum('nmg,gp->nmp', SPG_lmi, aokG)
                    #:tmp = numpy.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                    #:vppnl += numpy.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
            if p1 > 0:
                SPG_lmi = buf[:p1]
                SPG_lmi *= SI[ia].conj()
                SPG_lm_aoGs = jnp.dot(SPG_lmi, aokG)
                p1 = 0
                for l, proj in enumerate(pp[5:]):
                    rl, nl, hl = proj
                    if nl > 0:
                        p0, p1 = p1, p1+nl*(l*2+1)
                        hl = numpy.asarray(hl)
                        SPG_lm_aoG = SPG_lm_aoGs[p0:p1].reshape(nl,l*2+1,-1)
                        tmp = jnp.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                        vppnl += jnp.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
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
    return jnp.asarray(vpp)


@lib.dataclass
class FFTDF(fft.FFTDF):
    from pyscf.pbc.dft import gen_grid
    from pyscfad.pbc.dft import numint

    cell: Cell = lib.field(pytree_node=True)
    kpts: numpy.ndarray = numpy.zeros((1,3))

    stdout: Any = None
    verbose: Optional[int] = None
    max_memory: Optional[int] = None

    grids: Optional[gen_grid.UniformGrids] = None
    blockdim: int = getattr(__config__, 'pbc_df_df_DF_blockdim', 240)

    exxdiv: Optional[str] = None
    _numint: numint.KNumInt = numint.KNumInt()
    _rsh_df: dict = lib.field(default_factory = dict)

    def __post_init__(self):
        from pyscf.pbc.dft import gen_grid
        if self.stdout is None:
            self.stdout = self.cell.stdout
        if self.verbose is None:
            self.verbose = self.cell.verbose
        if self.max_memory is None:
            self.max_memory = self.cell.max_memory
        if self.grids is None:
            self.grids = gen_grid.UniformGrids(self.cell)
        self._keys = set(self.__dict__.keys())

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        from pyscfad.pbc.df import fft_jk
        if omega is not None:  # J/K for RSH functionals
            raise NotImplementedError
            #return _sub_df_jk_(self, dm, hermi, kpts, kpts_band,
            #                   with_j, with_k, omega, exxdiv)

        if kpts is None:
            if numpy.all(self.kpts == 0): # Gamma-point J/K by default
                kpts = numpy.zeros(3)
            else:
                kpts = self.kpts
        else:
            kpts = numpy.asarray(kpts)

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


    get_pp = get_pp
