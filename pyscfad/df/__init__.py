from . import df
from . import addons
from .df import DF

def density_fit(mf, auxbasis=None, with_df=None, only_dfj=False):
    # pylint: disable=import-outside-toplevel
    from pyscfad import util
    from pyscf.df import df_jk as pyscf_df_jk
    if with_df is None:
        #auxmol has to be defined before creating the DF object,
        #otherwise, its derivative won't be traced
        auxmol = addons.make_auxmol(mf.mol, auxbasis)
        with_df = DF(mf.mol, auxmol=auxmol)
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose
        with_df.auxbasis = auxbasis

    mf_class = mf.__class__
    kwargs = mf.__dict__.copy()
    mol = kwargs.pop("mol")

    # pylint: disable=abstract-method
    @util.pytree_node(['mol', 'with_df'], num_args=1)
    class DFHF(pyscf_df_jk._DFHF, mf_class):
        def __init__(self, mol, with_df=None, only_dfj=False, **kwargs):
            self.with_df = with_df
            if getattr(self.with_df, "mol", None) is not None:
                self.mol = self.with_df.mol
            else:
                self.mol = mol
            self.only_dfj = only_dfj
            self.__dict__.update(kwargs)

            self._eri = None
            self._keys = self._keys.union(['with_df', 'only_dfj'])

        def reset(self, mol=None):
            self.with_df.reset(mol)
            return mf_class.reset(self, mol)

        def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
                   omega=None):
            if dm is None:
                dm = self.make_rdm1()
            if self.with_df and self.only_dfj:
                vj = vk = None
                if with_j:
                    vj, vk = self.with_df.get_jk(dm, hermi, True, False,
                                                 self.direct_scf_tol, omega)
                if with_k:
                    vk = mf_class.get_jk(self, mol, dm, hermi, False, True, omega)[1]
            elif self.with_df:
                vj, vk = self.with_df.get_jk(dm, hermi, with_j, with_k,
                                             self.direct_scf_tol, omega)
            else:
                vj, vk = mf_class.get_jk(self, mol, dm, hermi, with_j, with_k, omega)
            return vj, vk

    return DFHF(mol, with_df, only_dfj, **kwargs)
