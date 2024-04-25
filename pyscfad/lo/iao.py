from pyscf import __config__
from pyscf.lib import logger
from pyscf.gto import format_atom
from pyscf.data.elements import is_ghost_atom
from jax import numpy as np
from jax import scipy
from pyscfad import gto
from .orth import vec_lowdin

MINAO = getattr(__config__, 'lo_iao_minao', 'minao')

def iao(mol, orbocc, minao=MINAO, kpts=None, lindep_threshold=1e-8):
    if mol.has_ecp() and minao == 'minao':
        logger.warn(mol, 'ECP/PP is used. MINAO is not a good reference AO basis in IAO.')

    pmol = reference_mol(mol, minao)
    # For PBC, we must use the pbc code for evaluating the integrals lest the
    # pbc conditions be ignored.
    has_pbc = getattr(mol, 'dimension', 0) > 0
    if has_pbc:
        from pyscfad.pbc import gto as pbcgto
        s1 = mol.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
        s2 = pmol.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
        s12 = pbcgto.cell.intor_cross('int1e_ovlp', mol, pmol, kpts=kpts)
    else:
        #s1 is the one electron overlap integrals (coulomb integrals)
        s1 = mol.intor_symmetric('int1e_ovlp')
        #s2 is the same as s1 except in minao
        s2 = pmol.intor_symmetric('int1e_ovlp')
        #overlap integrals of the two molecules
        s12 = gto.mole.intor_cross('int1e_ovlp', mol, pmol)

    def make_iaos(s1, s2, s12, mo):
        s21 = s12.conj().T
        # s2 is overlap in minimal reference basis and should never be singular:
        s2cd = scipy.linalg.cho_factor(s2)
        ctild = scipy.linalg.cho_solve(s2cd, np.dot(s21, mo))
        try:
            s1cd = scipy.linalg.cho_factor(s1)
            p12 = scipy.linalg.cho_solve(s1cd, s12)
            ctild = scipy.linalg.cho_solve(s1cd, np.dot(s12, ctild))
        # s1 can be singular in large basis sets: Use canonical orthogonalization in this case:
        except Exception: # pylint: disable=broad-exception-caught
            from pyscf.scf import addons
            x = addons.canonical_orth_(s1, lindep_threshold)
            p12 = np.linalg.multi_dot((x, x.conj().T, s12))
            ctild = np.dot(p12, ctild)
        # If there are no occupied orbitals at this k-point, all but the first term will vanish:
        if mo.shape[-1] == 0:
            return p12
        ctild = vec_lowdin(ctild, s1)
        ccs1 = np.linalg.multi_dot((mo, mo.conj().T, s1))
        ccs2 = np.linalg.multi_dot((ctild, ctild.conj().T, s1))
        #a is the set of IAOs in the original basis
        a = (p12 + 2*np.linalg.multi_dot((ccs1, ccs2, p12))
             - np.dot(ccs1, p12) - np.dot(ccs2, p12))
        return a

    # Molecules and Gamma-point only solids
    if s1[0].ndim == 1:
        iaos = make_iaos(s1, s2, s12, orbocc)
    # Solid with multiple k-points
    else:
        iaos = []
        for k in range(len(kpts)):
            iaos.append(make_iaos(s1[k], s2[k], s12[k], orbocc[k]))
        iaos = np.asarray(iaos)
    return iaos

def reference_mol(mol, minao=MINAO):
    '''Create a molecule which uses reference minimal basis'''
    pmol = mol.copy()
    atoms = format_atom(pmol.atom, unit=1)
    # remove ghost atoms
    pmol.atom = [atom for atom in atoms if not is_ghost_atom(atom[0])]
    if len(pmol.atom) != len(atoms):
        logger.info(mol, 'Ghost atoms found in system. '
                    'Current IAO does not support ghost atoms. '
                    'They are removed from IAO reference basis.')
    if getattr(pmol, 'rcut', None) is not None:
        pmol.rcut = None

    if getattr(mol, 'coords', None) is not None:
        assert pmol.coords is mol.coords
    pmol.build(dump_input=False, parse_arg=False, basis=minao,
               trace_exp=False, trace_ctr_coeff=False)
    return pmol

del MINAO
