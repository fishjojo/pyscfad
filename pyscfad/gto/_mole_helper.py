import numpy
from pyscf.gto.mole import (ATOM_OF, ANG_OF, NPRIM_OF, NCTR_OF,
                            KAPPA_OF, PTR_EXP, PTR_COEFF, PTR_ENV_START)

def uncontract(mol, shls_slice=None):
    """Uncontract basis functions.

    Parameters
    ----------
    mol : :class:`Mole` instance
        :class:`Mole` instance with the contracted basis functions.

    shls_slice : tuple
        Starting and ending indices of the shells being
        uncontracted. Default is `None`, meaning all
        shells are considered.

    Returns
    -------
    mol1 : :class:`Mole` instance
        :class:`Mole` instance with the uncontracted basis functions.

    Notes
    -----
    The uncontracted basis functions are neither sorted nor normalized.
    """
    if shls_slice is None:
        shls_slice = (0, mol.nbas)
    shl0, shl1 = shls_slice

    mol1 = mol.copy()
    tmp = []
    env = []
    bas = []
    ioff = istart = PTR_ENV_START + mol.natm * 4
    for i in range(shl0, shl1):
        iatm = mol._bas[i,ATOM_OF]
        l = mol._bas[i,ANG_OF]
        nprim = mol._bas[i,NPRIM_OF]
        kappa = mol._bas[i,KAPPA_OF]
        ptr_exp = mol._bas[i,PTR_EXP]
        if ptr_exp not in tmp:
            tmp.append(ptr_exp)
            for j in range(nprim):
                env.append([mol._env[ptr_exp+j], 1.])
                bas.append([iatm, l, 1, 1, kappa, ioff, ioff+1, ptr_exp])
                ioff += 2

    env = numpy.asarray(env).flatten()
    mol1._env = numpy.hstack((mol._env[:istart], env))

    bas = numpy.asarray(bas)
    ptr_exp = mol._bas[:,PTR_EXP]
    _bas = []
    for i in range(shl0, shl1):
        iatm = mol._bas[i,ATOM_OF]
        bas_tmp = bas[numpy.where(bas[:,-1] == ptr_exp[i])[0]]
        bas_tmp[:,ATOM_OF] = iatm
        _bas.append(bas_tmp)
    _bas = numpy.vstack(tuple(_bas))
    _bas[:,-1] = 0
    mol1._bas = _bas

    # stop tracing
    mol1.ctr_coeff = None
    mol1.exp = None
    return mol1

def shlmap_ctr2unctr(mol):
    """Mapping between contracted basis shells and
    uncontracted basis shells

    Parameters
    ----------
    mol : :class:`Mole` instance

    Returns
    -------
    map_c2u : array
        map_c2u[shell_id_contracted] = shell_id_uncontracted
    """
    nsh = 0
    map_c2u = []
    for i in range(mol.nbas):
        nprim = mol._bas[i,NPRIM_OF]
        map_c2u.append(nsh)
        nsh += nprim
    map_c2u.append(nsh)
    map_c2u = numpy.asarray(map_c2u)
    return map_c2u

def setup_exp(mol):
    """Find unique exponents of the basis functions.

    Parameters
    ----------
    mol : :class:`Mole` instance

    Returns
    -------
    es : array
        Unique exponents of the basis functions in `mol`.
        The exponents are stored shell by shell with the same
        sequence as the primitive Gaussians in `mol`.
    es_of : array
        The indices in `es` for the exponent of
        the first primitive Gaussian in each shell.
    env_of : array
        The indices in `mol._env` for each unique exponent in `es`.

    See also
    --------
    setup_ctr_coeff : Find unique contraction coefficients.
    """
    tmp = []
    es = numpy.empty([0], dtype=float)
    env_of = numpy.empty([0], dtype=numpy.int32)
    offset = 0
    es_of = []
    for i in range(mol.nbas):
        nprim = mol._bas[i,NPRIM_OF]
        ptr_exp = mol._bas[i,PTR_EXP]
        if ptr_exp not in tmp:
            tmp.append(ptr_exp)
            es = numpy.append(es, mol._env[ptr_exp : ptr_exp+nprim])
            env_of = numpy.append(env_of, numpy.arange(ptr_exp, ptr_exp+nprim))
            es_of.append(offset)
            offset += nprim
    tmp = numpy.asarray(tmp, dtype=numpy.int32)
    es_of = numpy.asarray(es_of, dtype=numpy.int32)
    ptr_exp = mol._bas[:,PTR_EXP]
    idx = []
    for ptr in ptr_exp:
        idx.append(numpy.where(ptr == tmp)[0])
    idx = numpy.asarray(idx).flatten()
    es_of = es_of[idx]
    return es, es_of, env_of

def setup_ctr_coeff(mol):
    """Find unique contraction coefficients of the basis functions.

    Parameters
    ----------
    mol : :class:`Mole` instance

    Returns
    -------
    cs : array
    cs_of : array
    env_of : array

    See also
    --------
    set_exp : Find unique exponents.
    """
    tmp = []
    cs = numpy.empty([0], dtype=float)
    env_of = numpy.empty([0], dtype=numpy.int32)
    offset = 0
    cs_of = []
    for i in range(mol.nbas):
        nprim = mol._bas[i,NPRIM_OF]
        nctr = mol._bas[i,NCTR_OF]
        ptr_coeff = mol._bas[i,PTR_COEFF]
        if ptr_coeff not in tmp:
            tmp.append(ptr_coeff)
            cs = numpy.append(cs, mol._env[ptr_coeff : ptr_coeff+nprim*nctr])
            env_of = numpy.append(env_of,
                                  numpy.arange(ptr_coeff,ptr_coeff+nprim*nctr))
            cs_of.append(offset)
            offset += nprim*nctr
    tmp = numpy.asarray(tmp, dtype=numpy.int32)
    cs_of = numpy.asarray(cs_of, dtype=numpy.int32)
    ptr_coeff = mol._bas[:,PTR_COEFF]
    idx = []
    for ptr in ptr_coeff:
        idx.append(numpy.where(ptr == tmp)[0])
    idx = numpy.asarray(idx).flatten()
    cs_of = cs_of[idx]
    return cs, cs_of, env_of

def get_fakemol_exp(mol, order=2, shls_slice=None):
    mol1 = uncontract(mol, shls_slice=shls_slice)
    mol1._bas[:,ANG_OF] += order
    return mol1

def get_fakemol_cs(mol, shls_slice=None):
    mol1 = uncontract(mol, shls_slice=shls_slice)
    return mol1
