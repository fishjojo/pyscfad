import numpy
from pyscf import numpy as np
from pyscf.gto.mole import (ATOM_OF, ANG_OF, NPRIM_OF, NCTR_OF,
                            KAPPA_OF, PTR_EXP, PTR_COEFF, PTR_ENV_START)

def uncontract(mol):
    """
    Uncontract basis shell by shell
    (exponents not sorted, basis not normalized)
    """
    mol1 = mol.copy()
    tmp = []
    env = []
    bas = []
    ioff = istart = PTR_ENV_START + mol.natm * 4
    for i in range(len(mol._bas)):
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
    for i, ptr in enumerate(ptr_exp):
        iatm = mol._bas[i,ATOM_OF]
        bas_tmp = bas[numpy.where(bas[:,-1] == ptr)[0]]
        bas_tmp[:,ATOM_OF] = iatm
        _bas.append(bas_tmp)
    _bas = numpy.vstack(tuple(_bas))
    _bas[:,-1] = 0
    mol1._bas = _bas
    return mol1

def setup_exp(mol):
    tmp = []
    es = np.empty([0], dtype=float)
    _env_of = numpy.empty([0], dtype=numpy.int32)
    offset = 0
    es_of = []
    for i in range(len(mol._bas)):
        nprim = mol._bas[i,NPRIM_OF]
        ptr_exp = mol._bas[i,PTR_EXP]
        if ptr_exp not in tmp:
            tmp.append(ptr_exp)
            es = np.append(es, mol._env[ptr_exp : ptr_exp+nprim])
            _env_of = numpy.append(_env_of, numpy.arange(ptr_exp,ptr_exp+nprim))
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
    return es, es_of, _env_of

def setup_ctr_coeff(mol):
    tmp = []
    cs = np.empty([0], dtype=float)
    _env_of = numpy.empty([0], dtype=numpy.int32)
    offset = 0
    cs_of = []
    for i in range(len(mol._bas)):
        nprim = mol._bas[i,NPRIM_OF]
        nctr = mol._bas[i,NCTR_OF]
        ptr_coeff = mol._bas[i,PTR_COEFF]
        if ptr_coeff not in tmp:
            tmp.append(ptr_coeff)
            cs = np.append(cs, mol._env[ptr_coeff : ptr_coeff+nprim*nctr])
            _env_of = numpy.append(_env_of, numpy.arange(ptr_coeff,ptr_coeff+nprim*nctr))
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
    return cs, cs_of, _env_of
