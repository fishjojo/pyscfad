from functools import partial
import ctypes
import numpy
from jax import custom_vjp
from jax.tree_util import tree_flatten, tree_unflatten

from pyscf import lib
from pyscf.lib import logger
from pyscf.gto import mole as pyscf_mole
from pyscfad.gto._pyscf_moleintor import (
    make_loc,
    make_cintopt,
    _stand_sym_code,
    _get_intor_and_comp,
)

from pyscfad.gto._mole_helper import (
    get_fakemol_exp,
    get_fakemol_cs,
    setup_exp,
    setup_ctr_coeff,
    shlmap_ctr2unctr,
)
from pyscfad.gto._moleintor_helper import (
    int1e_dr1_name,
    int2e_dr1_name,
)
from pyscfad.gto.moleintor import _intor
from pyscfadlib import libcgto_vjp as libcgto

def getints(mol, intor, shls_slice=None,
            comp=None, hermi=0, aosym='s1',
            out=None, grids=None):
    if intor.endswith('_spinor'):
        raise NotImplementedError('Integrals for spinors are not supported.')
    if grids is not None:
        raise NotImplementedError('Integrals on grids are not supported.')
    if out is not None:
        logger.warn(mol, f'Argument out = {out} will be ignored.')
    if hermi == 2:
        hermi = 0
        msg = f'Anti-hermitian symmetry is not supported. Setting hermi = {hermi}.'
        logger.warn(mol, msg)

    if (intor.startswith('int1e') or
        intor.startswith('int2c2e') or
        intor.startswith('ECP')):
        return getints2c(mol, intor, shls_slice, comp, hermi, out=None)
    elif intor.startswith('int2e'):
        return getints4c(mol, intor, shls_slice, comp, aosym, out=None)
    else:
        raise NotImplementedError(f'Integral {intor} is not supported.')

@partial(custom_vjp, nondiff_argnums=(1,2,3,4,5))
def getints2c(mol, intor, shls_slice=None, comp=None, hermi=0, out=None):
    return _intor(mol, intor, comp=comp, hermi=hermi,
                  shls_slice=shls_slice, out=out)

def getints2c_fwd(mol, intor, shls_slice, comp, hermi, out):
    y = getints2c(mol, intor, shls_slice, comp, hermi, out)
    return y, (mol,)

def getints2c_bwd(intor, shls_slice, comp, hermi, out,
                  res, ybar):
    mol = res[0]
    leaves = []

    if mol.coords is not None:
        vjp_coords = getints2c_coords_bwd(intor, shls_slice, comp, hermi, out,
                                          mol, ybar)
        leaves.append(vjp_coords)

    if mol.exp is not None:
        vjp_exp = getints2c_exp_bwd(intor, shls_slice, comp, hermi, out,
                                    mol, ybar)
        leaves.append(vjp_exp)

    if mol.ctr_coeff is not None:
        vjp_coeff = getints2c_coeff_bwd(intor, shls_slice, comp, hermi, out,
                                        mol, ybar)
        leaves.append(vjp_coeff)

    if mol.r0 is not None:
        raise NotImplementedError

    _, tree = tree_flatten(mol)
    molbar = tree_unflatten(tree, leaves)
    return (molbar,)

getints2c.defvjp(getints2c_fwd, getints2c_bwd)

@partial(custom_vjp, nondiff_argnums=(1,2,3,4,5))
def getints4c(mol, intor, shls_slice=None, comp=None, aosym='s1', out=None):
    return _intor(mol, intor, comp=comp, aosym=aosym,
                  shls_slice=shls_slice, out=out)

def getints4c_fwd(mol, intor, shls_slice, comp, aosym, out):
    y = getints4c(mol, intor, shls_slice, comp, aosym, out)
    return y, (mol,)

def getints4c_bwd(intor, shls_slice, comp, aosym, out,
                  res, ybar):
    mol = res[0]
    leaves = []

    if mol.coords is not None:
        vjp_coords = getints4c_coords_bwd(intor, shls_slice, comp, aosym, out,
                                          mol, ybar)
        leaves.append(vjp_coords)

    if mol.exp is not None:
        raise NotImplementedError
        #vjp_exp = getints4c_exp_bwd(intor, shls_slice, comp, aosym, out,
        #                            mol, ybar)

    if mol.ctr_coeff is not None:
        raise NotImplementedError
        #vjp_coeff = getints4c_coeff_bwd(intor, shls_slice, comp, aosym, out,
        #                                mol, ybar)

    _, tree = tree_flatten(mol)
    molbar = tree_unflatten(tree, leaves)
    return (molbar,)

getints4c.defvjp(getints4c_fwd, getints4c_bwd)


def _int1e_r0_deriv(intor_ip_bra, shls_slice, comp, hermi,
                    mol, ybar, rc_deriv=None, switch_ij=False):
    nbas = mol.nbas
    if shls_slice is None:
        shls_slice = (0, nbas, 0, nbas)
    i0, i1, j0, j1 = shls_slice[:4]
    assert i0 >= 0 and i1 <= nbas
    assert j0 >= 0 and j1 <= nbas
    assert i0 < i1 and j0 < j1

    ao_loc = make_loc(mol._bas, intor_ip_bra)
    ao_loc = numpy.asarray(ao_loc, order='C', dtype=numpy.int32)
    naoi = ao_loc[i1] - ao_loc[i0]
    naoj = ao_loc[j1] - ao_loc[j0]

    atm = numpy.asarray(mol._atm, order='C', dtype=numpy.int32)
    bas = numpy.asarray(mol._bas, order='C', dtype=numpy.int32)
    env = numpy.asarray(mol._env, order='C', dtype=numpy.double)

    cintopt = make_cintopt(atm, bas, env, intor_ip_bra)

    ybar = numpy.asarray(ybar).reshape(comp, naoi, naoj)
    if hermi == 1:
        assert i0 == j0 and i1 == j1
        ybar = ybar + ybar.transpose(0,2,1)
    elif switch_ij:
        ybar = ybar.transpose(0,2,1)
        shls_slice = (j0, j1, i0, i1)
    ybar = numpy.asarray(ybar, order='C', dtype=numpy.double)

    ndim = 3
    natm = len(atm)
    if rc_deriv is not None:
        vjp = numpy.zeros((ndim,), order='C', dtype=numpy.double)
        fn = getattr(libcgto, 'GTOint2c_rc_vjp')
    else:
        vjp = numpy.zeros((natm,ndim), order='C', dtype=numpy.double)
        fn = getattr(libcgto, 'GTOint2c_r0_vjp')

    fn(getattr(libcgto, intor_ip_bra),
       vjp.ctypes.data_as(ctypes.c_void_p),
       ybar.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(comp), ctypes.c_int(ndim), ctypes.c_int(hermi),
       (ctypes.c_int*4)(*(shls_slice[:4])),
       ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,
       atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(natm),
       bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nbas),
       env.ctypes.data_as(ctypes.c_void_p))
    return vjp

def _int1e_rc_deriv(intor_ip_bra, shls_slice, comp, hermi,
                    mol, ybar):
    intor_ip_bra = intor_ip_bra.replace('nuc', 'rinv')

    natm = mol.natm
    vjp = numpy.zeros((natm,3), dtype=numpy.double)
    for iatm in range(natm):
        with mol.with_rinv_at_nucleus(iatm):
            charge = -mol.atom_charge(iatm)
            vjp[iatm] += _int1e_r0_deriv(intor_ip_bra, shls_slice, comp, hermi,
                                         mol, ybar, rc_deriv=iatm) * charge
    return vjp

def getints2c_coords_bwd(intor, shls_slice, comp, hermi, out,
                         mol, ybar):
    log = logger.new_logger(mol)
    _, comp = _get_intor_and_comp(intor, comp)

    switch_ij = False
    intor_ip_bra, intor_ip_ket = int1e_dr1_name(intor)
    if not intor_ip_bra:
        #switch i and j shells due to derivative over ket
        switch_ij = True
        intor_ip_bra = intor_ip_ket

    vjp = _int1e_r0_deriv(intor_ip_bra, shls_slice, comp, hermi,
                          mol, ybar, switch_ij=switch_ij)
    if 'nuc' in intor_ip_bra:
        vjp += _int1e_rc_deriv(intor_ip_bra, shls_slice, comp, hermi,
                               mol, ybar)
    log.timer(f'getints2c_coords_bwd {intor}')
    del log
    return vjp

def getints2c_exp_bwd(intor, shls_slice, comp, hermi, out,
                      mol, ybar):
    log = logger.new_logger(mol)
    nbas = mol.nbas
    if shls_slice is None:
        shls_slice = (0, nbas, 0, nbas)
    i0, i1, j0, j1 = shls_slice[:4]
    assert i0 >= 0 and i1 <= nbas
    assert j0 >= 0 and j1 <= nbas
    assert i0 < i1 and j0 < j1

    if comp is None:
        comp = 1
    elif comp != 1:
        raise NotImplementedError

    order = 2 # first order derivative of Gaussians

    shlmap_c2u = shlmap_ctr2unctr(mol)
    shlmap_c2u = numpy.asarray(shlmap_c2u, order='C', dtype=numpy.int32)
    mol1 = get_fakemol_exp(mol, order)
    mol1._atm[:,pyscf_mole.CHARGE_OF] = 0 # set nuclear charge to zero

    ao_loc = make_loc(mol._bas, intor)
    ao_loc = numpy.asarray(ao_loc, order='C', dtype=numpy.int32)

    if intor.endswith('_sph'):
        cart = False
        intor = intor.replace('_sph', '_cart')
        ao_loc_cart = make_loc(mol._bas, intor)
        ao_loc_cart = numpy.asarray(ao_loc_cart, order='C', dtype=numpy.int32)
    elif intor.endswith('_cart'):
        cart = True
        ao_loc_cart = ao_loc
    else:
        raise NotImplementedError

    nbas1 = len(mol1._bas)
    shls_slice = shls_slice + (nbas, nbas+nbas1)
    if 'ECP' in intor:
        assert mol._ecp is not None
        bas1 = numpy.vstack((mol1._bas, mol._ecpbas))
    else:
        bas1 = mol1._bas
    atmc, basc, envc = pyscf_mole.conc_env(mol._atm, mol._bas, mol._env,
                                     mol1._atm, bas1, mol1._env)
    if 'ECP' in intor:
        envc[pyscf_mole.AS_ECPBAS_OFFSET] = nbas + nbas1
        envc[pyscf_mole.AS_NECPBAS] = len(mol._ecpbas)

    atmc = numpy.asarray(atmc, order='C', dtype=numpy.int32)
    basc = numpy.asarray(basc, order='C', dtype=numpy.int32)
    envc = numpy.asarray(envc, order='C', dtype=numpy.double)

    _, es_of, _ = setup_exp(mol)
    es_of = numpy.asarray(es_of, order='C', dtype=numpy.int32)

    nes = len(mol.exp)
    vjp = numpy.zeros((nes,), order='C', dtype=numpy.double)

    cintopt = make_cintopt(atmc, basc, envc, intor)

    if hermi == 1:
        ybar = ybar + ybar.T
    ybar = numpy.asarray(ybar, order='C', dtype=numpy.double)

    fn = getattr(libcgto, 'GTOint2c_exp_vjp')
    fn(getattr(libcgto, intor),
       vjp.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nes),
       ybar.ctypes.data_as(ctypes.c_void_p),
       shlmap_c2u.ctypes.data_as(ctypes.c_void_p),
       es_of.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(comp), ctypes.c_int(hermi),
       (ctypes.c_int*6)(*(shls_slice[:6])),
       ao_loc.ctypes.data_as(ctypes.c_void_p),
       ao_loc_cart.ctypes.data_as(ctypes.c_void_p), cintopt,
       atmc.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(atmc)),
       basc.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(basc)),
       envc.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(cart), ctypes.c_int(order))
    log.timer('getints2c_exp_bwd')
    del log
    return vjp

def getints2c_coeff_bwd(intor, shls_slice, comp, hermi, out,
                        mol, ybar):
    log = logger.new_logger(mol)
    nbas = mol.nbas
    if shls_slice is None:
        shls_slice = (0, nbas, 0, nbas)
    i0, i1, j0, j1 = shls_slice[:4]
    assert i0 >= 0 and i1 <= nbas
    assert j0 >= 0 and j1 <= nbas
    assert i0 < i1 and j0 < j1

    if comp is None:
        comp = 1
    elif comp != 1:
        raise NotImplementedError

    shlmap_c2u = shlmap_ctr2unctr(mol)
    shlmap_c2u = numpy.asarray(shlmap_c2u, order='C', dtype=numpy.int32)
    mol1 = get_fakemol_cs(mol)
    mol1._atm[:,pyscf_mole.CHARGE_OF] = 0 # set nuclear charge to zero

    ao_loc = make_loc(mol._bas, intor)
    ao_loc = numpy.asarray(ao_loc, order='C', dtype=numpy.int32)

    if intor.endswith('_sph'):
        cart = False
    elif intor.endswith('_cart'):
        cart = True
    else:
        raise NotImplementedError

    nbas1 = len(mol1._bas)
    shls_slice = shls_slice + (nbas, nbas+nbas1)
    if 'ECP' in intor:
        assert mol._ecp is not None
        bas1 = numpy.vstack((mol1._bas, mol._ecpbas))
    else:
        bas1 = mol1._bas
    atmc, basc, envc = pyscf_mole.conc_env(mol._atm, mol._bas, mol._env,
                                     mol1._atm, bas1, mol1._env)
    if 'ECP' in intor:
        envc[pyscf_mole.AS_ECPBAS_OFFSET] = nbas + nbas1
        envc[pyscf_mole.AS_NECPBAS] = len(mol._ecpbas)

    atmc = numpy.asarray(atmc, order='C', dtype=numpy.int32)
    basc = numpy.asarray(basc, order='C', dtype=numpy.int32)
    envc = numpy.asarray(envc, order='C', dtype=numpy.double)

    _, cs_of, _ = setup_ctr_coeff(mol)
    cs_of = numpy.asarray(cs_of, order='C', dtype=numpy.int32)

    ncs = len(mol.ctr_coeff)
    vjp = numpy.zeros((ncs,), order='C', dtype=numpy.double)

    cintopt = make_cintopt(atmc, basc, envc, intor)

    if hermi == 1:
        ybar = ybar + ybar.T
    ybar = numpy.asarray(ybar, order='C', dtype=numpy.double)

    fn = getattr(libcgto, 'GTOint2c_coeff_vjp')
    fn(getattr(libcgto, intor),
       vjp.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(ncs),
       ybar.ctypes.data_as(ctypes.c_void_p),
       shlmap_c2u.ctypes.data_as(ctypes.c_void_p),
       cs_of.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(comp), ctypes.c_int(hermi),
       (ctypes.c_int*6)(*(shls_slice[:6])),
       ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,
       atmc.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(atmc)),
       basc.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(basc)),
       envc.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(cart))
    log.timer('getints2c_coeff_bwd')
    del log
    return vjp


def getints4c_coords_bwd(intor, shls_slice, comp, aosym, out,
                         mol, ybar):
    log = logger.new_logger(mol)
    aosym = _stand_sym_code(aosym)
    if aosym != 's4':
        raise NotImplementedError

    atm = numpy.asarray(mol._atm, dtype=numpy.int32, order='C')
    bas = numpy.asarray(mol._bas, dtype=numpy.int32, order='C')
    env = numpy.asarray(mol._env, dtype=numpy.double, order='C')
    c_atm = atm.ctypes.data_as(ctypes.c_void_p)
    c_bas = bas.ctypes.data_as(ctypes.c_void_p)
    c_env = env.ctypes.data_as(ctypes.c_void_p)
    natm = atm.shape[0]
    nbas = bas.shape[0]

    if shls_slice is None:
        shls_slice = (0, nbas, 0, nbas, 0, nbas, 0, nbas)
    elif len(shls_slice) == 4:
        shls_slice = shls_slice + (0, nbas, 0, nbas)

    i0, i1, j0, j1, k0, k1, l0, l1 = shls_slice
    assert i0 >= 0 and i1 <= nbas
    assert j0 >= 0 and j1 <= nbas
    assert k0 >= 0 and k1 <= nbas
    assert l0 >= 0 and l1 <= nbas
    assert i0 < i1 and j0 < j1 and k0 < k1 and l0 < l1

    if comp is None:
        comp = 1
    elif comp != 1:
        raise NotImplementedError
    comp = 3 # first order

    intor1 = int2e_dr1_name(intor)[0]
    ao_loc = make_loc(bas, intor1)
    ao_loc = numpy.asarray(ao_loc, order='C', dtype=numpy.int32)

    naoi = ao_loc[i1] - ao_loc[i0]
    naoj = ao_loc[j1] - ao_loc[j0]
    naok = ao_loc[k1] - ao_loc[k0]
    naol = ao_loc[l1] - ao_loc[l0]

    if aosym in ('s4', 's2ij'):
        assert numpy.all(ao_loc[i0:i1]-ao_loc[i0] == ao_loc[j0:j1]-ao_loc[j0])
    if aosym in ('s4', 's2kl'):
        assert numpy.all(ao_loc[k0:k1]-ao_loc[k0] == ao_loc[l0:l1]-ao_loc[l0])

    drv = libcgto.GTOnr2e_fill_r0_vjp
    fill = getattr(libcgto, 'GTOnr2e_fill_r0_vjp_'+aosym)
    vjp = numpy.zeros((natm, comp), order='C', dtype=numpy.double)
    if aosym == 's4':
        ybar += ybar.T
    ybar = numpy.asarray(ybar, order='C', dtype=numpy.double)

    cintopt = make_cintopt(atm, bas, env, intor1)
    prescreen = lib.c_null_ptr()
    drv(getattr(libcgto, intor1), fill, prescreen,
        vjp.ctypes.data_as(ctypes.c_void_p),
        ybar.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(comp), (ctypes.c_int*8)(*shls_slice),
        ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt,
        c_atm, ctypes.c_int(natm), c_bas, ctypes.c_int(nbas), c_env)

    log.timer('getints4c_coords_bwd')
    del log
    return vjp
