#pylint: disable=unused-import
from functools import wraps
import warnings
import ctypes
import numpy
from pyscf import lib
from pyscf.gto import moleintor as molint

from pyscfadlib import libcgto_vjp
from pyscfad.experimental.util import replace_source_code

ANG_OF     = molint.ANG_OF
NPRIM_OF   = molint.NPRIM_OF
NCTR_OF    = molint.NCTR_OF
KAPPA_OF   = molint.KAPPA_OF
PTR_EXP    = molint.PTR_EXP
PTR_COEFF  = molint.PTR_COEFF
BAS_SLOTS  = molint.BAS_SLOTS
NGRIDS     = molint.NGRIDS
PTR_GRIDS  = molint.PTR_GRIDS


_cintoptHandler = molint._cintoptHandler
make_loc = molint.make_loc
_stand_sym_code = molint._stand_sym_code
ascint3 = molint.ascint3

_INTOR_FUNCTIONS = molint._INTOR_FUNCTIONS
_INTOR_FUNCTIONS.update({
    'int1e_ovlp_dr10'		: (3, 3),
    'int1e_ovlp_dr01'		: (3, 3),
    'int1e_kin_dr10'		: (3, 3),
    'int1e_kin_dr01'		: (3, 3),
    'int1e_nuc_dr10'		: (3, 3),
    'int1e_nuc_dr01'		: (3, 3),
    'int1e_rinv_dr10'		: (3, 3),
    'int1e_rinv_dr01'		: (3, 3),
    'int2c2e_dr10'		: (3, 3),
    'int2c2e_dr01'		: (3, 3),
    'int1e_r2_dr10'		: (3, 3),
    'int1e_r2_dr01'		: (3, 3),
    'int2e_dr1000'		: (3, 3),
    'int2e_dr0010'		: (3, 3),
    'int1e_ovlp_dr20'		: (9, 9),
    'int1e_ovlp_dr11'		: (9, 9),
    'int1e_ovlp_dr02'		: (9, 9),
    'int1e_kin_dr20'		: (9, 9),
    'int1e_kin_dr11'		: (9, 9),
    'int1e_kin_dr02'		: (9, 9),
    'int1e_nuc_dr20'		: (9, 9),
    'int1e_nuc_dr11'		: (9, 9),
    'int1e_nuc_dr02'		: (9, 9),
    'int1e_rinv_dr20'		: (9, 9),
    'int1e_rinv_dr11'		: (9, 9),
    'int1e_rinv_dr02'		: (9, 9),
    'int2c2e_dr20'		: (9, 9),
    'int2c2e_dr11'		: (9, 9),
    'int2c2e_dr02'		: (9, 9),
    'int1e_r2_dr20'		: (9, 9),
    'int1e_r2_dr11'		: (9, 9),
    'int1e_r2_dr02'		: (9, 9),
    'int2e_dr2000'		: (9, 9),
    'int2e_dr1100'		: (9, 9),
    'int2e_dr1010'		: (9, 9),
    'int2e_dr0020'		: (9, 9),
    'int2e_dr0011'		: (9, 9),
    'int1e_ovlp_dr30'		: (27, 27),
    'int1e_ovlp_dr21'		: (27, 27),
    'int1e_ovlp_dr12'		: (27, 27),
    'int1e_ovlp_dr03'		: (27, 27),
    'int1e_kin_dr30'		: (27, 27),
    'int1e_kin_dr21'		: (27, 27),
    'int1e_kin_dr12'		: (27, 27),
    'int1e_kin_dr03'		: (27, 27),
    'int1e_nuc_dr30'		: (27, 27),
    'int1e_nuc_dr21'		: (27, 27),
    'int1e_nuc_dr12'		: (27, 27),
    'int1e_nuc_dr03'		: (27, 27),
    'int1e_rinv_dr30'		: (27, 27),
    'int1e_rinv_dr21'		: (27, 27),
    'int1e_rinv_dr12'		: (27, 27),
    'int1e_rinv_dr03'		: (27, 27),
    'int2c2e_dr30'		: (27, 27),
    'int2c2e_dr21'		: (27, 27),
    'int2c2e_dr12'		: (27, 27),
    'int2c2e_dr03'		: (27, 27),
    'int1e_r2_dr30'		: (27, 27),
    'int1e_r2_dr21'		: (27, 27),
    'int1e_r2_dr12'		: (27, 27),
    'int1e_r2_dr03'		: (27, 27),
    'int2e_dr3000'		: (27, 27),
    'int2e_dr2100'		: (27, 27),
    'int2e_dr2010'		: (27, 27),
    'int2e_dr1200'		: (27, 27),
    'int2e_dr1110'		: (27, 27),
    'int2e_dr1020'		: (27, 27),
    'int2e_dr1011'		: (27, 27),
    'int2e_dr0030'		: (27, 27),
    'int2e_dr0021'		: (27, 27),
    'int2e_dr0012'		: (27, 27),
    'int1e_ovlp_dr40'		: (81, 81),
    'int1e_ovlp_dr31'		: (81, 81),
    'int1e_ovlp_dr22'		: (81, 81),
    'int1e_ovlp_dr13'		: (81, 81),
    'int1e_ovlp_dr04'		: (81, 81),
    'int1e_kin_dr40'		: (81, 81),
    'int1e_kin_dr31'		: (81, 81),
    'int1e_kin_dr22'		: (81, 81),
    'int1e_kin_dr13'		: (81, 81),
    'int1e_kin_dr04'		: (81, 81),
    'int1e_nuc_dr40'		: (81, 81),
    'int1e_nuc_dr31'		: (81, 81),
    'int1e_nuc_dr22'		: (81, 81),
    'int1e_nuc_dr13'		: (81, 81),
    'int1e_nuc_dr04'		: (81, 81),
    'int1e_rinv_dr40'		: (81, 81),
    'int1e_rinv_dr31'		: (81, 81),
    'int1e_rinv_dr22'		: (81, 81),
    'int1e_rinv_dr13'		: (81, 81),
    'int1e_rinv_dr04'		: (81, 81),
    'int2c2e_dr40'		: (81, 81),
    'int2c2e_dr31'		: (81, 81),
    'int2c2e_dr22'		: (81, 81),
    'int2c2e_dr13'		: (81, 81),
    'int2c2e_dr04'		: (81, 81),
    'int1e_r2_dr40'		: (81, 81),
    'int1e_r2_dr31'		: (81, 81),
    'int1e_r2_dr22'		: (81, 81),
    'int1e_r2_dr13'		: (81, 81),
    'int1e_r2_dr04'		: (81, 81),
    'int2e_dr4000'		: (81, 81),
    'int2e_dr3100'		: (81, 81),
    'int2e_dr3010'		: (81, 81),
    'int2e_dr2200'		: (81, 81),
    'int2e_dr2110'		: (81, 81),
    'int2e_dr2020'		: (81, 81),
    'int2e_dr2011'		: (81, 81),
    'int2e_dr1300'		: (81, 81),
    'int2e_dr1210'		: (81, 81),
    'int2e_dr1120'		: (81, 81),
    'int2e_dr1111'		: (81, 81),
    'int2e_dr1030'		: (81, 81),
    'int2e_dr1021'		: (81, 81),
    'int2e_dr1012'		: (81, 81),
    'int2e_dr0040'		: (81, 81),
    'int2e_dr0031'		: (81, 81),
    'int2e_dr0022'		: (81, 81),
    'int2e_dr0013'		: (81, 81),
})

def _get_intor_and_comp(intor_name, comp=None):
    intor_name = ascint3(intor_name)
    if comp is None:
        try:
            if '_spinor' in intor_name:
                fname = intor_name.replace('_spinor', '')
                comp = _INTOR_FUNCTIONS[fname][1]
            else:
                fname = intor_name.replace('_sph', '').replace('_cart', '')
                comp = _INTOR_FUNCTIONS[fname][0]
        except KeyError:
            warnings.warn(f'Function {intor_name} not found.  Set its comp to 1')
            comp = 1
    return intor_name, comp

make_cintopt = replace_source_code(molint.make_cintopt, locals(),
                                   'libcgto', 'libcgto_vjp')
getints2c = replace_source_code(molint.getints2c, locals(),
                                'libcgto', 'libcgto_vjp')
getints3c = replace_source_code(molint.getints3c, locals(),
                                'libcgto', 'libcgto_vjp')
getints4c = replace_source_code(molint.getints4c, locals(),
                                'libcgto', 'libcgto_vjp')
getints_by_shell = replace_source_code(molint.getints_by_shell, locals(),
                                       'libcgto', 'libcgto_vjp')

@wraps(molint.getints)
def getints(intor_name, atm, bas, env, shls_slice=None, comp=None, hermi=0,
            aosym='s1', ao_loc=None, cintopt=None, out=None):
    intor_name, comp = _get_intor_and_comp(intor_name, comp)
    if any(bas[:,ANG_OF] > 12):
        raise NotImplementedError('cint library does not support high angular (l>12) GTOs')

    if (intor_name.startswith('int1e') or
        intor_name.startswith('ECP') or
        intor_name.startswith('int2c2e')):
        return getints2c(intor_name, atm, bas, env, shls_slice, comp,
                         hermi, ao_loc, cintopt, out)
    elif intor_name.startswith('int2e') or intor_name.startswith('int4c1e'):
        return getints4c(intor_name, atm, bas, env, shls_slice, comp,
                         aosym, ao_loc, cintopt, out)
    elif intor_name.startswith('int3c'):
        return getints3c(intor_name, atm, bas, env, shls_slice, comp,
                         aosym, ao_loc, cintopt, out)
    else:
        raise KeyError(f'Unknown intor {intor_name}')
