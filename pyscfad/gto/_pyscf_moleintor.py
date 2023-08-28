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


_INTOR_FUNCTIONS = {
    # Functiona name            : (comp-for-scalar, comp-for-spinor)
    'int1e_ovlp'                : (1, 1),
    'int1e_nuc'                 : (1, 1),
    'int1e_kin'                 : (1, 1),
    'int1e_ia01p'               : (3, 3),
    'int1e_giao_irjxp'          : (3, 3),
    'int1e_cg_irxp'             : (3, 3),
    'int1e_giao_a11part'        : (9, 9),
    'int1e_cg_a11part'          : (9, 9),
    'int1e_a01gp'               : (9, 9),
    'int1e_igkin'               : (3, 3),
    'int1e_igovlp'              : (3, 3),
    'int1e_ignuc'               : (3, 3),
    'int1e_pnucp'               : (1, 1),
    'int1e_z'                   : (1, 1),
    'int1e_zz'                  : (1, 1),
    'int1e_r'                   : (3, 3),
    'int1e_r2'                  : (1, 1),
    'int1e_r4'                  : (1, 1),
    'int1e_rr'                  : (9, 9),
    'int1e_rrr'                 : (27, 27),
    'int1e_rrrr'                : (81, 81),
    'int1e_z_origj'             : (1, 1),
    'int1e_zz_origj'            : (1, 1),
    'int1e_r_origj'             : (3, 3),
    'int1e_rr_origj'            : (9, 9),
    'int1e_r2_origj'            : (1, 1),
    'int1e_r4_origj'            : (1, 1),
    'int1e_p4'                  : (1, 1),
    'int1e_prinvp'              : (1, 1),
    'int1e_prinvxp'             : (3, 3),
    'int1e_pnucxp'              : (3, 3),
    'int1e_irp'                 : (9, 9),
    'int1e_irrp'                : (27, 27),
    'int1e_irpr'                : (27, 27),
    'int1e_ggovlp'              : (9, 9),
    'int1e_ggkin'               : (9, 9),
    'int1e_ggnuc'               : (9, 9),
    'int1e_grjxp'               : (9, 9),
    'int2e'                     : (1, 1),
    'int2e_ig1'                 : (3, 3),
    'int2e_gg1'                 : (9, 9),
    'int2e_g1g2'                : (9, 9),
    'int2e_ip1v_rc1'            : (9, 9),
    'int2e_ip1v_r1'             : (9, 9),
    'int2e_ipvg1_xp1'           : (9, 9),
    'int2e_ipvg2_xp1'           : (9, 9),
    'int2e_p1vxp1'              : (3, 3),
    'int1e_inuc_rcxp'           : (3, 3),
    'int1e_inuc_rxp'            : (3, 3),
    'int1e_sigma'               : (12,3),
    'int1e_spsigmasp'           : (12,3),
    'int1e_srsr'                : (4, 1),
    'int1e_sr'                  : (4, 1),
    'int1e_srsp'                : (4, 1),
    'int1e_spsp'                : (4, 1),
    'int1e_sp'                  : (4, 1),
    'int1e_spnucsp'             : (4, 1),
    'int1e_sprinvsp'            : (4, 1),
    'int1e_srnucsr'             : (4, 1),
    'int1e_sprsp'               : (12,3),
    'int1e_govlp'               : (3, 3),
    'int1e_gnuc'                : (3, 3),
    'int1e_cg_sa10sa01'         : (36,9),
    'int1e_cg_sa10sp'           : (12,3),
    'int1e_cg_sa10nucsp'        : (12,3),
    'int1e_giao_sa10sa01'       : (36,9),
    'int1e_giao_sa10sp'         : (12,3),
    'int1e_giao_sa10nucsp'      : (12,3),
    'int1e_sa01sp'              : (12,3),
    'int1e_spgsp'               : (12,3),
    'int1e_spgnucsp'            : (12,3),
    'int1e_spgsa01'             : (36,9),
    'int2e_spsp1'               : (4, 1),
    'int2e_spsp1spsp2'          : (16,1),
    'int2e_srsr1'               : (4, 1),
    'int2e_srsr1srsr2'          : (16,1),
    'int2e_cg_sa10sp1'          : (12,3),
    'int2e_cg_sa10sp1spsp2'     : (48,3),
    'int2e_giao_sa10sp1'        : (12,3),
    'int2e_giao_sa10sp1spsp2'   : (48,3),
    'int2e_g1'                  : (12,3),
    'int2e_spgsp1'              : (12,3),
    'int2e_g1spsp2'             : (12,3),
    'int2e_spgsp1spsp2'         : (48,3),
    'int2e_pp1'                 : (1, 1),
    'int2e_pp2'                 : (1, 1),
    'int2e_pp1pp2'              : (1, 1),
    'int1e_spspsp'              : (4, 1),
    'int1e_spnuc'               : (4, 1),
    'int2e_spv1'                : (4, 1),
    'int2e_vsp1'                : (4, 1),
    'int2e_spsp2'               : (4, 1),
    'int2e_spv1spv2'            : (16,1),
    'int2e_vsp1spv2'            : (16,1),
    'int2e_spv1vsp2'            : (16,1),
    'int2e_vsp1vsp2'            : (16,1),
    'int2e_spv1spsp2'           : (16,1),
    'int2e_vsp1spsp2'           : (16,1),
    'int1e_ipovlp'              : (3, 3),
    'int1e_ipkin'               : (3, 3),
    'int1e_ipnuc'               : (3, 3),
    'int1e_iprinv'              : (3, 3),
    'int1e_rinv'                : (1, 1),
    'int1e_ipspnucsp'           : (12,3),
    'int1e_ipsprinvsp'          : (12,3),
    'int1e_ippnucp'             : (3, 3),
    'int1e_ipprinvp'            : (3, 3),
    'int2e_ip1'                 : (3, 3),
    'int2e_ip2'                 : (3, 3),
    'int2e_ipspsp1'             : (12,3),
    'int2e_ip1spsp2'            : (12,3),
    'int2e_ipspsp1spsp2'        : (48,3),
    'int2e_ipsrsr1'             : (12,3),
    'int2e_ip1srsr2'            : (12,3),
    'int2e_ipsrsr1srsr2'        : (48,3),
    'int2e_ssp1ssp2'            : (16,1),
    'int2e_ssp1sps2'            : (16,1),
    'int2e_sps1ssp2'            : (16,1),
    'int2e_sps1sps2'            : (16,1),
    'int2e_cg_ssa10ssp2'        : (48,3),
    'int2e_giao_ssa10ssp2'      : (18,3),
    'int2e_gssp1ssp2'           : (18,3),
    'int2e_gauge_r1_ssp1ssp2'   : (None, 1),
    'int2e_gauge_r1_ssp1sps2'   : (None, 1),
    'int2e_gauge_r1_sps1ssp2'   : (None, 1),
    'int2e_gauge_r1_sps1sps2'   : (None, 1),
    'int2e_gauge_r2_ssp1ssp2'   : (None, 1),
    'int2e_gauge_r2_ssp1sps2'   : (None, 1),
    'int2e_gauge_r2_sps1ssp2'   : (None, 1),
    'int2e_gauge_r2_sps1sps2'   : (None, 1),
    'int1e_ipipovlp'            : (9, 9),
    'int1e_ipovlpip'            : (9, 9),
    'int1e_ipipkin'             : (9, 9),
    'int1e_ipkinip'             : (9, 9),
    'int1e_ipipnuc'             : (9, 9),
    'int1e_ipnucip'             : (9, 9),
    'int1e_ipiprinv'            : (9, 9),
    'int1e_iprinvip'            : (9, 9),
    'int2e_ipip1'               : (9, 9),
    'int2e_ipvip1'              : (9, 9),
    'int2e_ip1ip2'              : (9, 9),
    'int1e_ipippnucp'           : (9, 9),
    'int1e_ippnucpip'           : (9, 9),
    'int1e_ipipprinvp'          : (9, 9),
    'int1e_ipprinvpip'          : (9, 9),
    'int1e_ipipspnucsp'         : (36,9),
    'int1e_ipspnucspip'         : (36,9),
    'int1e_ipipsprinvsp'        : (36,9),
    'int1e_ipsprinvspip'        : (36,9),
    'int3c2e'                   : (1, 1),
    'int3c2e_ip1'               : (3, 3),
    'int3c2e_ip2'               : (3, 3),
    'int3c2e_pvp1'              : (1, 1),
    'int3c2e_pvxp1'             : (3, 3),
    'int2c2e_ip1'               : (3, 3),
    'int2c2e_ip2'               : (3, 3),
    'int3c2e_ig1'               : (3, 3),
    'int3c2e_spsp1'             : (4, 1),
    'int3c2e_ipspsp1'           : (12,3),
    'int3c2e_spsp1ip2'          : (12,3),
    'int3c2e_ipip1'             : (9, 9),
    'int3c2e_ipip2'             : (9, 9),
    'int3c2e_ipvip1'            : (9, 9),
    'int3c2e_ip1ip2'            : (9, 9),
    'int2c2e_ip1ip2'            : (9, 9),
    'int2c2e_ipip1'             : (9, 9),
    'int3c1e'                   : (1, 1),
    'int3c1e_p2'                : (1, 1),
    'int3c1e_iprinv'            : (3, 3),
    'int2c2e'                   : (1, 1),
    'int2e_yp'                  : (1, 1),
    'int2e_stg'                 : (1, 1),
    'int2e_coulerf'             : (1, 1),
    'int1e_grids'               : (1, 1),
    'int1e_grids_ip'            : (3, 3),
    'int1e_grids_spvsp'         : (4, 1),
    'ECPscalar'                 : (1, None),
    'ECPscalar_ipnuc'           : (3, None),
    'ECPscalar_iprinv'          : (3, None),
    'ECPscalar_ignuc'           : (3, None),
    'ECPscalar_iprinvip'        : (9, None),
    'ECPso'                     : (3, 1),
    # alias
    'int1e_ovlp_dr10'           : (3, 3),
    'int1e_ovlp_dr01'           : (3, 3),
    'int1e_kin_dr10'            : (3, 3),
    'int1e_kin_dr01'            : (3, 3),
    'int1e_nuc_dr10'            : (3, 3),
    'int1e_nuc_dr01'            : (3, 3),
    'int1e_rinv_dr10'           : (3, 3),
    'int1e_rinv_dr01'           : (3, 3),
    'int1e_ovlp_dr20'           : (9, 9),
    'int1e_ovlp_dr11'           : (9, 9),
    'int1e_kin_dr20'            : (9, 9),
    'int1e_kin_dr11'            : (9, 9),
    'int1e_nuc_dr20'            : (9, 9),
    'int1e_nuc_dr11'            : (9, 9),
    'int1e_rinv_dr20'           : (9, 9),
    'int1e_rinv_dr11'           : (9, 9),
    'int1e_ovlp_dr30'           : (27, 27),
    'int1e_ovlp_dr21'           : (27, 27),
    'int1e_ovlp_dr12'           : (27, 27),
    'int1e_ovlp_dr03'           : (27, 27),
    'int1e_kin_dr30'            : (27, 27),
    'int1e_kin_dr21'            : (27, 27),
    'int1e_kin_dr12'            : (27, 27),
    'int1e_kin_dr03'            : (27, 27),
    'int1e_nuc_dr30'            : (27, 27),
    'int1e_nuc_dr21'            : (27, 27),
    'int1e_nuc_dr12'            : (27, 27),
    'int1e_nuc_dr03'            : (27, 27),
    'int1e_rinv_dr30'           : (27, 27),
    'int1e_rinv_dr21'           : (27, 27),
    'int1e_rinv_dr12'           : (27, 27),
    'int1e_rinv_dr03'           : (27, 27),
    'int2e_dr1000'              : (3, 3),
    'int2e_dr0010'              : (3, 3),
    'int2e_dr2000'              : (9, 9),
    'int2e_dr1100'              : (9, 9),
    'int2e_dr1010'              : (9, 9),
    'int2e_dr3000'              : (27, 27),
    'int2e_dr2100'              : (27, 27),
    'int2e_dr1200'              : (27, 27),
    'int2e_dr0030'              : (27, 27),
    'int2e_dr0021'              : (27, 27),
    'int2e_dr0012'              : (27, 27),
    'int2e_dr2010'              : (27, 27),
    'int2e_dr1020'              : (27, 27),
    'int2e_dr1110'              : (27, 27),
    'int2e_dr1011'              : (27, 27),
    'int3c2e_dr100'             : (3, 3),
    'int3c2e_dr001'             : (3, 3),
    'int3c2e_dr200'             : (9, 9),
    'int3c2e_dr002'             : (9, 9),
    'int3c2e_dr110'             : (9, 9),
    'int3c2e_dr101'             : (9, 9),
    'int2c2e_dr10'              : (3, 3),
    'int2c2e_dr01'              : (3, 3),
    'int2c2e_dr11'              : (9, 9),
    'int2c2e_dr20'              : (9, 9),
    'int1e_ovlp_dr40'           : (81, 81),
    'int1e_ovlp_dr31'           : (81, 81),
    'int1e_ovlp_dr22'           : (81, 81),
    'int1e_ovlp_dr13'           : (81, 81),
    'int1e_ovlp_dr04'           : (81, 81),
    'int1e_kin_dr40'            : (81, 81),
    'int1e_kin_dr31'            : (81, 81),
    'int1e_kin_dr22'            : (81, 81),
    'int1e_kin_dr13'            : (81, 81),
    'int1e_kin_dr04'            : (81, 81),
    'int1e_nuc_dr40'            : (81, 81),
    'int1e_nuc_dr31'            : (81, 81),
    'int1e_nuc_dr22'            : (81, 81),
    'int1e_nuc_dr13'            : (81, 81),
    'int1e_nuc_dr04'            : (81, 81),
    'int1e_rinv_dr40'           : (81, 81),
    'int1e_rinv_dr31'           : (81, 81),
    'int1e_rinv_dr22'           : (81, 81),
    'int1e_rinv_dr13'           : (81, 81),
    'int1e_rinv_dr04'           : (81, 81),
    'int2e_dr4000'              : (81, 81),
    'int2e_dr3100'              : (81, 81),
    'int2e_dr3010'              : (81, 81),
    'int2e_dr2200'              : (81, 81),
    'int2e_dr2110'              : (81, 81),
    'int2e_dr2020'              : (81, 81),
    'int2e_dr2011'              : (81, 81),
    'int2e_dr1300'              : (81, 81),
    'int2e_dr1210'              : (81, 81),
    'int2e_dr1120'              : (81, 81),
    'int2e_dr1111'              : (81, 81),
    'int2e_dr1030'              : (81, 81),
    'int2e_dr1021'              : (81, 81),
    'int2e_dr1012'              : (81, 81),
    'int2e_dr0040'              : (81, 81),
    'int2e_dr0031'              : (81, 81),
    'int2e_dr0022'              : (81, 81),
    'int2e_dr0013'              : (81, 81),
    ####
}


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
