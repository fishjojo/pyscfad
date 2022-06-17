import jax
from pyscfad import scf

def test_df_hf_nuc_grad(get_h2):
    mol = get_h2
    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    dm = mf.make_rdm1()
    def hf_energy(mf, dm0=None):
        mf.reset()
        e_tot = mf.kernel(dm0=dm0)
        return e_tot
    jac = jax.grad(hf_energy)(mf)
    assert abs(jac.mol.coords - jac.with_df.mol.coords).max() < 1e-10
    g = jac.with_df.mol.coords+jac.with_df.auxmol.coords

    jac = jax.grad(hf_energy)(mf, dm)
    g0 = jac.with_df.mol.coords+jac.with_df.auxmol.coords
    assert abs(g - g0).max() < 1e-6
    assert abs(g0[1,2] - 0.007562137919067152) < 1e-6 # finite difference reference
