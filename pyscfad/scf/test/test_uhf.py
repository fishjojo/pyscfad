from pyscfad import scf

def test_nuc_grad(get_h2o_plus):
    mol = get_h2o_plus
    mf = scf.UHF(mol)
    g1 = mf.energy_grad().coords
    mf.kernel()
    g2 = mf.energy_grad().coords
    g0 = mf.nuc_grad_method().kernel()
    assert abs(g1-g0).max() < 1e-6
    assert abs(g2-g0).max() < 1e-6
