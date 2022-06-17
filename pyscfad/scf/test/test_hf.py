from pyscfad import scf

def test_nuc_grad(get_h2o):
    mol = get_h2o
    mf = scf.RHF(mol)
    g1 = mf.energy_grad().coords
    mf.kernel()
    g2 = mf.energy_grad().coords
    g0 = mf.Gradients().grad()
    assert abs(g1-g0).max() < 1e-6
    assert abs(g2-g0).max() < 1e-6

def test_nuc_grad_deg(get_n2):
    mol = get_n2
    mf = scf.RHF(mol)
    g1 = mf.energy_grad().coords
    mf.kernel()
    g2 = mf.energy_grad().coords
    g0 = mf.Gradients().grad()
    assert abs(g1-g0).max() < 1e-6
    assert abs(g2-g0).max() < 1e-6
