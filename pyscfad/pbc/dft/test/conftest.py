import pytest
from pyscfad.pbc import gto

@pytest.fixture
def get_Si2():
    a = 5.431020511
    basis = 'gth-szv'
    pseudo = 'gth-pade'
    lattice = [[0., a/2, a/2],
              [a/2, 0., a/2],
              [a/2, a/2, 0.]]
    disp = 0.01
    atom = [['Si', [0., 0., 0.]],
            ['Si', [a/4+disp, a/4+disp, a/4+disp]]]

    cell = gto.Cell()
    cell.atom = atom
    cell.a = lattice
    cell.basis = basis
    cell.pseudo = pseudo
    cell.max_memory = 7000
    cell.build()
    return cell
