from pyscfad import pbc

class Cell(pbc.gto.Cell):
    def get_mm_ewald_pot():
        # TODO don't forget to use xtb param
        # TODO don't forget to broadcast pot0 to shells
        pass

    def get_qm_ewald_hess():
        # TODO don't forget to use xtb param
        # TODO don't forget to broadcast hess0 to shells
        pass