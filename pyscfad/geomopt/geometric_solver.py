import os
import uuid
import tempfile
import geometric

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.geomopt.addons import dump_mol_geometry

class PySCFADEngine(geometric.engine.Engine):
    def __init__(self, mol, value_and_grad,
                 maxsteps=100, callback=None):
        molecule = geometric.molecule.Molecule()
        molecule.elem = [mol.atom_symbol(i) for i in range(mol.natm)]
        molecule.xyzs = [mol.atom_coords()*lib.param.BOHR]  # In Angstrom
        super().__init__(molecule)

        self.mol = mol
        self.value_and_grad = value_and_grad

        self.cycle = 0
        self.maxsteps = maxsteps
        self.callback = callback
        self.e_last = 0
        #self.assert_convergence = assert_convergence

    def calc_new(self, coords, dirname):
        if self.cycle >= self.maxsteps:
            raise NotConvergedError( 'Geometry optimization is not converged in '
                                    f'{self.maxsteps} iterations')

        mol = self.mol
        value_and_grad = self.value_and_grad
        self.cycle += 1
        logger.note(mol, '\nGeometry optimization cycle %d', self.cycle)

        # geomeTRIC requires coords and gradients in atomic unit
        coords = numpy.asarray(coords.reshape(-1,3))
        if mol.verbose >= logger.NOTE:
            dump_mol_geometry(mol, coords*lib.param.BOHR)

        if mol.symmetry:
            pass

        mol.coords = None
        mol = mol.set_geom_(coords, unit='Bohr', inplace=False)
        mol.build(trace_coords=True, trace_exp=False, trace_ctr_coeff=False)
        energy, gradients = value_and_grad(mol)
        energy = numpy.asarray(energy)
        gradients = numpy.asarray(gradients)
        logger.note(mol, 'cycle %d: E = %.12g  dE = %g  norm(grad) = %g',
                    self.cycle, energy, energy - self.e_last, numpy.linalg.norm(gradients))
        self.e_last = energy
        self.mol = mol

        if callable(self.callback):
            self.callback(locals())
        return {'energy': energy, 'gradient': gradients.ravel()}

def kernel(mol, value_and_grad,
           constraints=None, callback=None,
           maxsteps=100, **kwargs):
    engine = PySCFADEngine(mol, value_and_grad)
    engine.callback = callback
    engine.maxsteps = maxsteps

    if engine.mol.symmetry:
        pass

    if (not os.path.exists(os.path.abspath(
            os.path.join(geometric.optimize.__file__, '..', 'log.ini')))
            and kwargs.get('logIni') is None):
        kwargs['logIni'] = os.path.abspath(os.path.join(__file__, '..', 'log.ini'))

    with tempfile.TemporaryDirectory(dir=lib.param.TMPDIR) as tmpdir:
        tmpf = os.path.join(tmpdir, str(uuid.uuid4()))
        try:
            geometric.optimize.run_optimizer(customengine=engine, input=tmpf,
                                             constraints=constraints, **kwargs)
            conv = True
        except NotConvergedError as e:
            logger.note(mol, str(e))
            conv = False
    return conv, engine.mol

class NotConvergedError(RuntimeError):
    pass
