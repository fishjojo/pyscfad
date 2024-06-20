import h5py
from pyscf.lib.chkfile import save_mol
from pyscfad import ops
from pyscfad.lib.chkfile import save

def dump_scf(mol, chkfile, e_tot, mo_energy, mo_coeff, mo_occ,
             overwrite_mol=True):
    if h5py.is_hdf5(chkfile) and not overwrite_mol:
        with h5py.File(chkfile, 'a') as fh5:
            if 'mol' not in fh5:
                fh5['mol'] = mol.dumps()
    else:
        save_mol(mol, chkfile)

    e_tot = ops.to_numpy(e_tot)
    mo_energy = ops.to_numpy(mo_energy)
    mo_occ = ops.to_numpy(mo_occ)
    mo_coeff = ops.to_numpy(mo_coeff)
    scf_dic = {'e_tot'    : e_tot,
               'mo_energy': mo_energy,
               'mo_occ'   : mo_occ,
               'mo_coeff' : mo_coeff}
    save(chkfile, 'scf', scf_dic)
