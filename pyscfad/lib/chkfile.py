import h5py

# pylint: disable=consider-using-f-string
def dump(chkfile, key, value):
    def save_as_group(key, value, root):
        if isinstance(value, dict):
            root1 = root.create_group(key)
            for k in value:
                save_as_group(k, value[k], root1)
        elif isinstance(value, (tuple, list, range)):
            root1 = root.create_group(key + '__from_list__')
            for k, v in enumerate(value):
                save_as_group('%06d'%k, v, root1)
        else:
            try:
                root[key] = getattr(value, 'val', value)
            except (TypeError, ValueError) as e:
                if not (e.args[0] == "Object dtype dtype('O') has no native HDF5 equivalent" or
                        e.args[0].startswith('could not broadcast input array')):
                    raise e
                root1 = root.create_group(key + '__from_list__')
                for k, v in enumerate(value):
                    save_as_group('%06d'%k, v, root1)

    if h5py.is_hdf5(chkfile):
        with h5py.File(chkfile, 'r+') as fh5:
            if key in fh5:
                del fh5[key]
            elif key + '__from_list__' in fh5:
                del fh5[key+'__from_list__']
            save_as_group(key, value, fh5)
    else:
        with h5py.File(chkfile, 'w') as fh5:
            save_as_group(key, value, fh5)
dump_chkfile_key = save = dump
