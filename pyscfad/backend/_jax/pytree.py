# Copyright 2021-2025 Xing Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import warnings
from jax import tree_util

def _dict_hash(this):
    from pyscf.lib.misc import finger
    fg = []
    leaves, tree = tree_util.tree_flatten(this)
    fg.append(hash(tree))
    for v in leaves:
        if hasattr(v, "size"): # arrays
            fg.append(finger(v))
        elif isinstance(v, set):
            fg.append(_dict_hash(tuple(sorted(v))))
        else:
            try:
                fg.append(hash(v))
            except TypeError as e:
                raise e
    return hash(tuple(fg))


def _dict_equality(d1, d2):
    leaves1, tree1 = tree_util.tree_flatten(d1)
    leaves2, tree2 = tree_util.tree_flatten(d2)
    if tree1 != tree2:
        return False

    for v1, v2 in zip(leaves1, leaves2):
        if v1 is v2:
            neq = False
        else:
            if hasattr(v1, "size") and hasattr(v2, "size"): # arrays
                if v1.size != v2.size:
                    neq = True
                elif v1.size == 0 and v2.size == 0:
                    neq = False
                else:
                    try:
                        neq = not v1 == v2
                    except ValueError as e:
                        try:
                            neq = not (v1 == v2).all()
                        except Exception:
                            raise e from None
            else:
                try:
                    neq = not v1 == v2
                except ValueError as e:
                    raise e
        if neq:
            return False
    return True


class _AuxData:
    def __init__(self, data, exclude_name=()):
        self.data = dict(sorted(data.items()))
        self.exclude_name = exclude_name

    @property
    def data_for_hash(self):
        if self.exclude_name:
            return {k : v for k, v in self.data.items() if k not in self.exclude_name}
        else:
            return self.data

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, _AuxData):
            return False
        return _dict_equality(self.data_for_hash, other.data_for_hash)

    def __hash__(self):
        return _dict_hash(self.data_for_hash)


def class_as_pytree_node(cls, leaf_names, num_args=0, exclude_aux_name=()):
    def tree_flatten(obj):
        keys = obj.__dict__.keys()
        for leaf_name in leaf_names:
            if leaf_name not in keys:
                raise KeyError(f'Pytree leaf {leaf_name} is not defined in class {cls}.')
        children =  tuple(getattr(obj, leaf_name, None) for leaf_name in leaf_names)
        if len(children) <= 0:
            #raise KeyError("Empty pytree node is not supported.")
            warnings.warn(f'Not taking derivatives w.r.t. the leaves in '
                          f'the node {obj.__class__} as none of those was specified.')

        aux_keys = list(set(keys) - set(leaf_names))
        aux_data = {k : getattr(obj, k, None) for k in aux_keys}
        metadata = (num_args, _AuxData(aux_data, exclude_name=exclude_aux_name))
        return children, metadata

    def tree_unflatten(metadata, children):
        num_args = metadata[0]
        auxdata = metadata[1]
        leaves_args = children[:num_args]
        leaves_kwargs = dict(zip(leaf_names[num_args:], children[num_args:]))
        kwargs = {**leaves_kwargs, **(auxdata.data)}
        obj = cls(*leaves_args, **kwargs)
        return obj

    tree_util.register_pytree_node(cls, tree_flatten, tree_unflatten)
    return cls


class PytreeNodeMeta(type):
    def __new__(mcls, name, bases, dct, **kwargs):
        cls = super().__new__(mcls, name, bases, dct, **kwargs)

        def _collect_attr(name):
            out = []
            for base in reversed(cls.__mro__):
                if hasattr(base, name):
                    out.extend(getattr(base, name))
            return tuple(dict.fromkeys(out))

        dyn_keys = _collect_attr("_dynamic_attr")
        sta_keys = _collect_attr("_static_attr")
        ign_keys = _collect_attr("_ignore_attr")
        allow_missing_dynamic = getattr(cls, "_allow_missing_dynamic", True)

        cls._dynamic_attr = dyn_keys
        cls._static_attr = sta_keys
        cls._ignore_attr = ign_keys

        def _flatten_impl(obj, with_keys=False):
            children = []
            for key in dyn_keys:
                if hasattr(obj, key):
                    val = getattr(obj, key)
                else:
                    if allow_missing_dynamic:
                        val = None
                    else:
                        raise AttributeError(f"Missing dynamic attribute '{key}' in {type(obj).__name__}")
                children.append((tree_util.GetAttrKey(key), val) if with_keys else val)

            obj_dict = vars(obj)
            if sta_keys:
                aux_keys = set(obj_dict.keys()) & set(sta_keys)
            else:
                aux_keys = set(obj_dict.keys()) - set(dyn_keys)
            aux_data = {key: obj_dict[key] for key in aux_keys}
            return children, _AuxData(aux_data, exclude_name=ign_keys)

        def _unflatten_impl(aux_data, children):
            obj = object.__new__(cls)
            for key, value in zip(dyn_keys, children):
                object.__setattr__(obj, key, value)
            for key, value in aux_data.data.items():
                object.__setattr__(obj, key, value)
            return obj

        tree_util.register_pytree_with_keys(
            cls,
            partial(_flatten_impl, with_keys=True),
            _unflatten_impl,
            flatten_func=partial(_flatten_impl, with_keys=False)
        )
        return cls


class PytreeNode(metaclass=PytreeNodeMeta):
    """Subclassing ``PytreeNode`` to register the class as a pytree,
    which works with jax transformations.

    Attributes
    ----------
    _dynamic_attr : list of str, optional
        Names of dynamic attributes. These are typically arrays,
        whose values can change during the calculation.
    _static_attr : list of str, optional
        Names of static attributes. These are typically configuration
        flags, whose values should not change during the calculation.
        Modifying static attributes will triger recompilation. If not
        explicitly specified, all attributes except those specified by
        ``_dynamic_attr`` and ``_ignore_attr`` are treated as static.
        If explicitly specified, only these are considered as static.
    _ignore_attr : list of str, optional
        Names of ignored attributes. These are typically temporary data,
        which are used before jax transformations.
        The values of them may not be carried over during jax transformations.
    _allow_missing_dynamic : bool, optional
        Whether to allow missing dynamic attributes. Default is True,
        which will assign ``None`` to missing dynamic attributes.

    Example
    -------

    .. code-block:: python

        class Foo(PytreeNode):
            _dynamic_attr = ["params", "state"]
            _static_attr  = ["basis", "config"]
            _ignore_attr  = ["_cache"]
            _allow_missing_dynamic = False

            def __init__(self, params, state, basis, config):
                self.params = params
                self.state = state
                self.basis = basis
                self.config = config
                self._cache = {}
    """
    pass

