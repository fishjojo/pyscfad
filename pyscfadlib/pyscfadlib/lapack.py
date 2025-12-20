import numpy as np

import jax
try:
    from jax import ffi
except ImportError:
    from jax.extend import ffi

from pyscfadlib import pyscfad_lapack as lp

for _name, _value in lp.registrations().items():
    ffi.register_ffi_target(
        _name,
        _value,
        platform="cpu",
        api_version=(1 if _name.endswith("_ffi") else 0),
    )

LAPACK_DTYPE_PREFIX = {
    np.float32: "s",
    np.float64: "d",
    np.complex64: "c",
    np.complex128: "z",
}

def prepare_lapack_call(fn_base, dtype):
    lp.initialize()
    try:
        prefix = (LAPACK_DTYPE_PREFIX.get(dtype, None) or 
                  LAPACK_DTYPE_PREFIX[dtype.type])
        return f"lapack_{prefix}{fn_base}"
    except KeyError:
        raise NotImplementedError(f"Unsupported dtype {dtype}")

if jax.__version__ < "0.8":
    import jaxlib.mlir.ir as ir
    import jaxlib.mlir.dialects.stablehlo as hlo
    from jaxlib.hlo_helpers import (
        custom_call,
        hlo_s32,
        ensure_hlo_s32,
        mk_result_types_and_shapes,
    )

    def sygvd_hlo(dtype, a, b, a_shape_vals, b_shape_vals,
                  lower=False, itype=1):
        lp.initialize()
        a_type = ir.RankedTensorType(a.type)
        b_type = ir.RankedTensorType(b.type)
        assert len(a_shape_vals) >= 2
        m, n = a_shape_vals[-2:]
        # Non-batch dimensions must be static
        assert type(m) is int and type(n) is int and m == n, a_shape_vals

        batch_dims_vals = a_shape_vals[:-2]
        num_bd = len(a_shape_vals) - 2

        i32_type = ir.IntegerType.get_signless(32)
        #workspace: list[ShapeTypePair]
        if dtype == np.float32:
            fn = "lapack_ssygvd"
            eigvals_type = ir.F32Type.get()
            workspace = [
                ([lp.sygvd_work_size(n)], a_type.element_type),
                ([lp.sygvd_iwork_size(n)], i32_type),
            ]
        elif dtype == np.float64:
            fn = "lapack_dsygvd"
            eigvals_type = ir.F64Type.get()
            workspace = [
                ([lp.sygvd_work_size(n)], a_type.element_type),
                ([lp.sygvd_iwork_size(n)], i32_type),
            ]
        elif dtype == np.complex64:
            fn = "lapack_chegvd"
            eigvals_type = ir.F32Type.get()
            workspace = [
                ([lp.hegvd_work_size(n)], a_type.element_type),
                ([lp.hegvd_rwork_size(n)], eigvals_type),
                ([lp.sygvd_iwork_size(n)], i32_type),
            ]
        elif dtype == np.complex128:
            fn = "lapack_zhegvd"
            eigvals_type = ir.F64Type.get()
            workspace = [
                ([lp.hegvd_work_size(n)],  a_type.element_type),
                ([lp.hegvd_rwork_size(n)], eigvals_type),
                ([lp.sygvd_iwork_size(n)], i32_type),
            ]
        else:
            raise NotImplementedError(f"Unsupported dtype {dtype}")

        batch_size_val = hlo_s32(1)
        for b_v in batch_dims_vals:
            batch_size_val = hlo.multiply(batch_size_val, ensure_hlo_s32(b_v))

        scalar_layout = []
        shape_layout = [0]
        workspace_layouts = [shape_layout] * len(workspace)
        layout = (num_bd, num_bd + 1) + tuple(range(num_bd - 1, -1, -1))

        result_types, result_shapes = mk_result_types_and_shapes(
            [(a_shape_vals, a_type.element_type),
             (batch_dims_vals + (n,),  eigvals_type),
             (batch_dims_vals, i32_type),
             (b_shape_vals, b_type.element_type)] + workspace
        )

        out = custom_call(
            fn,
            result_types=result_types,
            operands=[ensure_hlo_s32(itype),
                      hlo_s32(1 if lower else 0),
                      batch_size_val,
                      ensure_hlo_s32(n), a, b],
            operand_layouts=[scalar_layout] * 4 + [layout, layout],
            result_layouts=[
                layout,
                tuple(range(num_bd, -1, -1)),
                tuple(range(num_bd - 1, -1, -1)),
                layout, # for b
            ] + workspace_layouts,
            operand_output_aliases={4: 0, 5: 3},
            result_shapes=result_shapes,
        ).results
        return out[:3]
