#include "xla/ffi/api/ffi.h"
#include "pyscfadlib/cuda/vendor.h"
#include "cuint.h"

namespace pyscfad {
namespace cuint {

namespace ffi = xla::ffi;

ffi::Error OverlapDispatch(
    cudaStream_t stream, int batch_count,
    int i_angular, int j_angular, int is_screened,
    int n_pairs, int n_primitives, int n_functions,
    int atm_stride, int bas_stride, int env_stride,
    ffi::Buffer<ffi::F64> result_in,
    ffi::Buffer<ffi::S32> pair_indices, ffi::Buffer<ffi::S32> primitive_to_function,
    ffi::Buffer<ffi::S32> atm, ffi::Buffer<ffi::S32> bas, ffi::Buffer<ffi::F64> env,
    ffi::Result<ffi::Buffer<ffi::F64>> result_out)
{
  auto* result_in_data = result_in.typed_data();
  auto* result_out_data = result_out->typed_data();
  if (result_in_data != result_out_data) {
    cudaMemcpyAsync(result_out_data, result_in_data, result_in.size_bytes(),
                    cudaMemcpyDeviceToDevice, stream);
  }

  auto* pair_indices_data = pair_indices.typed_data();
  auto* primitive_to_function_data = primitive_to_function.typed_data();
  auto* atm_data = atm.typed_data();
  auto* bas_data = bas.typed_data();
  auto* env_data = env.typed_data();

  overlap(result_out_data, pair_indices_data, n_pairs,
          n_primitives, primitive_to_function_data,
          n_functions, atm_data, atm_stride,
          bas_data, bas_stride, env_data,
          env_stride, batch_count,
          i_angular, j_angular, is_screened);

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
  OverlapFfi, OverlapDispatch,
  ffi::Ffi::Bind()
    .Ctx<ffi::PlatformStream<cudaStream_t>>()
    .Attr<int>("batch_count")
    .Attr<int>("i_angular")
    .Attr<int>("j_angular")
    .Attr<int>("is_screened")
    .Attr<int>("n_pairs")
    .Attr<int>("n_primitives")
    .Attr<int>("n_functions")
    .Attr<int>("atm_stride")
    .Attr<int>("bas_stride")
    .Attr<int>("env_stride")
    .Arg<ffi::Buffer<ffi::F64>>(/*result_in*/)
    .Arg<ffi::Buffer<ffi::S32>>(/*pair_indices*/)
    .Arg<ffi::Buffer<ffi::S32>>(/*primitive_to_function*/)
    .Arg<ffi::Buffer<ffi::S32>>(/*atm*/)
    .Arg<ffi::Buffer<ffi::S32>>(/*bas*/)
    .Arg<ffi::Buffer<ffi::F64>>(/*env*/)
    .Ret<ffi::Buffer<ffi::F64>>(/*result_out*/)
);

ffi::Error OverlapGradientDispatch(
    cudaStream_t stream, int batch_count,
    int i_angular, int j_angular, int is_screened,
    int n_pairs, int n_primitives, int n_functions,
    int atm_stride, int bas_stride, int env_stride,
    ffi::Buffer<ffi::F64> result_in,
    ffi::Buffer<ffi::S32> pair_indices, ffi::Buffer<ffi::S32> primitive_to_function,
    ffi::Buffer<ffi::S32> atm, ffi::Buffer<ffi::S32> bas, ffi::Buffer<ffi::F64> env,
    ffi::Result<ffi::Buffer<ffi::F64>> result_out)
{
  auto* result_in_data = result_in.typed_data();
  auto* result_out_data = result_out->typed_data();
  if (result_in_data != result_out_data) {
    cudaMemcpyAsync(result_out_data, result_in_data, result_in.size_bytes(),
                    cudaMemcpyDeviceToDevice, stream);
  }

  auto* pair_indices_data = pair_indices.typed_data();
  auto* primitive_to_function_data = primitive_to_function.typed_data();
  auto* atm_data = atm.typed_data();
  auto* bas_data = bas.typed_data();
  auto* env_data = env.typed_data();

  overlap_gradient(result_out_data, pair_indices_data, n_pairs,
                   n_primitives, primitive_to_function_data,
                   n_functions, atm_data, atm_stride,
                   bas_data, bas_stride, env_data,
                   env_stride, batch_count,
                   i_angular, j_angular, is_screened);

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
  OverlapGradientFfi, OverlapGradientDispatch,
  ffi::Ffi::Bind()
    .Ctx<ffi::PlatformStream<cudaStream_t>>()
    .Attr<int>("batch_count")
    .Attr<int>("i_angular")
    .Attr<int>("j_angular")
    .Attr<int>("is_screened")
    .Attr<int>("n_pairs")
    .Attr<int>("n_primitives")
    .Attr<int>("n_functions")
    .Attr<int>("atm_stride")
    .Attr<int>("bas_stride")
    .Attr<int>("env_stride")
    .Arg<ffi::Buffer<ffi::F64>>(/*result_in*/)
    .Arg<ffi::Buffer<ffi::S32>>(/*pair_indices*/)
    .Arg<ffi::Buffer<ffi::S32>>(/*primitive_to_function*/)
    .Arg<ffi::Buffer<ffi::S32>>(/*atm*/)
    .Arg<ffi::Buffer<ffi::S32>>(/*bas*/)
    .Arg<ffi::Buffer<ffi::F64>>(/*env*/)
    .Ret<ffi::Buffer<ffi::F64>>(/*result_out*/)
);

} // namespace cuint
} // namespace pyscfad
