#include "xla/ffi/api/ffi.h"
#include "pyscfadlib/cuda/vendor.h"
#include "pyscfadlib/ffi_helpers.h"
#include "cuint.h"

namespace pyscfad {
namespace cuint {

namespace ffi = xla::ffi;

ffi::Error LatOverlapDispatch(
    cudaStream_t stream,
    ffi::Buffer<ffi::F64> result_in,
    ffi::Buffer<ffi::S32> pair_indices, ffi::Buffer<ffi::S32> primitive_to_function,
    ffi::Buffer<ffi::S32> atm, ffi::Buffer<ffi::S32> bas, ffi::Buffer<ffi::F64> env,
    ffi::Buffer<ffi::F64> Ls, ffi::Buffer<ffi::S32> mask,
    ffi::Result<ffi::Buffer<ffi::F64>> result_out,
    int i_angular, int j_angular, int is_screened,
    int n_pairs, int n_primitives, int n_functions,
    int atm_stride, int bas_stride, int env_stride,
    int n_images, int reduce_over_images)
{
  auto* result_in_data = result_in.typed_data();
  auto* result_out_data = result_out->typed_data();
  if (result_in_data != result_out_data) {
    cudaMemcpyAsync(result_out_data, result_in_data, result_in.size_bytes(),
                    cudaMemcpyDeviceToDevice, stream);
  }

  auto [nmat, nao_i, nao_j] = SplitBatch2D(result_in.dimensions());
  auto batch_count = nmat / n_images;
  if (nao_i != nao_j || static_cast<int>(nao_i) != n_functions) {
    return ffi::Error::InvalidArgument(
      "Overlap: out buffer has wrong shape.");
  }
  auto* pair_indices_data = pair_indices.typed_data();
  auto* primitive_to_function_data = primitive_to_function.typed_data();
  auto* atm_data = atm.typed_data();
  auto* bas_data = bas.typed_data();
  auto* env_data = env.typed_data();
  auto* Ls_data = Ls.typed_data();
  auto* mask_data = mask.typed_data();

  pbc_overlap(stream, result_out_data, pair_indices_data, n_pairs,
              n_primitives, primitive_to_function_data,
              n_functions, atm_data, atm_stride,
              bas_data, bas_stride, env_data,
              env_stride, static_cast<int>(batch_count),
              Ls_data, n_images, mask_data,
              i_angular, j_angular, is_screened, reduce_over_images);

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
  LatOverlapFfi, LatOverlapDispatch,
  ffi::Ffi::Bind()
    .Ctx<ffi::PlatformStream<cudaStream_t>>()
    .Arg<ffi::Buffer<ffi::F64>>(/*result_in*/)
    .Arg<ffi::Buffer<ffi::S32>>(/*pair_indices*/)
    .Arg<ffi::Buffer<ffi::S32>>(/*primitive_to_function*/)
    .Arg<ffi::Buffer<ffi::S32>>(/*atm*/)
    .Arg<ffi::Buffer<ffi::S32>>(/*bas*/)
    .Arg<ffi::Buffer<ffi::F64>>(/*env*/)
    .Arg<ffi::Buffer<ffi::F64>>(/*Ls*/)
    .Arg<ffi::Buffer<ffi::S32>>(/*mask*/)
    .Ret<ffi::Buffer<ffi::F64>>(/*result_out*/)
    .Attr<int>("i_angular")
    .Attr<int>("j_angular")
    .Attr<int>("is_screened")
    .Attr<int>("n_pairs")
    .Attr<int>("n_primitives")
    .Attr<int>("n_functions")
    .Attr<int>("atm_stride")
    .Attr<int>("bas_stride")
    .Attr<int>("env_stride")
    .Attr<int>("n_images")
    .Attr<int>("reduce_over_images")
);

ffi::Error LatOverlapGradientDispatch(
    cudaStream_t stream,
    ffi::Buffer<ffi::F64> result_in,
    ffi::Buffer<ffi::S32> pair_indices, ffi::Buffer<ffi::S32> primitive_to_function,
    ffi::Buffer<ffi::S32> atm, ffi::Buffer<ffi::S32> bas, ffi::Buffer<ffi::F64> env,
    ffi::Buffer<ffi::F64> Ls, ffi::Buffer<ffi::S32> mask,
    ffi::Result<ffi::Buffer<ffi::F64>> result_out,
    int i_angular, int j_angular, int is_screened,
    int n_pairs, int n_primitives, int n_functions,
    int atm_stride, int bas_stride, int env_stride,
    int n_images, int reduce_over_images)
{
  auto* result_in_data = result_in.typed_data();
  auto* result_out_data = result_out->typed_data();
  if (result_in_data != result_out_data) {
    cudaMemcpyAsync(result_out_data, result_in_data, result_in.size_bytes(),
                    cudaMemcpyDeviceToDevice, stream);
  }

  auto [nmat3, nao_i, nao_j] = SplitBatch2D(result_in.dimensions());
  auto batch_count = nmat3 / n_images / 3;
  if (nao_i != nao_j || static_cast<int>(nao_i) != n_functions) {
    return ffi::Error::InvalidArgument(
      "Overlap: out buffer has wrong shape.");
  }
  auto* pair_indices_data = pair_indices.typed_data();
  auto* primitive_to_function_data = primitive_to_function.typed_data();
  auto* atm_data = atm.typed_data();
  auto* bas_data = bas.typed_data();
  auto* env_data = env.typed_data();
  auto* Ls_data = Ls.typed_data();
  auto* mask_data = mask.typed_data();

  pbc_overlap_gradient(stream, result_out_data, pair_indices_data, n_pairs,
                       n_primitives, primitive_to_function_data,
                       n_functions, atm_data, atm_stride,
                       bas_data, bas_stride, env_data,
                       env_stride, static_cast<int>(batch_count),
                       Ls_data, n_images, mask_data,
                       i_angular, j_angular, is_screened, reduce_over_images);

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
  LatOverlapGradientFfi, LatOverlapGradientDispatch,
  ffi::Ffi::Bind()
    .Ctx<ffi::PlatformStream<cudaStream_t>>()
    .Arg<ffi::Buffer<ffi::F64>>(/*result_in*/)
    .Arg<ffi::Buffer<ffi::S32>>(/*pair_indices*/)
    .Arg<ffi::Buffer<ffi::S32>>(/*primitive_to_function*/)
    .Arg<ffi::Buffer<ffi::S32>>(/*atm*/)
    .Arg<ffi::Buffer<ffi::S32>>(/*bas*/)
    .Arg<ffi::Buffer<ffi::F64>>(/*env*/)
    .Arg<ffi::Buffer<ffi::F64>>(/*Ls*/)
    .Arg<ffi::Buffer<ffi::S32>>(/*mask*/)
    .Ret<ffi::Buffer<ffi::F64>>(/*result_out*/)
    .Attr<int>("i_angular")
    .Attr<int>("j_angular")
    .Attr<int>("is_screened")
    .Attr<int>("n_pairs")
    .Attr<int>("n_primitives")
    .Attr<int>("n_functions")
    .Attr<int>("atm_stride")
    .Attr<int>("bas_stride")
    .Attr<int>("env_stride")
    .Attr<int>("n_images")
    .Attr<int>("reduce_over_images")
);

} // namespace cuint
} // namespace pyscfad
