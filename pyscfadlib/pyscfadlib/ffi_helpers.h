#ifndef PYSCFADLIB_FFI_HELPERS_H_
#define PYSCFADLIB_FFI_HELPERS_H_

#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace pyscfad {

template <typename T>
inline T MaybeCastNoOverflow(std::int64_t value)
{
    if constexpr (sizeof(T) == sizeof(std::int64_t)) {
        return value;
    } else {
        if (value > std::numeric_limits<T>::max()) [[unlikely]] {
            throw std::runtime_error("overflow when casting " +
                                     std::to_string(value) + " to " +
                                     typeid(T).name());
        }
        return static_cast<T>(value);
    }
}

template <::xla::ffi::DataType dtype>
auto AllocateScratchMemory(std::size_t size)
    -> std::unique_ptr<std::remove_extent_t<::xla::ffi::NativeType<dtype>>[]>
{
    using ValueType = std::remove_extent_t<::xla::ffi::NativeType<dtype>>;
    return std::unique_ptr<ValueType[]>(new ValueType[size]);
}

template <typename T>
inline auto AllocateWorkspace(::xla::ffi::ScratchAllocator& scratch,
                              int64_t size, std::string_view name)
{
    auto maybe_workspace = scratch.Allocate(sizeof(T) * size);
    if (!maybe_workspace.has_value()) {
        throw std::runtime_error("Unable to allocate workspace for " +
                                 std::string(name));
    }
    return static_cast<T*>(maybe_workspace.value());
}

inline int64_t GetBatchSize(::xla::ffi::Span<const int64_t> dims) {
    return std::accumulate(dims.begin(), dims.end(),
                           1LL, std::multiplies<int64_t>());
}

inline std::tuple<int64_t, int64_t, int64_t> SplitBatch2D(
    ::xla::ffi::Span<const int64_t> dims)
{
    auto trailingDims = dims.last(2);
    return std::make_tuple(GetBatchSize(dims.first(dims.size() - 2)),
                         trailingDims.front(), trailingDims.back());
}

} //namespace pyscfad

#endif
