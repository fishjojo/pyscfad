#include "pyscfadlib/cuda/handle_pool.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

namespace pyscfad {
namespace cuda {

namespace {

// Opt-in cuSOLVER FP32/FP64 emulation, configured by environment variables and applied once
// when a pooled handle is created (see ConfigureEmulation). Parsed once and cached.
enum class EmuStrategy { kDefault, kPerformant, kEager };

struct EmuConfig {
    bool fp32 = false;
    bool fp64 = false;
    EmuStrategy strategy = EmuStrategy::kDefault;
};

bool EnvTruthy(const char* v) {
    if (!v) {
        return false;
    }
    std::string s(v);
    for (auto& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s == "1" || s == "true" || s == "on" || s == "yes";
}

const EmuConfig& EmulationConfig() {
    static const EmuConfig cfg = [] {
        EmuConfig c;
        c.fp32 = EnvTruthy(std::getenv("PYSCFADLIB_CUSOLVER_FP32_EMULATION"));
        c.fp64 = EnvTruthy(std::getenv("PYSCFADLIB_CUSOLVER_FP64_EMULATION"));
        const char* s = std::getenv("PYSCFADLIB_CUSOLVER_EMULATION_STRATEGY");
        std::string sv = s ? std::string(s) : "";
        for (auto& ch : sv) {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }
        c.strategy = (sv == "performant") ? EmuStrategy::kPerformant
                   : (sv == "eager")      ? EmuStrategy::kEager
                   : EmuStrategy::kDefault;
        return c;
    }();
    return cfg;
}

void WarnOnce(const char* msg) {
    static std::once_flag flag;
    std::call_once(flag, [&] { std::fprintf(stderr, "[pyscfadlib] %s\n", msg); });
}

#if defined(PYSCFAD_HAVE_CUSOLVER_EMULATION)  // CUDA >= 13.0
// Maps the version-agnostic strategy onto the CUDA enum (native values, no casting).
cudaEmulationStrategy_t EmulationStrategy(EmuStrategy s) {
    switch (s) {
        case EmuStrategy::kPerformant: return CUDA_EMULATION_STRATEGY_PERFORMANT;
        case EmuStrategy::kEager:      return CUDA_EMULATION_STRATEGY_EAGER;
        case EmuStrategy::kDefault:    break;
    }
    return CUDA_EMULATION_STRATEGY_DEFAULT;
}
#endif

// Sets the cuSOLVER math mode + emulation strategy from the cached env config, once when a
// handle is created. The mode is dtype-independent: cuSOLVER applies each emulation only to
// routines of the matching precision and runs the rest natively, so a single pooled handle
// correctly serves both FP32 and FP64 solves (combined mode when both are on). Makes no API
// calls when emulation is off, preserving the native path. Non-fatal on failure (warns once).
void ConfigureEmulation(cusolverDnHandle_t handle) {
#if defined(PYSCFAD_HAVE_CUSOLVER_EMULATION)  // CUDA >= 13.0: math-mode/strategy API + FP32 BF16x9
    const EmuConfig& cfg = EmulationConfig();
    cusolverMathMode_t mode = CUSOLVER_DEFAULT_MATH;
    bool emulate = false;
#if defined(PYSCFAD_HAVE_CUSOLVER_FP64_EMULATION)  // CUDA >= 13.2: FP32, FP64 and combined modes
    if (cfg.fp32 && cfg.fp64) {
        mode = CUSOLVER_FP32_FP64_EMULATED_MATH;
        emulate = true;
    } else if (cfg.fp32) {
        mode = CUSOLVER_FP32_EMULATED_BF16X9_MATH;
        emulate = true;
    } else if (cfg.fp64) {
        mode = CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH;
        emulate = true;
    }
#else  // CUDA 13.0/13.1: only FP32 BF16x9 emulation exists
    if (cfg.fp64) {
        WarnOnce("FP64 emulation requires CUDA >= 13.2; plugin built older. Native FP64 used.");
    }
    if (cfg.fp32) {
        mode = CUSOLVER_FP32_EMULATED_BF16X9_MATH;
        emulate = true;
    }
#endif
    if (emulate) {
        if (cusolverDnSetMathMode(handle, mode) == CUSOLVER_STATUS_SUCCESS) {
            cusolverDnSetEmulationStrategy(handle, EmulationStrategy(cfg.strategy));  // best-effort
        } else {
            WarnOnce("cusolverDnSetMathMode failed; emulation ignored for this handle.");
        }
    }
#else  // cuda12 / CUDA < 13.0: no math-mode API
    const EmuConfig& cfg = EmulationConfig();
    if (cfg.fp32 || cfg.fp64) {
        WarnOnce("PYSCFADLIB_CUSOLVER_FP32/FP64_EMULATION set, but this plugin was built "
                 "against CUDA < 13.0; ignoring (native precision used).");
    }
    (void)handle;
#endif
}

}  // namespace

template <>
SolverHandlePool::Handle SolverHandlePool::Borrow(cudaStream_t stream) {
    SolverHandlePool* pool = Instance();
    std::lock_guard<std::mutex> lock(pool->mu_);

    cusolverDnHandle_t handle;
    auto& free_handles = pool->handles_[stream];
    if (free_handles.empty()) {
        cusolverStatus_t status = cusolverDnCreate(&handle);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            throw std::runtime_error(
                "cusolverDnCreate failed with status " +
                std::to_string(static_cast<int>(status)));
        }
        // Configure emulation once, at creation: the env-driven config is process-global,
        // so a handle later reused from the pool already carries the right math mode.
        ConfigureEmulation(handle);
    } else {
        handle = free_handles.back();
        free_handles.pop_back();
    }

    if (stream) {
        cusolverStatus_t status = cusolverDnSetStream(handle, stream);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            throw std::runtime_error(
                "cusolverDnSetStream failed with status " +
                std::to_string(static_cast<int>(status)));
        }
    }

    return Handle(pool, handle, stream);
}

} // namespace cuda
} // namespace pyscfad
