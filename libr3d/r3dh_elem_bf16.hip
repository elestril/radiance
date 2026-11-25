/* mul_weight and logit_softcap: libhalo's kernels and shims (libhalo/halo_model_bf16.hip), taken
 * out of that unit with the halo_ -> r3dh_ rename. The rest of halo_model_bf16.hip duplicates ops
 * libr3d already serves (r3d_model_bf16.hip), so only these two came over. The chained forms are
 * left out: libr3d builds no chained form (README, "No collective rows and no chained forms").
 *
 *   mul_weight     y = x * w, w one element (a scalar for the whole op) or one row of n
 *                  broadcast down the rows; bf16 x and y, a bf16 or f32 w, the f32 product rounded
 *                  once (ref_mul_weight). Gemma's per-layer scalar on the residual stream.
 *   logit_softcap  y = cap * tanh(x / cap), the final-logit softcap, over bf16 or f32 logits
 *                  (ref_logit_softcap). A row is spread over up to 64 blocks, because decode has
 *                  few rows and a row is a whole vocabulary.
 */
#include "r3dh_args.h"
#include "r3dh_common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

constexpr uint32_t kThreads = 256;

__device__ __forceinline__ float ld_f32(uint16_t v) { return bf16_to_f32(v); }
__device__ __forceinline__ float ld_f32(float v) { return v; }

/* ref_mul_weight: one block a row; `w_step` 0 reads the one element for every column. */
template <typename TW>
__global__ __launch_bounds__(kThreads) void mul_weight_kernel(
    const uint16_t* __restrict__ x, const TW* __restrict__ w, uint16_t* __restrict__ y,
    uint32_t n, uint32_t x_ld, uint32_t y_ld, uint32_t w_step, uint32_t M) {
    for (uint32_t m = blockIdx.x; m < M; m += gridDim.x) {
        const uint16_t* __restrict__ xr = x + (size_t)m * x_ld;
        uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
        for (uint32_t i = threadIdx.x; i < n; i += kThreads)
            yr[i] = f32_to_bf16(bf16_to_f32(xr[i]) * ld_f32(w[(size_t)i * w_step]));
    }
}

/* ref_logit_softcap, both legs: the bf16 one rounds the result once, the f32 one not at all. */
template <typename T>
__global__ __launch_bounds__(kThreads) void logit_softcap_kernel(
    const T* __restrict__ x, T* __restrict__ y, uint32_t n, float cap, uint32_t x_ld,
    uint32_t y_ld) {
    const uint32_t m = blockIdx.x;
    const T* __restrict__ xr = x + (size_t)m * x_ld;
    T* __restrict__ yr = y + (size_t)m * y_ld;
    for (uint32_t i = blockIdx.y * kThreads + threadIdx.x; i < n; i += gridDim.y * kThreads) {
        const float v = cap * tanhf(ld_f32(xr[i]) / cap);
        if constexpr (sizeof(T) == 2) yr[i] = f32_to_bf16(v);
        else                          yr[i] = v;
    }
}

/* Blocks a softcap row: eight elements a thread, at most 64 blocks. */
uint32_t softcap_blocks_per_row(long long n) {
    long long b = (n + kThreads * 8 - 1) / (kThreads * 8);
    return (uint32_t)(b < 1 ? 1 : (b > 64 ? 64 : b));
}

}  // namespace

extern "C" int r3dh_rad_mul_weight(const RadArgs* a, RadStream s) {
    const RadTensor* X = r3dh_opt(a, 0);
    const RadTensor* W = r3dh_opt(a, 1);
    const RadTensor* Y = r3dh_opt(a, 2);
    if (!X || !W || !Y) return R3DH_NO(RAD_E_INVAL);
    if (X->dtype != RAD_BF16 || Y->dtype != RAD_BF16) return R3DH_NO(RAD_E_DTYPE);
    if (W->dtype != RAD_BF16 && W->dtype != RAD_F32) return R3DH_NO(RAD_E_DTYPE);
    const long long n = r3dh_geti_or(a, "n", r3dh_dim(Y, (int)Y->rank - 1));
    if (n <= 0) return R3DH_NO(RAD_E_SHAPE);
    long long M = 0, Mx = 0;
    const long long y_ld = r3dh_rows_pitch(Y, n, &M);
    const long long x_ld = r3dh_rows_pitch(X, n, &Mx);
    if (y_ld < 0 || x_ld < 0) return R3DH_NO(RAD_E_STRIDE);
    if (M <= 0 || Mx < M) return R3DH_NO(RAD_E_SHAPE);
    uint32_t w_step;
    if (r3dh_numel(W) == 1) w_step = 0;
    else if (r3dh_numel(W) == n && r3dh_contig(W)) w_step = 1;
    else return R3DH_NO(RAD_E_SHAPE);
    const dim3 grid((uint32_t)std::min<long long>(M, 65535));
    if (W->dtype == RAD_F32)
        hipLaunchKernelGGL(mul_weight_kernel<float>, grid, dim3(kThreads), 0, (hipStream_t)s,
                           (const uint16_t*)X->data, (const float*)W->data, (uint16_t*)Y->data,
                           (uint32_t)n, (uint32_t)x_ld, (uint32_t)y_ld, w_step, (uint32_t)M);
    else
        hipLaunchKernelGGL(mul_weight_kernel<uint16_t>, grid, dim3(kThreads), 0, (hipStream_t)s,
                           (const uint16_t*)X->data, (const uint16_t*)W->data, (uint16_t*)Y->data,
                           (uint32_t)n, (uint32_t)x_ld, (uint32_t)y_ld, w_step, (uint32_t)M);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int r3dh_rad_logit_softcap(const RadArgs* a, RadStream s) {
    const RadTensor* x = r3dh_opt(a, 0);
    const RadTensor* y = r3dh_opt(a, 1);
    if (!x || !y) return R3DH_NO(RAD_E_INVAL);
    const bool leg_bf16 = x->dtype == RAD_BF16 && y->dtype == RAD_BF16;
    const bool leg_f32  = x->dtype == RAD_F32 && y->dtype == RAD_F32;
    if (!leg_bf16 && !leg_f32) return R3DH_NO(RAD_E_DTYPE);
    /* A zero cap would divide by zero in every lane. */
    const float cap = (float)r3dh_getf(a, "cap", 30.0);
    if (!(cap > 0.0f)) return R3DH_NO(RAD_E_INVAL);
    const long long n = r3dh_geti_or(a, "n", r3dh_dim(y, (int)y->rank - 1));
    if (n <= 0) return R3DH_NO(RAD_E_SHAPE);
    long long M = 0, My = 0;
    const long long x_ld = r3dh_rows_pitch(x, n, &M);
    const long long y_ld = r3dh_rows_pitch(y, n, &My);
    if (x_ld < 0 || y_ld < 0) return R3DH_NO(RAD_E_STRIDE);
    if (M <= 0 || My < M || M > 0x7fffffff) return R3DH_NO(RAD_E_SHAPE);
    const dim3 grid((uint32_t)M, softcap_blocks_per_row(n));
    if (leg_bf16)
        hipLaunchKernelGGL(logit_softcap_kernel<uint16_t>, grid, dim3(kThreads), 0,
                           (hipStream_t)s, (const uint16_t*)x->data, (uint16_t*)y->data,
                           (uint32_t)n, cap, (uint32_t)x_ld, (uint32_t)y_ld);
    else
        hipLaunchKernelGGL(logit_softcap_kernel<float>, grid, dim3(kThreads), 0, (hipStream_t)s,
                           (const float*)x->data, (float*)y->data, (uint32_t)n, cap,
                           (uint32_t)x_ld, (uint32_t)y_ld);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
