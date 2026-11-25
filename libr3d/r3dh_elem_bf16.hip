/* mul_weight, logit_softcap, rmsnorm_p and gelu_mul: libhalo's kernels and shims
 * (libhalo/halo_model_bf16.hip), taken out of that unit with the halo_ -> r3dh_ rename. The rest
 * of halo_model_bf16.hip duplicates ops libr3d already serves (r3d_model_bf16.hip), so only these
 * four came over.
 * The chained forms are left out: libr3d builds no chained form (README, "No collective rows and
 * no chained forms").
 *
 *   mul_weight     y = x * w, w one element (a scalar for the whole op) or one row of n
 *                  broadcast down the rows; bf16 x and y, a bf16 or f32 w, the f32 product rounded
 *                  once (ref_mul_weight). Gemma's per-layer scalar on the residual stream.
 *   logit_softcap  y = cap * tanh(x / cap), the final-logit softcap, over bf16 or f32 logits
 *                  (ref_logit_softcap). A row is spread over up to 64 blocks, because decode has
 *                  few rows and a row is a whole vocabulary.
 *   rmsnorm_p      y = x * rsqrt(mean(x^2) + eps), no gain (ref_rmsnorm_p): Gemma 4's value norm
 *                  in the gemma4_bf16 plugin. The sum is a wave butterfly then the wave partials in
 *                  wave order, rmsnorm's tree.
 *   gelu_mul       y = gelu_tanh(gate_up[i]) * gate_up[n + i], the gate the first half
 *                  (ref_gelu_mul), the dense feed-forward's product.
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

/* ref_rmsnorm_p: one block a row, the block sum a wave butterfly and then the wave partials in
 * order. */
__global__ __launch_bounds__(kThreads) void rmsnorm_p_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ y, uint32_t n, float eps,
    uint32_t x_ld, uint32_t y_ld) {
    __shared__ float lds[kThreads / 32];
    const uint32_t m = blockIdx.x;
    const uint16_t* __restrict__ xr = x + (size_t)m * x_ld;
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
        const float v = bf16_to_f32(xr[i]);
        ss += v * v;
    }
    ss = r3dh_wave_sum(ss);
    if ((threadIdx.x & 31) == 0) lds[threadIdx.x / 32] = ss;
    __syncthreads();
    float t = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < kThreads / 32; ++w) t += lds[w];
    const float sc = rsqrtf(t / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += kThreads) yr[i] = f32_to_bf16(bf16_to_f32(xr[i]) * sc);
}

/* ref_gelu_mul: the tanh GELU of the first half times the second, one rounding. */
__global__ __launch_bounds__(kThreads) void gelu_mul_kernel(
    const uint16_t* __restrict__ gu, uint16_t* __restrict__ y, uint32_t n, uint32_t gu_ld,
    uint32_t y_ld) {
    const uint32_t m = blockIdx.x;
    const uint16_t* __restrict__ gr = gu + (size_t)m * gu_ld;
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
    for (uint32_t i = blockIdx.y * kThreads + threadIdx.x; i < n; i += gridDim.y * kThreads) {
        const float g = bf16_to_f32(gr[i]);
        const float k0 = 0.7978845608028654f, k1 = 0.044715f;
        const float gelu = 0.5f * g * (1.0f + tanhf(k0 * (g + k1 * g * g * g)));
        yr[i] = f32_to_bf16(gelu * bf16_to_f32(gr[n + i]));
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

extern "C" int r3dh_rad_rmsnorm_p(const RadArgs* a, RadStream s) {
    const RadTensor* x = r3dh_opt(a, 0);
    const RadTensor* y = r3dh_opt(a, 1);
    if (!x || !y) return R3DH_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || y->dtype != RAD_BF16) return R3DH_NO(RAD_E_DTYPE);
    const long long n = r3dh_geti_or(a, "n", r3dh_dim(x, (int)x->rank - 1));
    if (n <= 0) return R3DH_NO(RAD_E_SHAPE);
    long long M = 0, My = 0;
    const long long x_ld = r3dh_rows_pitch(x, n, &M);
    const long long y_ld = r3dh_rows_pitch(y, n, &My);
    if (x_ld < 0 || y_ld < 0) return R3DH_NO(RAD_E_STRIDE);
    if (M <= 0 || My < M || M > 0x7fffffff) return R3DH_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(rmsnorm_p_kernel, dim3((uint32_t)M), dim3(kThreads), 0, (hipStream_t)s,
                       (const uint16_t*)x->data, (uint16_t*)y->data, (uint32_t)n,
                       (float)r3dh_getf(a, "eps", 1e-6), (uint32_t)x_ld, (uint32_t)y_ld);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int r3dh_rad_gelu_mul(const RadArgs* a, RadStream s) {
    const RadTensor* gu = r3dh_opt(a, 0);
    const RadTensor* y  = r3dh_opt(a, 1);
    if (!gu || !y) return R3DH_NO(RAD_E_INVAL);
    if (gu->dtype != RAD_BF16 || y->dtype != RAD_BF16) return R3DH_NO(RAD_E_DTYPE);
    const long long n = r3dh_geti_or(a, "n", r3dh_dim(y, (int)y->rank - 1));
    if (n <= 0) return R3DH_NO(RAD_E_SHAPE);
    long long M = 0, Mg = 0;
    const long long y_ld  = r3dh_rows_pitch(y, n, &M);
    const long long gu_ld = r3dh_rows_pitch(gu, 2 * n, &Mg);
    if (y_ld < 0 || gu_ld < 0) return R3DH_NO(RAD_E_STRIDE);
    if (M <= 0 || Mg < M || M > 0x7fffffff) return R3DH_NO(RAD_E_SHAPE);
    const uint32_t by = (uint32_t)std::min<long long>((n + kThreads * 8 - 1) / (kThreads * 8), 64);
    hipLaunchKernelGGL(gelu_mul_kernel, dim3((uint32_t)M, by < 1 ? 1 : by), dim3(kThreads), 0,
                       (hipStream_t)s, (const uint16_t*)gu->data, (uint16_t*)y->data, (uint32_t)n,
                       (uint32_t)gu_ld, (uint32_t)y_ld);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
