/* mul_weight, logit_softcap, rmsnorm_p, gelu_mul and kv_store_i8: libhalo's kernels and shims
 * (libhalo/halo_model_bf16.hip), taken out of that unit with the halo_ -> r3dh_ rename. The rest
 * of halo_model_bf16.hip duplicates ops libr3d already serves (r3d_model_bf16.hip), so only these
 * five came over.
 * libhalo's chained forms are left out; logit_softcap has libr3d's own (r3d_chain.h, marked
 * `libr3d:`).
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
 *   kv_store_i8    k and v into the paged int8 cache (`kv-cache-dtype: i8`): a head's K and its V
 *                  as int8 codes with one f16 scale each (see the kernel's comment).
 */
#include "r3dh_args.h"
#include "r3dh_common.h"
#include "r3d_chain.h"   /* libr3d: the soft-cap's chained form */

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
template <typename T, class CH = HaloUnchained>
__global__ __launch_bounds__(kThreads) void logit_softcap_kernel(
    const T* __restrict__ x, T* __restrict__ y, uint32_t n, float cap, uint32_t x_ld,
    uint32_t y_ld) {
    R3D_CHAIN_SCOPE(CH);   /* libr3d: the chained form's bracket; no return, no barrier */
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
    HaloChainScope chain_(r3dh_chain_small(a));   /* libr3d: the chained form at decode heights */
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
        R3D_CHAIN_LAUNCH(logit_softcap_kernel, (uint16_t,), grid, dim3(kThreads), (hipStream_t)s,
                         (const uint16_t*)x->data, (uint16_t*)y->data, (uint32_t)n, cap,
                         (uint32_t)x_ld, (uint32_t)y_ld);
    else
        R3D_CHAIN_LAUNCH(logit_softcap_kernel, (float,), grid, dim3(kThreads), (hipStream_t)s,
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

/* ------------------------------------------------------------------ kv_store into the int8 cache
 * libhalo's kv_store_i8 (halo_model_bf16.hip), for `kv-cache-dtype: i8`: a head's K and its V each
 * as int8 codes against one f16 scale, the head's absmax / 127, written after the codes. One
 * workgroup a (token, head) and one dim a lane; the scale is rounded to f16 first and the codes are
 * taken against the rounded value, clamped so 127 times it stays an f16; an all-zero head stores
 * scale 0 and codes 0. The chained form is left out. */
namespace {
struct KvI8 { uint8_t k, v; uint16_t ks, vs; };
__device__ __forceinline__ KvI8 kv_i8_quant(const uint16_t* __restrict__ k,
                                            const uint16_t* __restrict__ v, uint32_t head_dim,
                                            uint32_t k_ld, uint32_t v_ld, float (*s_m)[32]) {
    const uint32_t t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const bool in = d < head_dim;
    const float kf = in ? bf16_to_f32(k[(size_t)t * k_ld + (size_t)h * head_dim + d]) : 0.0f;
    const float vf = in ? bf16_to_f32(v[(size_t)t * v_ld + (size_t)h * head_dim + d]) : 0.0f;
    float km = r3dh_wave_max(fabsf(kf)), vm = r3dh_wave_max(fabsf(vf));
    if ((d & 31u) == 0u) { s_m[0][d >> 5] = km; s_m[1][d >> 5] = vm; }
    __syncthreads();
    km = 0.0f; vm = 0.0f;
    for (uint32_t w = 0; w < blockDim.x / 32u; ++w) {
        km = fmaxf(km, s_m[0][w]);
        vm = fmaxf(vm, s_m[1][w]);
    }
    const __fp16 k16 = (__fp16)fminf(km / 127.0f, 515.0f);
    const __fp16 v16 = (__fp16)fminf(vm / 127.0f, 515.0f);
    const float ksc = (float)k16, vsc = (float)v16;
    const float kr = ksc > 0.0f ? rintf(kf / ksc) : 0.0f;
    const float vr = vsc > 0.0f ? rintf(vf / vsc) : 0.0f;
    KvI8 q;
    q.k  = (uint8_t)(int8_t)(int)fminf(fmaxf(kr, -127.0f), 127.0f);
    q.v  = (uint8_t)(int8_t)(int)fminf(fmaxf(vr, -127.0f), 127.0f);
    q.ks = __builtin_bit_cast(uint16_t, k16);
    q.vs = __builtin_bit_cast(uint16_t, v16);
    return q;
}
__global__ __launch_bounds__(1024) void kv_store_i8_kernel(
    const uint16_t* __restrict__ k, const uint16_t* __restrict__ v,
    const int32_t* __restrict__ slot, uint8_t* __restrict__ kv, uint32_t head_dim,
    uint32_t kv_heads, uint32_t block_size, uint32_t n_blocks, uint32_t k_ld, uint32_t v_ld) {
    __shared__ float s_m[2][32];
    const uint32_t t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const KvI8 q = kv_i8_quant(k, v, head_dim, k_ld, v_ld, s_m);
    const int32_t s = slot[t];
    if (d >= head_dim || s < 0 || (uint32_t)s / block_size >= n_blocks) return;
    const uint32_t blk = (uint32_t)s / block_size, off = (uint32_t)s % block_size;
    const size_t row = 2 * (size_t)head_dim + 4;
    uint8_t* c = kv + (((size_t)blk * kv_heads + h) * block_size + off) * row;
    c[d]            = q.k;
    c[head_dim + d] = q.v;
    if (d == 0u) {
        *(uint16_t*)(c + 2 * head_dim)     = q.ks;
        *(uint16_t*)(c + 2 * head_dim + 2) = q.vs;
    }
}
}  // namespace

/* Operands k, v, slot_mapping, kv_cache (libref's kv_store); the cache exactly
 * [n_blocks, kv_heads, block_size, 2 * head_dim + 4] int8. */
extern "C" int r3dh_rad_kv_store_i8(const RadArgs* a, RadStream s) {
    const RadTensor *k = r3dh_opt(a, 0), *v = r3dh_opt(a, 1), *sm = r3dh_opt(a, 2),
                    *kv = r3dh_opt(a, 3);
    if (!k || !v || !sm || !kv) return R3DH_NO(RAD_E_INVAL);
    if (k->dtype != RAD_BF16 || v->dtype != RAD_BF16 || sm->dtype != RAD_I32 ||
        kv->dtype != RAD_I8) return R3DH_NO(RAD_E_DTYPE);
    if (!r3dh_contig(sm) || !r3dh_contig(kv)) return R3DH_NO(RAD_E_STRIDE);
    const long long hd = r3dh_geti_or(a, "head_dim", r3dh_dim(k, (int)k->rank - 1));
    const long long bs = r3dh_geti_or(a, "block_size", 0);
    const long long kvh = r3dh_geti_or(a, "n_head_kv", kv->rank == 4 ? kv->shape[1] : 0);
    if (hd <= 0 || bs <= 0 || kvh <= 0 || hd % 4 || hd > 1024) return R3DH_NO(RAD_E_SHAPE);
    if (kv->rank != 4 || kv->shape[1] != kvh || kv->shape[2] != bs || kv->shape[3] != 2 * hd + 4)
        return R3DH_NO(RAD_E_SHAPE);
    long long M = 0, Mv = 0;
    const long long k_ld = r3dh_rows_pitch(k, kvh * hd, &M);
    const long long v_ld = r3dh_rows_pitch(v, kvh * hd, &Mv);
    if (k_ld < 0 || v_ld < 0) return R3DH_NO(RAD_E_STRIDE);
    const long long T = r3dh_numel(sm);
    if (T <= 0 || M < T || Mv < T || T > 0x7fffffff) return R3DH_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(kv_store_i8_kernel, dim3((uint32_t)T, (uint32_t)kvh),
                       dim3((uint32_t)((hd + 31) / 32 * 32)), 0, (hipStream_t)s,
                       (const uint16_t*)k->data, (const uint16_t*)v->data,
                       (const int32_t*)sm->data, (uint8_t*)kv->data, (uint32_t)hd, (uint32_t)kvh,
                       (uint32_t)bs, (uint32_t)kv->shape[0], (uint32_t)k_ld, (uint32_t)v_ld);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
