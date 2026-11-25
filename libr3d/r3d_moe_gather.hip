/* r3d_moe_gather.hip -- moe_gather / moe_gather_inv: fold the expert outputs back to the token.
 *
 *   y[t, c] = sum_j expert_w[t, j] * y_expert[slot(t, j), c]
 *
 * summed in ascending slot order (reproducibility: the expert-sorted order would differ in the last
 * bits). One block per token; the slot table (slot(t, j) and the weight) is hoisted to LDS so the
 * per-column `ye` loads pipeline.
 *
 * Two arms:
 *   - moe_gather: `slot` is the inverse of `sorted_tok`; the block scans `sorted` for its token's
 *     slots (the self-scan form, no prepass).
 *   - moe_gather_inv: `slot` is the `inverse` table moe_scatter_inv wrote (read straight).
 *
 * An optional shared arm adds `shared[t, c] * act(gate[t])` (two narrows, matching the unfused
 * scale_rows pair). `act` is "sigmoid" or "none" (the gate as it is; absent = the
 * identity).
 *
 * The wide-prefill form (a wave per token, 8 cols/lane) is not implemented; this is the decode and
 * short-pass form (one block per token), which covers every height the engine routes here.
 */
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_codecs.h"
#include "r3d_launch.h"

namespace {

constexpr uint32_t kGatherThr = 256;
constexpr uint32_t kGatherSlots = 64;   // top_k <= 64 for the LDS-cached slot table

template <bool F32W>
__device__ __forceinline__ float gather_weight(const void* ew, size_t i) {
    if constexpr (F32W) return ((const float*)ew)[i];
    else return bf16_to_f32(((const uint16_t*)ew)[i]);
}

/* One enum arg (not two bools) so the launch macro sees a single, comma-free template argument. */
enum class Gk { bf16_sorted, bf16_inv, f32_sorted, f32_inv };

template <Gk MODE>
__global__ __launch_bounds__(kGatherThr) void gather_kernel(
    const uint16_t* __restrict__ ye, const void* __restrict__ ew,
    const int32_t* __restrict__ inv, uint16_t* __restrict__ y,
    uint32_t M, uint32_t n, uint32_t top_k, uint32_t ye_ld, uint32_t ew_ld, uint32_t y_ld,
    const int32_t* __restrict__ sorted, uint32_t T,
    const uint16_t* __restrict__ sh, const uint16_t* __restrict__ sg,
    uint32_t sh_ld, uint32_t sg_ld, int sig) {
    constexpr bool F32W = (MODE == Gk::f32_sorted || MODE == Gk::f32_inv);
    constexpr bool GIVEN = (MODE == Gk::bf16_inv || MODE == Gk::f32_inv);
    __shared__ int32_t s_i[kGatherSlots];
    __shared__ float s_w[kGatherSlots];
    const uint32_t t = blockIdx.x;
    if (t >= M) return;
    const bool cached = top_k <= kGatherSlots;

    float gate = 0.0f;
    if (sh) {
        const float v0 = bf16_to_f32(sg[(size_t)t * sg_ld]);
        gate = sig ? (1.0f / (1.0f + __expf(-v0))) : v0;
    }

    if (!GIVEN) {
        /* The self-scan form: scan `sorted` for this token's slots. An unclaimed slot reads -1, so
         * the fill and the scan are separated by a barrier. */
        for (uint32_t j = threadIdx.x; j < top_k; j += kGatherThr) s_i[j] = -1;
        __syncthreads();
        const int32_t lo = (int32_t)(t * top_k), hi = lo + (int32_t)top_k;
        for (uint32_t i = threadIdx.x; i < T; i += kGatherThr) {
            const int32_t f = sorted[i];
            if (f >= lo && f < hi) s_i[f - lo] = (int32_t)i;
        }
        for (uint32_t j = threadIdx.x; j < top_k; j += kGatherThr)
            s_w[j] = gather_weight<F32W>(ew, (size_t)t * ew_ld + j);
        __syncthreads();
    } else if (cached) {
        for (uint32_t j = threadIdx.x; j < top_k; j += kGatherThr) {
            s_i[j] = inv[t * top_k + j];
            s_w[j] = gather_weight<F32W>(ew, (size_t)t * ew_ld + j);
        }
        __syncthreads();
    }

    for (uint32_t c = blockIdx.y * kGatherThr + threadIdx.x; c < n; c += kGatherThr * gridDim.y) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < top_k; ++j) {
            const int32_t i = cached ? s_i[j] : inv[t * top_k + j];
            if (i < 0) continue;
            const float wj = cached ? s_w[j] : gather_weight<F32W>(ew, (size_t)t * ew_ld + j);
            acc += wj * bf16_to_f32(ye[(size_t)i * ye_ld + c]);
        }
        const uint16_t g = f32_to_bf16(acc);
        if (!sh) { y[(size_t)t * y_ld + c] = g; continue; }
        const uint16_t p = f32_to_bf16(bf16_to_f32(sh[(size_t)t * sh_ld + c]) * gate);
        y[(size_t)t * y_ld + c] = f32_to_bf16(bf16_to_f32(p) + bf16_to_f32(g));
    }
}


/* The prefill form of the self-scan (moe_gather): a block TPB tokens by 512 columns. The decode
 * form above is a block a (token, 256 columns) and each block scans all of `sorted` for its
 * token: at a 2048-token chunk of the 26B-A4B (16384 slots, 2816 columns) that is 11 x 2048 scans
 * of 64 KB, 1.0 ms a launch for 100 MB of gather. Here one scan serves TPB tokens and twice the
 * columns, a thread a column pair (one 4-byte load a slot); each output is the same sum in the
 * same ascending slot order, so the bytes are the decode form's. */
template <bool F32W, uint32_t TPB>
__global__ __launch_bounds__(kGatherThr) void gather_pf_kernel(
    const uint16_t* __restrict__ ye, const void* __restrict__ ew, uint16_t* __restrict__ y,
    uint32_t M, uint32_t n, uint32_t top_k, uint32_t ye_ld, uint32_t ew_ld, uint32_t y_ld,
    const int32_t* __restrict__ sorted, uint32_t T,
    const uint16_t* __restrict__ sh, const uint16_t* __restrict__ sg,
    uint32_t sh_ld, uint32_t sg_ld, int sig) {
    __shared__ int32_t s_i[TPB * kGatherSlots];
    __shared__ float s_w[TPB * kGatherSlots];
    const uint32_t t0 = blockIdx.x * TPB;
    const uint32_t nt = M - t0 < TPB ? M - t0 : TPB;
    for (uint32_t j = threadIdx.x; j < nt * top_k; j += kGatherThr) {
        s_i[j] = -1;
        s_w[j] = gather_weight<F32W>(ew, (size_t)(t0 + j / top_k) * ew_ld + j % top_k);
    }
    __syncthreads();
    const int32_t lo = (int32_t)(t0 * top_k), hi = (int32_t)((t0 + nt) * top_k);
    for (uint32_t i = threadIdx.x; i < T; i += kGatherThr) {
        const int32_t f = sorted[i];
        if (f >= lo && f < hi) s_i[f - lo] = (int32_t)i;
    }
    __syncthreads();
    for (uint32_t tt = 0; tt < nt; ++tt) {
        const uint32_t t = t0 + tt;
        float gate = 0.0f;
        if (sh) {
            const float v0 = bf16_to_f32(sg[(size_t)t * sg_ld]);
            gate = sig ? (1.0f / (1.0f + __expf(-v0))) : v0;
        }
        const uint32_t c = blockIdx.y * 2 * kGatherThr + 2 * threadIdx.x;   /* a column pair */
        if (c < n) {
            float acc0 = 0.0f, acc1 = 0.0f;
            for (uint32_t j = 0; j < top_k; ++j) {
                const int32_t i = s_i[tt * top_k + j];
                if (i < 0) continue;
                const float wj = s_w[tt * top_k + j];
                const uint32_t v = *(const uint32_t*)(ye + (size_t)i * ye_ld + c);
                acc0 += wj * bf16_to_f32((uint16_t)(v & 0xffffu));
                acc1 += wj * bf16_to_f32((uint16_t)(v >> 16));
            }
            uint16_t g0 = f32_to_bf16(acc0), g1 = f32_to_bf16(acc1);
            if (sh) {
                const uint16_t p0 = f32_to_bf16(bf16_to_f32(sh[(size_t)t * sh_ld + c]) * gate);
                const uint16_t p1 = f32_to_bf16(bf16_to_f32(sh[(size_t)t * sh_ld + c + 1]) * gate);
                g0 = f32_to_bf16(bf16_to_f32(p0) + bf16_to_f32(g0));
                g1 = f32_to_bf16(bf16_to_f32(p1) + bf16_to_f32(g1));
            }
            *(uint32_t*)(y + (size_t)t * y_ld + c) = (uint32_t)g0 | ((uint32_t)g1 << 16);
        }
    }
}

}  // namespace

extern "C" {

int r3d_rad_moe_gather(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 3) return R3D_NO(RAD_E_INVAL);
    const RadTensor* ye = r3d_opt(a, 0), *ew = r3d_opt(a, 1), *sorted = r3d_opt(a, 2),
                    *y = r3d_opt(a, 5);
    if (!r3d_p(ye) || !r3d_p(ew) || !r3d_p(sorted) || !r3d_wp(y)) return R3D_NO(RAD_E_INVAL);
    if (ye->dtype != RAD_BF16 || y->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);
    if (ew->dtype != RAD_BF16 && ew->dtype != RAD_F32) return R3D_NO(RAD_E_DTYPE);
    if (sorted->dtype != RAD_I32) return R3D_NO(RAD_E_DTYPE);

    const long long M = r3d_geti_or(a, "M", 0), n = r3d_geti_or(a, "n", 0),
                    top_k = r3d_geti_or(a, "top_k", 0);
    if (M <= 0 || n <= 0 || top_k <= 0) return R3D_NO(RAD_E_SHAPE);
    if (top_k > kGatherSlots) return R3D_NO(RAD_E_SHAPE);   // the LDS-cached form only

    // Row pitches are the tensors' own strides: a view's rows can be wider than its shape.
    const uint32_t ye_ld = (uint32_t)ye->stride[0];
    const uint32_t ew_ld = (uint32_t)ew->stride[0];
    const uint32_t y_ld = (uint32_t)y->stride[0];
    const uint32_t T = (uint32_t)r3d_numel(sorted);
    /* M is the batch capacity, not the token count: the engine pads the batch, and the slot
     * table covers only the tokens present, so it may be shorter than M * top_k (the kernel
     * zeroes the padding rows). It may never be longer. */
    if (T > (uint64_t)M * top_k) return R3D_NO(RAD_E_SHAPE);
    /* The launch covers y's rows, not the band's capacity: a decode step issues one row against a
     * band whose M is the step's ceiling, and a block a padding row only zeroes it. Rows of y past
     * its view are not y's. */
    const long long y_rows = y->rank >= 1 ? y->shape[0] : M;
    const long long Mg = y_rows < M ? y_rows : M;

    // The optional shared arm (operands 3 and 4, present together).
    const RadTensor* sh = r3d_opt(a, 3), *sg = r3d_opt(a, 4);
    const uint16_t* shp = sh ? (const uint16_t*)r3d_p(sh) : nullptr;
    const uint16_t* sgp = sg ? (const uint16_t*)r3d_p(sg) : nullptr;
    if (sh && !sg) return R3D_NO(RAD_E_INVAL);
    // The gate is one value a token ([M] or [M, 1]): its pitch is stride[0], not a shape.
    const uint32_t sh_ld = sh ? (uint32_t)sh->stride[0] : 0;
    const uint32_t sg_ld = sg ? (uint32_t)sg->stride[0] : 1;
    int sig = 0;
    if (const char* act = r3d_gets(a, "act")) {
        if (std::string(act) == "sigmoid") sig = 1;
        else if (std::string(act) != "none") return R3D_NO(RAD_E_INVAL);
    }

    const bool f32w = (ew->dtype == RAD_F32);
    /* Prefill heights: the TPB-token form, one scan of `sorted` a block (column pairs, so an even
     * n and even pitches at 4-byte aligned planes). */
    constexpr uint32_t kTpb = 8;
    if (Mg >= 256 && n % 2 == 0 && ye_ld % 2 == 0 && y_ld % 2 == 0 && (!sh || sh_ld % 2 == 0) &&
        r3d_aligned(r3d_p(ye), 4) && r3d_aligned(r3d_wp(y), 4)) {
        const dim3 pgrid((unsigned)((Mg + kTpb - 1) / kTpb),
                         (unsigned)((n + 2 * kGatherThr - 1) / (2 * kGatherThr))), pblk(kGatherThr);
#define R3D_GPF(F32V)                                                                             \
    gather_pf_kernel<F32V, kTpb><<<pgrid, pblk, 0, (hipStream_t)s>>>(                             \
        (const uint16_t*)r3d_p(ye), r3d_p(ew), (uint16_t*)r3d_wp(y), (uint32_t)Mg, (uint32_t)n,    \
        (uint32_t)top_k, ye_ld, ew_ld, y_ld, (const int32_t*)r3d_p(sorted), T, shp, sgp, sh_ld,    \
        sg_ld, sig)
        if (f32w) R3D_GPF(true); else R3D_GPF(false);
#undef R3D_GPF
        return (int)hipGetLastError();
    }
    dim3 grid((unsigned)Mg, (unsigned)((n + kGatherThr - 1) / kGatherThr)), blk(kGatherThr);
    if (f32w)
        R3D_KLAUNCH(gather_kernel<Gk::f32_sorted>, grid, blk, (hipStream_t)s,
                    (const uint16_t*)r3d_p(ye), r3d_p(ew), (const int32_t*)r3d_p(sorted),
                    (uint16_t*)r3d_wp(y), (uint32_t)M, (uint32_t)n, (uint32_t)top_k, ye_ld, ew_ld,
                    y_ld, (const int32_t*)r3d_p(sorted), T, shp, sgp, sh_ld, sg_ld, sig);
    else
        R3D_KLAUNCH(gather_kernel<Gk::bf16_sorted>, grid, blk, (hipStream_t)s,
                    (const uint16_t*)r3d_p(ye), r3d_p(ew), (const int32_t*)r3d_p(sorted),
                    (uint16_t*)r3d_wp(y), (uint32_t)M, (uint32_t)n, (uint32_t)top_k, ye_ld, ew_ld,
                    y_ld, (const int32_t*)r3d_p(sorted), T, shp, sgp, sh_ld, sg_ld, sig);
    return (int)hipGetLastError();
}

int r3d_rad_moe_gather_inv(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 3) return R3D_NO(RAD_E_INVAL);
    const RadTensor* ye = r3d_opt(a, 0), *ew = r3d_opt(a, 1), *inv = r3d_opt(a, 2),
                    *y = r3d_opt(a, 5);
    if (!r3d_p(ye) || !r3d_p(ew) || !r3d_p(inv) || !r3d_wp(y)) return R3D_NO(RAD_E_INVAL);
    if (ye->dtype != RAD_BF16 || y->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);
    if (ew->dtype != RAD_BF16 && ew->dtype != RAD_F32) return R3D_NO(RAD_E_DTYPE);
    if (inv->dtype != RAD_I32) return R3D_NO(RAD_E_DTYPE);

    const long long M = r3d_geti_or(a, "M", 0), n = r3d_geti_or(a, "n", 0),
                    top_k = r3d_geti_or(a, "top_k", 0);
    if (M <= 0 || n <= 0 || top_k <= 0) return R3D_NO(RAD_E_SHAPE);
    if (top_k > kGatherSlots) return R3D_NO(RAD_E_SHAPE);   // the LDS-cached form only

    // Row pitches are the tensors' own strides: a view's rows can be wider than its shape.
    const uint32_t ye_ld = (uint32_t)ye->stride[0];
    const uint32_t ew_ld = (uint32_t)ew->stride[0];
    const uint32_t y_ld = (uint32_t)y->stride[0];
    const uint32_t T = (uint32_t)r3d_numel(inv);
    if (T < M * top_k) return R3D_NO(RAD_E_SHAPE);

    const RadTensor* sh = r3d_opt(a, 3), *sg = r3d_opt(a, 4);
    const uint16_t* shp = sh ? (const uint16_t*)r3d_p(sh) : nullptr;
    const uint16_t* sgp = sg ? (const uint16_t*)r3d_p(sg) : nullptr;
    if (sh && !sg) return R3D_NO(RAD_E_INVAL);
    // The gate is one value a token ([M] or [M, 1]): its pitch is stride[0], not a shape.
    const uint32_t sh_ld = sh ? (uint32_t)sh->stride[0] : 0;
    const uint32_t sg_ld = sg ? (uint32_t)sg->stride[0] : 1;
    int sig = 0;
    if (const char* act = r3d_gets(a, "act")) {
        if (std::string(act) == "sigmoid") sig = 1;
        else if (std::string(act) != "none") return R3D_NO(RAD_E_INVAL);
    }

    const bool f32w = (ew->dtype == RAD_F32);
    dim3 grid((unsigned)M, (unsigned)((n + kGatherThr - 1) / kGatherThr)), blk(kGatherThr);
    if (f32w)
        R3D_KLAUNCH(gather_kernel<Gk::f32_inv>, grid, blk, (hipStream_t)s,
                    (const uint16_t*)r3d_p(ye), r3d_p(ew), (const int32_t*)r3d_p(inv),
                    (uint16_t*)r3d_wp(y), (uint32_t)M, (uint32_t)n, (uint32_t)top_k, ye_ld, ew_ld,
                    y_ld, (const int32_t*)nullptr, T, shp, sgp, sh_ld, sg_ld, sig);
    else
        R3D_KLAUNCH(gather_kernel<Gk::bf16_inv>, grid, blk, (hipStream_t)s,
                    (const uint16_t*)r3d_p(ye), r3d_p(ew), (const int32_t*)r3d_p(inv),
                    (uint16_t*)r3d_wp(y), (uint32_t)M, (uint32_t)n, (uint32_t)top_k, ye_ld, ew_ld,
                    y_ld, (const int32_t*)nullptr, T, shp, sgp, sh_ld, sg_ld, sig);
    return (int)hipGetLastError();
}

}  // extern "C"
