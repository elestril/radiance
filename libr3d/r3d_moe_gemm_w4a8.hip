/* r3d_moe_gemm_w4a8.hip -- moe_gemm_q dtype w4a8: the grouped int4 GEMM over the expert-sorted rows.
 *
 *   y[T,N] = a[sorted_tok[i] / top_k, :] @ w[e(i)]^T, with a scale fold per (row, weight group).
 *
 * The routed rows are in SORTED order (one row per (token, expert slot)); row i reads its
 * activation from `a[sorted_tok[i] / top_k]` (or a[i] under a_order "sorted") and its weight from
 * expert e(i)'s table, where e(i) is the expert owning sorted row i (the one whose [off, off') span
 * contains i). The experts are weight tables (RAD_OPD_WTAB, r3d_moe_tab.h): the even experts in
 * `w` / `w_scale`, the odd in `w_odd` / `w_odd_scale`, expert e at entry e/2 of class e%2, resolved
 * on the device (a pointer array or a stacked plane).
 *
 * The weight is two's-complement int4 (nibbles, -8..7) packed two per byte, with one bf16 scale per
 * (row, weight scale group of K: `group` 128, or 32); the activation's scale is per (row, 128 of
 * K). The output is bf16 in sorted order. Two arms, picked by the activation dtype:
 *   * int8 a  -> the native int8 wave-matrix (v_wmma_i32_16x16x16_iu8, exact i32 accumulate): the
 *     int4 weight is widened to int8, so no f16 rounding anywhere in the chain.
 *   * E4M3 a  -> the f16 wave-matrix (v_wmma_f32_16x16x16_f16, f32 accumulate): the E4M3 activation
 *     and the int4 weight are decoded exactly to f16 (libr4d's w4a8 form).
 * Each weight group of K is accumulated raw, then folded by (sub * scale_a[src][b]) *
 * scale_w[e][n][g] in f32, b the group's 128-wide block of K -- the exact order the libref
 * reference runs, so the int8 arm agrees bit for bit.
 *
 * A 16-row block walks the expert runs in its rows; where the grid is small (decode) the runs are
 * spread over grid.z, one block a run. An E4M3 activation takes the decode GEMV up to 32 sorted
 * rows and the run-aligned prefill tile above (both below). STATUS: the w4a8 form is served (both
 * activation arms). Remaining: the codebook NL forms (w4nl64a8h / w4nl32a8h / w5nl64a8h,
 * magnitude-table decode), the int8 arm's prefill tile, and the SK/UW/RS tuning.
 */
#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_launch.h"
#include "r3d_codecs.h"
#include "r3d_fp8_dequant.h"
#include "r3d_moe_tab.h"

namespace {

constexpr int kThreads = 256;           // eight waves
constexpr int kWaves   = kThreads / kWave;
constexpr int kGroup   = 128;           // a scale block: one k tile, eight wmma steps
constexpr int kPitch   = 33;            // dwords a staged row: 33 mod 32 = 1, bank-free


__device__ __forceinline__ int32_t nib_i8(uint32_t nib)
{
    return (nib & 8u) ? (int32_t)nib - 16 : (int32_t)(nib & 7u);
}


__device__ __forceinline__ float bf16_f32(uint16_t h) { return __builtin_bit_cast(float, uint32_t(h) << 16); }

__device__ __forceinline__ _Float16 h16(uint32_t w) { return __builtin_bit_cast(_Float16, (uint16_t)w); }

// The block's rows' source tokens + the expert runs, shared by both arms. `a_sorted` selects the
// gather (token: sorted row i reads a[sorted[i]/top_k]; sorted: a[i]). -1 marks a dropped slot.
__device__ __forceinline__ void moe_w4a8_setup(
    const int32_t* __restrict__ sorted, const int32_t* __restrict__ off, uint32_t r0, uint32_t r1,
    uint32_t tid, int T, int n_expert, int top_k, int a_sorted, int32_t* s_tok, int32_t* s_nruns,
    int32_t* s_rs, int32_t* s_re, int32_t* s_e)
{
    if (tid < 16) {
        const uint32_t i = r0 + tid;
        const int32_t f = (i < (uint32_t)T) ? sorted[i] : -1;
        s_tok[tid] = (f < 0) ? -1 : (a_sorted ? (int32_t)i : f / top_k);
    }
    if (tid == 0) {
        int nruns = 0, i = (int)r0;
        while (i < (int)r1) {
            int32_t ee = -1;
            for (int32_t x = 0; x < n_expert; ++x) if (off[x] <= i && i < off[x + 1]) ee = x;
            const int re = (ee < 0) ? (int)r1 : (int)(off[ee + 1] < r1 ? off[ee + 1] : r1);
            s_rs[nruns] = i; s_re[nruns] = re; s_e[nruns] = ee;
            ++nruns; i = re;
        }
        *s_nruns = nruns;
    }
    group_barrier();
}

/* ------------------------------------------------------------------ int8 arm (i32 wave-matrix) */

template <int MR, int WV, int WG>
__global__ __launch_bounds__(WV * kWave) R3D_WAVES(WV) void moe_w4a8_i8_kernel(
    const int8_t* __restrict__ a, const float* __restrict__ a_scale,
    const int32_t* __restrict__ sorted, const int32_t* __restrict__ off, R3dMoeTabs tw,
    uint16_t* __restrict__ y, int T, int N, int K, int a_ld, int as_ld, int y_ld,
    int n_expert, int top_k, int a_sorted)
{
    __shared__ __align__(16) uint32_t s_a[MR * kPitch];
    __shared__ int32_t s_tok[MR];
    __shared__ int32_t s_nruns;
    __shared__ int32_t s_rs[MR], s_re[MR], s_e[MR];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & (kWave - 1);
    const uint32_t wave = tid / kWave;
    const uint32_t r = lane & 15;

    const uint32_t r0 = (uint32_t)blockIdx.y * MR;
    const uint32_t ncol = ((uint32_t)blockIdx.x * WV + wave) * 16;   // the wave's 16 columns
    const uint32_t r1 = (r0 + MR < (uint32_t)T) ? r0 + MR : (uint32_t)T;

    moe_w4a8_setup(sorted, off, r0, r1, tid, T, n_expert, top_k, a_sorted, s_tok, &s_nruns, s_rs,
                   s_re, s_e);
    const int nruns = s_nruns;

    const uint32_t nc = ncol + r;
    const int n_real = (int)(nc < (uint32_t)N ? nc : (uint32_t)N - 1);

    for (int run = (int)blockIdx.z; run < nruns; run += (int)gridDim.z) {
        const int rs = s_rs[run], re = s_re[run], e = s_e[run];
        v8float acc = v8float{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
        if (e >= 0) {
        const int cls = e & 1, e2 = e >> 1;
        const uint8_t* wrow = r3d_tab_at(tw.w[cls], e2) + (size_t)n_real * tw.w[cls].ld / 2;
        const uint16_t* wsb = (const uint16_t*)r3d_tab_at(tw.s[cls], e2);
        const int ws_ld = tw.s[cls].ld;

        for (int b = 0; b < K / kGroup; ++b) {
            const int kt = b * kGroup;

            // Stage 16 rows x 128 i8 of the gathered activation; padding rows read the run's first.
            for (int i = (int)tid; i < MR * (kGroup / 4); i += (WV * kWave)) {
                const int row = i / (kGroup / 4), d = i % (kGroup / 4);
                const int src = (rs <= (int)(r0 + row) && (int)(r0 + row) < re) ? s_tok[row]
                                                                                  : s_tok[rs - (int)r0];
                const int rr = (src < 0) ? 0 : src;
                s_a[(size_t)row * kPitch + d] = *(const uint32_t*)(a + (size_t)rr * a_ld + kt + 4 * d);
            }
            group_barrier();

            const uint32_t* ap = s_a + (size_t)r * kPitch;
            v8int sub = v8int{ 0, 0, 0, 0, 0, 0, 0, 0 };
#pragma unroll
            for (int k = 0; k < kGroup; k += 16) {
                const r3d_u32x4 aq = *(const r3d_u32x4*)(ap + k / 4);
                const uint2 wp = *(const uint2*)(wrow + (kt + k) / 2);   // 8 bytes = 16 nibbles
                int32_t v[16];
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const int byt = j / 2;
                    const bool high = (j & 1);
                    const uint32_t byte = (byt < 4) ? ((wp.x >> (8 * byt)) & 0xFFu)
                                                      : ((wp.y >> (8 * (byt - 4))) & 0xFFu);
                    v[j] = nib_i8(high ? (byte >> 4) : (byte & 0xFu));
                }
                v4int bf{ (int)((uint32_t)(v[0] & 0xFF) | (uint32_t)(v[1] & 0xFF) << 8 |
                            (uint32_t)(v[2] & 0xFF) << 16 | (uint32_t)(v[3] & 0xFF) << 24),
                          (int)((uint32_t)(v[4] & 0xFF) | (uint32_t)(v[5] & 0xFF) << 8 |
                            (uint32_t)(v[6] & 0xFF) << 16 | (uint32_t)(v[7] & 0xFF) << 24),
                          (int)((uint32_t)(v[8] & 0xFF) | (uint32_t)(v[9] & 0xFF) << 8 |
                            (uint32_t)(v[10] & 0xFF) << 16 | (uint32_t)(v[11] & 0xFF) << 24),
                          (int)((uint32_t)(v[12] & 0xFF) | (uint32_t)(v[13] & 0xFF) << 8 |
                            (uint32_t)(v[14] & 0xFF) << 16 | (uint32_t)(v[15] & 0xFF) << 24) };
                sub = wmma_i8(v4int{ (int)aq.x, (int)aq.y, (int)aq.z, (int)aq.w }, bf, sub);
                if ((k + 16) % WG) continue;
                // Fold a weight group's scales: (sub * scale_a[src][b]) * scale_w[e][n][group],
                // libref's order; WG 128 folds once a block, WG 32 four times.
                const float sw = bf16_f32(wsb[(size_t)n_real * ws_ld + b * (kGroup / WG) + k / WG]);
#pragma unroll
                for (int g2 = 0; g2 < 8; ++g2) {
                    const int row = (int)(g2 * 2 + (lane >> 4));
                    const int src = (rs <= (int)(r0 + row) && (int)(r0 + row) < re) ? s_tok[row]
                                                                                      : s_tok[rs - (int)r0];
                    const float sa = a_scale[(size_t)(src < 0 ? 0 : src) * as_ld + b];
                    acc[g2] += (float)sub[g2] * sa * sw;
                }
                sub = v8int{ 0, 0, 0, 0, 0, 0, 0, 0 };
            }
            group_barrier();
        }

        }
        // Store only the run's rows.
#pragma unroll
        for (int g2 = 0; g2 < 8; ++g2) {
            const int row = (int)(g2 * 2 + (lane >> 4));
            const int m = (int)(r0 + row);
            if (rs <= m && m < re && nc < (uint32_t)N) y[(size_t)m * y_ld + nc] = f32_to_bf16(acc[g2]);
        }
    }
}

/* ------------------------------------------------------------- E4M3 arm (f16 wave-matrix) */

template <int MR, int WV, int WG>
__global__ __launch_bounds__(WV * kWave) R3D_WAVES(WV) void moe_w4a8_e4m3_kernel(
    const uint8_t* __restrict__ a, const float* __restrict__ a_scale,
    const int32_t* __restrict__ sorted, const int32_t* __restrict__ off, R3dMoeTabs tw,
    uint16_t* __restrict__ y, int T, int N, int K, int a_ld, int as_ld, int y_ld,
    int n_expert, int top_k, int a_sorted)
{
    __shared__ __align__(16) uint32_t s_a[16 * kPitch];
    __shared__ __align__(16) uint32_t s_w[kWaves * 16 * kPitch];   // the block's 128 cols x 128 K
    __shared__ __align__(16) uint16_t s_tbl[R3D_E4M3_TBL];
    __shared__ int32_t s_tok[16];
    __shared__ int32_t s_nruns;
    __shared__ int32_t s_rs[16], s_re[16], s_e[16];

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & (kWave - 1);
    const uint32_t wave = tid / kWave;
    const uint32_t r = lane & 15;

    r3d_e4m3_f16_init(s_tbl, (int)tid, kThreads);

    const uint32_t r0 = (uint32_t)blockIdx.y * MR;
    const uint32_t n0 = (uint32_t)blockIdx.x * kWaves * 16;
    const uint32_t r1 = (r0 + MR < (uint32_t)T) ? r0 + MR : (uint32_t)T;

    moe_w4a8_setup(sorted, off, r0, r1, tid, T, n_expert, top_k, a_sorted, s_tok, &s_nruns, s_rs,
                   s_re, s_e);
    const int nruns = s_nruns;

    const uint32_t nc = n0 + wave * 16 + r;   // this wave's output column
    const int n_real = (int)(nc < (uint32_t)N ? nc : (uint32_t)N - 1);

    for (int run = (int)blockIdx.z; run < nruns; run += (int)gridDim.z) {
        const int rs = s_rs[run], re = s_re[run], e = s_e[run];
        v8float acc = v8float{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
        if (e >= 0) {
        const int cls = e & 1, e2 = e >> 1;
        const uint8_t* wbase = r3d_tab_at(tw.w[cls], e2);
        const int w_ld = tw.w[cls].ld;
        const uint16_t* wsb = (const uint16_t*)r3d_tab_at(tw.s[cls], e2);
        const int ws_ld = tw.s[cls].ld;

        for (int b = 0; b < K / kGroup; ++b) {
            const int kt = b * kGroup;

            // Stage 16 rows x 128 E4M3 of the gathered activation; padding rows read the run's first.
            for (int i = (int)tid; i < MR * (kGroup / 4); i += (WV * kWave)) {
                const int row = i / (kGroup / 4), d = i % (kGroup / 4);
                const int src = (rs <= (int)(r0 + row) && (int)(r0 + row) < re) ? s_tok[row]
                                                                                  : s_tok[rs - (int)r0];
                const int rr = (src < 0) ? 0 : src;
                s_a[(size_t)row * kPitch + d] = *(const uint32_t*)(a + (size_t)rr * a_ld + kt + 4 * d);
            }
            // Stage the block's 128 cols x 128 K of the int4 weight (packed nibbles).
            for (int i = (int)tid; i < kWaves * 16 * (kGroup / 8); i += (WV * kWave)) {
                const int col = i / (kGroup / 8), d = i % (kGroup / 8);
                const int nn = (int)(n0 + col < (uint32_t)N ? n0 + col : (uint32_t)N - 1);
                s_w[(size_t)col * kPitch + d] =
                    *(const uint32_t*)(wbase + (size_t)nn * w_ld / 2 + kt / 2 + 4 * d);
            }
            group_barrier();

            const uint32_t* ap = s_a + (size_t)r * kPitch;
            const uint32_t* wp = s_w + (size_t)(wave * 16 + r) * kPitch;   // this wave's 16 cols
            v8float blk = v8float{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
#pragma unroll
            for (int k = 0; k < kGroup; k += 16) {
                uint32_t ain[4], af[8], bf[8];
#pragma unroll
                for (int g = 0; g < 4; ++g) ain[g] = ap[k / 4 + g];
                r3d_e4m3x16_decode<0>(s_tbl, ain, af);
                // Decode the 16 packed int4 weights of this k tile to f16, two per word (as af[8]).
                const uint2 wq = *(const uint2*)(wp + k / 8);   // 8 bytes = 16 nibbles
#pragma unroll
                for (int g = 0; g < 8; ++g) {
                    const uint32_t byte = (g < 4) ? ((wq.x >> (8 * g)) & 0xFFu)
                                                     : ((wq.y >> (8 * (g - 4))) & 0xFFu);
                    const _Float16 h0 = (_Float16)(float)nib_i8(byte & 0xFu);        // weight 2g
                    const _Float16 h1 = (_Float16)(float)nib_i8((byte >> 4) & 0xFu); // weight 2g+1
                    bf[g] = (uint32_t)__builtin_bit_cast(uint16_t, h0)
                          | ((uint32_t)__builtin_bit_cast(uint16_t, h1) << 16);
                }
                v16half fa, fb;
#pragma unroll
                for (int g = 0; g < 8; ++g) { fa[2 * g] = h16(af[g] & 0xffffu); fa[2 * g + 1] = h16(af[g] >> 16); }
#pragma unroll
                for (int g = 0; g < 8; ++g) { fb[2 * g] = h16(bf[g] & 0xffffu); fb[2 * g + 1] = h16(bf[g] >> 16); }
                blk = wmma_f16(fa, fb, blk);
                if ((k + 16) % WG) continue;
                // Fold a weight group's scales: (dot * scale_a[src][b]) * scale_w[e][n][group],
                // libref's order; WG 128 folds once a block, WG 32 four times.
                const float sw = bf16_f32(wsb[(size_t)n_real * ws_ld + b * (kGroup / WG) + k / WG]);
#pragma unroll
                for (int g = 0; g < 8; ++g) {
                    const int row = (int)(g * 2 + (lane >> 4));
                    const int src = (rs <= (int)(r0 + row) && (int)(r0 + row) < re) ? s_tok[row]
                                                                                      : s_tok[rs - (int)r0];
                    const float sa = a_scale[(size_t)(src < 0 ? 0 : src) * as_ld + b];
                    acc[g] += blk[g] * sa * sw;
                }
                blk = v8float{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
            }
            group_barrier();
        }

        }
        // Store only the run's rows.
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int row = (int)(g * 2 + (lane >> 4));
            const int m = (int)(r0 + row);
            if (rs <= m && m < re && nc < (uint32_t)N) y[(size_t)m * y_ld + nc] = f32_to_bf16(acc[g]);
        }
    }
}

/* ------------------------------------ E4M3 arm, prefill (run-aligned 64 x 256 tiles, f16 LDS)
 *
 * The 16-row block above re-decodes its 128 columns of weight for every 16 rows of a run, nibble
 * by nibble through f32, and stages each 128-wide K block behind a barrier with nothing in flight:
 * at a 2048-token chunk of the 26B-A4B (16384 sorted rows) gate_up ran 14.4 ms and down 7.8 ms a
 * launch (skewed routings, RX 7900 XT), 10 and 9 TF/s of the f16 wave matrix's 103. Here a
 * block is one 64-row tile of ONE expert's run (libhalo's run-aligned form, kAlignRows) by 256
 * columns. Each 32-wide K pass is staged into LDS already decoded to exact f16 -- the activation's
 * E4M3 by bit moves (e4m3x2_f16x2), the weight's nibbles by the 0x6400 magic -- so a staged weight
 * value serves 64 rows and an activation value 256 columns. The LDS is double-buffered: pass p
 * multiplies one buffer while pass p + 1 is decoded into the other and pass p + 3's raw bytes are
 * loaded, one LDS-only barrier a pass. A wave owns a 64 x 32 patch (4 x 2 wave-matrix tiles), six
 * fragment reads for eight WMMAs: the fragment reads, not the WMMAs, were the 64 x 128 tile's limit
 * (2 x 2 patches, 3.3 ms). Tiles of a short run skip their empty 16-row tiles.
 *
 * The tile map is computed in every wave from the offsets, no LDS, no barrier, no scratch: expert e
 * owns ceil(rows_e / 64) tiles in expert order and the grid's y axis is an upper bound on their sum
 * (ceil(T / 64) + n_expert); blocks past it leave. Rows past the live count belong to no run, and
 * a token-order dropped slot inside one is not stored either: both are left as the caller set them,
 * as libref leaves them.
 *
 * The numbers are the 16-row arm's, element for element: each 128-block's product sum is one
 * v_wmma_f32_16x16x16_f16 chain of eight 16-K steps in ascending k from zero (carried over the four
 * passes of the block), on the same exact f16 operands, then folded as
 * acc += blk * a_scale[src][b] * w_scale[e][n][group] in ascending block order (at weight group 32,
 * four folds a block, as there). So the two arms store the same bytes (checked at 2048 tokens, both
 * orders, both groups). Measured at a 2048-token chunk, skewed routings: gate_up 2.77 ms (51 TF/s),
 * down 1.67 ms; at group 32 the fold every pass spills 79 registers and gate_up runs 7.9 ms (the
 * 16-row arm 15.0). */
constexpr int kPfM = 64;                // rows a tile: one run's, or part of one
constexpr int kPfN = 256;               // columns a tile: eight waves of 32
constexpr int kPfK = 32;                // K a staged pass: a quarter of a scale block
constexpr int kPfP = kPfK + 8;          // f16 a staged row: 80 bytes, 16-byte aligned, banks spread
constexpr int kPfMaxE = 256;            // experts the in-wave tile map covers (8 a lane)

__device__ __forceinline__ uint32_t e4m3x2_f16x2(uint32_t v);   // below, with the decode GEMV

/* This block's (expert, first row, end row), or false when blockIdx.y is past the last tile. */
__device__ __forceinline__ bool moe_pf_tile(const int32_t* __restrict__ off, int n_expert, int T,
                                            int tile, int lane, int& e, int& r0, int& r1)
{
    constexpr int EPL = kPfMaxE / kWave;
    const int x0 = lane * EPL;
    int nt[EPL], tot = 0;
#pragma unroll
    for (int j = 0; j < EPL; ++j) {
        const int x = x0 + j;
        int t = 0;
        if (x < n_expert) {
            const int c = off[x + 1] - off[x];
            t = (c + kPfM - 1) / kPfM;
        }
        nt[j] = t;
        tot += t;
    }
    int inc = tot;
#pragma unroll
    for (int d = 1; d < kWave; d <<= 1) {
        const int v = __shfl_up(inc, d);
        if (lane >= d) inc += v;
    }
    if (tile >= __shfl(inc, kWave - 1)) return false;
    const int owner = __ffsll((unsigned long long)__ballot(tile < inc)) - 1;
    int ee = 0, rr = 0;
    {
        int q = tile - (inc - tot);
#pragma unroll
        for (int j = 0; j < EPL; ++j) {
            if (q >= 0 && q < nt[j]) { ee = x0 + j; rr = q; }
            q -= nt[j];
        }
    }
    e = __shfl(ee, owner);
    rr = __shfl(rr, owner);
    const int o1 = off[e + 1] < T ? off[e + 1] : T;
    r0 = off[e] + rr * kPfM;
    r1 = r0 + kPfM < o1 ? r0 + kPfM : o1;
    return r0 < r1;
}

/* A barrier that orders LDS only (libhalo's lds_barrier): __syncthreads() is a workgroup fence over
 * every address space, so it would wait for the next pass's global loads in flight. */
__device__ __forceinline__ void moe_pf_barrier()
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

/* A wave-matrix fragment from LDS: 16 f16 at one row, read as two 16-byte words and reinterpreted.
 * Read as a v16half the compiler moves every half through a 16-bit register op (two VALU a dword
 * the wave matrix waits behind). */
typedef unsigned int moe_u32x8 __attribute__((ext_vector_type(8)));
typedef moe_u32x8 moe_v16hf;
__device__ __forceinline__ moe_v16hf moe_pf_frag(const uint16_t* p)
{
    const r3d_u32x4 lo = *(const r3d_u32x4*)p, hi = *(const r3d_u32x4*)(p + 8);
    return moe_u32x8{ lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w };
}
/* v_wmma_f32_16x16x16_f16 on two fragments held as dwords. Through the builtin the fragments are
 * f16 vectors, and gfx11's real-true16 codegen then moves the high half of every fragment dword
 * through a 16-bit op of its own (128 v_mov_b16 a pass, the wave matrix waiting behind them). The
 * A and B operands here are LDS loads, never an earlier WMMA's result, so the WMMA-reads-WMMA
 * hazard the compiler would pad does not arise. */
__device__ __forceinline__ v8float moe_pf_wmma(const moe_u32x8& a, const moe_u32x8& b, v8float c)
{
    asm("v_wmma_f32_16x16x16_f16 %0, %1, %2, %0" : "+v"(c) : "v"(a), "v"(b));
    return c;
}

template <int WG>
__global__ __launch_bounds__(kThreads) void moe_w4a8_e4m3_pf_kernel(
    const uint8_t* __restrict__ a, const float* __restrict__ a_scale,
    const int32_t* __restrict__ sorted, const int32_t* __restrict__ off, R3dMoeTabs tw,
    uint16_t* __restrict__ y, int T, int N, int K, int a_ld, int as_ld, int y_ld,
    int n_expert, int top_k, int a_sorted)
{
    __shared__ __align__(16) uint16_t s_a[2][kPfM * kPfP];
    __shared__ __align__(16) uint16_t s_b[2][kPfN * kPfP];
    __shared__ int32_t s_tok[kPfM];
    __shared__ float s_sa[2][kPfM];              // a block's a-scales, by block parity

    const int tid = (int)threadIdx.x, lane = tid & (kWave - 1);
    const int wave = __builtin_amdgcn_readfirstlane(tid / kWave);
    const int wn = wave;                         // the wave's 64 x 32 patch: every row, cols wn*32

    int e, r0, r1;
    if (!moe_pf_tile(off, n_expert, T, (int)blockIdx.y, lane, e, r0, r1)) return;
    const int n0 = (int)blockIdx.x * kPfN;

    // The source row of each of the tile's rows; -1 a token-order dropped slot, which the
    // reference leaves as the caller set it (read as row 0, never stored). Rows past the run read
    // its first.
    if (tid < kPfM) {
        const int i = r0 + tid < r1 ? r0 + tid : r0;
        s_tok[tid] = a_sorted ? i : (sorted[i] < 0 ? -1 : sorted[i] / top_k);
    }

    const int cls = e & 1, e2 = e >> 1;
    const uint8_t* wbase = r3d_tab_at(tw.w[cls], e2);
    const int w_rb = tw.w[cls].ld / 2;           // bytes a weight row
    const uint16_t* wsb = (const uint16_t*)r3d_tab_at(tw.s[cls], e2);
    const int ws_ld = tw.s[cls].ld;

    // Staging, one piece a thread a pass: the activation as 64 rows x 4 eight-byte pieces (eight
    // E4M3 codes), the weight as 256 columns x 16 bytes (32 nibbles).
    const int arow = tid >> 2, aseg = (tid & 3) * 8;
    const int bcol = tid, bseg = 0;
    const int bn = n0 + bcol < N ? n0 + bcol : N - 1;
    const uint8_t* bsrc = wbase + (size_t)bn * w_rb + bseg;
    moe_pf_barrier();                             // s_tok
    const uint8_t* asrc = a + (size_t)(s_tok[arow] < 0 ? 0 : s_tok[arow]) * a_ld + aseg;
    const int sa_row = tid < kPfM ? (s_tok[tid] < 0 ? 0 : s_tok[tid]) : 0;

    // The wave's columns (for the weight scales), and the 16-row tiles the run reaches.
    const int nmt = (r1 - r0 + 15) / 16;
    int ncol[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int n = n0 + wn * 32 + j * 16 + (lane & 15);
        ncol[j] = n < N ? n : N - 1;
    }

    const int np = K / kPfK;                      // passes; a multiple of four (K % 128)
    uint2 ra[2];                                  // a pass's raw bytes, by pass parity
    r3d_u32x4 rb[2];
    float sa_n = 0.f;                             // the a-scale of the next block's first pass
    float sw[2][2];                               // a pass's weight scales, by pass parity
    auto issue = [&](int q, uint2& xa, r3d_u32x4& xb) {
        xa = *(const uint2*)(asrc + q * kPfK);
        xb = *(const r3d_u32x4*)(bsrc + q * (kPfK / 2));
        if ((q % (kGroup / kPfK)) == 0 && tid < kPfM)
            sa_n = a_scale[(size_t)sa_row * as_ld + q / (kGroup / kPfK)];
    };
    // Decode pass q's raw bytes into buffer `buf`, with its a-scales (a block's first pass) and
    // its weight scales (the pass a fold closes).
    auto stage = [&](int q, int buf, const uint2& xa, const r3d_u32x4& xb, float (&swq)[2]) {
        {
            // E4M3 -> f16: bytes b0..b3 of a dword to the f16 pairs (b0, b1), (b2, b3).
            const uint32_t in[2] = { xa.x, xa.y };
            uint32_t h[4];
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                h[2 * t]     = e4m3x2_f16x2(__builtin_amdgcn_perm(0u, in[t], 0x010c000cu));
                h[2 * t + 1] = e4m3x2_f16x2(__builtin_amdgcn_perm(0u, in[t], 0x030c020cu));
            }
            *(r3d_u32x4*)(s_a[buf] + arow * kPfP + aseg) = r3d_u32x4{ h[0], h[1], h[2], h[3] };
        }
        {
            // int4 -> f16: byte j holds k 2j (low nibble) and 2j+1 (high); (n ^ 8) under the
            // 0x6400 exponent reads 1024 + (n ^ 8), less 1032 is n, exactly.
            typedef _Float16 h2 __attribute__((ext_vector_type(2)));
            const uint32_t in[4] = { xb.x, xb.y, xb.z, xb.w };
            uint32_t h[16];
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const uint32_t x = in[t] ^ 0x88888888u;
                const uint32_t lo = x & 0x0f0f0f0fu, hi = (x >> 4) & 0x0f0f0f0fu;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const uint32_t hb = __builtin_amdgcn_perm(hi, lo, 0x0c040c00u + 0x00010001u * k)
                                      | 0x64006400u;
                    const h2 hv = __builtin_bit_cast(h2, hb) - h2{ (_Float16)1032.0f, (_Float16)1032.0f };
                    h[4 * t + k] = __builtin_bit_cast(uint32_t, hv);
                }
            }
            r3d_u32x4* d = (r3d_u32x4*)(s_b[buf] + bcol * kPfP);
#pragma unroll
            for (int t = 0; t < 4; ++t) d[t] = r3d_u32x4{ h[4 * t], h[4 * t + 1], h[4 * t + 2], h[4 * t + 3] };
        }
        const int b = q / (kGroup / kPfK);
        if ((q % (kGroup / kPfK)) == 0 && tid < kPfM) s_sa[b & 1][tid] = sa_n;
        // The weight group a fold at the end of pass q closes: the block's at group 128 (its last
        // pass), the pass's own at 32.
        const bool closes = WG == kGroup ? (q % (kGroup / kPfK)) == (kGroup / kPfK) - 1 : true;
        if (closes) {
            const int g = WG == kGroup ? b : q;
#pragma unroll
            for (int j = 0; j < 2; ++j) swq[j] = bf16_f32(wsb[(size_t)ncol[j] * ws_ld + g]);
        }
    };

    v8float acc[4][2], blk[4][2];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            acc[i][j] = v8float{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
            blk[i][j] = acc[i][j];
        }

    // Pass p multiplies buffer p & 1 while pass p + 1 is decoded into the other, and pass p + 3's
    // bytes are loaded into the registers pass p + 1's just left: one barrier a pass, and the
    // decode's VALU interleaves with this pass's wave-matrix work.
    auto step = [&](int p, const int S) {
        {
#pragma unroll
            for (int s = 0; s < kPfK / 16; ++s) {
                moe_v16hf fb[2];
#pragma unroll
                for (int j = 0; j < 2; ++j)
                    fb[j] = moe_pf_frag(s_b[S] + (wn * 32 + j * 16 + (lane & 15)) * kPfP + s * 16);
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    if (i < nmt) {
                        const moe_v16hf fa = moe_pf_frag(s_a[S] + (i * 16 + (lane & 15)) * kPfP + s * 16);
#pragma unroll
                        for (int j = 0; j < 2; ++j) blk[i][j] = moe_pf_wmma(fa, fb[j], blk[i][j]);
                    }
            }
            // A weight group closes: (dot * a_scale[src][b]) * w_scale[e][n][group], libref's order.
            const bool fold = WG == kGroup ? (p % (kGroup / kPfK)) == (kGroup / kPfK) - 1 : true;
            if (fold) {
                const float* sap = s_sa[(p / (kGroup / kPfK)) & 1];
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int g = 0; g < 8; ++g) {
                        const float sa = sap[i * 16 + g * 2 + (lane >> 4)];
#pragma unroll
                        for (int j = 0; j < 2; ++j) acc[i][j][g] += blk[i][j][g] * sa * sw[S][j];
                    }
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 2; ++j) blk[i][j] = v8float{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
            }
        }
        if (p + 1 < np) {
            stage(p + 1, S ^ 1, ra[S ^ 1], rb[S ^ 1], sw[S ^ 1]);
            if (p + 3 < np) issue(p + 3, ra[S ^ 1], rb[S ^ 1]);
        }
        moe_pf_barrier();
    };

    issue(0, ra[0], rb[0]);
    issue(1, ra[1], rb[1]);
    stage(0, 0, ra[0], rb[0], sw[0]);
    issue(2, ra[0], rb[0]);
    moe_pf_barrier();
    for (int p = 0; p < np; p += 2) {
        step(p, 0);
        step(p + 1, 1);
    }

#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int row = i * 16 + g * 2 + (lane >> 4);
            const int m = r0 + row;
            if (m >= r1 || s_tok[row] < 0) continue;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int n = n0 + wn * 32 + j * 16 + (lane & 15);
                if (n < N) y[(size_t)m * y_ld + n] = f32_to_bf16(acc[i][j][g]);
            }
        }
}

/* --------------------------------------------- E4M3 arm, decode GEMV (fdot2, one run a block)
 *
 * At decode a run is one to a few rows and the 16x128 block above is latency-bound: 22 staged
 * k tiles behind a barrier each. Here block (x, y) takes the y-th non-empty expert run and a
 * slice of its columns. The run's activation rows (kGvRows at a time) are decoded once into LDS as
 * exact f16 pairs (k, k+1). A lane owns one (column, 128-block of K): it loads the block's 64
 * bytes of packed weight, decodes the nibbles to f16 with the 0x6400 magic and runs one fdot2
 * chain over the block in ascending k; a weight group closes as the reference closes it,
 * (dot * a_scale[src][b]) * w_scale[e][n][group]; and the column's group terms are added in
 * ascending group order by a readlane walk over the lanes that hold them. So the order is the
 * reference's: k ascending, one accumulator a group and one a column; the only difference is
 * fdot2's one rounding a pair (the dot2 rows' documented difference, README "Numerics"). A dropped
 * token-order slot (sorted < 0) is left as the caller set it, as in the reference; a sorted-order
 * row is computed. */
constexpr int kGvRows = 4;              // activation rows a pass over LDS

__device__ __forceinline__ uint32_t e4m3x2_f16x2(uint32_t v)
{
    // v holds an E4M3 code in bits 8..15 of each half; the result is the two codes as exact f16.
    const uint32_t h = (v & 0x80008000u) | ((v >> 1) & 0x3f803f80u);   // value * 2^-8
    typedef _Float16 h2 __attribute__((ext_vector_type(2)));
    const h2 s = __builtin_bit_cast(h2, h) * h2{ (_Float16)256.0f, (_Float16)256.0f };
    return __builtin_bit_cast(uint32_t, s);
}

__device__ __forceinline__ float readlane_f(float v, int l)
{
    return __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), l));
}

template <int WG, int CS>
__global__ __launch_bounds__(kThreads) void moe_w4a8_e4m3_gemv_kernel(
    const uint8_t* __restrict__ a, const float* __restrict__ a_scale,
    const int32_t* __restrict__ sorted, const int32_t* __restrict__ off, R3dMoeTabs tw,
    uint16_t* __restrict__ y, int T, int N, int K, int a_ld, int as_ld, int y_ld,
    int n_expert, int top_k, int a_sorted, int a_rows)
{
    constexpr int TPL = kGroup / WG;     // weight groups (scale folds) a lane's block holds
    R3D_DYN_LDS(uint32_t, s_act);        // kGvRows rows x K/2 dwords of f16 pairs
    __shared__ float s_sa[kGvRows][32];  // a_scale of the rows, K/128 <= 32

    const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;

    // The y-th non-empty expert run, ranked by every wave on its own (ballots over the offsets, no
    // barrier), so the weight loads below go out as soon as the run is known.
    int e = -1, rs = 0, re = 0;
    {
        int base = 0;
        for (int e0 = 0; e0 < n_expert && e < 0; e0 += kWave) {
            const int ee = e0 + (int)lane;
            const int o0 = ee < n_expert ? off[ee] : 0, o1 = ee < n_expert ? off[ee + 1] : 0;
            const uint32_t bal = (uint32_t)__ballot(ee < n_expert && o1 > o0);
            const int cnt = __popc(bal);
            if (base + cnt > (int)blockIdx.y) {
                uint32_t m = bal;
                for (int t = 0; t < (int)blockIdx.y - base; ++t) m &= m - 1u;
                const int l = __ffs((int)m) - 1;
                e = e0 + l;
                rs = __builtin_amdgcn_readlane(o0, l);
                re = __builtin_amdgcn_readlane(o1, l);
            }
            base += cnt;
        }
    }
    if (e < 0) return;
    re = re < T ? re : T;

    const int cls = e & 1, e2 = e >> 1;
    const uint8_t* wbase = r3d_tab_at(tw.w[cls], e2);
    const int w_ld = tw.w[cls].ld;
    const uint16_t* wsb = (const uint16_t*)r3d_tab_at(tw.s[cls], e2);
    const int ws_ld = tw.s[cls].ld;
    const int nb = K / kGroup;               // lanes a column
    const int cpw = kWave / nb;              // columns a wave a column set
    const int cl = (int)lane / nb, b = (int)lane % nb;
    const bool live_lane = cl < cpw;
    const int col0 = (int)blockIdx.x * (kWaves * cpw * CS) + (int)wave * cpw * CS;

    // The lane's weight: CS columns' 128-block b, 64 bytes each, and the block's scales.
    r3d_u32x4 w[CS][4];
    float sw[CS][TPL];
#pragma unroll
    for (int c = 0; c < CS; ++c) {
        int n = col0 + c * cpw + cl;
        n = (live_lane && n < N) ? n : N - 1;
        const uint8_t* wp = wbase + (size_t)n * w_ld / 2 + (size_t)b * 64;
#pragma unroll
        for (int t = 0; t < 4; ++t)
            w[c][t] = __builtin_nontemporal_load(
                (const __attribute__((address_space(1))) r3d_u32x4*)(wp + 16 * t));
#pragma unroll
        for (int t = 0; t < TPL; ++t) sw[c][t] = bf16_f32(wsb[(size_t)n * ws_ld + b * TPL + t]);
    }

    for (int r0 = rs; r0 < re; r0 += kGvRows) {
        const int nr = (re - r0) < kGvRows ? re - r0 : kGvRows;
        // Every thread reads the chunk's sources itself (a few shared words), no barrier.
        int s_src[kGvRows];
#pragma unroll
        for (int j = 0; j < kGvRows; ++j) {
            int src = -1;
            if (j < nr) {
                const int i = r0 + j;
                // The reference's rule: a token-order row with a dropped slot (or one past a's
                // rows) is left as the caller set it; a sorted-order row is always computed.
                src = a_sorted ? i : (sorted[i] >= 0 ? sorted[i] / top_k : -1);
                if (src >= a_rows) src = -1;
            }
            s_src[j] = src;
        }
        // Decode the rows: one 8-code chunk a thread a turn, as the f16 pairs (k, k+1).
        for (int j = 0; j < nr; ++j) {
            const int src = s_src[j] < 0 ? 0 : s_src[j];
            const uint8_t* ar = a + (size_t)src * a_ld;
            for (int c8 = (int)tid; c8 < K / 8; c8 += kThreads) {
                const uint2 x = *(const uint2*)(ar + 8 * c8);
                r3d_u32x4 o;
                o.x = e4m3x2_f16x2(__builtin_amdgcn_perm(x.x, x.x, 0x010c000cu));
                o.y = e4m3x2_f16x2(__builtin_amdgcn_perm(x.x, x.x, 0x030c020cu));
                o.z = e4m3x2_f16x2(__builtin_amdgcn_perm(x.y, x.y, 0x010c000cu));
                o.w = e4m3x2_f16x2(__builtin_amdgcn_perm(x.y, x.y, 0x030c020cu));
                *(r3d_u32x4*)(s_act + (size_t)j * (K / 2) + 4 * c8) = o;
            }
            for (int bb = (int)tid; bb < nb; bb += kThreads) s_sa[j][bb] = a_scale[(size_t)src * as_ld + bb];
        }
        group_barrier();

        for (int j = 0; j < nr; ++j) {
            const int src = s_src[j];
            if (src < 0) continue;
            const float sa = s_sa[j][b];
            const uint32_t* ap = s_act + (size_t)j * (K / 2) + (size_t)b * 64;
            float term[CS][TPL];
            float d[CS];
#pragma unroll
            for (int c = 0; c < CS; ++c) d[c] = 0.f;
#pragma unroll
            for (int t = 0; t < 4; ++t) {   // 32 codes a step, ascending k
                const r3d_u32x4 a0 = *(const r3d_u32x4*)(ap + 16 * t);
                const r3d_u32x4 a1 = *(const r3d_u32x4*)(ap + 16 * t + 4);
                const r3d_u32x4 a2 = *(const r3d_u32x4*)(ap + 16 * t + 8);
                const r3d_u32x4 a3 = *(const r3d_u32x4*)(ap + 16 * t + 12);
                const uint32_t af[16] = { a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w,
                                          a2.x, a2.y, a2.z, a2.w, a3.x, a3.y, a3.z, a3.w };
#pragma unroll
                for (int c = 0; c < CS; ++c) {
                    typedef _Float16 h2 __attribute__((ext_vector_type(2)));
                    const uint32_t wv4[4] = { w[c][t].x, w[c][t].y, w[c][t].z, w[c][t].w };
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        // A word's nibbles k..k+7 as the f16 pairs (k, k+1): (n ^ 8) | 0x6400 is
                        // 1032 + n, the low nibble of byte i the even k.
                        const uint32_t x = wv4[u] ^ 0x88888888u;
                        const uint32_t lo = x & 0x0f0f0f0fu, hi = (x >> 4) & 0x0f0f0f0fu;
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const uint32_t hb = __builtin_amdgcn_perm(hi, lo, 0x0c040c00u + 0x00010001u * i) | 0x64006400u;
                            const h2 hv = __builtin_bit_cast(h2, hb) - h2{ (_Float16)1032.0f, (_Float16)1032.0f };
                            d[c] = __builtin_amdgcn_fdot2(__builtin_bit_cast(h2, af[4 * u + i]), hv, d[c], false);
                        }
                    }
                    if (WG == 32) { term[c][t % TPL] = (d[c] * sa) * sw[c][t % TPL]; d[c] = 0.f; }
                }
            }
            if (WG == 128) {
#pragma unroll
                for (int c = 0; c < CS; ++c) term[c][0] = (d[c] * sa) * sw[c][0];
            }
            // Each column's group terms, added in ascending group order: lane (cl, b) holds the
            // terms of block b.
#pragma unroll
            for (int c = 0; c < CS; ++c) {
                for (int cc = 0; cc < cpw; ++cc) {
                    float acc = 0.f;
                    for (int bb = 0; bb < nb; ++bb)
#pragma unroll
                        for (int t = 0; t < TPL; ++t) acc = acc + readlane_f(term[c][t], cc * nb + bb);
                    const int n = col0 + c * cpw + cc;
                    if ((int)lane == cc * nb && n < N) y[(size_t)(r0 + j) * y_ld + n] = f32_to_bf16(acc);
                }
            }
        }
        group_barrier();   // the next row chunk's decode must not race these LDS reads
    }
}

}  // namespace

namespace {
/* --------------------------------- E4M3 arm, decode GEMV with K split over the lanes (libr3d)
 *
 * A wave takes kLsCols columns of one run's expert, two at a time, and its lanes split K: in each
 * 256-wide chunk lane l owns k = 256c + 8l .. 8l + 7, one dword of packed weight a column, so a
 * column's chunk is one coalesced 128-byte read. The run's activation row is decoded once into
 * registers (NCH chunks x 4 f16 pairs a lane) and its scales a lane's group a chunk, so there is
 * no LDS and no barrier. A lane's chunk term is (dot8 * a_scale) * w_scale over its eight k, the
 * terms summed in chunk order and the lanes by the wave butterfly: the reference's order within
 * eight k, another association across them (the sum's last bits). */
constexpr int kLsThreads = 128;                 // four waves
constexpr int kLsCols = 32;                     // columns a wave (8: 89.4, 16: 92.4, 64: 90.5 tok/s; 32: 92.9)

template <int WG, int NCH>
__global__ __launch_bounds__(kLsThreads) void moe_w4a8_e4m3_gemv_ls_kernel(
    const uint8_t* __restrict__ a, const float* __restrict__ a_scale,
    const int32_t* __restrict__ sorted, const int32_t* __restrict__ off, R3dMoeTabs tw,
    uint16_t* __restrict__ y, int T, int N, int K, int a_ld, int as_ld, int y_ld,
    int n_expert, int top_k, int a_sorted, int a_rows)
{
    typedef _Float16 h2 __attribute__((ext_vector_type(2)));
    const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;

    /* the y-th non-empty run, offsets loaded four 32s at once */
    int e = -1, rs = 0, re = 0;
    {
        int base = 0;
        for (int e0 = 0; e0 < n_expert && e < 0; e0 += 4 * kWave) {
            int o0[4], o1[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int ee = e0 + u * kWave + (int)lane, ec = ee < n_expert ? ee : n_expert - 1;
                const int a0 = off[ec], a1 = off[ec + 1];
                o0[u] = ee < n_expert ? a0 : 0;
                o1[u] = ee < n_expert ? a1 : 0;
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const uint32_t bal = (uint32_t)__ballot(o1[u] > o0[u]);
                const int cnt = __popc(bal);
                if (e < 0 && base + cnt > (int)blockIdx.y) {
                    uint32_t m = bal;
                    for (int t = 0; t < (int)blockIdx.y - base; ++t) m &= m - 1u;
                    const int l = __ffs((int)m) - 1;
                    e = e0 + u * kWave + l;
                    rs = __builtin_amdgcn_readlane(o0[u], l);
                    re = __builtin_amdgcn_readlane(o1[u], l);
                }
                base += cnt;
            }
        }
    }
    if (e < 0) return;
    re = re < T ? re : T;
    const int cls = e & 1, e2 = e >> 1;
    const uint8_t* wbase = r3d_tab_at(tw.w[cls], e2);
    const int w_ld = tw.w[cls].ld;
    const uint16_t* wsb = (const uint16_t*)r3d_tab_at(tw.s[cls], e2);
    const int ws_ld = tw.s[cls].ld;
    const int col0 = ((int)blockIdx.x * (kLsThreads / kWave) + (int)wave) * kLsCols;
    if (col0 >= N) return;   /* a wave's own; no barrier follows */

    for (int i = rs; i < re; ++i) {
        const int src = a_sorted ? i : (sorted[i] >= 0 ? sorted[i] / top_k : -1);
        if (src < 0 || src >= a_rows) continue;
        /* the lane's activation, NCH chunks of eight codes as four exact f16 pairs, and its
         * activation scale a chunk (group 128 of the activation) */
        uint32_t af[NCH][4];
        float sa[NCH];
        const uint8_t* ar = a + (size_t)src * a_ld + 8 * lane;
#pragma unroll
        for (int c = 0; c < NCH; ++c) {
            const uint2 x = *(const uint2*)(ar + 256 * c);
            af[c][0] = e4m3x2_f16x2(__builtin_amdgcn_perm(x.x, x.x, 0x010c000cu));
            af[c][1] = e4m3x2_f16x2(__builtin_amdgcn_perm(x.x, x.x, 0x030c020cu));
            af[c][2] = e4m3x2_f16x2(__builtin_amdgcn_perm(x.y, x.y, 0x010c000cu));
            af[c][3] = e4m3x2_f16x2(__builtin_amdgcn_perm(x.y, x.y, 0x030c020cu));
            sa[c] = a_scale[(size_t)src * as_ld + (256 * c + 8 * (int)lane) / kGroup];
        }
        for (int cc = 0; cc < kLsCols; cc += 2) {
            uint32_t wv[2][NCH];
            float sw[2][NCH];
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                int n = col0 + cc + q;
                n = n < N ? n : N - 1;
                const uint8_t* wp = wbase + (size_t)n * w_ld / 2 + 4 * lane;
                const uint16_t* sp = wsb + (size_t)n * ws_ld;
#pragma unroll
                for (int c = 0; c < NCH; ++c) {
                    wv[q][c] = __builtin_nontemporal_load((const uint32_t*)(wp + 128 * c));
                    sw[q][c] = bf16_f32(sp[(256 * c + 8 * (int)lane) / WG]);
                }
            }
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                float acc = 0.f;
#pragma unroll
                for (int c = 0; c < NCH; ++c) {
                    const uint32_t x = wv[q][c] ^ 0x88888888u;
                    const uint32_t lo = x & 0x0f0f0f0fu, hi = (x >> 4) & 0x0f0f0f0fu;
                    float d = 0.f;
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const uint32_t hb = __builtin_amdgcn_perm(hi, lo, 0x0c040c00u + 0x00010001u * k) | 0x64006400u;
                        const h2 hv = __builtin_bit_cast(h2, hb) - h2{ (_Float16)1032.0f, (_Float16)1032.0f };
                        d = __builtin_amdgcn_fdot2(__builtin_bit_cast(h2, af[c][k]), hv, d, false);
                    }
                    acc = acc + (d * sa[c]) * sw[q][c];
                }
                acc = wave_sum(acc);
                const int n = col0 + cc + q;
                if (lane == 0 && n < N) y[(size_t)i * y_ld + n] = f32_to_bf16(acc);
            }
        }
    }
}

}  // namespace

extern "C" {

/* moe_gemm_q dtype w4a8: the grouped int4 GEMM. The arm is picked by the activation dtype (int8 ->
 * the exact i32 wave-matrix; E4M3 -> the f16 wave-matrix, libr4d's form). The weight scale is bf16
 * and the output bf16 in both arms. */
int r3d_rad_moe_gemm_w4a8(const RadArgs* a, RadStream s)
{
    R3dMoeCall c;
    int rc = r3d_moe_call(a, true, &c);
    if (rc != RAD_OK) return rc;
    if (!c.a_scale) return R3D_NO(RAD_E_INVAL);
    if (c.a->dtype != RAD_I8 && c.a->dtype != RAD_F8E4M3) return R3D_NO(RAD_E_DTYPE);
    if (c.a_scale->dtype != RAD_F32 || c.y->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);
    if (c.K % kGroup) return R3D_NO(RAD_E_SHAPE);   // the N tail clamps
    /* The weight's scale group: 128 (one scale a block), or 32 (Q4_0's block, which a QAT
     * checkpoint is trained to); the activation's scale stays per 128 of K. */
    const long long wg = r3d_geti_or(a, "group", kGroup);
    if (wg != 128 && wg != 32) return R3D_NO(RAD_E_SHAPE);

    R3dMoeTabs tw;
    rc = r3d_moe_q_tabs(c, RAD_I4, 4, RAD_BF16, 16, c.N, c.K / wg, &tw);
    if (rc != RAD_OK) return rc;
    const int a_ld = (int)c.a->stride[0], as_ld = (int)c.a_scale->stride[0];
    const int y_ld = (int)c.y->stride[0];


    const int32_t* ps = (const int32_t*)r3d_p(c.sorted);
    const int32_t* po = (const int32_t*)r3d_p(c.off);
    const float* psa = (const float*)r3d_p(c.a_scale);
    uint16_t* py = (uint16_t*)r3d_wp(c.y);
    const int mrows = r3d_geti_or(a, "mrows", 16);
    const int wv = r3d_geti_or(a, "wv", 8);
    const int ctile = wv * 16;
    dim3 grid((unsigned)((c.N + ctile - 1) / ctile), (unsigned)((c.T + mrows - 1) / mrows)), blk(wv * kWave);
    /* Few blocks (decode: one token's top_k rows are top_k runs of one row in one 16-row block):
     * spread a block's runs over grid.z, block z taking runs z, z + gridDim.z, ... Each run is
     * still one block's, accumulated in the same order, so the output is the same bytes; the
     * blocks past a block's run count leave after the setup. */
    /* Decode (at most 32 sorted rows, an E4M3 activation, K <= 4096): the GEMV, one run a block. */
    constexpr int kGvCS = 2;
    if (c.a->dtype == RAD_F8E4M3 && c.T <= 32 && c.K / kGroup <= 32) {
        long long ar = 0, ak = 0;
        r3d_rows_of(c.a, &ar, &ak);
        const int cpw = kWave / (int)(c.K / kGroup);
        const int cols = kWaves * cpw * kGvCS;
        const dim3 vgrid((unsigned)((c.N + cols - 1) / cols),
                         (unsigned)(c.T < c.n_expert ? c.T : c.n_expert));
        const size_t lds = (size_t)kGvRows * c.K * 2;
        const uint8_t* pa8 = (const uint8_t*)r3d_p(c.a);
        /* The lane-split form at the two widths it is instantiated for (the 26B-A4B's gate_up and
         * down), weight group 128: gate_up 37.8 -> 31.1 us, down 24.9 -> 21.2 a launch (cold
         * routings, RX 7900 XT); the 26B-A4B's decode 87.8 -> 91.8 tok/s. Elsewhere the
         * run-a-block GEMV.
         *
         * Group 32 was left on the run-a-block GEMV when the split form's gate_up measured rel_l2
         * 1.3e-4 against libref there. That is the order of summation and no error: one output
         * that lands on the other side of a bf16 rounding boundary moves the rel_l2 of one token's
         * 12288 outputs by 3.5e-5 to 7.0e-5 times its size over the rms. With the split form
         * forced at group 32, 40 seeded batches of one and four tokens had no output of either
         * form, at either group, further from the exact sum than f32 summation in another order
         * allows, while single batches reached 7e-5 to 8e-5 on one output of 1.4 to 1.8 times the
         * rms: the split form at both groups, the run-a-block GEMV at group 128, and libref's
         * order itself against the sum in f64. So the split form's numbers are as good at group
         * 32; it is not instantiated there and has not been timed. */
        if (wg == 128 && (c.K == 2816 || c.K == 768)) {
            const dim3 lgrid((unsigned)((c.N + (kLsThreads / kWave) * kLsCols - 1) /
                                        ((kLsThreads / kWave) * kLsCols)),
                             (unsigned)(c.T < c.n_expert ? c.T : c.n_expert));
#define R3D_LS1(WGV, NCHV)                                                                       \
    R3D_KLAUNCH((moe_w4a8_e4m3_gemv_ls_kernel<WGV, NCHV>), lgrid, dim3(kLsThreads),              \
                (hipStream_t)s, pa8, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld,    \
                as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted, (int)ar)
#define R3D_LS(NCHV)                                                                             \
    R3D_LS1(128, NCHV)
            if (c.K == 2816) R3D_LS(11);
            else             R3D_LS(3);
#undef R3D_LS
#undef R3D_LS1
            return (int)hipGetLastError();
        }
        if (wg == 32)
            R3D_KLAUNCH_S((moe_w4a8_e4m3_gemv_kernel<32, kGvCS>), vgrid, dim3(kThreads), lds, (hipStream_t)s,
                          pa8, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld,
                          (int)c.n_expert, (int)c.top_k, c.a_sorted, (int)ar);
        else
            R3D_KLAUNCH_S((moe_w4a8_e4m3_gemv_kernel<128, kGvCS>), vgrid, dim3(kThreads), lds, (hipStream_t)s,
                          pa8, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld,
                          (int)c.n_expert, (int)c.top_k, c.a_sorted, (int)ar);
        return (int)hipGetLastError();
    }
    /* Above the GEMV's 32 rows an E4M3 activation takes the run-aligned prefill tile (faster than
     * the 16-row arm from 5 tokens up: 142 against 538 us at the 26B-A4B's gate_up), where its
     * operands are read 8 and 16 bytes at a time. The mrows / wv axes are the 16-row arm's and do
     * not reach it. */
    {
        const bool pf_w = (tw.w[0].ld / 2) % 16 == 0 && (tw.w[1].ld / 2) % 16 == 0 &&
                          tw.w[0].s0 % 16 == 0 && tw.w[1].s0 % 16 == 0;
        if (c.a->dtype == RAD_F8E4M3 && c.T > 32 && c.n_expert <= kPfMaxE && c.K % kPfK == 0 &&
            a_ld % 16 == 0 && r3d_aligned(r3d_p(c.a), 16) && pf_w) {
            const dim3 pgrid((unsigned)((c.N + kPfN - 1) / kPfN),
                             (unsigned)((c.T + kPfM - 1) / kPfM + c.n_expert));
            const uint8_t* pa8 = (const uint8_t*)r3d_p(c.a);
#define R3D_PF(WGV)                                                                              \
    R3D_KLAUNCH((moe_w4a8_e4m3_pf_kernel<WGV>), pgrid, dim3(kThreads), (hipStream_t)s, pa8, psa, \
                ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld, (int)c.n_expert,  \
                (int)c.top_k, c.a_sorted)
            if (wg == 32) R3D_PF(32); else R3D_PF(128);
#undef R3D_PF
            return (int)hipGetLastError();
        }
    }
    if (grid.x * grid.y < 256u) grid.z = (unsigned)(c.T < mrows ? c.T : mrows);
    if (c.a->dtype == RAD_I8) {
        const int8_t* pa8 = (const int8_t*)r3d_p(c.a);
        if (mrows == 8 && wv == 4)
            { if (wg == 32) moe_w4a8_i8_kernel<8,4,32><<<grid, blk, 0, (hipStream_t)s>>>( pa8, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted); else moe_w4a8_i8_kernel<8,4,128><<<grid, blk, 0, (hipStream_t)s>>>( pa8, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted); }
        else if (mrows == 8)
            { if (wg == 32) moe_w4a8_i8_kernel<8,8,32><<<grid, blk, 0, (hipStream_t)s>>>( pa8, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted); else moe_w4a8_i8_kernel<8,8,128><<<grid, blk, 0, (hipStream_t)s>>>( pa8, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted); }
        else if (wv == 4)
            { if (wg == 32) moe_w4a8_i8_kernel<16,4,32><<<grid, blk, 0, (hipStream_t)s>>>( pa8, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted); else moe_w4a8_i8_kernel<16,4,128><<<grid, blk, 0, (hipStream_t)s>>>( pa8, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted); }
        else
            { if (wg == 32) moe_w4a8_i8_kernel<16,8,32><<<grid, blk, 0, (hipStream_t)s>>>( pa8, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted); else moe_w4a8_i8_kernel<16,8,128><<<grid, blk, 0, (hipStream_t)s>>>( pa8, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted); }
    } else {
        const uint8_t* pa8 = (const uint8_t*)r3d_p(c.a);
        if (mrows == 8 && wv == 4)
            { if (wg == 32) moe_w4a8_e4m3_kernel<8,4,32><<<grid, blk, 0, (hipStream_t)s>>>( pa8, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted); else moe_w4a8_e4m3_kernel<8,4,128><<<grid, blk, 0, (hipStream_t)s>>>( pa8, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted); }
        else if (mrows == 8)
            { if (wg == 32) moe_w4a8_e4m3_kernel<8,8,32><<<grid, blk, 0, (hipStream_t)s>>>( pa8, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted); else moe_w4a8_e4m3_kernel<8,8,128><<<grid, blk, 0, (hipStream_t)s>>>( pa8, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted); }
        else if (wv == 4)
            { if (wg == 32) moe_w4a8_e4m3_kernel<16,4,32><<<grid, blk, 0, (hipStream_t)s>>>( pa8, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted); else moe_w4a8_e4m3_kernel<16,4,128><<<grid, blk, 0, (hipStream_t)s>>>( pa8, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted); }
        else
            { if (wg == 32) moe_w4a8_e4m3_kernel<16,8,32><<<grid, blk, 0, (hipStream_t)s>>>( pa8, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted); else moe_w4a8_e4m3_kernel<16,8,128><<<grid, blk, 0, (hipStream_t)s>>>( pa8, psa, ps, po, tw, py, (int)c.T, (int)c.N, (int)c.K, a_ld, as_ld, y_ld, (int)c.n_expert, (int)c.top_k, c.a_sorted); }
    }
    return (int)hipGetLastError();
}

}  // extern "C"
