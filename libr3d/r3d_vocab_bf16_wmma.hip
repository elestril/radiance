/* r3d_vocab_bf16_wmma.hip -- the prefill tiled GEMM on the wave-matrix: the WMMA twin of the
 * ALU fdot2 kernel in r3d_vocab_bf16.hip (the `gemm_nt_bf16_tiled` row's compute-dual).
 *
 *   y[M,N] = a[M,K] @ w[N,K]^T, bf16 in, f32 accumulate, bf16 out (one narrowing at the store).
 *
 * The ALU arm owns 64 f32 accumulators a lane (an 8x8 square) and runs v_fdot2; this arm runs the
 * same 128x128 tile on the wave-matrix, which is the matrix core at 512 FLOPS/clk/CU against the
 * ALU's ~256, so it is the faster of the pair at prefill heights. The tile is split eight ways
 * across the eight waves by column: wave w owns cols [w*16, w*16+16) and all 128 rows, so it
 * accumulates eight 16x16 wave-matrix tiles (one per 16-row block) -- 64 f32 accumulators a wave,
 * the same count the ALU arm keeps a lane. Every wave does unique work (the 16x128 wave-filling
 * retile), so no wave re-runs a tile.
 *
 * K is walked in 32-deep chunks (two 16-wide wave-matrix steps); the operands stage to LDS as bf16
 * (16 dwords a row, +1 bank-free) and load straight into the bf16 wave-matrix fragments.
 */
#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_codecs.h"
#include "r3d_launch.h"

namespace {

constexpr int kThreads = 256;           // eight waves
constexpr int kWaves   = kThreads / kWave;
constexpr int kTM      = 128;           // activation rows a block
constexpr int kTN      = 128;           // weight cols a block
constexpr int kKC      = 32;            // k codes a chunk: two 16-wide wave-matrix steps
constexpr int kDW      = kKC / 2;       // dwords a row: 32 bf16 are 16 dwords (two bf16 a dword)
constexpr int kPitch   = kDW + 1;       // dwords a staged row: +1, bank-free

__global__ __launch_bounds__(kThreads) R3D_WAVES(kWaves) void vocab_bf16_wmma_kernel(
    const uint16_t* __restrict__ a,
    const uint16_t* __restrict__ w,
    uint16_t* __restrict__ y, int M, int N, int K, int a_ld, int w_ld, int y_ld)
{
    __shared__ __align__(16) uint32_t sA[kTM * kPitch];   // 128 rows x 16 dwords
    __shared__ __align__(16) uint32_t sB[kTN * kPitch];   // 128 cols x 16 dwords

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & (kWave - 1);
    const uint32_t wv = tid / kWave;          // the wave index (0..7): owns cols wv*16..wv*16+16
    const uint32_t r = lane & 15;

    const uint32_t m0 = (uint32_t)blockIdx.y * kTM;
    const uint32_t n0 = (uint32_t)blockIdx.x * kTN;

    // acc[8]: one 16x16 wave-matrix tile per 16-row block (128 rows / 16).
    v8float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = v8float{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };

    for (int k0 = 0; k0 < K; k0 += kKC) {
        // Stage 128 activation rows and 128 weight cols (16 dwords each), clamped at the edge.
        for (int i = (int)tid; i < kTM * kDW; i += kThreads) {
            const int row = i / kDW, d = i % kDW;
            const int mm = (int)(m0 + row < (uint32_t)M ? m0 + row : (uint32_t)M - 1);
            sA[row * kPitch + d] = *(const uint32_t*)(a + (size_t)mm * a_ld + k0 + 2 * d);
        }
        for (int i = (int)tid; i < kTN * kDW; i += kThreads) {
            const int col = i / kDW, d = i % kDW;
            const int nn = (int)(n0 + col < (uint32_t)N ? n0 + col : (uint32_t)N - 1);
            sB[col * kPitch + d] = *(const uint32_t*)(w + (size_t)nn * w_ld + k0 + 2 * d);
        }
        group_barrier();

        // Wave wv owns cols wv*16..wv*16+16, all eight 16-row blocks.
        const uint32_t* wp = sB + (size_t)(wv * 16 + r) * kPitch;
        #pragma unroll
        for (int rb = 0; rb < 8; ++rb) {
            const uint32_t* ap = sA + (size_t)(rb * 16 + r) * kPitch;
            #pragma unroll
            for (int k = 0; k < kKC; k += 16) {
                v16short fa, fb;
                #pragma unroll
                for (int g = 0; g < 8; ++g) {
                    const uint32_t dw = ap[k / 2 + g];
                    fa[2 * g] = (short)(dw & 0xffffu); fa[2 * g + 1] = (short)(dw >> 16);
                    const uint32_t dwb = wp[k / 2 + g];
                    fb[2 * g] = (short)(dwb & 0xffffu); fb[2 * g + 1] = (short)(dwb >> 16);
                }
                acc[rb] = wmma_bf16(fa, fb, acc[rb]);
            }
        }
        group_barrier();
    }

    // Store: C-frag lane l, VGPR g -> row 2*(l>>4)+g, col l&15; offset by the row block and wave.
    #pragma unroll
    for (int rb = 0; rb < 8; ++rb)
        #pragma unroll
        for (int g = 0; g < 8; ++g) {
            const int m = m0 + rb * 16 + 2 * g + (lane >> 4);
            const int n = n0 + wv * 16 + r;
            if (m < M && n < N) y[m * y_ld + n] = f32_to_bf16(acc[rb][g]);
        }
}

/* The same 128x128 tile in libhalo's halo_gemm_bf16_nt_tiled shape, where K % 64 == 0: eight waves
 * as 2 (M) x 4 (N), each a 64 x 32 patch (4 x 2 wave-matrix tiles), so a k step reads six
 * fragments for eight WMMAs where the kernel above reads nine for eight; K staged 64 deep with
 * 16-byte loads, the next tile's loaded into registers while this one multiplies, and blocks taken
 * in groups of eight M bands so blocks in flight share their weight tiles in L2. Each output is the
 * kernel above's chain -- one WMMA per 16 of K in ascending order from zero, the same operands in
 * the same slots -- so the two store the same bytes. The barriers order LDS only: a workgroup fence
 * over every address space would wait for the next tile's loads. */
constexpr int kT2K = 64;                 // K a staged tile: four wave-matrix steps
constexpr int kT2P = kT2K + 8;           // bf16 a staged row: 144 bytes, 16-byte aligned

__device__ __forceinline__ void t2_barrier()
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

__device__ __forceinline__ v16short t2_frag(const uint16_t* p)
{
    const r3d_u32x4 lo = *(const r3d_u32x4*)p, hi = *(const r3d_u32x4*)(p + 8);
    typedef unsigned int u32x8 __attribute__((ext_vector_type(8)));
    return __builtin_bit_cast(v16short, u32x8{ lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w });
}

__global__ __launch_bounds__(kThreads) void gemm_bf16_t2_kernel(
    const uint16_t* __restrict__ a, const uint16_t* __restrict__ w, uint16_t* __restrict__ y,
    int M, int N, int K, int a_ld, int w_ld, int y_ld)
{
    constexpr int TM = 4, TN = 2, CPR = kT2K / 8;         // 16-byte chunks a staged row
    constexpr int AL = kTM * CPR / kThreads, WL = kTN * CPR / kThreads;
    __shared__ __align__(16) uint16_t sA[kTM * kT2P];
    __shared__ __align__(16) uint16_t sW[kTN * kT2P];

    const int tid = (int)threadIdx.x, lane = tid & (kWave - 1);
    const int wave = __builtin_amdgcn_readfirstlane(tid / kWave);
    const int wm = wave / 4, wn = wave % 4;

    // Groups of eight M bands, M fastest within a group (libhalo's remap).
    constexpr unsigned kGroupM = 8;
    const unsigned per = kGroupM * gridDim.x, lin = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned first = lin / per * kGroupM, rr = lin % per;
    const unsigned gm = min(gridDim.y - first, kGroupM);
    const int m0 = (int)(first + rr % gm) * kTM, n0 = (int)(rr / gm) * kTN;

    const uint16_t* asrc[AL];
    const uint16_t* wsrc[WL];
    int adst[AL], wdst[WL];
#pragma unroll
    for (int i = 0; i < AL; ++i) {
        const int c = tid + kThreads * i, r = c / CPR, e = (c % CPR) * 8;
        const int gr = m0 + r < M ? m0 + r : M - 1;
        asrc[i] = a + (size_t)gr * a_ld + e;
        adst[i] = r * kT2P + e;
    }
#pragma unroll
    for (int i = 0; i < WL; ++i) {
        const int c = tid + kThreads * i, r = c / CPR, e = (c % CPR) * 8;
        const int gr = n0 + r < N ? n0 + r : N - 1;
        wsrc[i] = w + (size_t)gr * w_ld + e;
        wdst[i] = r * kT2P + e;
    }
    const uint16_t* fa = sA + (wm * TM * 16 + (lane & 15)) * kT2P;
    const uint16_t* fw = sW + (wn * TN * 16 + (lane & 15)) * kT2P;

    v8float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = v8float{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };

    r3d_u32x4 ra[AL], rw[WL];
#pragma unroll
    for (int i = 0; i < AL; ++i) ra[i] = *(const r3d_u32x4*)asrc[i];
#pragma unroll
    for (int i = 0; i < WL; ++i) rw[i] = *(const r3d_u32x4*)wsrc[i];

    const int nk = K / kT2K;
    for (int kt = 0; kt < nk; ++kt) {
        t2_barrier();                                    // the last tile's fragments are all read
#pragma unroll
        for (int i = 0; i < AL; ++i) *(r3d_u32x4*)(sA + adst[i]) = ra[i];
#pragma unroll
        for (int i = 0; i < WL; ++i) *(r3d_u32x4*)(sW + wdst[i]) = rw[i];
        t2_barrier();
        if (kt + 1 < nk) {
            const int k1 = (kt + 1) * kT2K;
#pragma unroll
            for (int i = 0; i < AL; ++i) ra[i] = *(const r3d_u32x4*)(asrc[i] + k1);
#pragma unroll
            for (int i = 0; i < WL; ++i) rw[i] = *(const r3d_u32x4*)(wsrc[i] + k1);
        }
#pragma unroll
        for (int st = 0; st < kT2K / 16; ++st) {
            v16short af[TM], bf[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) af[i] = t2_frag(fa + i * 16 * kT2P + st * 16);
#pragma unroll
            for (int j = 0; j < TN; ++j) bf[j] = t2_frag(fw + j * 16 * kT2P + st * 16);
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = wmma_bf16(af[i], bf[j], acc[i][j]);
            /* A step's fragment reads, then its products (libhalo: 6-9% on this tile). */
            __builtin_amdgcn_sched_group_barrier(0x100, TM + TN, 0);
            __builtin_amdgcn_sched_group_barrier(0x008, TM * TN, 0);
        }
    }

#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int n = n0 + (wn * TN + j) * 16 + (lane & 15);
            const int mb = m0 + (wm * TM + i) * 16;
#pragma unroll
            for (int g = 0; g < 8; ++g) {
                const int m = mb + 2 * g + (lane >> 4);
                if (m < M && n < N) y[(size_t)m * y_ld + n] = f32_to_bf16(acc[i][j][g]);
            }
        }
}

int r3d_vocab_bf16_wmma(long av, long wv, long yv, int M, int N, int K, int a_ld, int w_ld,
                        int y_ld, long stream)
{
    if (!av || !wv || !yv) return -1;
    if (M < 1 || N < 1 || K < 1) return -2;
    if (K % kKC) return -2;   // the tile's k-chunk; the M/N tails clamp and are dropped at the store
    if (!a_ld) a_ld = K;
    if (!w_ld) w_ld = K;
    if (!y_ld) y_ld = N;
    if (a_ld < K || w_ld < K || y_ld < N) return -3;
    if ((av & 15) || (wv & 15) || (yv & 15)) return -5;

    const uint16_t* a = (const uint16_t*)(uintptr_t)av;
    const uint16_t* w = (const uint16_t*)(uintptr_t)wv;
    uint16_t* y = (uint16_t*)(uintptr_t)yv;
    hipStream_t s = (hipStream_t)(uintptr_t)stream;

    dim3 grid((unsigned)(N + kTN - 1) / kTN, (unsigned)(M + kTM - 1) / kTM), blk(kThreads);
    if (K % kT2K == 0 && (a_ld % 8) == 0 && (w_ld % 8) == 0)
        gemm_bf16_t2_kernel<<<grid, blk, 0, s>>>(a, w, y, M, N, K, a_ld, w_ld, y_ld);
    else
        vocab_bf16_wmma_kernel<<<grid, blk, 0, s>>>(a, w, y, M, N, K, a_ld, w_ld, y_ld);
    return (int)hipGetLastError();
}

}  // namespace

extern "C" {

int r3d_gemm_nt_bf16_tiled_wmma(long a, long b, long y, int M, int N, int K, int a_ld, int w_ld,
                                int y_ld, long stream)
{ return r3d_vocab_bf16_wmma(a, b, y, M, N, K, a_ld, w_ld, y_ld, stream); }

int r3d_rad_gemm_nt_bf16_tiled_wmma(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 3) return R3D_NO(RAD_E_INVAL);
    const RadTensor* at = &a->t[0], *wt = &a->t[1], *yt = &a->t[2];
    if (!r3d_p(at) || !r3d_p(wt) || !r3d_wp(yt)) return R3D_NO(RAD_E_INVAL);
    if (at->dtype != RAD_BF16 || wt->dtype != RAD_BF16 || yt->dtype != RAD_BF16)
        return R3D_NO(RAD_E_DTYPE);   // the plain bf16 head; the quantised form is the ALU arm's
    long long M = 0, K = 0;
    if (r3d_rows_of(at, &M, &K) != 0) return R3D_NO(RAD_E_INVAL);
    const long long N = (wt->rank >= 2) ? wt->shape[0] : 0;
    if (M <= 0 || N <= 0 || K <= 0 || (K & 31))
        return R3D_NO(RAD_E_SHAPE);   // the 128x128 tile, 32-deep k chunks; the M/N tails clamp
    const int a_ld = (int)at->stride[0];
    const int w_ld = (int)wt->stride[0];
    const int y_ld = (int)yt->stride[0];
    return r3d_vocab_bf16_wmma((long)(uintptr_t)r3d_p(at), (long)(uintptr_t)r3d_p(wt),
                               (long)(uintptr_t)r3d_wp(yt), (int)M, (int)N, (int)K, a_ld, w_ld,
                               y_ld, (long)(uintptr_t)s);
}

}  // extern "C"
