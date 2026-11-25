/* r3d_gemm_i8a8.hip -- the int8 trunk GEMM: gemm_nt_q at dtype i8a8 and i8a8h, every M.
 *
 *   y[M,N] = a[M,K] @ b[N,K]^T, int8 a against signed int8 codes, the scale group 128.
 *
 * The operands are quant_act_i8g's (i8a8) or had_quant_act_i8g's (i8a8h): the activation's codes
 * are row-major, with one f32 scale per (row, group); the weight is the container's I8 [N, K] as
 * stored, with one bf16 scale per (row, group). i8a8h is the same launch: the rotation is in the
 * stored weight and in the activation's quantiser, and the weight's scale carries the /128. This
 * is the w8a8 wave-matrix arm (r3d_gemm_w8a8_nt_m64.hip) with those three differences: no step
 * order to unpermute, the activation scale read per group, and a bf16 weight scale.
 *
 * ---- the group law --------------------------------------------------------------------------
 *
 * libref's quant_dot: a group's 128 products are |a * code| <= 128 * 128, their sum at most 2^21,
 * an exact integer far inside i32 and exact in f32, so the eight wmma steps of a group chain an
 * i32 sub-sum with no rounding, and the group term is
 *
 *   acc += (sub * a_scale[m, g]) * b_scale[n, g]
 *
 * in f32, in ascending group order, then one bf16 store. Each of the three operations rounds
 * as the reference's does (two multiplies, then the add), so the row is bit-identical to
 * libref's quant_dot, not merely inside gemm_nt_q's tolerance.
 *
 * ---- the shape of a block -------------------------------------------------------------------
 *
 * A wave is one 16-column tile of N over MT 16-row tiles of M; a block is eight waves, and the
 * grid is (N / 128, M / (16 * MT)). The B fragment (the weight column) is loaded once per 16-k
 * step and reused over the MT row tiles. MT is 1 for M <= 16 (decode: no second tile of clamped
 * duplicate rows) and 2 for 17 <= M < 128. A block reads its 128 weight rows once, so the weight
 * is read ceil(M / (16 * MT)) times in all; from M = 128 the tile kernel below serves the launch.
 *
 * ---- the B pointer --------------------------------------------------------------------------
 *
 * The weight row base `w = B + n_real * b_rb` carries NO group offset; the group's k-base
 * enters the fragment read exactly once, as `w + kt + k` with kt = g * 128.
 *
 * ---- the register budget --------------------------------------------------------------------
 *
 * Read from the built object (.vgpr_count): 104 at MT = 1 and 117 at MT = 2, no spills, of the
 * 128 a wave that eight waves a SIMD leave on gfx11.
 */

#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_common.h"
#include "r3d_codecs.h"
#include "r3d_launch.h"

#include <algorithm>

namespace {

constexpr int kWaves        = 8;
constexpr int kBlockThreads = kWaves * kWave;
constexpr int kGroup        = 128;            /* the scale group: one k tile, eight wmma steps */
constexpr int kPitch        = 33;             /* dwords a staged a row: 33 mod 32 = 1, bank-free */


/* Signed i8 in, signed i32 out: the builtin's sign selectors take the operands' signedness as
 * compile-time constants, both signed for this grid. */

template <bool NT>
__device__ __forceinline__ r3d_u32x4 ldw(const uint8_t* p)
{
    if constexpr (NT) return *(const __attribute__((address_space(1))) r3d_u32x4*)p;
    else return __builtin_bit_cast(r3d_u32x4, *(const uint4*)p);
}

/* y[M,N] = a[M,K] @ b[N,K]^T, MT 16-row tiles a wave, one 16-column tile a wave, one scale
 * group a k tile. The dot2 arm's contract: K a multiple of the group, the group term the
 * reference's, in ascending group order. */
template <bool NT, int MT>
__global__ __launch_bounds__(kBlockThreads) R3D_WAVES(8) void gemm_i8a8_wmma_kernel(
    const int8_t* __restrict__ a, const float* __restrict__ sa, int sa_rb,
    const uint8_t* __restrict__ B, int b_rb, const uint16_t* __restrict__ sb,
    int sb_rb, uint16_t* __restrict__ Y,
    int M, int N, int K, int a_ld, int y_ld)
{
    __shared__ __align__(16) uint32_t s_a[MT * 16 * kPitch];

    const uint32_t tid  = threadIdx.x;
    const uint32_t lane = tid & (kWave - 1);
    const uint32_t wave = tid / kWave;
    const uint32_t r = lane & 15;                   // the replicated row/column index

    const uint32_t n0 = (uint32_t)blockIdx.x * kWaves * 16 + wave * 16 + r;
    const int n_real = (int)(n0 < (uint32_t)N ? n0 : (uint32_t)(N - 1));
    const int mbase = (int)blockIdx.y * MT * 16;    // this block's first row

    const uint8_t* w = B + (size_t)n_real * b_rb;   // the row base: NO group offset
    v8float acc[MT];
    #pragma unroll
    for (int i = 0; i < MT; ++i) acc[i] = v8float{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };

    for (int g = 0; g < K / kGroup; ++g) {
        const int kt = g * kGroup;

        /* Stage MT*16 rows x 128 i8, row-major as the quantiser stored them. A row past M
         * re-reads the last real row, whose products the store drops. */
        for (int i = (int)tid; i < MT * 16 * (kGroup / 4); i += kBlockThreads) {
            const int row = i / (kGroup / 4), d = i % (kGroup / 4);
            const int rr = mbase + row;
            const uint32_t src =
                *(const uint32_t*)(a + (size_t)(rr < M ? rr : M - 1) * a_ld + kt + 4 * d);
            s_a[(size_t)row * kPitch + d] = src;
        }
        group_barrier();

        #pragma unroll
        for (int i = 0; i < MT; ++i) {
            const uint32_t* ap = s_a + (size_t)(i * 16 + r) * kPitch;
            v8int sub = v8int{ 0, 0, 0, 0, 0, 0, 0, 0 };
            for (int k = 0; k < kGroup; k += 16) {
                const r3d_u32x4 aq = *(const r3d_u32x4*)(ap + k / 4);
                // The group's k-base enters once (kt); the row base carries none.
                const r3d_u32x4 bq = ldw<NT>(w + kt + k);
                sub = wmma_i8(v4int{ (int)aq.x, (int)aq.y, (int)aq.z, (int)aq.w },
                              v4int{ (int)bq.x, (int)bq.y, (int)bq.z, (int)bq.w }, sub);
            }
            /* The group term, the reference's parentheses and the reference's roundings: the
             * product (sub * a_scale) * b_scale is rounded to f32 and then added, a v_mul and a
             * v_add, never a v_fmac. A fused multiply-add keeps the product unrounded, which
             * lands about half the launches of a model one bf16 step from quant_dot at their
             * worst element; libref and every unit here are built -ffp-contract=off, and an asm
             * fmac is outside that flag's reach. The b_scale load, its drain, its convert and
             * the eight terms run as one asm block: the load goes straight from global, the
             * drain is a vmcnt this asm owns, and the scale's register is written only by this
             * block's global_load, so no fragment load can clobber it mid-group. v20 (the scale)
             * and v21 (the rounded product) are named in the clobber because this LLVM aliases a
             * constrained asm output with a dead input register; the clobber keeps every
             * t/acc/sub register out of both across the block. The weight's scale is bf16: its
             * f32 is the word shifted up 16. The activation's scale is this group's, one per
             * (row, group). */
            float t[8];
            #pragma unroll
            for (int g2 = 0; g2 < 8; ++g2) {
                const int m = mbase + i * 16 + g2 * 2 + (int)(lane >> 4);
                t[g2] = (float)sub[g2] * sa[(size_t)(m < M ? m : M - 1) * sa_rb + g];
            }
            {
                const uint64_t ga = (uint64_t)(sb + (size_t)n_real * sb_rb + g);
                asm volatile(
                    "global_load_u16 v20, %[ga], off\n\t"
                    "s_waitcnt vmcnt(0)\n\t"
                    "v_lshlrev_b32_e32 v20, 16, v20\n\t"
                    "v_mul_f32_e32 v21, %[t0], v20\n\t"
                    "v_add_f32_e32 %[a0], %[a0], v21\n\t"
                    "v_mul_f32_e32 v21, %[t1], v20\n\t"
                    "v_add_f32_e32 %[a1], %[a1], v21\n\t"
                    "v_mul_f32_e32 v21, %[t2], v20\n\t"
                    "v_add_f32_e32 %[a2], %[a2], v21\n\t"
                    "v_mul_f32_e32 v21, %[t3], v20\n\t"
                    "v_add_f32_e32 %[a3], %[a3], v21\n\t"
                    "v_mul_f32_e32 v21, %[t4], v20\n\t"
                    "v_add_f32_e32 %[a4], %[a4], v21\n\t"
                    "v_mul_f32_e32 v21, %[t5], v20\n\t"
                    "v_add_f32_e32 %[a5], %[a5], v21\n\t"
                    "v_mul_f32_e32 v21, %[t6], v20\n\t"
                    "v_add_f32_e32 %[a6], %[a6], v21\n\t"
                    "v_mul_f32_e32 v21, %[t7], v20\n\t"
                    "v_add_f32_e32 %[a7], %[a7], v21"
                    : [a0] "+v"(acc[i][0]), [a1] "+v"(acc[i][1]), [a2] "+v"(acc[i][2]),
                      [a3] "+v"(acc[i][3]), [a4] "+v"(acc[i][4]), [a5] "+v"(acc[i][5]),
                      [a6] "+v"(acc[i][6]), [a7] "+v"(acc[i][7])
                    : [ga] "v"(ga),
                      [t0] "v"(t[0]), [t1] "v"(t[1]), [t2] "v"(t[2]), [t3] "v"(t[3]),
                      [t4] "v"(t[4]), [t5] "v"(t[5]), [t6] "v"(t[6]), [t7] "v"(t[7])
                    : "v20", "v21");
            }
        }
        group_barrier();                      // the next tile's stage must not race this one's reads
    }

    const bool keep = (n0 < (uint32_t)N);
    if (!keep) return;
    #pragma unroll
    for (int i = 0; i < MT; ++i)
    #pragma unroll
        for (int g2 = 0; g2 < 8; ++g2) {
            const int m = mbase + i * 16 + g2 * 2 + (int)(lane >> 4);
            if (m < M) Y[(size_t)m * y_ld + n0] = f32_to_bf16(acc[i][g2]);
        }
}

/* The decode GEMV: y[M,N] for M <= 4, on the integer dot (v_dot4_i32_iu8), no LDS and no barrier.
 *
 * Eight lanes cover one weight row's group (8 x 16 bytes = 128 codes), so a wave reads four rows a
 * column set, each as one whole 128-byte line, and CS column sets; U groups are loaded ahead of
 * their use. A lane's four dot4 give its 16 codes' i32 sum, the three xor steps (an integer add,
 * exact in any order) give the group's, and every lane of the eight then closes the group term
 * with the reference's parentheses and roundings, (sub * a_scale[m, g]) * b_scale[n, g] added to
 * the f32 accumulator, in ascending group order: bit-identical to quant_dot, as the wave-matrix arm
 * is. The activation rows are read straight from global (they are small and every wave shares
 * them). Rows past M re-read the last real row and are not stored. */
typedef r3d_u32x4 r3d_u32x4_a4v __attribute__((aligned(4)));

__device__ __forceinline__ r3d_u32x4 ldw_stream(const uint8_t* p)
{
    return __builtin_nontemporal_load((const __attribute__((address_space(1))) r3d_u32x4*)p);
}

__device__ __forceinline__ int sdot4(uint32_t a, uint32_t b, int c)
{
    return __builtin_amdgcn_sudot4(true, (int)a, true, (int)b, c, false);
}

template <int MR, int CS, int NU>
__device__ __forceinline__ void gemv_i8a8_groups(
    int g, const int8_t* const (&ap)[MR], const uint8_t* const (&wp)[CS],
    const uint16_t* const (&sp)[CS], const float* __restrict__ sa, int sa_rb, int M,
    float (&acc)[MR][CS])
{
    r3d_u32x4 w[NU][CS], x[NU][MR];
    uint32_t s[NU][CS];
    #pragma unroll
    for (int u = 0; u < NU; ++u) {
        #pragma unroll
        for (int c = 0; c < CS; ++c) {
            w[u][c] = ldw_stream(wp[c] + (size_t)(g + u) * kGroup);
            s[u][c] = ldg_u16(sp[c] + g + u);
        }
        #pragma unroll
        for (int m = 0; m < MR; ++m) x[u][m] = *(const r3d_u32x4_a4v*)(ap[m] + (size_t)(g + u) * kGroup);
    }
    #pragma unroll
    for (int u = 0; u < NU; ++u) {
        #pragma unroll
        for (int m = 0; m < MR; ++m) {
            const float sam = sa[(size_t)(m < M ? m : M - 1) * sa_rb + g + u];
            #pragma unroll
            for (int c = 0; c < CS; ++c) {
                int sub = sdot4(x[u][m].x, w[u][c].x, 0);
                sub = sdot4(x[u][m].y, w[u][c].y, sub);
                sub = sdot4(x[u][m].z, w[u][c].z, sub);
                sub = sdot4(x[u][m].w, w[u][c].w, sub);
                sub += (int)r3d_xor_lane_u32<1>((uint32_t)sub);
                sub += (int)r3d_xor_lane_u32<2>((uint32_t)sub);
                sub += (int)r3d_xor_lane_u32<4>((uint32_t)sub);
                const float t = (float)sub * sam;
                const float sbv = __uint_as_float(s[u][c] << 16);
                acc[m][c] = acc[m][c] + t * sbv;
            }
        }
    }
}

template <int MR, int CS, int U>
__global__ __launch_bounds__(kBlockThreads) R3D_WAVES(8) void gemv_i8a8_kernel(
    const int8_t* __restrict__ a, const float* __restrict__ sa, int sa_rb,
    const uint8_t* __restrict__ B, int b_rb, const uint16_t* __restrict__ sb,
    int sb_rb, uint16_t* __restrict__ Y,
    int M, int N, int K, int a_ld, int y_ld)
{
    const uint32_t lane = threadIdx.x & (kWave - 1);
    const uint32_t q = lane & 7, cg = lane >> 3;
    const uint32_t wv = (uint32_t)blockIdx.x * (kBlockThreads / kWave) + threadIdx.x / kWave;

    int nc[CS];
    const uint8_t* wp[CS];
    const uint16_t* sp[CS];
    #pragma unroll
    for (int c = 0; c < CS; ++c) {
        nc[c] = (int)((wv * CS + c) * 4 + cg);
        const int nr = nc[c] < N ? nc[c] : N - 1;
        wp[c] = B + (size_t)nr * b_rb + 16 * q;
        sp[c] = sb + (size_t)nr * sb_rb;
    }
    const int8_t* ap[MR];
    #pragma unroll
    for (int m = 0; m < MR; ++m) ap[m] = a + (size_t)(m < M ? m : M - 1) * a_ld + 16 * q;

    float acc[MR][CS];
    #pragma unroll
    for (int m = 0; m < MR; ++m)
        #pragma unroll
        for (int c = 0; c < CS; ++c) acc[m][c] = 0.f;

    const int G = K / kGroup;
    int g = 0;
    for (; g + U <= G; g += U) gemv_i8a8_groups<MR, CS, U>(g, ap, wp, sp, sa, sa_rb, M, acc);
    for (; g < G; ++g) gemv_i8a8_groups<MR, CS, 1>(g, ap, wp, sp, sa, sa_rb, M, acc);

    if (q) return;
    #pragma unroll
    for (int m = 0; m < MR; ++m)
        #pragma unroll
        for (int c = 0; c < CS; ++c)
            if (m < M && nc[c] < N) Y[(size_t)m * y_ld + nc[c]] = f32_to_bf16(acc[m][c]);
}

/* The prefill tile: y[M,N] over 128 x 128 output tiles, the wave-matrix twin of the bf16 tile in
 * r3d_vocab_bf16_wmma.hip. A block stages one scale group of both operands in LDS (128 activation
 * rows and 128 weight rows, 128 codes each) and the 128 activation scales of the group; wave w owns
 * columns [w*16, w*16+16) and all eight 16-row tiles. Per row tile the group's eight wave-matrix
 * steps chain an exact i32 sub-sum, and the group closes into that tile's f32 accumulator with the
 * reference's parentheses and roundings, written in C (this unit is built -ffp-contract=off):
 *
 *   acc = acc + ((float)sub * a_scale[m, g]) * b_scale[n, g]
 *
 * The weight is read ceil(M / 128) times in all, a quarter of the 32-row block's reads. Rows past
 * M and columns past N are staged clamped to the last real one and dropped at the store.
 *
 * Registers: a block holds 34,304 bytes of LDS, and with that much the compiler does not hold the
 * kernel to an eight-waves-a-SIMD budget (a draft declared so was allocated 256 registers). The
 * kernel declares four waves a SIMD, 256 registers a wave, and the built object's .vgpr_count is
 * 163 with no spill: 64 accumulators, the group's 32 weight-fragment registers (read from LDS
 * once a group and kept over the eight row tiles), the sub-sum, an activation fragment and the
 * store's addresses. */
constexpr int kTM = 128;                      /* activation rows a block */
constexpr int kTN = 128;                      /* weight rows (output columns) a block */
constexpr int kGD = kGroup / 4;               /* dwords a staged row of one group */

/* The activation's 16-byte load: the shim holds `a` and its pitch to 4 bytes, not 16. */
typedef r3d_u32x4 r3d_u32x4_a4 __attribute__((aligned(4)));

/* A value the optimiser cannot see through: an empty asm that takes and returns it in a VGPR. */
__device__ __forceinline__ uint32_t lds_base(uint32_t v)
{
    asm volatile("" : "+v"(v));
    return v;
}

template <bool NT>
__global__ __launch_bounds__(kBlockThreads) R3D_WAVES(4) void gemm_i8a8_tile_kernel(
    const int8_t* __restrict__ a, const float* __restrict__ sa, int sa_rb,
    const uint8_t* __restrict__ B, int b_rb, const uint16_t* __restrict__ sb,
    int sb_rb, uint16_t* __restrict__ Y,
    int M, int N, int K, int a_ld, int y_ld)
{
    __shared__ __align__(16) uint32_t s_a[kTM * kPitch];
    __shared__ __align__(16) uint32_t s_b[kTN * kPitch];
    __shared__ float s_sa[kTM];

    const uint32_t tid  = threadIdx.x;
    const uint32_t lane = tid & (kWave - 1);
    const uint32_t wave = tid / kWave;
    const uint32_t r = lane & 15;                   // the replicated row/column index

    const int m0 = (int)blockIdx.y * kTM;
    const int nb = (int)blockIdx.x * kTN;
    const uint32_t n0 = (uint32_t)nb + wave * 16 + r;
    const int n_real = (int)(n0 < (uint32_t)N ? n0 : (uint32_t)(N - 1));

    v8float acc[kTM / 16];
    #pragma unroll
    for (int i = 0; i < kTM / 16; ++i) acc[i] = v8float{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };

    /* The LDS row bases of this lane, one a row tile, each made opaque to the optimiser. Left
     * visible, every read below is `one base + a constant up to 15 KB`, the 2-address LDS load
     * reaches 1 KB, and the compiler keeps a register for the address of every pair of dwords
     * (about 130 of them, hoisted out of the group loop). With an opaque base a row tile's 32
     * dwords are within the load's own offset field. */
    uint32_t ao[kTM / 16];
    #pragma unroll
    for (int rb = 0; rb < kTM / 16; ++rb) ao[rb] = lds_base((uint32_t)(rb * 16 + r) * kPitch);
    const uint32_t bo = lds_base((wave * 16 + r) * kPitch);
    const uint32_t so = lds_base(lane >> 4);

    for (int g = 0; g < K / kGroup; ++g) {
        const int kt = g * kGroup;

        /* Stage the group: 16 bytes a thread a turn, the weight straight from its stored rows. */
        for (int i = (int)tid; i < kTM * (kGD / 4); i += kBlockThreads) {
            const int row = i / (kGD / 4), q = i % (kGD / 4);
            const int rr = m0 + row;
            const r3d_u32x4 v = *(const r3d_u32x4_a4*)(a + (size_t)(rr < M ? rr : M - 1) * a_ld + kt + 16 * q);
            uint32_t* d = s_a + (size_t)row * kPitch + 4 * q;
            d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w;
        }
        for (int i = (int)tid; i < kTN * (kGD / 4); i += kBlockThreads) {
            const int col = i / (kGD / 4), q = i % (kGD / 4);
            const int nn = nb + col;
            const r3d_u32x4 v = ldw<NT>(B + (size_t)(nn < N ? nn : N - 1) * b_rb + kt + 16 * q);
            uint32_t* d = s_b + (size_t)col * kPitch + 4 * q;
            d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w;
        }
        if (tid < (uint32_t)kTM) {
            const int rr = m0 + (int)tid;
            s_sa[tid] = ldg_f32(sa + (size_t)(rr < M ? rr : M - 1) * sa_rb + g);
        }
        group_barrier();

        const float sbv = bf16_to_f32((uint16_t)ldg_u16(sb + (size_t)n_real * sb_rb + g));
        const uint32_t* bp = s_b + bo;
        const float* sp = s_sa + so;
        #pragma unroll
        for (int rb = 0; rb < kTM / 16; ++rb) {
            const uint32_t* ap = s_a + ao[rb];
            v8int sub = v8int{ 0, 0, 0, 0, 0, 0, 0, 0 };
            #pragma unroll
            for (int k = 0; k < kGD; k += 4)
                sub = wmma_i8(v4int{ (int)ap[k], (int)ap[k + 1], (int)ap[k + 2], (int)ap[k + 3] },
                              v4int{ (int)bp[k], (int)bp[k + 1], (int)bp[k + 2], (int)bp[k + 3] }, sub);
            #pragma unroll
            for (int g2 = 0; g2 < 8; ++g2) {
                const float t = (float)sub[g2] * sp[rb * 16 + g2 * 2];
                acc[rb][g2] = acc[rb][g2] + t * sbv;
            }
        }
        group_barrier();                      // the next group's stage must not race these reads
    }

    if (n0 >= (uint32_t)N) return;
    #pragma unroll
    for (int rb = 0; rb < kTM / 16; ++rb)
    #pragma unroll
        for (int g2 = 0; g2 < 8; ++g2) {
            const int m = m0 + rb * 16 + g2 * 2 + (int)(lane >> 4);
            if (m < M) Y[(size_t)m * y_ld + n0] = f32_to_bf16(acc[rb][g2]);
        }
}

}  // namespace

/* ==================================================================================================
 * The RadArgs shim. Schema positions: a 0, a_scale 1, b 2, b_scale 3, y 4, a_sum 5, b_ref 6.
 * ================================================================================================== */
extern "C" int r3d_rad_gemm_nt_q_i8a8(const RadArgs* a, RadStream s)
{
    if (!a) return R3D_NO(RAD_E_INVAL);
    const char* dt = r3d_gets(a, "dtype");
    if (!dt || (std::strcmp(dt, "i8a8") != 0 && std::strcmp(dt, "i8a8h") != 0))
        return R3D_NO(RAD_E_DTYPE);
    if (r3d_geti_or(a, "group", kGroup) != kGroup) return R3D_NO(RAD_E_SHAPE);

    const RadTensor* A = r3d_opt(a, 0);
    const RadTensor* AS = r3d_opt(a, 1);
    const RadTensor* B = r3d_opt(a, 2);
    const RadTensor* BS = r3d_opt(a, 3);
    const RadTensor* Y = r3d_opt(a, 4);
    const RadTensor* ASUM = r3d_opt(a, 5);
    const RadTensor* BREF = r3d_opt(a, 6);
    if (!r3d_p(A) || !r3d_p(AS) || !r3d_p(B) || !r3d_p(BS) || !r3d_p(Y))
        return R3D_NO(RAD_E_INVAL);
    if (r3d_has(ASUM) || r3d_has(BREF)) return R3D_NO(RAD_E_DTYPE);

    if (!r3d_is_i8(A)) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_f32(AS)) return R3D_NO(RAD_E_DTYPE);
    if (B->dtype != RAD_I8) return R3D_NO(RAD_E_DTYPE);
    if (BS->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_bf16(Y)) return R3D_NO(RAD_E_DTYPE);

    const long long N = r3d_geti_or(a, "N", 0), K = r3d_geti_or(a, "K", 0);
    if (N < 1 || K < 1) return R3D_NO(RAD_E_SHAPE);
    if (K % kGroup) return R3D_NO(RAD_E_SHAPE);
    const long long G = K / kGroup;

    const long long ynel = r3d_numel(Y);
    if (ynel % N) return R3D_NO(RAD_E_SHAPE);
    const long long M = ynel / N;
    if (M < 1 || M > (1 << 24)) return R3D_NO(RAD_E_SHAPE);

    if (r3d_numel(A) < M * K) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(AS) < M * G) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(B) < N * K) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(BS) < N * G) return R3D_NO(RAD_E_SHAPE);

    const long long ap = r3d_row_pitch(A);
    if (ap < K || ap % 4) return R3D_NO(RAD_E_STRIDE);
    const long long asp = M > 1 ? r3d_row_pitch(AS) : G;
    if (asp < G) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_contig(B)) return R3D_NO(RAD_E_STRIDE);
    const long long bp = r3d_row_pitch(B), bsp = r3d_row_pitch(BS);
    if (bp < K || (bp & 15) || bsp < G) return R3D_NO(RAD_E_STRIDE);
    long long yrows = 0;
    const long long yp = r3d_rows_pitch(Y, N, &yrows);
    if (yp < N) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_aligned(r3d_p(A), 4) || !r3d_aligned(r3d_p(B), 16)) return R3D_NO(RAD_E_ALIGN);

    R3dGemmCfg c;
    r3d_cfg_gemm(a, &c);

    const dim3 block(kBlockThreads);
    if (M >= kTM) {
        const dim3 tgrid((unsigned)((N + kTN - 1) / kTN), (unsigned)((M + kTM - 1) / kTM));
#define R3D_I8A8T(NTV)                                                                            \
    R3D_KLAUNCH((gemm_i8a8_tile_kernel<NTV>), tgrid, block, (hipStream_t)s,                       \
                (const int8_t*)r3d_p(A), (const float*)r3d_p(AS), (int)asp,                       \
                (const uint8_t*)r3d_p(B), (int)bp, (const uint16_t*)r3d_p(BS), (int)bsp,          \
                (uint16_t*)r3d_wp(Y), (int)M, (int)N, (int)K, (int)ap, (int)yp)
        if (c.nt) R3D_I8A8T(true); else R3D_I8A8T(false);
#undef R3D_I8A8T
        return hipGetLastError() == hipSuccess ? RAD_OK : R3D_NO(RAD_E_DEVICE);
    }
    /* M <= 4, decode: the integer-dot GEMV, two column sets (eight weight rows) a wave. */
    if (M <= 4) {
        constexpr int kCS = 2;
        const long long waves = (N + 4 * kCS - 1) / (4 * kCS);
        const dim3 vgrid((unsigned)((waves + kWaves - 1) / kWaves));
#define R3D_I8A8V(MRV, UV)                                                                        \
    R3D_KLAUNCH((gemv_i8a8_kernel<MRV, kCS, UV>), vgrid, block, (hipStream_t)s,                   \
                (const int8_t*)r3d_p(A), (const float*)r3d_p(AS), (int)asp,                       \
                (const uint8_t*)r3d_p(B), (int)bp, (const uint16_t*)r3d_p(BS), (int)bsp,          \
                (uint16_t*)r3d_wp(Y), (int)M, (int)N, (int)K, (int)ap, (int)yp)
        if (M == 1) R3D_I8A8V(1, 4); else if (M == 2) R3D_I8A8V(2, 4); else R3D_I8A8V(4, 2);
#undef R3D_I8A8V
        return hipGetLastError() == hipSuccess ? RAD_OK : R3D_NO(RAD_E_DEVICE);
    }
    const int mt = M <= 16 ? 1 : 2;
    const dim3 grid((unsigned)((N + kWaves * 16 - 1) / (kWaves * 16)),
                    (unsigned)((M + mt * 16 - 1) / (mt * 16)));
#define R3D_I8A8(NTV, MTV)                                                                        \
    R3D_KLAUNCH((gemm_i8a8_wmma_kernel<NTV, MTV>), grid, block, (hipStream_t)s,                   \
                (const int8_t*)r3d_p(A), (const float*)r3d_p(AS), (int)asp,                       \
                (const uint8_t*)r3d_p(B), (int)bp, (const uint16_t*)r3d_p(BS), (int)bsp,          \
                (uint16_t*)r3d_wp(Y), (int)M, (int)N, (int)K, (int)ap, (int)yp)
    if (c.nt) { if (mt == 1) R3D_I8A8(true, 1); else R3D_I8A8(true, 2); }
    else      { if (mt == 1) R3D_I8A8(false, 1); else R3D_I8A8(false, 2); }
#undef R3D_I8A8
    return hipGetLastError() == hipSuccess ? RAD_OK : R3D_NO(RAD_E_DEVICE);
}
