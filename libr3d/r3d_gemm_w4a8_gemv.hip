/* r3d_gemm_w4a8_gemv.hip -- the decode form of gemm_w4a8_nt_m64 (Gemma 4 31B's trunk): M <= 4.
 *
 *   C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T, int8 A with an f32 scale a row, signed 4-bit W with an
 *   f16 scale per 128 of K, bf16 C.
 *
 * libhalo's kernel (r3dh_gemm_w4a8_nt_m64.hip, carried verbatim) runs a 16x16x16 int8 wave
 * matrix a k step: at M = 1 fifteen of its sixteen rows are clamped copies, and every step pays
 * three cross-half shuffles to rebuild the gfx11 fragment from the gfx12 fragment order both
 * operands are stored in. A group's integer sub-sum is exact however it is formed, so this form
 * computes it with v_dot4_i32_iu8 straight from the stored order: in a k step, lane half h holds
 * the column's K quarters h and h + 2 (two four-code dwords after the 16 * code unpack), and the
 * activation's eight bytes of that half are the same two quarters (the quantiser's in-step
 * shuffle). One xor-16 integer add joins the halves. Everything after the sub-sum is the kernel's:
 * acc = fmaf((float)sub, scale, acc) in ascending group order within the law's split, the splits
 * summed from 0.f in ascending order through LDS, then v * Ascale[m] * 0.0625f to bf16. The split
 * is the one the m64 kernel would take (the same cfg_for), so the output is the same bytes.
 * Every weight byte is read once by one lane, sixteen bytes a load, U groups ahead.
 */
#include "r3dh.h"
#include "r3dh_gemm_args.h"
#include "r3dh_common.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <type_traits>

extern "C" int r3dh_rad_gemm_w4a8_m64(const RadArgs* a, RadStream s);

namespace {

typedef unsigned uint4w __attribute__((ext_vector_type(4)));
typedef unsigned uint2w __attribute__((ext_vector_type(2)));

__device__ __forceinline__ int sdot4(unsigned a, unsigned b, int c)
{
    return __builtin_amdgcn_sudot4(true, (int)a, true, (int)b, c, false);
}

__device__ __forceinline__ int xor16(int v)
{
    return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xFEDCBA98u, false, false);
}

template <int MR, int U>
__global__ __launch_bounds__(1024) void w4a8_gemv_kernel(
    const signed char* __restrict__ A, const float* __restrict__ Ascale,
    const uint4w* __restrict__ Wq, const unsigned* __restrict__ Ws,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int WV, int SK)
{
    extern __shared__ float red[];                 // [MR][WV][SK][16]
    const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
    const int col_local = wave / SK, ksplit = wave % SK;
    const int nt0 = blockIdx.x * WV + col_local;
    const int Ks = K / SK, groups = Ks / 128, kbase = ksplit * Ks;
    const int ntlast = N / 16 - 1;
    const int t = nt0 < ntlast ? nt0 : ntlast;
    const int h = lane >> 4;
    const uint4w* pw = Wq + (size_t)(t * (K / 64) + kbase / 64) * 32 + lane;
    const unsigned* ps = Ws + (size_t)(t * (K / 128) + kbase / 128) * 16 + (lane & 15);
    const signed char* pa[MR];
#pragma unroll
    for (int m = 0; m < MR; ++m) pa[m] = A + (size_t)(m < M ? m : M - 1) * K + kbase + 8 * h;

    float acc[MR];
#pragma unroll
    for (int m = 0; m < MR; ++m) acc[m] = 0.f;

    auto groups_at = [&](int g, auto nu_tag) {
        constexpr int NU = decltype(nu_tag)::value;
        uint4w w[NU][2];
        float sc[NU];
#pragma unroll
        for (int u = 0; u < NU; ++u) {
#pragma unroll
            for (int b = 0; b < 2; ++b) w[u][b] = __builtin_nontemporal_load(&pw[(size_t)((g + u) * 2 + b) * 32]);
            sc[u] = __half2float(__ushort_as_half((unsigned short)(ps[(size_t)(g + u) * 16] & 0xFFFFu)));
        }
#pragma unroll
        for (int u = 0; u < NU; ++u) {
#pragma unroll
            for (int m = 0; m < MR; ++m) {
                const uint2w* ap = (const uint2w*)(pa[m] + (size_t)(g + u) * 128);
                int sub = 0;
#pragma unroll
                for (int b = 0; b < 2; ++b)
#pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        const uint2w av = ap[2 * (4 * b + s)];
                        const unsigned dw = w[u][b][s];
                        sub = sdot4(av[0], (dw << 4) & 0xF0F0F0F0u, sub);
                        sub = sdot4(av[1], dw & 0xF0F0F0F0u, sub);
                    }
                sub += xor16(sub);
                acc[m] = fmaf((float)sub, sc[u], acc[m]);
            }
        }
    };
    int g = 0;
    for (; g + U <= groups; g += U) groups_at(g, std::integral_constant<int, U>{});
    for (; g < groups; ++g) groups_at(g, std::integral_constant<int, 1>{});

    if (lane < 16)
#pragma unroll
        for (int m = 0; m < MR; ++m) red[((m * WV + col_local) * SK + ksplit) * 16 + lane] = acc[m];
    __syncthreads();
    for (int idx = tid; idx < MR * WV * 16; idx += WV * SK * 32) {
        const int m = idx / (WV * 16), col = (idx / 16) % WV, c = idx % 16;
        float v = 0.f;
        for (int s = 0; s < SK; ++s) v += red[((m * WV + col) * SK + s) * 16 + c];
        const int n = (blockIdx.x * WV + col) * 16 + c;
        // 0.0625 undoes the 16 from placing the nibble in the byte's high half.
        if (n < N && m < M) C[(size_t)m * N + n] = (__hip_bfloat16)(v * Ascale[m] * 0.0625f);
    }
}

}  // namespace

/* gemm_w4a8_nt_m64's entry: the decode form for M <= 4 on the symmetric grid without a chained
 * launch, the libhalo kernel for everything else. */
extern "C" int r3d_rad_gemm_w4a8_m64(const RadArgs* a, RadStream s)
{
    GemmOps g;
    int rc = gemm_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    const bool chain = r3dh_geti_or(a, "chain", 0) != 0;
    if (g.M > 4 || g.N % 16 || !g.ascale || is_asym(a) || chain || g.K % 128)
        return r3dh_rad_gemm_w4a8_m64(a, s);
    Cfg c;
    rc = cfg_for(a, g.K, g.M, R3DH_GEMM_W4A8_GROUP, BR_NPW, &c);
    if (rc != RAD_OK) return rc;
    if (c.wv * c.sk * 32 > 1024) return r3dh_rad_gemm_w4a8_m64(a, s);
    const int MR = g.M == 1 ? 1 : (g.M == 2 ? 2 : 4);
    const int ntiles = (int)(g.N / 16);
    const dim3 grid((unsigned)((ntiles + c.wv - 1) / c.wv)), block((unsigned)(c.wv * c.sk * 32));
    const size_t smem = (size_t)MR * c.wv * c.sk * 16 * sizeof(float);
    const signed char* Ap = (const signed char*)r3dh_addr(g.a);
    const float* Sp = (const float*)r3dh_addr(g.ascale);
    const uint4w* Wp = (const uint4w*)r3dh_addr(g.b);
    const unsigned* Zp = (const unsigned*)r3dh_addr(g.bscale);
    __hip_bfloat16* Cp = (__hip_bfloat16*)r3dh_addr(g.y);
    hipStream_t st = (hipStream_t)s;
    const int M = (int)g.M, K = (int)g.K, N = (int)g.N;
    if (MR == 1)      hipLaunchKernelGGL((w4a8_gemv_kernel<1, 4>), grid, block, smem, st, Ap, Sp, Wp, Zp, Cp, M, K, N, c.wv, c.sk);
    else if (MR == 2) hipLaunchKernelGGL((w4a8_gemv_kernel<2, 4>), grid, block, smem, st, Ap, Sp, Wp, Zp, Cp, M, K, N, c.wv, c.sk);
    else              hipLaunchKernelGGL((w4a8_gemv_kernel<4, 2>), grid, block, smem, st, Ap, Sp, Wp, Zp, Cp, M, K, N, c.wv, c.sk);
    return hipGetLastError() == hipSuccess ? RAD_OK : R3DH_NO(RAD_E_DEVICE);
}
