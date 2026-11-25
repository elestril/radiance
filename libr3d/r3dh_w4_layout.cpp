/* The w4 weight layout hooks of libhalo/halo_layout.cpp (its helpers and the w4 section, the
 * halo_ -> r3dh_ rename of identifiers only), for the w4a8 GEMM ported from libhalo
 * (r3dh_gemm_w4a8_nt_m64.hip; libhalo's tiled form, halo_gemm_w4a8_tiled.hip, is not carried). The
 * layout tags are libr4d's and libhalo's (r4d.w4.*): the same bytes. The MoE views and every other
 * format are left out. */
#include "r3d_plugin.h"
#include "r3dh_args.h"
#include "r3dh_enc.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

/* ------------------------------------------------------------------ half-precision
 *
 * Written out so the file builds without ROCm or a float16 extension: __fp16 is not portable and
 * _Float16 is not available on every host compiler. */

static uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t man = x & 0x7FFFFFu;

    if (((x >> 23) & 0xFFu) == 0xFFu)                    /* inf / nan */
        return (uint16_t)(sign | 0x7C00u | (man ? 0x200u : 0u));
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);  /* overflow -> inf */
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;            /* underflow -> zero */
        man |= 0x800000u;                                /* subnormal: shift in the hidden bit */
        const int shift = 14 - exp;
        const uint32_t v = man >> shift;
        const uint32_t rem = man & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1);
        uint32_t out = v;
        if (rem > half || (rem == half && (v & 1u))) ++out;   /* round to nearest even */
        return (uint16_t)(sign | out);
    }
    uint32_t v = ((uint32_t)exp << 10) | (man >> 13);
    const uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (v & 1u))) ++v;
    return (uint16_t)(sign | v);
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t man  = h & 0x3FFu;
    uint32_t out;
    if (exp == 0) {
        if (man == 0) out = sign;
        else {
            int e = -1;
            uint32_t m = man;
            do { m <<= 1; ++e; } while (!(m & 0x400u));
            out = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 0x1F) {
        out = sign | 0x7F800000u | (man << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

/* The scale dword: a group's f16 scale in the low half and the f16 of -(1024 + zero) in the high
 * half, broadcast into a v_pk_add_f16. The kernel builds 1024 + code and adds that constant, which
 * subtracts an integer zero point exactly. A symmetric 4-bit grid uses zero == 8, so one packed
 * weight feeds both the f16 and the int8 kernel. */
static uint32_t scale_zero_dword(float scale, int zero) {
    const uint32_t lo = f32_to_f16(scale);
    const uint32_t hi = f32_to_f16(-(1024.0f + (float)zero));
    return lo | (hi << 16);
}

/* Geometry the layout hooks read out of the declared parameters. */
struct WGeom { long long N, K, group; };

static int wgeom(const RadParam* p, int n_p, long long dflt_group, WGeom* g) {
    g->N = r3dh_pgeti_or(p, n_p, "N", 0);
    g->K = r3dh_pgeti_or(p, n_p, "K", 0);
    g->group = r3dh_pgeti_or(p, n_p, "group", dflt_group);
    if (g->N <= 0 || g->K <= 0 || g->group <= 0) return RAD_E_INVAL;
    if (g->N % R3DH_NTILE) return RAD_E_SHAPE;
    if (g->K % g->group) return RAD_E_SHAPE;
    return RAD_OK;
}

/* Operand indices in `gemm_nt_q` (see oGemmNtQ in r3dh_rows.cpp). */
enum { Q_B = 2, Q_BSCALE = 3, Q_BREF = 6 };

static void fill(RadLayout* o, const char* tag, uint32_t dtype, long long r0, long long r1,
                 long long bytes) {
    std::memset(o, 0, sizeof(*o));
    o->tag = tag;
    o->dtype = dtype;
    o->rank = 2;
    o->shape[0] = r0;
    o->shape[1] = r1;
    /* Every stream is read with 128-bit loads; 256 bytes is a cacheline on the target
     * architectures and the .rad plane alignment. */
    o->align = 256;
    o->bytes = bytes;
}

/* The asymmetric 4-bit grid is a different weight, and `dtype` says which. `w4a8_asym` kernels
 * read the nibble unsigned and subtract a per-group integer zero from the scale dword's high half;
 * the symmetric grid stores a two's-complement nibble with the zero fixed at 8. Packing, extents
 * and byte counts are identical, so a plane read as the other grid decodes eight levels off. This
 * is the kernels' own test (r3dh_gemm_args.h is_asym), so the layout hook accepts only the matching
 * encoding (i4 without a zero, u4 with one) and refuses the other at declare. */
static bool w4_asym(const RadParam* p, int n_p) {
    const char* d = r3dh_pgets(p, n_p, "dtype");
    return d && std::strstr(d, "_asym") != nullptr;
}

/* ================================================================== moe w4
 *
 * The grouped MoE GEMM's 4-bit expert weight. Unlike the dense w4 it is not permuted into fragment
 * order: the grouped kernel stages its weight through LDS (eight M tiles share one N tile), so a
 * permutation would only have to be undone.
 *
 *   weight  [N][K/2] bytes, row-major, two's complement codes -8..7: an ordinary symmetric int4
 *           grid, as GPTQ writes.
 *   scale   [N][K/group] bf16, one per row per 128 of K. w ~= code * scale.
 *
 * Decode: V_PERM_B32 permutes the bytes of an eight-byte source, so one table holds eight entries
 * and a two's complement nibble needs sixteen. Two perms index a positive and a negative table
 * with the low three bits, and a third selects on bit 3 by adding 4 to a selector byte (which
 * cannot carry): all sixteen levels for five more VALU per eight weights than a fifteen-level
 * sign-magnitude grid. See r3dh_moe.hip w4_e4m3.
 *
 * The integers -8..7 are exact in E4M3 (three mantissa bits reach 16), so expanding a code to an
 * E4M3 byte is lossless and the weight runs on the existing f32_16x16x16_fp8_fp8 inner loop; only
 * the staging differs.
 *
 * Nibble order: byte b of a packed dword holds element b in its low nibble and element b+4 in its
 * high nibble, so the two masks yield elements 0..3 and 4..7 already in order, as in the dense
 * w4a8 kernel.
 *
 * Not shardable: a packed row is K/2 bytes and a column shard could cut a byte in half. Under
 * expert parallelism an expert is whole on one rank, the only configuration that declares this
 * format; rad_block_moe_fp8.h refuses a width-sharded declaration.
 */
/* `dtype=w4a8h` serves `i4*bf16[1x128]/fwht128`: the same bytes, but every group of K was
 * multiplied by a `group`-wide Hadamard before quantisation. Nothing in packing, extents or byte
 * count shows this, and a weight served the other way reads as noise, so the layout hook accepts
 * only the encoding the dtype names.
 *
 * The kernel does not branch on it: <Hw, Ha> = group * <w, a>, and the quantiser folds the
 * residual factor into the stored scale (docs/MOE-W4.md). What must agree is the activation, which
 * had_quant_act_fp8 quantises with the same rotation width. That pairing is a property of the
 * graph and is not checked here; the arch block that declares one declares the other. */
/* `dtype=w4nla8h` serves `u4*bf16[1x128]/fwht128` whose codes index a sixteen-entry `table` plane
 * (libquant `table=w4nl`), and the kernel decodes only its own table (r3dh_args.h). The codes view
 * selects the table plane and the relayout compares it with the kernel's value for value, so a
 * container quantised against another table is refused at load. */

/* ================================================================== the hooks
 *
 * Codes and scales arrive chosen by a quantiser; the hooks only arrange them as the kernel reads
 * them (fragment order, the (scale, zero) dword, a tile-major scale plane, a nibble order). Each
 * refuses an encoding its kernel does not read with RAD_E_DTYPE. */
namespace {

using r3dh_enc::selects;
using r3dh_enc::dims;

/* A selected plane's dense bytes. */
const uint8_t* pdata(const RadTensor& t) { return (const uint8_t*)t.data; }

/* The f16 scale of a canonical plane as f32; scale_zero_dword narrows it back exactly. */
float f16_at(const RadTensor& t, long long i) { return f16_to_f32(((const uint16_t*)t.data)[i]); }

/* Zero point `gi` of row `n` of a [N, nsz] u4 or u8 plane. Rows start on a byte (rad_encoding.h),
 * so a u4 row of an odd number of groups ends in a pad nibble and the next row starts after it. */
uint8_t* zero_row(const RadTensor& t, long long n, long long nsz) {
    return (uint8_t*)t.data + n * rad_enc_row_bytes(t.dtype, nsz);
}
int zero_at(const RadTensor& t, long long n, long long nsz, long long gi) {
    return (int)rad_load_f32(zero_row(t, n, nsz), t.dtype, gi);
}

/* The dense w4/w8/w2 families' scale operand: (scale) for a symmetric grid, (scale, zero) for an
 * asymmetric one, tile-major [N/16][(K/group)*16] in dwords or, for w8a8, halves. */
int sz_layout(const WGeom& g, const RadEncoding* enc, const int* sel, const RadTensor* pl, int n,
              bool asym, const char* tag, uint32_t dt, int bytes_each, RadLayout* out) {
    const long long nsz = g.K / g.group;
    if (asym ? !selects(enc, sel, n, "scale", "zero") : !selects(enc, sel, n, "scale"))
        return RAD_E_SHAPE;
    for (int i = 0; i < n; ++i)
        if (!dims(pl[i], g.N, nsz)) return RAD_E_SHAPE;
    fill(out, tag, dt, g.N / R3DH_NTILE, nsz * R3DH_NTILE, g.N * nsz * bytes_each);
    return RAD_OK;
}

/* Write the tile-major (scale, zero) dwords. `zfixed` >= 0 is a symmetric grid's constant zero. */
void sz_write(const WGeom& g, const RadTensor* pl, int zfixed, uint32_t* sz) {
    const long long nt = g.N / R3DH_NTILE, nsz = g.K / g.group;
    for (long long t = 0; t < nt; ++t)
        for (long long r = 0; r < R3DH_NTILE; ++r)
            for (long long gi = 0; gi < nsz; ++gi) {
                const long long n = t * R3DH_NTILE + r;
                const int z = zfixed >= 0 ? zfixed : zero_at(pl[1], n, nsz, gi);
                const float sc = f16_at(pl[0], n * nsz + gi);
                sz[(t * nsz + gi) * R3DH_NTILE + r] = scale_zero_dword(sc, z);
            }
}

/* Whether every zero point fits in `bits` bits. libhalo's int8 prefill kernel
 * (halo_gemm_wpa8i8_tiled.hip, which libr3d does not carry) reads five bits of the zero out of the scale dword where the f16 kernel reads the whole value,
 * so a larger zero would make the two disagree. */
bool sz_zeros_fit(const WGeom& g, const RadTensor& zp, int bits) {
    const long long nsz = g.K / g.group;
    for (long long n = 0; n < g.N; ++n)
        for (long long gi = 0; gi < nsz; ++gi) {
            const int z = zero_at(zp, n, nsz, gi);
            if (z < 0 || z >= (1 << bits)) return false;
        }
    return true;
}

void sz_read(const WGeom& g, const uint32_t* sz, const RadTensor* pl, bool asym) {
    const long long nt = g.N / R3DH_NTILE, nsz = g.K / g.group;
    for (long long t = 0; t < nt; ++t)
        for (long long r = 0; r < R3DH_NTILE; ++r)
            for (long long gi = 0; gi < nsz; ++gi) {
                const long long n = t * R3DH_NTILE + r;
                const uint32_t d = sz[(t * nsz + gi) * R3DH_NTILE + r];
                ((uint16_t*)pl[0].data)[n * nsz + gi] = (uint16_t)(d & 0xFFFFu);
                if (asym) {
                    const int z = (int)(-f16_to_f32((uint16_t)(d >> 16)) - 1024.0f + 0.5f);
                    rad_store_code(zero_row(pl[1], n, nsz), pl[1].dtype, gi, z);
                }
            }
}

/* A rotated asymmetric grid (ParoQuant): codes of dtype `codes`, an f16 scale and a zero point
 * per group, optionally with a `givens` transform and its tables. The bytes are the unrotated
 * grid's; the rotation is applied to the activation. */
static bool paro_asym_enc(const RadEncoding* e, uint32_t codes, long long group) {
    if (!e || !rad_enc_is(e, "affine") || e->n_planes < 3) return false;
    const bool giv = std::strncmp(e->transform, "givens", RAD_ENC_STR) == 0;
    if (!giv && e->transform[0]) return false;
    if (!giv && e->n_planes != 3) return false;
    if (giv && (!rad_enc_valid(e) || rad_enc_find(e, "t.pairs") < 0)) return false;
    if (std::strncmp(e->plane[0].role, "codes", RAD_ENC_STR) != 0 || e->plane[0].dtype != codes ||
        e->plane[0].block[0] != 1 || e->plane[0].block[1] != 1) return false;
    const int s = rad_enc_find(e, "scale"), z = rad_enc_find(e, "zero");
    if (s != 1 || z != 2) return false;
    return e->plane[s].dtype == RAD_F16 && e->plane[s].kind == RAD_PLANE_TILED &&
           e->plane[s].block[0] == 1 && e->plane[s].block[1] == group &&
           (e->plane[z].dtype == RAD_U8 || e->plane[z].dtype == codes) &&
           e->plane[z].kind == RAD_PLANE_TILED && e->plane[z].block[0] == 1 &&
           e->plane[z].block[1] == group;
}

/* ---- w4: a symmetric grid is i4 codes with an f16 scale; `_asym` is u4 codes with a zero. */
int w4_enc_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                  const RadTensor* pl, int n, RadLayout* out, WGeom* gout) {
    if (!out || !pl) return RAD_E_INVAL;
    WGeom g;
    int rc = wgeom(p, n_p, R3DH_W4_GROUP, &g);
    if (rc != RAD_OK) return rc;
    if (g.group != R3DH_W4_GROUP) return RAD_E_DTYPE;
    if (g.K % R3DH_W4_KPB) return RAD_E_SHAPE;
    const bool asym = w4_asym(p, n_p);
    const bool ok = asym ? (r3dh_enc::affine(enc, RAD_U4, RAD_F16, 1, g.group, RAD_U8) ||
                            r3dh_enc::affine(enc, RAD_U4, RAD_F16, 1, g.group, RAD_U4) ||
                            paro_asym_enc(enc, RAD_U4, g.group))
                         : r3dh_enc::affine(enc, RAD_I4, RAD_F16, 1, g.group);
    if (!ok) return RAD_E_DTYPE;
    if (gout) *gout = g;
    if (operand == Q_BSCALE)
        return sz_layout(g, enc, sel, pl, n, asym, asym ? "r4d.w4u.sz.g128" : "r4d.w4.sz.g128",
                         R3DH_DT_W4_SZ, 4, out);
    if (operand != Q_B) return RAD_E_UNSUPPORTED;
    if (!selects(enc, sel, n, "codes") || !dims(pl[0], g.N, g.K)) return RAD_E_SHAPE;
    fill(out, asym ? "r4d.w4u.frag.g128" : "r4d.w4.frag.g128", R3DH_DT_W4_FRAG, g.N, g.K,
         g.N * g.K / 2);
    return RAD_OK;
}

/* Where element (n, k) of a w4 plane lands: the 32-bit word and the nibble within it. */
void w4_slot(const WGeom& g, long long n, long long k, long long* word, int* nib) {
    const long long kpb = g.K / R3DH_W4_KPB;
    const long long t = n / R3DH_NTILE, r = n % R3DH_NTILE;
    const long long kb = k / R3DH_W4_KPB, kin = k % R3DH_W4_KPB;
    const long long s = kin / 16, kk = kin % 16;
    const long long lh = (kk % 8) / 4, e = (kk / 8) * 4 + (kk % 4);
    *nib = (int)((e < 4) ? (2 * e) : (2 * (e - 4) + 1));
    *word = ((t * kpb + kb) * R3DH_WAVE + lh * 16 + r) * 4 + s;
}

}  /* namespace */

extern "C" int r3dh_rad_layout_w4(const RadParam* p, int n_p, int operand,
                                     const RadEncoding* enc, const int* sel, const RadTensor* pl,
                                     int n, RadLayout* out) {
    return w4_enc_layout(p, n_p, operand, enc, sel, pl, n, out, nullptr);
}

extern "C" int r3dh_rad_relayout_w4(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                   const int* sel, const RadTensor* pl, int n, void* dst,
                                   int64_t dst_bytes) {
    RadLayout L;
    WGeom g;
    const int rc = w4_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (dst_bytes != L.bytes) return RAD_E_SHAPE;
    std::memset(dst, 0, (size_t)dst_bytes);
    if (operand == Q_BSCALE) {
        const bool asym = w4_asym(p, n_p);
        /* A rotated (ParoQuant) grid's zero must fit a 4-bit code. */
        if (asym && std::strncmp(enc->transform, "givens", RAD_ENC_STR) == 0 &&
            !sz_zeros_fit(g, pl[1], 4)) return RAD_E_FORMAT;
        sz_write(g, pl, asym ? -1 : 8, (uint32_t*)dst);
        return RAD_OK;
    }
    uint32_t* wq = (uint32_t*)dst;
    for (long long nn = 0; nn < g.N; ++nn) {
        const uint8_t* row = pdata(pl[0]) + nn * (g.K / 2);
        for (long long k = 0; k < g.K; ++k) {
            long long w;
            int nib;
            w4_slot(g, nn, k, &w, &nib);
            wq[w] |= (uint32_t)((row[k >> 1] >> (4 * (k & 1))) & 0xF) << (4 * nib);
        }
    }
    return RAD_OK;
}

extern "C" int r3dh_rad_unrelayout_w4(const RadParam* p, int n_p, int operand,
                                     const RadEncoding* enc, const int* sel, const void* src,
                                     int64_t src_bytes, const RadTensor* pl, int n) {
    RadLayout L;
    WGeom g;
    const int rc = w4_enc_layout(p, n_p, operand, enc, sel, pl, n, &L, &g);
    if (rc != RAD_OK) return rc;
    if (src_bytes != L.bytes) return RAD_E_SHAPE;
    if (operand == Q_BSCALE) {
        sz_read(g, (const uint32_t*)src, pl, w4_asym(p, n_p));
        return RAD_OK;
    }
    const uint32_t* wq = (const uint32_t*)src;
    uint8_t* d = (uint8_t*)pl[0].data;
    std::memset(d, 0, (size_t)(g.N * g.K / 2));
    for (long long nn = 0; nn < g.N; ++nn)
        for (long long k = 0; k < g.K; ++k) {
            long long w;
            int nib;
            w4_slot(g, nn, k, &w, &nib);
            const uint32_t q = (wq[w] >> (4 * nib)) & 0xF;
            d[nn * (g.K / 2) + (k >> 1)] |= (uint8_t)(q << (4 * (k & 1)));
        }
    return RAD_OK;
}
