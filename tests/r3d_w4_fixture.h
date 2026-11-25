/* The dense four-bit weight of gemm_nt_q at w4a8 (libr3d/r3dh_w4_layout.cpp), for its host and
 * card tests: a canonical weight, its stored form through a row's layout hooks, and the order
 * gemm_w4a8_nt_m64's kernels read that stored form in, written out here from the kernels and not
 * from the hooks. */
#pragma once
#include "rad_abi.h"
#include "rad_plugin.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
int r3dh_rad_layout_w4(const RadParam* params, int count, int operand, const RadEncoding* enc, const int* sel,
                       const RadTensor* planes, int plane_count, RadLayout* out);
int r3dh_rad_relayout_w4(const RadParam* params, int count, int operand, const RadEncoding* enc, const int* sel,
                         const RadTensor* planes, int plane_count, void* dst, int64_t dst_bytes);
int r3dh_rad_unrelayout_w4(const RadParam* params, int count, int operand, const RadEncoding* enc, const int* sel,
                           const void* src, int64_t src_bytes, const RadTensor* planes, int plane_count);
}

namespace w4_fixture {
inline void check(bool ok, const char* expr, int line) {
    if (!ok) {
        std::fprintf(stderr, "w4 fixture line %d: %s\n", line, expr);
        std::exit(1);
    }
}
#define W4_CHECK(expr) ::w4_fixture::check((expr), #expr, __LINE__)
using Params = std::array<RadParam, 5>;
inline Params params(int M, int N, int K) {
    return {RAD_INT("M", M), RAD_INT("N", N), RAD_INT("K", K), RAD_INT("group", 128), RAD_STR("dtype", "w4a8")};
}
/* gemm_nt_q's operands b and b_scale, and the encoding's plane each one selects */
constexpr int kCodes = 2, kScale = 3;
constexpr int plane_ids[]{0, 1};
inline RadTensor matrix(void* data, uint32_t type, int64_t rows, int64_t columns) {
    RadTensor result{};
    result.data = data;
    result.dtype = type;
    result.rank = 2;
    result.shape[0] = rows;
    result.shape[1] = result.stride[0] = columns;
    result.stride[1] = 1;
    return result;
}
/* A symmetric int4 weight as a container holds it: two's-complement codes -8..7, two a byte with
 * the lower column in the low nibble, and an f16 scale per (row, 128 columns). */
struct Weight {
    int N, K;
    RadEncoding enc{};
    std::vector<uint8_t> codes;
    std::vector<uint16_t> scale;
    Weight(int N, int K, uint32_t seed) : N(N), K(K), codes(size_t(N) * K / 2), scale(size_t(N) * (K / 128)) {
        rad_enc_copy_str(enc.scheme, "affine");
        rad_enc_add_plane(&enc, "codes", RAD_I4, 1, 1);
        rad_enc_add_plane(&enc, "scale", RAD_F16, 1, 128);
        uint32_t random = seed;
        for (auto& packed : codes) {
            random = 1664525u * random + 1013904223u;
            packed = random >> 24;
        }
        for (auto& value : scale) {
            random = 1664525u * random + 1013904223u;
            value = rad_f32_to_f16(float(64 + (random >> 24)) / 8192);
        }
    }
    int code(int n, int k) const {
        const int nibble = (codes[size_t(n) * (K / 2) + k / 2] >> (4 * (k % 2))) & 15;
        return nibble < 8 ? nibble : nibble - 16;
    }
    std::array<RadTensor, 2> planes() {
        return {matrix(codes.data(), RAD_I4, N, K), matrix(scale.data(), RAD_F16, N, K / 128)};
    }
};
inline RadLayout describe(const Params& geometry, Weight& weight, int operand, RadLayoutFn layout) {
    const auto planes = weight.planes();
    const int plane = operand - kCodes;
    RadLayout result{};
    W4_CHECK(layout(geometry.data(), geometry.size(), operand, &weight.enc, plane_ids + plane, planes.data() + plane,
                    1, &result) == RAD_OK);
    return result;
}
inline std::vector<uint8_t> pack(const Params& geometry, Weight& weight, int operand, RadLayoutFn layout,
                                 RadRelayoutFn relayout) {
    const auto planes = weight.planes();
    const int plane = operand - kCodes;
    const RadLayout result = describe(geometry, weight, operand, layout);
    W4_CHECK(result.bytes > 0 && result.align == 256);
    std::vector<uint8_t> bytes(result.bytes);
    W4_CHECK(relayout(geometry.data(), geometry.size(), operand, &weight.enc, plane_ids + plane,
                      planes.data() + plane, 1, bytes.data(), bytes.size()) == RAD_OK);
    return bytes;
}

/* What the kernels read (r3dh_gemm_w4a8_nt_m64.hip, dequant_step and dequant8; the decode form in
 * r3d_gemm_w4a8_gemv.hip reads the same dwords). Weight rows go in tiles of 16 and columns in
 * blocks of 64. A block of a tile is 32 lanes of four dwords, one dword a 16-column step. In a
 * step, the lane of row r holds the row's first and third column quarters and lane 16 + r its
 * second and fourth; a dword's even nibbles are the earlier quarter's four codes and its odd
 * nibbles the later quarter's. */
inline int stored_code(const std::vector<uint8_t>& stored, int K, int n, int k) {
    const int tile = n / 16, row = n % 16, block = k / 64, step = k % 64 / 16, quarter = k % 16 / 4;
    const size_t lane = (size_t(tile) * (K / 64) + block) * 32 + 16 * (quarter % 2) + row;
    uint32_t word;
    std::memcpy(&word, stored.data() + (lane * 4 + step) * 4, 4);
    const int nibble = (word >> (4 * (2 * (k % 4) + quarter / 2))) & 15;
    return nibble < 8 ? nibble : nibble - 16;
}
/* The scale operand: a tile's 16 rows side by side for each 128 columns, a dword each with the
 * f16 scale in its low half. The high half is the f16 of -(1024 + 8), the fixed zero point the
 * f16-activation kernels of this weight read; the w4a8 kernels do not read it. */
inline uint32_t stored_scale(const std::vector<uint8_t>& stored, int K, int n, int group) {
    uint32_t word;
    std::memcpy(&word, stored.data() + ((size_t(n / 16) * (K / 128) + group) * 16 + n % 16) * 4, 4);
    return word;
}
}
