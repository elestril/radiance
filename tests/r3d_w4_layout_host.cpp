/* The dense w4 layout hooks (libr3d/r3dh_w4_layout.cpp) on the host: the stored form is the order
 * gemm_w4a8_nt_m64's kernels read (r3d_w4_fixture.h), the hooks are each other's inverse, and a
 * geometry or an encoding the kernels do not read is refused. */
#include "r3d_w4_fixture.h"

using namespace w4_fixture;
static void roundtrip(int N, int K) {
    const auto geometry = params(1, N, K);
    Weight original(N, K, 7 + N + K), copy(N, K, 33);
    const RadLayout codes = describe(geometry, original, kCodes, r3dh_rad_layout_w4);
    W4_CHECK(!std::strcmp(codes.tag, "r4d.w4.frag.g128") && codes.rank == 2 && codes.shape[0] == N
        && codes.shape[1] == K && codes.bytes == int64_t(N) * K / 2);
    const RadLayout scales = describe(geometry, original, kScale, r3dh_rad_layout_w4);
    W4_CHECK(!std::strcmp(scales.tag, "r4d.w4.sz.g128") && scales.rank == 2 && scales.shape[0] == N / 16
        && scales.shape[1] == K / 128 * 16 && scales.bytes == int64_t(N) * (K / 128) * 4);
    W4_CHECK(codes.dtype != scales.dtype && codes.dtype != RAD_I4 && scales.dtype != RAD_F16);

    auto packed = pack(geometry, original, kCodes, r3dh_rad_layout_w4, r3dh_rad_relayout_w4);
    auto packed_scale = pack(geometry, original, kScale, r3dh_rad_layout_w4, r3dh_rad_relayout_w4);
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) W4_CHECK(stored_code(packed, K, n, k) == original.code(n, k));
        for (int group = 0; group < K / 128; ++group)
            W4_CHECK(stored_scale(packed_scale, K, n, group)
                == (original.scale[size_t(n) * (K / 128) + group] | uint32_t(rad_f32_to_f16(-1032.0f)) << 16));
    }

    auto planes = copy.planes();
    auto unpack = [&](int operand, const std::vector<uint8_t>& data) {
        return r3dh_rad_unrelayout_w4(geometry.data(), geometry.size(), operand, &copy.enc, plane_ids + operand - kCodes,
            data.data(), data.size(), planes.data() + operand - kCodes, 1);
    };
    W4_CHECK(unpack(kCodes, packed) == RAD_OK && unpack(kScale, packed_scale) == RAD_OK);
    W4_CHECK(original.codes == copy.codes && original.scale == copy.scale);

    /* a stored form of another size is not this geometry's */
    packed.pop_back();
    W4_CHECK(unpack(kCodes, packed) == RAD_E_SHAPE);
    W4_CHECK(r3dh_rad_relayout_w4(geometry.data(), geometry.size(), kCodes, &copy.enc, plane_ids, planes.data(), 1,
        packed.data(), packed.size()) == RAD_E_SHAPE);
    packed_scale.push_back(0);
    W4_CHECK(unpack(kScale, packed_scale) == RAD_E_SHAPE);
}
int main() {
    /* one tile and one scale group, then several of each, an odd count of groups among them */
    const int shapes[][2]{{16, 128}, {32, 256}, {48, 384}, {16, 640}, {80, 128}};
    for (const auto& shape : shapes) roundtrip(shape[0], shape[1]);

    Weight weight(32, 256, 5);
    auto planes = weight.planes();
    RadLayout layout{};
    auto inspect = [&](Params geometry, int operand, const int* ids) {
        return r3dh_rad_layout_w4(geometry.data(), geometry.size(), operand, &weight.enc, ids,
            planes.data() + (operand == kScale), 1, &layout);
    };
    W4_CHECK(inspect(params(1, 32, 256), kCodes, plane_ids) == RAD_OK);
    /* the activation is not a weight; a plane must be the one its operand reads */
    W4_CHECK(inspect(params(1, 32, 256), 0, plane_ids) == RAD_E_UNSUPPORTED);
    W4_CHECK(inspect(params(1, 32, 256), kCodes, plane_ids + 1) == RAD_E_SHAPE);
    W4_CHECK(inspect(params(1, 32, 256), kScale, plane_ids) == RAD_E_SHAPE);
    /* whole tiles of 16 rows, whole scale groups of 128 columns, and the planes' own extents */
    W4_CHECK(inspect(params(1, 24, 256), kCodes, plane_ids) == RAD_E_SHAPE);
    W4_CHECK(inspect(params(1, 32, 320), kCodes, plane_ids) == RAD_E_SHAPE);
    W4_CHECK(inspect(params(1, 48, 256), kCodes, plane_ids) == RAD_E_SHAPE);
    W4_CHECK(inspect(params(1, 32, 384), kScale, plane_ids + 1) == RAD_E_SHAPE);
    /* a scale group of 128 only */
    auto geometry = params(1, 32, 256);
    geometry[3] = RAD_INT("group", 64);
    W4_CHECK(inspect(geometry, kCodes, plane_ids) == RAD_E_DTYPE);
    /* the symmetric grid is i4 codes with an f16 scale and nothing else; the asymmetric dtype
     * reads another weight */
    geometry = params(1, 32, 256);
    geometry[4] = RAD_STR("dtype", "w4a8_asym");
    W4_CHECK(inspect(geometry, kCodes, plane_ids) == RAD_E_DTYPE);
    weight.enc.plane[0].dtype = RAD_U4;
    W4_CHECK(inspect(params(1, 32, 256), kCodes, plane_ids) == RAD_E_DTYPE);
    weight.enc.plane[0].dtype = RAD_I4;
    weight.enc.plane[1].dtype = RAD_BF16;
    W4_CHECK(inspect(params(1, 32, 256), kScale, plane_ids + 1) == RAD_E_DTYPE);
    weight.enc.plane[1].dtype = RAD_F16;
    weight.enc.plane[1].block[1] = 64;
    W4_CHECK(inspect(params(1, 32, 256), kScale, plane_ids + 1) == RAD_E_DTYPE);
    weight.enc.plane[1].block[1] = 128;
    rad_enc_copy_str(weight.enc.transform, "fwht128");
    W4_CHECK(inspect(params(1, 32, 256), kCodes, plane_ids) == RAD_E_DTYPE);
}
