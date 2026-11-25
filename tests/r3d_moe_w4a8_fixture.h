/* moe_gemm_q at dtype w4a8, by value: routed batches and a host reference for them.
 *
 *   y[i, n] = sum over the weight scale groups g of (sum_g * a_scale[src, g's 128-block of K])
 *             * w_scale[e, n, g]
 *
 * for sorted row i, its expert e and its activation row src; sum_g is the group's sum of
 * activation * weight code. The reference runs that in f32, groups ascending, with every group sum
 * exact and rounded once, and stores one bf16.
 *
 * The tolerance. With an int8 activation a group sum is exact in i32 and the kernel folds in the
 * same order, so its bytes must be the reference's. With an E4M3 activation every product is still
 * exact (two exact f16 operands, at most 8 significant bits), but a kernel adds the products in
 * f32 in its own order: 16 k a wave-matrix step and a fold a group (the 16-row block and the
 * prefill tile), a fused pair a step and a fold a group (the decode GEMV), or eight k a lane, a
 * fold a lane and the lanes' terms added by a butterfly (the lane-split GEMV). Each of them adds
 * pieces of 8 to `group` products, scales a piece's sum by a_scale and then w_scale, and adds the
 * scaled sums. An f32 addition is within u = 2^-24 of its result, relative, so a piece of s
 * products is within (s - 1) u of the sum of its magnitudes, the two scale multiplies add 2 u, and
 * adding K / s scaled sums in any order adds (K / s - 1) u. Over s from 8 to `group` that is at
 * most
 *
 *   E = max(8 + K / 8, group + K / group) u sum_k |activation_k weight_k| a_scale w_scale
 *
 * from the exact result, and an output passes when it is the bf16 of some value that close: at
 * these shapes the reference's bf16 or a neighbour, and more only where the terms cancel. The
 * relative L2 error against the reference, bf16 against bf16, is printed for each case and not
 * judged: one output of the rms's size on the other side of a rounding boundary moves it by
 * 0.004 to 0.008 over the square root of the output count.
 */
#pragma once
#include "rad_plugin.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace w4a8_fixture {
inline void check(bool ok, const char* expr, int line) {
    if (!ok) {
        std::fprintf(stderr, "w4a8 fixture line %d: %s\n", line, expr);
        std::exit(1);
    }
}
#define W4A8_CHECK(expr) ::w4a8_fixture::check((expr), #expr, __LINE__)

constexpr int experts = 128;

/* The 26B-A4B's two projections with fewer output columns (a prefill tile and a part of one): the
 * gate_up reads a row a token and is 2816 wide, the down projection a row a (token, slot) and is
 * 768 wide. `other` is a width neither of them has. */
struct Shape {
    const char* name;
    int outputs, width;
    bool sorted_input;
};
constexpr Shape gate_up{"gate_up", 320, 2816, false}, down{"down", 448, 768, true},
    other{"other", 320, 1024, false};

enum class Routing { uniform, skewed, sparse };
constexpr const char* routing_name[]{"uniform", "skewed", "sparse"};

struct Case {
    Shape shape;
    int group;
    bool e4m3;
    int tokens, top_k;
    Routing routing;
    int rows() const { return tokens * top_k; }
    int input_rows() const { return shape.sorted_input ? rows() : tokens; }
};

using Params = std::array<RadParam, 9>;
inline Params params(const Case& c) {
    return {RAD_INT("M", c.tokens), RAD_INT("N", c.shape.outputs), RAD_INT("K", c.shape.width),
        RAD_INT("n_expert", experts), RAD_INT("top_k", c.top_k), RAD_INT("group", c.group),
        RAD_INT("a_group", 128), RAD_STR("dtype", "w4a8"),
        RAD_STR("a_order", c.shape.sorted_input ? "sorted" : "token")};
}

struct Random {
    uint32_t state;
    uint32_t next() {
        state = 1664525u * state + 1013904223u;
        return state >> 8;
    }
    /* A float in [low, 2 * low) with a full mantissa. */
    float scale(float low) { return low * (1.0f + float(next()) / 16777216.0f); }
};

/* Every expert's codes, two a byte with the even k in the low half, and its bf16 scales, one for
 * each (output, group of K). */
struct Experts {
    Shape shape;
    int group;
    std::vector<uint8_t> codes;
    std::vector<uint16_t> scales;
    Experts(const Shape& shape, int group) : shape(shape), group(group),
        codes(size_t(experts) * shape.outputs * shape.width / 2),
        scales(size_t(experts) * shape.outputs * (shape.width / group)) {
        Random random{0x51ed270bu + uint32_t(shape.width)};
        for (auto& byte : codes) byte = uint8_t(random.next());
        for (auto& scale : scales) scale = rad_f32_to_bf16(random.scale(1.0f / 128));
    }
    size_t code_bytes() const { return size_t(shape.outputs) * shape.width / 2; }
    size_t scale_count() const { return size_t(shape.outputs) * (shape.width / group); }
};

/* The activation: every code but the NaN (E4M3) or -128 (int8), and an f32 scale for each (row,
 * 128 of K). */
struct Input {
    std::vector<uint8_t> codes;
    std::vector<float> scales;
    Input(const Case& c, Random& random) : codes(size_t(c.input_rows()) * c.shape.width),
        scales(size_t(c.input_rows()) * (c.shape.width / 128)) {
        for (auto& code : codes) {
            code = uint8_t(random.next());
            if (c.e4m3 ? (code & 0x7f) == 0x7f : code == 0x80) code ^= 1;
        }
        for (auto& scale : scales) scale = random.scale(1.0f / 256);
    }
};

/* Each token's top_k distinct experts, sorted by expert as the scatter sorts them: `slots` is
 * sorted_tok (token * top_k + slot for each sorted row) and `offsets` is expert_offset. Uniform
 * draws from every expert. Skewed gives every token's first slot to one expert, so that expert's
 * run is as long as the batch. Sparse draws from twelve experts, so the others receive no row,
 * among them more than 32 in a row. */
struct Route {
    std::vector<int32_t> slots, offsets;
    Route(const Case& c, Random& random) : slots(c.rows()), offsets(experts + 1, 0) {
        constexpr int hot = 37;
        constexpr std::array<int, 12> live{0, 1, 2, 3, 34, 35, 64, 97, 98, 99, 126, 127};
        std::vector<int> picks(c.rows());
        for (int token = 0; token < c.tokens; ++token) {
            int* mine = picks.data() + size_t(token) * c.top_k;
            for (int slot = 0; slot < c.top_k; ++slot) {
                int expert;
                do {
                    expert = c.routing == Routing::sparse ? live[random.next() % live.size()]
                        : c.routing == Routing::skewed && slot == 0 ? hot : int(random.next() % experts);
                } while (std::find(mine, mine + slot, expert) != mine + slot);
                mine[slot] = expert;
            }
        }
        for (int expert : picks) ++offsets[expert + 1];
        for (int expert = 0; expert < experts; ++expert) offsets[expert + 1] += offsets[expert];
        std::vector<int32_t> next(offsets.begin(), offsets.end() - 1);
        for (int flat = 0; flat < c.rows(); ++flat) slots[next[picks[flat]]++] = flat;
    }
};

/* For each output the reference's bf16, and for an E4M3 activation the lowest and highest bf16
 * the tolerance admits (for int8 both are the reference). */
struct Expected {
    std::vector<uint16_t> value, low, high;
};

inline int32_t dot(const int16_t* a, const int16_t* b, int count) {
    int32_t sum = 0;
    for (int index = 0; index < count; ++index) sum += int32_t(a[index]) * b[index];
    return sum;
}

inline Expected reference(const Case& c, const Experts& weights, const Input& input, const Route& route) {
    const int outputs = c.shape.outputs, width = c.shape.width, groups = width / c.group;
    /* An E4M3 value is a whole number of 2^-9 steps, 229376 of them at most: `high` * 256 + `low`. */
    std::array<int32_t, 256> steps{};
    for (int code = 0; code < 256; ++code)
        steps[code] = c.e4m3 ? int32_t(std::lround(std::ldexp(double(rad_fp8e4m3_to_f32(uint8_t(code))), 9)))
                             : int8_t(code);
    const double unit = c.e4m3 ? std::ldexp(1.0, -9) : 1.0;
    const double bound = 1.001 * std::max(8 + width / 8, c.group + groups) * std::ldexp(1.0, -24);
    Expected expected;
    expected.value.resize(size_t(c.rows()) * outputs);
    expected.low.resize(expected.value.size());
    expected.high.resize(expected.value.size());
    /* One piece of work is up to 32 rows of one expert's run. */
    struct Piece { int expert, first, last; };
    std::vector<Piece> pieces;
    for (int expert = 0; expert < experts; ++expert)
        for (int row = route.offsets[expert]; row < route.offsets[expert + 1]; row += 32)
            pieces.push_back({expert, row, std::min(row + 32, route.offsets[expert + 1])});
    std::atomic<size_t> taken{0};
    auto work = [&] {
        /* The expert's codes and the row's steps, and the magnitudes of both for the bound. */
        std::vector<int16_t> codes(size_t(outputs) * width), sizes(codes.size());
        std::vector<int16_t> high(width), low(width), high_size(width), low_size(width);
        int unpacked = -1;
        for (size_t at; (at = taken++) < pieces.size();) {
            const Piece piece = pieces[at];
            if (unpacked != piece.expert) {
                const uint8_t* packed = weights.codes.data() + piece.expert * weights.code_bytes();
                for (size_t index = 0; index < codes.size(); ++index) {
                    const int nibble = (packed[index / 2] >> (4 * (index % 2))) & 15;
                    codes[index] = int16_t(nibble < 8 ? nibble : nibble - 16);
                    sizes[index] = int16_t(std::abs(codes[index]));
                }
                unpacked = piece.expert;
            }
            const uint16_t* scales = weights.scales.data() + piece.expert * weights.scale_count();
            for (int row = piece.first; row < piece.last; ++row) {
                const int source = c.shape.sorted_input ? row : route.slots[row] / c.top_k;
                const uint8_t* activation = input.codes.data() + size_t(source) * width;
                const float* factors = input.scales.data() + size_t(source) * (width / 128);
                for (int inner = 0; inner < width; ++inner) {
                    const int32_t value = steps[activation[inner]], size = std::abs(value);
                    low[inner] = int16_t(c.e4m3 ? value & 255 : value);
                    high[inner] = int16_t((value - low[inner]) / 256);
                    low_size[inner] = int16_t(size & 255);
                    high_size[inner] = int16_t(size / 256);
                }
                for (int output = 0; output < outputs; ++output) {
                    const int16_t* weight = codes.data() + size_t(output) * width;
                    const int16_t* weight_size = sizes.data() + size_t(output) * width;
                    float sum = 0;
                    double exact = 0, slack = 0;
                    for (int group = 0; group < groups; ++group) {
                        const int first = group * c.group;
                        const int32_t raw = dot(low.data() + first, weight + first, c.group)
                            + 256 * dot(high.data() + first, weight + first, c.group);
                        const float factor = factors[first / 128];
                        const float scale = rad_bf16_to_f32(scales[size_t(output) * groups + group]);
                        const float folded = float(float(raw) * float(unit)) * factor;
                        sum += folded * scale;
                        exact += raw * unit * factor * scale;
                        if (!c.e4m3) continue;
                        const int32_t size = dot(low_size.data() + first, weight_size + first, c.group)
                            + 256 * dot(high_size.data() + first, weight_size + first, c.group);
                        slack += size * unit * factor * scale;
                    }
                    const size_t index = size_t(row) * outputs + output;
                    expected.value[index] = expected.low[index] = expected.high[index] = rad_f32_to_bf16(sum);
                    if (!c.e4m3) continue;
                    slack *= bound;
                    expected.low[index] = rad_f32_to_bf16(std::nextafter(float(exact - slack), -INFINITY));
                    expected.high[index] = rad_f32_to_bf16(std::nextafter(float(exact + slack), INFINITY));
                }
            }
        }
    };
    std::vector<std::thread> threads(std::min(16u, std::max(1u, std::thread::hardware_concurrency())) - 1);
    for (auto& thread : threads) thread = std::thread(work);
    work();
    for (auto& thread : threads) thread.join();
    return expected;
}
}
