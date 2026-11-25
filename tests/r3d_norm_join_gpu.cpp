/* r3d_norm_join_gpu -- Gemma 4's sandwich-norm joins by value, on a card.
 *
 * libref has no row for norm_add_norm, norm_add_norm_route, norm_add_norm_route_i8g and
 * join_add_scale, so rad-kbench has nothing to compare them with. Each fused row is launched here
 * and every value it stores is compared with a host computation of the chain the gemma4 plugin
 * issues where no library serves the join (arch/gemma4/gemma4.cpp):
 *
 *   norm_add_norm            rmsnorm, add, rmsnorm
 *   norm_add_norm_route      the same, then the router's rmsnorm (with `mult`) and
 *                            rmsnorm_quant_fp8: E4M3 codes of the bf16-rounded norm, a scale of
 *                            amax / 448 a 128 columns
 *   norm_add_norm_route_i8g  the same with the experts' input as int8: each 128 columns of the
 *                            bf16-rounded norm through an unnormalised Hadamard, a scale of
 *                            amax / 127
 *   join_add_scale           rmsnorm of each arm, add, rmsnorm, add, mul_weight, and with `w_n`
 *                            the next rmsnorm over the x just stored
 *
 * in the chain's arithmetic: a norm is bf16(((v * s) * w) * mult) with s = 1 / sqrt(mean(v^2) +
 * eps), an add bf16(u + v), the layer scalar bf16(x * s), and a value the chain stores is rounded
 * to bf16 where the chain stores it.
 *
 * Tolerance. Every step but two is one IEEE f32 operation or a rounding, the same on the host and
 * on the card, so given its inputs a stored value is exact. The two are the norm factor and the
 * reciprocal of a quantiser's scale. The card sums a row's squares over threads and waves in the
 * kernel's order (at 256 threads 23 adds a thread, five butterfly rounds and seven waves, in the
 * 1024-thread decode form two, five and 31, and the square: at most 39 roundings of a sum of
 * non-negative terms, 1.2e-6 on the factor), then divides and takes
 * a reciprocal square root, neither of which a card has to round correctly. The host therefore
 * takes each factor as an interval, kFactorTol (2^-19, 1.9e-6) either side of
 * 1 / sqrt(mean + eps) computed in double from the exact squares, and each reciprocal as three
 * ulps either side of the f32 quotient. An interval is carried through the chain (every operation
 * is monotone) and a stored value must lie in the range that results. For all but a few elements
 * in a thousand the range is a single value, which the card must match bit for bit; for the rest
 * it is two neighbouring ones. Where a later stage reads a plane the row stores (x), the host
 * continues from the card's bytes, as the kernel does.
 *
 * The rotated int8 codes of a group depend on all of its 128 inputs. Where one of them has two
 * possible values, the host computes the group for each choice in the kernel's butterfly order,
 * and the card must match one of them.
 *
 * Elements whose range is not a single value are counted, printed and bounded (kOpenShare), so
 * the comparison cannot become an empty one; the int8 groups compared against a choice are
 * counted and printed too. Padding columns and the row past M must come back untouched.
 *
 * Forms. A launch of at most 64 rows and 3072 columns takes the 1024-thread form (not the int8
 * route), a wider row 24 columns a thread, the rest 12. Each launch of at most 64 rows is also
 * made with the parameter `chain` = 1, as the engine resolves a row it chains: the kernels'
 * chained forms. No engine record is in the argument segment here, so a chained form neither
 * waits nor adds, and what is compared is its arithmetic.
 */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "../core/rad_internal.h"
#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr double kFactorTol = 0x1p-19;   /* a norm factor, relative */
constexpr int kInvUlps = 3;              /* a quantiser's 1 / scale */
constexpr double kOpenShare = 0.02;      /* elements a case may leave with two values */
constexpr int kGroup = 128;

void check(bool ok, const char* expr, int line) {
    if (!ok) {
        std::fprintf(stderr, "norm join line %d: %s\n", line, expr);
        std::exit(1);
    }
}
#define NJ_CHECK(expr) check((expr), #expr, __LINE__)
void hip_check(hipError_t status) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "HIP: %s\n", hipGetErrorString(status));
        std::exit(1);
    }
}

struct Buffers {
    std::vector<void*> owned;
    template<class Value> void* upload(const std::vector<Value>& values) {
        void* device = nullptr;
        const size_t bytes = values.size() * sizeof(Value);
        hip_check(hipMalloc(&device, bytes));
        owned.push_back(device);
        hip_check(hipMemcpy(device, values.data(), bytes, hipMemcpyHostToDevice));
        return device;
    }
    template<class Value> static void download(std::vector<Value>& values, const void* device) {
        hip_check(hipMemcpy(values.data(), device, values.size() * sizeof(Value), hipMemcpyDeviceToHost));
    }
    ~Buffers() { for (void* data : owned) hip_check(hipFree(data)); }
};
RadTensor rows(void* data, uint32_t dtype, int64_t count, int64_t width, int64_t pitch) {
    RadTensor result{};
    result.data = data;
    result.dtype = dtype;
    result.rank = 2;
    result.shape[0] = count;
    result.shape[1] = width;
    result.stride[0] = pitch;
    result.stride[1] = 1;
    return result;
}
RadTensor vec(void* data, uint32_t dtype, int64_t count) {
    RadTensor result{};
    result.data = data;
    result.dtype = dtype;
    result.rank = 1;
    result.shape[0] = count;
    result.stride[0] = 1;
    return result;
}

/* ------------------------------------------------------------------ the host chain */

float b2f(uint16_t code) { return rad_bf16_to_f32(code); }
uint16_t f2b(float value) { return rad_f32_to_bf16(value); }
float rb(float value) { return b2f(f2b(value)); }

/* The values a quantity may take: every f32 between lo and hi. */
struct Iv { float lo, hi; };
Iv point(float value) { return {value, value}; }
bool same_bits(float a, float b) { return !std::memcmp(&a, &b, sizeof a); }
Iv ordered(float a, float b) { return {std::min(a, b), std::max(a, b)}; }
/* v * f for a positive f, and v * g for a gain of either sign: the f32 products of the ends. */
Iv times(Iv v, Iv f) { return {std::min(v.lo * f.lo, v.lo * f.hi), std::max(v.hi * f.lo, v.hi * f.hi)}; }
Iv gain(Iv v, float g) { return ordered(v.lo * g, v.hi * g); }
Iv plus(Iv a, Iv b) { return {a.lo + b.lo, a.hi + b.hi}; }
Iv rounded(Iv v) { return {rb(v.lo), rb(v.hi)}; }

float outward(double value, bool up) {
    float result = (float)value;
    if (up ? (double)result < value : (double)result > value)
        result = std::nextafterf(result, up ? INFINITY : -INFINITY);
    return result;
}
/* 1 / sqrt(mean + eps) for a sum of squares between ss_lo and ss_hi. */
Iv factor(double ss_lo, double ss_hi, int n, float eps) {
    return {outward((1.0 - kFactorTol) / std::sqrt(ss_hi / n + (double)eps), false),
            outward((1.0 + kFactorTol) / std::sqrt(ss_lo / n + (double)eps), true)};
}
void add_square(Iv v, double* lo, double* hi) {
    const double a = (double)v.lo * v.lo, b = (double)v.hi * v.hi;
    *lo += v.lo <= 0 && v.hi >= 0 ? 0.0 : std::min(a, b);
    *hi += std::max(a, b);
}
/* 1 / scale as the card may compute it. */
Iv reciprocal(Iv scale) {
    Iv result{1.0f / scale.hi, 1.0f / scale.lo};
    for (int step = 0; step < kInvUlps; ++step) {
        result.lo = std::nextafterf(result.lo, 0.0f);
        result.hi = std::nextafterf(result.hi, INFINITY);
    }
    return result;
}

struct Tally {
    const char* plane = "";
    long compared = 0, open = 0, bad = 0;
    void miss(int row, int column, double got, Iv want) {
        if (++bad <= 4)
            std::fprintf(stderr, "  %s row %d column %d: got %.9g, expected %.9g .. %.9g\n", plane, row, column,
                         got, (double)want.lo, (double)want.hi);
    }
    /* A bf16 value against its range. */
    void bf16(uint16_t got, Iv want, int row, int column) {
        ++compared;
        const uint16_t lo = f2b(want.lo), hi = f2b(want.hi);
        if (lo == hi) {
            if (got != lo) miss(row, column, b2f(got), want);
            return;
        }
        ++open;
        if (!(b2f(got) >= want.lo && b2f(got) <= want.hi)) miss(row, column, b2f(got), want);
    }
    /* An f32 against its range. */
    void f32(float got, Iv want, int row, int column) {
        ++compared;
        if (same_bits(want.lo, want.hi)) {
            if (!same_bits(got, want.lo)) miss(row, column, got, want);
            return;
        }
        ++open;
        if (!(got >= want.lo && got <= want.hi)) miss(row, column, got, want);
    }
};

/* rmsnorm_quant_fp8 over one group of the bf16-rounded norm: the scale and the E4M3 codes. */
void fp8_group(const Iv* value, const uint8_t* codes, float scale, int row, int first, Tally& tq, Tally& ts) {
    float amax_lo = 0.0f, amax_hi = 0.0f;
    for (int j = 0; j < kGroup; ++j) {
        const Iv v = value[j];
        amax_lo = std::max(amax_lo, v.lo >= 0 ? v.lo : v.hi <= 0 ? -v.hi : 0.0f);
        amax_hi = std::max(amax_hi, std::max(std::fabs(v.lo), std::fabs(v.hi)));
    }
    const Iv want = ordered(amax_lo > 0.0f ? amax_lo * (1.0f / 448.0f) : 1.0f,
                            amax_hi > 0.0f ? amax_hi * (1.0f / 448.0f) : 1.0f);
    ts.f32(scale, want, row, first / kGroup);
    const Iv inv = reciprocal(want);
    for (int j = 0; j < kGroup; ++j) {
        const Iv v = times(value[j], inv);
        const uint8_t lo = rad_f32_to_fp8e4m3(std::fmin(std::fmax(v.lo, -448.0f), 448.0f));
        const uint8_t hi = rad_f32_to_fp8e4m3(std::fmin(std::fmax(v.hi, -448.0f), 448.0f));
        const float got = rad_fp8e4m3_to_f32(codes[j]);
        const Iv range{rad_fp8e4m3_to_f32(lo), rad_fp8e4m3_to_f32(hi)};
        ++tq.compared;
        if (lo == hi) {
            if (codes[j] != lo) tq.miss(row, first + j, got, range);
            continue;
        }
        ++tq.open;
        if (!(got >= range.lo && got <= range.hi)) tq.miss(row, first + j, got, range);
    }
}

/* The bf16 values from lo to hi, at most four; empty where there are more. */
std::vector<float> bf16_between(Iv v) {
    std::vector<float> result{v.lo};
    while (result.back() < v.hi) {
        if (result.size() == 4) return {};
        const float last = result.back();
        uint16_t code = f2b(last);
        if (last > 0) ++code;
        else if (last < 0) --code;
        else code = 0x0001;
        float next = b2f(code);
        if (last < 0 && (code & 0x7fffu) == 0) next = 0.0f;
        result.push_back(next);
    }
    return result;
}
/* had_quant_act_i8g over one group of the bf16-rounded norm, for one choice of its inputs: the
 * butterfly by ascending stride, the low index of a pair keeping the sum. */
bool i8_group_is(const float* value, const int8_t* codes, float scale, long* open) {
    float h[kGroup];
    std::copy(value, value + kGroup, h);
    for (int stride = 1; stride < kGroup; stride <<= 1)
        for (int j = 0; j < kGroup; ++j)
            if (!(j & stride)) {
                const float a = h[j], b = h[j + stride];
                h[j] = a + b;
                h[j + stride] = a - b;
            }
    float amax = 0.0f;
    for (float v : h) amax = std::max(amax, std::fabs(v));
    const float want = amax > 0.0f ? amax * (1.0f / 127.0f) : 1.0f;
    if (!same_bits(scale, want)) return false;
    const Iv inv = reciprocal(point(want));
    long wide = 0;
    for (int j = 0; j < kGroup; ++j) {
        const Iv v = times(point(h[j]), inv);
        const int lo = (int)std::rint(std::fmin(std::fmax(v.lo, -127.0f), 127.0f));
        const int hi = (int)std::rint(std::fmin(std::fmax(v.hi, -127.0f), 127.0f));
        if (codes[j] < lo || codes[j] > hi) return false;
        wide += lo != hi;
    }
    *open = wide;
    return true;
}
struct I8Tally { long groups = 0, choices = 0, unresolved = 0; };
void i8_group(const Iv* value, const int8_t* codes, float scale, int row, int first, Tally& tq, I8Tally& ti) {
    std::vector<int> free_at;
    std::vector<std::vector<float>> options;
    float chosen[kGroup];
    for (int j = 0; j < kGroup; ++j) {
        chosen[j] = value[j].lo;
        if (same_bits(value[j].lo, value[j].hi)) continue;
        auto between = bf16_between(value[j]);
        if (between.empty() || free_at.size() == 8) { ++ti.unresolved; return; }
        free_at.push_back(j);
        options.push_back(std::move(between));
    }
    ++ti.groups;
    tq.compared += kGroup;
    ti.choices += !free_at.empty();
    std::vector<size_t> pick(free_at.size(), 0);
    for (;;) {
        for (size_t f = 0; f < free_at.size(); ++f) chosen[free_at[f]] = options[f][pick[f]];
        long open = 0;
        if (i8_group_is(chosen, codes, scale, &open)) {
            tq.open += open;
            return;
        }
        size_t f = 0;
        while (f < pick.size() && ++pick[f] == options[f].size()) pick[f++] = 0;
        if (f == pick.size()) break;
    }
    tq.bad += kGroup;
    if (tq.bad <= 4 * kGroup)
        std::fprintf(stderr, "  q row %d columns %d..%d: the codes and scale %.9g match no choice of the %zu open inputs\n",
                     row, first, first + kGroup - 1, (double)scale, free_at.size());
}

/* ------------------------------------------------------------------ the cases */

std::vector<uint16_t> draw_bf16(std::mt19937& rng, size_t count, float sd) {
    std::normal_distribution<float> dist(0.0f, sd);
    std::vector<uint16_t> result(count);
    for (auto& value : result) value = f2b(dist(rng));
    return result;
}
std::vector<float> draw_gains(std::mt19937& rng, int n) {
    std::normal_distribution<float> dist(1.0f, 0.4f);
    std::vector<float> result((size_t)n);
    for (auto& value : result) value = dist(rng);
    return result;
}
/* Elements of a plane outside [0, M) x [0, n) that the launch changed. */
template<class Value> long touched(const std::vector<Value>& got, const std::vector<Value>& before, int M, int n, int pitch) {
    long count = 0;
    for (size_t index = 0; index < got.size(); ++index)
        if ((int)(index / pitch) >= M || (int)(index % pitch) >= n) count += got[index] != before[index];
    return count;
}
int verdict(const std::string& name, std::initializer_list<const Tally*> tallies, long pad, const I8Tally* i8 = nullptr) {
    long compared = 0, open = 0, bad = 0;
    for (const Tally* t : tallies) { compared += t->compared; open += t->open; bad += t->bad; }
    const bool vacuous = (double)open > kOpenShare * (double)compared + 16.0;
    const bool unresolved = i8 && i8->unresolved;
    const bool ok = !bad && !pad && !vacuous && !unresolved;
    std::printf("%-58s %9ld values, %5ld with two possible, %ld outside, padding touched %ld", name.c_str(), compared,
                open, bad, pad);
    if (i8) std::printf("; %ld int8 groups, %ld against a choice, %ld unresolved", i8->groups, i8->choices, i8->unresolved);
    std::printf(" -> %s\n", ok ? "ok" : vacuous && !bad ? "FAIL (too many open)" : "FAIL");
    return ok ? 0 : 1;
}

/* A join's parameters as the gemma4 plugin states them; `chain` as the engine adds it. */
struct Params {
    std::vector<RadParam> params;
    Params(int M, int n, bool chain = false) {
        params = {RAD_INT("M", M), RAD_INT("n", n), RAD_F64("eps", 1e-6f), RAD_STR("dtype", "bf16")};
        if (chain) params.push_back(RAD_INT("chain", 1));
    }
};

/* route 0: norm_add_norm; 1: norm_add_norm_route; 2: norm_add_norm_route_i8g. `zero_group` zeroes
 * the experts' gain over the second 128 columns: a group whose amax is 0 takes the scale 1. */
int case_attention(const RadKernelInfo& kernel, hipStream_t stream, Params geometry, int M, int n, int route,
                   bool zero_group, const char* note) {
    const int pitch = n + 16, R = M + 1, G = n / kGroup, scale_pitch = G + 3;
    const float eps = 1e-6f, mult = (float)(1.0 / std::sqrt((double)n));
    std::mt19937 rng(1000u * (unsigned)n + 10u * (unsigned)M + (unsigned)route);
    const auto a = draw_bf16(rng, (size_t)R * pitch, 3.0f), x0 = draw_bf16(rng, (size_t)R * pitch, 1.0f),
               y0 = draw_bf16(rng, (size_t)R * pitch, 1.0f), r0 = draw_bf16(rng, (size_t)R * pitch, 1.0f);
    const auto wa = draw_gains(rng, n), wy = draw_gains(rng, n), wr = draw_gains(rng, n);
    auto we = draw_gains(rng, n);
    if (zero_group) std::fill(we.begin() + kGroup, we.begin() + 2 * kGroup, 0.0f);
    std::vector<uint8_t> q0((size_t)R * pitch);
    for (auto& code : q0) code = (uint8_t)(rng() & 0x3f);
    const std::vector<float> s0((size_t)R * scale_pitch, -7.0f);

    Buffers buffers;
    auto x = x0, y = y0, r = r0;
    auto q = q0;
    auto s = s0;
    RadTensor tensors[10] = {
        rows(buffers.upload(a), RAD_BF16, M, n, pitch), vec(buffers.upload(wa), RAD_F32, n),
        rows(buffers.upload(x), RAD_BF16, M, n, pitch), vec(buffers.upload(wy), RAD_F32, n),
        rows(buffers.upload(y), RAD_BF16, M, n, pitch), vec(buffers.upload(wr), RAD_F32, n),
        rows(buffers.upload(r), RAD_BF16, M, n, pitch), vec(buffers.upload(we), RAD_F32, n),
        rows(buffers.upload(q), route == 2 ? RAD_I8 : RAD_F8E4M3, M, n, pitch),
        rows(buffers.upload(s), RAD_F32, M, G, scale_pitch)};
    if (route) {
        geometry.params.push_back(RAD_F64("mult", 1.0 / std::sqrt((double)n)));
        geometry.params.push_back(RAD_INT("group", kGroup));
    }
    RadArgs args{};
    args.t = tensors;
    args.n_t = route ? 10 : 5;
    args.p = geometry.params.data();
    args.n_p = (int)geometry.params.size();
    NJ_CHECK(kernel.launch(&args, stream) == RAD_OK);
    hip_check(hipStreamSynchronize(stream));
    Buffers::download(x, tensors[2].data);
    Buffers::download(y, tensors[4].data);
    Buffers::download(r, tensors[6].data);
    Buffers::download(q, tensors[8].data);
    Buffers::download(s, tensors[9].data);

    Tally tx{"x"}, ty{"y"}, tr{"r"}, tq{"q"}, ts{"qs"};
    I8Tally ti;
    std::vector<Iv> experts((size_t)n);
    for (int m = 0; m < M; ++m) {
        const size_t o = (size_t)m * pitch;
        double ss = 0.0;
        for (int i = 0; i < n; ++i) ss += (double)b2f(a[o + i]) * b2f(a[o + i]);
        const Iv fa = factor(ss, ss, n, eps);
        ss = 0.0;
        for (int i = 0; i < n; ++i) {
            const Iv t = rounded(gain(times(point(b2f(a[o + i])), fa), wa[(size_t)i]));     /* rmsnorm */
            tx.bf16(x[o + i], rounded(plus(point(b2f(x0[o + i])), t)), m, i);                /* add */
            ss += (double)b2f(x[o + i]) * b2f(x[o + i]);
        }
        const Iv fx = factor(ss, ss, n, eps);
        for (int i = 0; i < n; ++i) {
            const Iv normed = times(point(b2f(x[o + i])), fx);
            ty.bf16(y[o + i], rounded(gain(normed, wy[(size_t)i])), m, i);                    /* rmsnorm */
            if (!route) continue;
            tr.bf16(r[o + i], rounded(gain(gain(normed, wr[(size_t)i]), mult)), m, i);        /* the router's */
            experts[(size_t)i] = rounded(gain(normed, we[(size_t)i]));
        }
        for (int g = 0; route && g < G; ++g) {
            const float scale = s[(size_t)m * scale_pitch + g];
            if (route == 1) {
                fp8_group(&experts[(size_t)g * kGroup], &q[o + (size_t)g * kGroup], scale, m, g * kGroup, tq, ts);
            } else {
                ++ts.compared;
                i8_group(&experts[(size_t)g * kGroup], (const int8_t*)&q[o + (size_t)g * kGroup], scale, m, g * kGroup, tq, ti);
            }
        }
    }
    long pad = touched(x, x0, M, n, pitch) + touched(y, y0, M, n, pitch);
    if (route) pad += touched(r, r0, M, n, pitch) + touched(q, q0, M, n, pitch) + touched(s, s0, M, G, scale_pitch);
    else pad += touched(r, r0, 0, 0, pitch) + touched(q, q0, 0, 0, pitch) + touched(s, s0, 0, 0, scale_pitch);
    const std::string name = std::string(kernel.name) + " M " + std::to_string(M) + " n " + std::to_string(n) + note;
    return verdict(name, {&tx, &ty, &tr, &tq, &ts}, pad, route == 2 ? &ti : nullptr);
}

/* `next`: with w_n and y, the norm that reads the x the join stores. */
int case_feed_forward(const RadKernelInfo& kernel, hipStream_t stream, Params geometry, int M, int n, bool next,
                      const char* note) {
    const int pitch = n + 16, R = M + 1;
    const float eps = 1e-6f;
    std::mt19937 rng(1000u * (unsigned)n + 10u * (unsigned)M + 7u + (unsigned)next);
    /* the two arms as one [2M, n] plane, as the plugin holds them: the dense rows, then the routed */
    const auto arms = draw_bf16(rng, (size_t)2 * R * pitch, 2.0f), x0 = draw_bf16(rng, (size_t)R * pitch, 4.0f),
               y0 = draw_bf16(rng, (size_t)R * pitch, 1.0f);
    const auto wd = draw_gains(rng, n), wr = draw_gains(rng, n), wj = draw_gains(rng, n), wn = draw_gains(rng, n);
    const std::vector<uint16_t> scalar{f2b(0.4375f + (float)(rng() % 64) / 128.0f)};

    Buffers buffers;
    auto x = x0, y = y0;
    uint16_t* arms_device = (uint16_t*)buffers.upload(arms);
    RadTensor tensors[9] = {
        rows(arms_device, RAD_BF16, M, n, pitch), vec(buffers.upload(wd), RAD_F32, n),
        rows(arms_device + (size_t)M * pitch, RAD_BF16, M, n, pitch), vec(buffers.upload(wr), RAD_F32, n),
        vec(buffers.upload(wj), RAD_F32, n), rows(buffers.upload(x), RAD_BF16, M, n, pitch),
        vec(buffers.upload(scalar), RAD_BF16, 1), vec(buffers.upload(wn), RAD_F32, n),
        rows(buffers.upload(y), RAD_BF16, M, n, pitch)};
    RadArgs args{};
    args.t = tensors;
    args.n_t = next ? 9 : 7;
    args.p = geometry.params.data();
    args.n_p = (int)geometry.params.size();
    NJ_CHECK(kernel.launch(&args, stream) == RAD_OK);
    hip_check(hipStreamSynchronize(stream));
    Buffers::download(x, tensors[5].data);
    Buffers::download(y, tensors[8].data);

    Tally tx{"x"}, ty{"y"};
    const float sc = b2f(scalar[0]);
    std::vector<Iv> sum((size_t)n);
    for (int m = 0; m < M; ++m) {
        const size_t o = (size_t)m * pitch, o2 = (size_t)(M + m) * pitch;
        double sd = 0.0, sr = 0.0;
        for (int i = 0; i < n; ++i) {
            sd += (double)b2f(arms[o + i]) * b2f(arms[o + i]);
            sr += (double)b2f(arms[o2 + i]) * b2f(arms[o2 + i]);
        }
        const Iv fd = factor(sd, sd, n, eps), fr = factor(sr, sr, n, eps);
        double lo = 0.0, hi = 0.0;
        for (int i = 0; i < n; ++i) {
            const Iv u = rounded(gain(times(point(b2f(arms[o + i])), fd), wd[(size_t)i]));    /* rmsnorm (arm 1) */
            const Iv v = rounded(gain(times(point(b2f(arms[o2 + i])), fr), wr[(size_t)i]));   /* rmsnorm (arm 2) */
            sum[(size_t)i] = rounded(plus(u, v));                                             /* add */
            add_square(sum[(size_t)i], &lo, &hi);
        }
        const Iv fj = factor(lo, hi, n, eps);
        double sx = 0.0;
        for (int i = 0; i < n; ++i) {
            const Iv j = rounded(gain(times(sum[(size_t)i], fj), wj[(size_t)i]));             /* rmsnorm */
            const Iv added = rounded(plus(point(b2f(x0[o + i])), j));                         /* add */
            tx.bf16(x[o + i], rounded(gain(added, sc)), m, i);                                /* mul_weight */
            sx += (double)b2f(x[o + i]) * b2f(x[o + i]);
        }
        if (!next) continue;
        const Iv fn = factor(sx, sx, n, eps);
        for (int i = 0; i < n; ++i)                                                           /* the next rmsnorm */
            ty.bf16(y[o + i], rounded(gain(times(point(b2f(x[o + i])), fn), wn[(size_t)i])), m, i);
    }
    const long pad = touched(x, x0, M, n, pitch) + touched(y, y0, next ? M : 0, next ? n : 0, pitch);
    const std::string name = std::string(kernel.name) + (next ? " with the next norm" : "") + " M " + std::to_string(M) +
                             " n " + std::to_string(n) + note;
    return verdict(name, {&tx, &ty}, pad);
}

}

int main(int argc, char** argv) {
    const bool catalogue = argc == 3 && !std::strcmp(argv[2], "--catalogue-only");
    NJ_CHECK(argc == 2 || catalogue);
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) { std::fprintf(stderr, "%s\n", dlerror()); return 1; }
    const auto row_count = reinterpret_cast<decltype(&rad_kernel_count)>(dlsym(library, "rad_kernel_count"));
    const auto row_at = reinterpret_cast<decltype(&rad_kernel_at)>(dlsym(library, "rad_kernel_at"));
    NJ_CHECK(row_count && row_at);
    /* The one row of `op` whose constraints hold at the geometry, as the engine selects it. */
    auto match = [&](const char* op, const std::vector<RadParam>& params) {
        rad::Geometry geometry;
        for (const auto& param : params) {
            if (param.kind == RAD_P_INT) geometry.set_i(param.key, param.ival);
            else if (param.kind == RAD_P_STR) geometry.set_s(param.key, param.sval);
            else geometry.set_f(param.key, param.dval);
        }
        const RadKernelInfo* row = nullptr;
        for (int index = 0; index < row_count(); ++index) {
            const auto* candidate = row_at(index);
            if (candidate->domain != RAD_DOMAIN_DEVICE || std::strcmp(candidate->op, op)) continue;
            if (std::all_of(candidate->constraints, candidate->constraints + candidate->n_constraints,
                    [&](const RadConstraint& constraint) { return rad::constraint_holds(constraint, geometry); })) {
                NJ_CHECK(!row);
                row = candidate;
            }
        }
        return row;
    };
    auto routed = [](Params geometry, int n, int group) {
        geometry.params.push_back(RAD_F64("mult", 1.0 / std::sqrt((double)n)));
        geometry.params.push_back(RAD_INT("group", group));
        return geometry.params;
    };
    /* Gemma 4 26B-A4B, 12B and 31B (text_config.hidden_size). */
    constexpr int widths[]{2816, 3840, 5376};
    constexpr int heights[]{1, 2, 8, 64, 512};
    const char* const ops[]{"norm_add_norm", "norm_add_norm_route", "norm_add_norm_route_i8g", "join_add_scale"};
    const char* const names[]{"norm_add_norm_bf16", "norm_add_norm_route_bf16", "norm_add_norm_route_i8g_bf16",
                              "join_add_scale_bf16"};
    const RadKernelInfo* kernels[4]{};
    for (int n : widths) {
        const Params plain(512, n);
        for (int op = 0; op < 4; ++op) {
            const bool route = op == 1 || op == 2;
            const auto params = route ? routed(plain, n, kGroup) : plain.params;
            const auto* row = kernels[op] = match(ops[op], params);
            NJ_CHECK(row && !std::strcmp(row->name, names[op]) && row->launch && row->opd_shape && !row->scratch);
            const int operands = op == 0 ? 5 : route ? 10 : 9;
            RadOpdDesc desc{};
            for (int operand = 0; operand < operands; ++operand)
                NJ_CHECK(row->opd_shape(params.data(), (int)params.size(), operand, &desc) == RAD_OK);
            NJ_CHECK(row->opd_shape(params.data(), (int)params.size(), operands, &desc) != RAD_OK);
            if (!route) continue;
            NJ_CHECK(row->opd_shape(params.data(), (int)params.size(), 8, &desc) == RAD_OK && desc.rank == 2 &&
                     desc.shape[0] == 512 && desc.shape[1] == n && desc.dtype == (op == 2 ? RAD_I8 : RAD_F8E4M3));
            NJ_CHECK(row->opd_shape(params.data(), (int)params.size(), 9, &desc) == RAD_OK && desc.rank == 2 &&
                     desc.shape[0] == 512 && desc.shape[1] == n / kGroup && desc.dtype == RAD_F32);
            /* whole 128-column groups at a group of 128 only */
            NJ_CHECK(!match(ops[op], routed(plain, n, 64)));
            NJ_CHECK(!match(ops[op], routed(Params(512, n + 64), n + 64, kGroup)));
        }
    }
    /* the widest row the kernels hold in registers is 6144 */
    NJ_CHECK(match("norm_add_norm", Params(1, 6144).params) && !match("norm_add_norm", Params(1, 6145).params));
    NJ_CHECK(match("join_add_scale", Params(1, 6144).params) && !match("join_add_scale", Params(1, 6145).params));
    if (catalogue) { dlclose(library); return 0; }

    int count = 0;
    const auto status = hipGetDeviceCount(&count);
    if (status == hipErrorNoDevice) return 77;
    hip_check(status);
    int chosen = -1;
    for (int device = 0; device < count; ++device) {
        hipDeviceProp_t props{};
        hip_check(hipGetDeviceProperties(&props, device));
        if (std::string(props.gcnArchName).substr(0, 7) == "gfx1100") { chosen = device; break; }
    }
    if (chosen < 0) { std::fprintf(stderr, "no visible gfx1100\n"); return 77; }
    hip_check(hipSetDevice(chosen));
    hipStream_t stream{};
    hip_check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    int failures = 0;
    for (int M : heights)
        for (int n : widths)
            for (bool chain : {false, true}) {
                if (chain && M > 64) continue;
                const Params geometry(M, n, chain);
                const char* note = chain ? " chained" : "";
                for (int route = 0; route < 3; ++route)
                    failures += case_attention(*kernels[route], stream, geometry, M, n, route, false, note);
                for (bool next : {false, true})
                    failures += case_feed_forward(*kernels[3], stream, geometry, M, n, next, note);
            }
    for (int route = 1; route < 3; ++route)
        failures += case_attention(*kernels[route], stream, Params(8, 2816), 8, 2816, route, true, " (a zero group)");
    hip_check(hipStreamDestroy(stream));
    dlclose(library);
    std::printf("norm joins: %d case(s) failed\n", failures);
    return failures ? 1 : 0;
}
