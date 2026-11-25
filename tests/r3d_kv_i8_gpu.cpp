/* r3d_kv_i8_gpu -- libr3d's int8 KV cache against its bf16 cache, on the card.
 *
 * The same random keys and values go through kv_store_i8 into an int8 cache and through
 * kv_store_bf16 into a bf16 one. Then the int8 windowed attention row and the bf16 row of the same
 * geometry run on the same queries, and the two outputs must agree within what the int8 step
 * allows. That is done at every (head_dim, gqa) that has an int8 row, at a decode step (q_len 1),
 * a verify-sized step and a prefill-shaped one, without a window and under a window shorter than
 * the context. The qrow rows run on the same data with q at a row pitch and must write the window
 * rows' bytes.
 *
 * The store. A slot of the int8 cache is K's and V's int8 codes and an f16 scale for each
 * (abi/rad_builder.h): the scale is the head's largest magnitude in that token over 127, rounded
 * to f16 and at most 515, and a code is rint(x / scale) clamped to +-127 (docs/OPS.md). The scales
 * read back from the cache must be exactly that, and a value x must come back as code * scale to
 * within
 *
 *     e = max(scale / 2, absmax - 127 * scale)
 *
 * which is the rounding of the code, or the clamp where the f16 rounding took the scale down. At a
 * normal f16 scale the first term is the larger: the rounding moves the scale by at most 2^-11 of
 * itself, and 127 * 2^-11 < 1/2. A subnormal scale is a multiple of 2^-24, so a head whose absmax
 * is under 127 * 2^-14 (7.8e-3) keeps fewer than 127 steps, and one under 127 * 2^-25 (3.8e-6)
 * gets scale 0: all its codes are 0 and e is its absmax. The third KV head here is such a head in
 * every token, and some tokens of the second are just above it.
 *
 * The tolerance. A query row q scores key j as s_j = scale * (q . K_j). K_j is off by at most eK_j
 * an element, so s_j moves by at most
 *
 *     eps = scale * sum|q| * max_j eK_j.
 *
 * A softmax weight p_j = exp(s_j) / sum_i exp(s_i) then changes by a factor inside exp(+-2 eps),
 * its numerator and its denominator each moving by a factor inside exp(+-eps), so the weights
 * move by at most exp(2 eps) - 1 in sum. An output element is sum_j p_j V_j, V_j is off by at most
 * eV_j, and the weights sum to one, so the int8 cache moves an output element by at most
 *
 *     (exp(2 eps) - 1) * max|V| + max_j eV_j.
 *
 * To that is added what the two kernels' own arithmetic may differ by on equal inputs. Each rounds
 * its output to bf16 (2^-9 of the value) and the WMMA kernel of a prefill-shaped step holds its
 * weights in f16 (2^-11 of each): under 2^-7 * max|V| together. Over the int8 cache the WMMA kernel
 * also rounds weight times V scale to f16, which is off by up to 2^-25 where that product is
 * subnormal, times a code of at most 127: 127 * 2^-24 a key covers it and the bf16 kernel's own
 * subnormal weights.
 *
 * The bound is the worst case of every code rounding against the query at once, so the observed
 * difference is far below it; the test prints both. A query of the first two KV heads is a
 * multiple of one key (in reach, ahead of the row or just outside the window), so that the output
 * is that key's V, or a mean without it: an output well above the bound, which the test also
 * checks, since a bound wider than the outputs would pass anything.
 */
#include "../core/rad_internal.h"
#include "rad_plugin.h"
#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

namespace {
void check(bool ok, const char* expr, int line) {
    if (!ok) {
        std::fprintf(stderr, "r3d_kv_i8_gpu line %d: %s\n", line, expr);
        std::exit(1);
    }
}
#define KV_CHECK(expr) check((expr), #expr, __LINE__)
void hip_check(hipError_t status) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "HIP: %s\n", hipGetErrorString(status));
        std::exit(1);
    }
}
struct Buffers {
    std::vector<void*> owned;
    void* raw(size_t bytes, int fill) {
        void* device = nullptr;
        hip_check(hipMalloc(&device, bytes));
        owned.push_back(device);
        /* the fill runs on the null stream, which the launches' stream does not wait for */
        hip_check(hipMemset(device, fill, bytes));
        hip_check(hipDeviceSynchronize());
        return device;
    }
    template<class Value> void* upload(const std::vector<Value>& values) {
        void* device = raw(values.size() * sizeof(Value), 0);
        hip_check(hipMemcpy(device, values.data(), values.size() * sizeof(Value), hipMemcpyHostToDevice));
        return device;
    }
    ~Buffers() { for (void* data : owned) hip_check(hipFree(data)); }
};
template<class Value> std::vector<Value> download(const void* device, size_t count) {
    std::vector<Value> values(count);
    hip_check(hipMemcpy(values.data(), device, count * sizeof(Value), hipMemcpyDeviceToHost));
    return values;
}
/* A dense tensor of up to four extents; `pitch` (elements) replaces the leading stride. */
RadTensor tensor(void* data, uint32_t dtype, std::vector<int64_t> shape, int64_t pitch = 0) {
    RadTensor result{};
    result.data = data;
    result.dtype = dtype;
    result.rank = (uint32_t)shape.size();
    int64_t stride = 1;
    for (int axis = (int)shape.size() - 1; axis >= 0; --axis) {
        result.shape[axis] = shape[axis];
        result.stride[axis] = stride;
        stride *= shape[axis];
    }
    if (pitch) result.stride[0] = pitch;
    return result;
}

int (*row_count)(void);
const RadKernelInfo* (*row_at)(int);
/* The device row of this name, which must declare the geometry: every constraint holds on it. */
const RadKernelInfo* row(const std::string& name, const std::vector<RadParam>& params) {
    rad::Geometry geometry;
    for (const auto& param : params) {
        if (param.kind == RAD_P_INT) geometry.set_i(param.key, param.ival);
        else geometry.set_s(param.key, param.sval);
    }
    for (int index = 0; index < row_count(); ++index) {
        const auto* candidate = row_at(index);
        if (candidate->domain != RAD_DOMAIN_DEVICE || name != candidate->name) continue;
        KV_CHECK(std::all_of(candidate->constraints, candidate->constraints + candidate->n_constraints,
            [&](const RadConstraint& constraint) { return rad::constraint_holds(constraint, geometry); }));
        return candidate;
    }
    std::fprintf(stderr, "r3d_kv_i8_gpu: no row %s\n", name.c_str());
    std::exit(1);
}
void launch(const RadKernelInfo* kernel, std::vector<RadTensor>& tensors, std::vector<RadParam>& params,
            Buffers& buffers, hipStream_t stream) {
    RadArgs args{};
    args.t = tensors.data();
    args.n_t = (int)tensors.size();
    args.p = params.data();
    args.n_p = (int)params.size();
    args.world_size = 1;
    if (kernel->scratch) {
        args.scratch_bytes = kernel->scratch(&args);
        KV_CHECK(args.scratch_bytes >= 0);
        if (args.scratch_bytes) args.scratch = buffers.raw((size_t)args.scratch_bytes, 0);
    }
    const int status = kernel->launch(&args, stream);
    if (status != RAD_OK) {
        std::fprintf(stderr, "r3d_kv_i8_gpu: %s returned %d\n", kernel->name, status);
        std::exit(1);
    }
    hip_check(hipStreamSynchronize(stream));
}

constexpr int64_t kBlock = 16, kHeads = 3, kContext = 997;   /* three KV heads; a partly filled last block */
constexpr int64_t kLogical = (kContext + kBlock - 1) / kBlock, kPhysical = kLogical + 2;
constexpr uint16_t kNan = 0x7fc0;
/* The physical block of a logical one: every block out of order, two never written. */
int64_t block_of(int64_t logical) { return (logical * 37 + 5) % kPhysical; }

/* The largest magnitudes of K (first) and V (second) in KV head `head` of token `token`. The
 * second head is the first at other ranges, with every eleventh token a few steps of a subnormal
 * scale; the third is under 127 * 2^-25 throughout. */
std::pair<float, float> amplitude(int64_t token, int64_t head) {
    if (head == 0) return {1.0f, 2.0f};
    if (head == 1) return token % 11 == 7 ? std::pair<float, float>{1.2e-5f, 2.5e-5f}
                                         : std::pair<float, float>{4.0f, 0.5f};
    return {3.0e-6f, 3.5e-6f};
}

struct Lcg {
    uint32_t state;
    float unit() {                                         /* [-1, 1) */
        state = state * 1664525u + 1013904223u;
        return (float)(state >> 8) / 8388608.0f - 1.0f;
    }
    uint32_t below(uint32_t bound) {
        state = state * 1664525u + 1013904223u;
        return (state >> 8) % bound;
    }
};

/* One (head_dim, gqa): the two caches, and what the store left in the int8 one. */
struct Caches {
    int64_t head_dim, gqa, slot;
    std::vector<uint16_t> keys, values;                    /* [token, kv head, head_dim] bf16 */
    void* int8 = nullptr;
    void* bf16 = nullptr;
    void* table = nullptr;
    void* used = nullptr;
    /* per KV head: the largest code error of K and of V over the tokens, and the largest |V| */
    float key_error[kHeads] = {}, value_error[kHeads] = {}, value_max[kHeads] = {};
};

void store(Caches& caches, Buffers& buffers, hipStream_t stream) {
    const int64_t head_dim = caches.head_dim, lane = kHeads * head_dim;
    caches.slot = 2 * head_dim + 4;
    Lcg random{0x2545f491u + (uint32_t)head_dim + (uint32_t)caches.gqa};
    caches.keys.resize(size_t(kContext * lane));
    caches.values.resize(size_t(kContext * lane));
    std::vector<int32_t> slots(kContext), table(kLogical), used{(int32_t)kContext};
    for (int64_t token = 0; token < kContext; ++token) {
        slots[token] = (int32_t)(block_of(token / kBlock) * kBlock + token % kBlock);
        for (int64_t head = 0; head < kHeads; ++head) {
            const auto [key_range, value_range] = amplitude(token, head);
            for (int64_t dim = 0; dim < head_dim; ++dim) {
                const size_t index = size_t(token * lane + head * head_dim + dim);
                caches.keys[index] = rad_f32_to_bf16(key_range * random.unit());
                caches.values[index] = rad_f32_to_bf16(value_range * random.unit());
            }
        }
    }
    for (int64_t logical = 0; logical < kLogical; ++logical) table[logical] = (int32_t)block_of(logical);
    /* Unwritten slots hold NaN in both caches (0x7f7f is an f16 NaN), so a key read from past the
     * context or from a block the table does not name shows in the output. */
    const size_t int8_bytes = size_t(kPhysical * kHeads * kBlock * caches.slot);
    const size_t bf16_count = size_t(kPhysical * kHeads * kBlock * 2 * head_dim);
    caches.int8 = buffers.raw(int8_bytes, 0x7f);
    caches.bf16 = buffers.upload(std::vector<uint16_t>(bf16_count, kNan));
    caches.table = buffers.upload(table);
    caches.used = buffers.upload(used);
    void* keys = buffers.upload(caches.keys);
    void* values = buffers.upload(caches.values);
    void* mapping = buffers.upload(slots);
    for (const bool wide : {false, true}) {
        std::vector<RadParam> params{RAD_INT("M", kContext), RAD_INT("head_dim", head_dim),
            RAD_INT("n_head_kv", kHeads), RAD_INT("block_size", kBlock), RAD_STR("kv_dtype", wide ? "bf16" : "i8")};
        std::vector<RadTensor> tensors{
            tensor(keys, RAD_BF16, {kContext, kHeads, head_dim}),
            tensor(values, RAD_BF16, {kContext, kHeads, head_dim}),
            tensor(mapping, RAD_I32, {kContext}),
            wide ? tensor(caches.bf16, RAD_BF16, {kPhysical, kHeads, kBlock, 2 * head_dim})
                 : tensor(caches.int8, RAD_I8, {kPhysical, kHeads, kBlock, caches.slot})};
        launch(row(wide ? "kv_store_bf16" : "kv_store_i8", params), tensors, params, buffers, stream);
    }

    const auto wide = download<uint16_t>(caches.bf16, bf16_count);
    const auto narrow = download<uint8_t>(caches.int8, int8_bytes);
    int zero_scales = 0, subnormal_scales = 0, clamped = 0;
    for (int64_t token = 0; token < kContext; ++token) {
        for (int64_t head = 0; head < kHeads; ++head) {
            const size_t at = size_t((slots[token] / kBlock * kHeads + head) * kBlock + slots[token] % kBlock);
            const uint16_t* exact = &wide[at * 2 * head_dim];
            const uint8_t* codes = &narrow[at * caches.slot];
            for (int half = 0; half < 2; ++half) {
                const uint16_t* source = (half ? caches.values : caches.keys).data() + token * lane + head * head_dim;
                /* the bf16 cache is the input, bit for bit */
                KV_CHECK(!std::memcmp(exact + half * head_dim, source, size_t(head_dim) * 2));
                float largest = 0;
                for (int64_t dim = 0; dim < head_dim; ++dim)
                    largest = std::max(largest, std::fabs(rad_bf16_to_f32(source[dim])));
                uint16_t stored;
                std::memcpy(&stored, codes + 2 * head_dim + 2 * half, 2);
                KV_CHECK(stored == rad_f32_to_f16(std::fmin(largest / 127.0f, 515.0f)));
                const float scale = rad_f16_to_f32(stored);
                const float error = std::max(scale / 2, largest - 127 * scale);
                zero_scales += stored == 0;
                subnormal_scales += stored != 0 && stored < 0x400;
                clamped += largest > 127.5f * scale;
                for (int64_t dim = 0; dim < head_dim; ++dim) {
                    const int code = (int8_t)codes[half * head_dim + dim];
                    KV_CHECK(code >= -127 && code <= 127);
                    KV_CHECK(std::fabs(rad_bf16_to_f32(source[dim]) - (float)code * scale) <= error * 1.0001f);
                    KV_CHECK(scale != 0 || code == 0);
                }
                float& worst = (half ? caches.value_error : caches.key_error)[head];
                worst = std::max(worst, error);
                if (half) caches.value_max[head] = std::max(caches.value_max[head], largest);
            }
        }
    }
    /* The third head's scales are all 0, and the second's quiet tokens are the subnormal ones. */
    KV_CHECK(zero_scales == 2 * kContext);
    KV_CHECK(subnormal_scales > 0 && clamped > 2 * kContext);
    std::printf("h%lld gqa%lld store: %lld scales exact; %d zero (their heads store 0 codes), %d subnormal, "
                "%d heads clamped at 127 codes\n", (long long)head_dim, (long long)caches.gqa,
                (long long)(2 * kContext * kHeads), zero_scales, subnormal_scales, clamped - zero_scales);
}

/* One step of `q_len` rows at the end of the context, on both caches. */
void attend(const Caches& caches, int64_t q_len, int64_t window, hipStream_t stream) {
    const int64_t head_dim = caches.head_dim, gqa = caches.gqa, heads = kHeads * gqa;
    const int64_t width = heads * head_dim, lane = kHeads * head_dim;
    const float softmax_scale = 1.0f / std::sqrt((float)head_dim);
    Buffers buffers;
    Lcg random{0x9e3779b9u + (uint32_t)(q_len * 131 + window)};
    std::vector<uint16_t> query(size_t(q_len * width));
    for (int64_t row_index = 0; row_index < q_len; ++row_index) {
        const int64_t position = kContext - q_len + row_index;
        for (int64_t head = 0; head < heads; ++head) {
            const int64_t kv_head = head / gqa;
            uint16_t* out = &query[size_t(row_index * width + head * head_dim)];
            if (kv_head == 2) {
                for (int64_t dim = 0; dim < head_dim; ++dim) out[dim] = rad_f32_to_bf16(random.unit());
                continue;
            }
            /* The key this query is a multiple of: a recent one, any earlier one, one ahead of the
             * row (which a causal row must not see), or the first one a window leaves out. */
            int64_t target = position;
            switch ((row_index * heads + head) % 4) {
                case 0: target = position - 1 - random.below(8); break;
                case 1: target = random.below((uint32_t)position + 1); break;
                case 2: target = std::min(position + 2, kContext - 1); break;
                default: target = window ? position - window : random.below((uint32_t)position + 1);
            }
            if (kv_head == 1 && target % 11 == 7) --target;          /* not a quiet token */
            /* Its own key scores 8 (in the exponent), the others about 8 / sqrt(head_dim). */
            const float range = amplitude(target, kv_head).first;
            const float gain = 3 * 8 / (std::sqrt((float)head_dim) * range * range);
            const uint16_t* key = &caches.keys[size_t(target * lane + kv_head * head_dim)];
            for (int64_t dim = 0; dim < head_dim; ++dim)
                out[dim] = rad_f32_to_bf16(gain * rad_bf16_to_f32(key[dim]));
        }
    }
    /* The same rows two rows of NaN apart, for the qrow rows. */
    const int64_t pitch = 3 * width;
    std::vector<uint16_t> pitched(size_t(q_len * pitch), kNan);
    for (int64_t row_index = 0; row_index < q_len; ++row_index)
        std::copy_n(&query[size_t(row_index * width)], width, &pitched[size_t(row_index * pitch)]);
    void* dense = buffers.upload(query);
    void* apart = buffers.upload(pitched);

    const std::string geometry = "h" + std::to_string(head_dim) + "_gqa" + std::to_string(gqa);
    std::vector<uint16_t> result[2];
    int qrow_rows = 0;
    for (const bool wide : {false, true}) {
        std::vector<RadParam> params{RAD_INT("q_len", q_len), RAD_INT("head_dim", head_dim), RAD_INT("gqa", gqa),
            RAD_INT("block_size", kBlock), RAD_INT("causal", 1), RAD_INT("window", window),
            RAD_STR("q_dtype", "bf16"), RAD_STR("kv_dtype", wide ? "bf16" : "i8"), RAD_INT("n_head", heads),
            RAD_INT("max_ctx", kLogical * kBlock), RAD_INT("max_seqs", 1)};
        const std::string cache = wide ? "_bf16kv" : "_i8kv";
        for (const bool qrow : {false, true}) {
            /* h512 gqa16 has no qrow row */
            if (qrow && gqa == 16) continue;
            std::vector<RadTensor> tensors(8);
            tensors[0] = tensor(qrow ? apart : dense, RAD_BF16, {q_len, heads, head_dim}, qrow ? pitch : 0);
            tensors[1] = wide ? tensor(caches.bf16, RAD_BF16, {kPhysical, kHeads, kBlock, 2 * head_dim})
                              : tensor(caches.int8, RAD_I8, {kPhysical, kHeads, kBlock, caches.slot});
            tensors[2] = tensor(caches.table, RAD_I32, {1, kLogical});
            tensors[3] = tensor(caches.used, RAD_I32, {1});
            tensors[7] = tensor(buffers.upload(std::vector<uint16_t>(query.size(), kNan)), RAD_BF16,
                                {q_len, heads, head_dim});
            launch(row((qrow ? "attn_qrow_" : "attn_window_") + geometry + cache, params), tensors, params,
                   buffers, stream);
            const auto got = download<uint16_t>(tensors[7].data, query.size());
            if (!qrow) result[wide] = got;
            else { KV_CHECK(got == result[wide]); ++qrow_rows; }
        }
    }

    /* The prefill-shaped step is the WMMA kernel's (from 64 query rows). */
    const float subnormal = q_len >= 64 ? (float)kContext * 127 * std::ldexp(1.0f, -24) : 0;
    double worst = 0, part = 0, bound_max[kHeads] = {}, output_max[kHeads] = {}, quiet = 0;
    for (int64_t row_index = 0; row_index < q_len; ++row_index) {
        for (int64_t head = 0; head < heads; ++head) {
            const int64_t kv_head = head / gqa;
            const size_t base = size_t(row_index * width + head * head_dim);
            double sum = 0;
            for (int64_t dim = 0; dim < head_dim; ++dim) sum += std::fabs(rad_bf16_to_f32(query[base + dim]));
            const double eps = softmax_scale * sum * caches.key_error[kv_head];
            const double bound = std::expm1(2 * eps) * caches.value_max[kv_head] + caches.value_error[kv_head]
                + std::ldexp(caches.value_max[kv_head], -7) + subnormal;
            bound_max[kv_head] = std::max(bound_max[kv_head], bound);
            for (int64_t dim = 0; dim < head_dim; ++dim) {
                const float narrow = rad_bf16_to_f32(result[0][base + dim]);
                const float wide = rad_bf16_to_f32(result[1][base + dim]);
                KV_CHECK(std::isfinite(narrow) && std::isfinite(wide));
                const double difference = std::fabs((double)narrow - wide);
                if (difference > bound) {
                    std::fprintf(stderr, "%s q_len %lld window %lld row %lld head %lld dim %lld: int8 %g bf16 %g, "
                                 "bound %g\n", geometry.c_str(), (long long)q_len, (long long)window,
                                 (long long)row_index, (long long)head, (long long)dim, narrow, wide, bound);
                    std::exit(1);
                }
                worst = std::max(worst, difference);
                part = std::max(part, difference / bound);
                output_max[kv_head] = std::max(output_max[kv_head], (double)std::fabs(wide));
                if (kv_head == 2) quiet = std::max(quiet, (double)std::fabs(narrow));
            }
        }
    }
    /* A bound is a check only where the outputs are above it. */
    for (int head = 0; head < 2; ++head) KV_CHECK(output_max[head] >= 2 * bound_max[head]);
    std::printf("%s q_len %lld window %lld: largest |int8 - bf16| %.3g, at most %.3f of its bound; bounds %.3g "
                "%.3g %.3g against bf16 outputs up to %.3g %.3g %.3g (the third head over int8: %.3g); %d qrow "
                "rows byte for byte\n", geometry.c_str(), (long long)q_len, (long long)window, worst, part,
                bound_max[0], bound_max[1], bound_max[2], output_max[0], output_max[1], output_max[2], quiet,
                qrow_rows);
}
}

int main(int argc, char** argv) {
    KV_CHECK(argc == 2);
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) { std::fprintf(stderr, "%s\n", dlerror()); return 1; }
    row_count = reinterpret_cast<decltype(&rad_kernel_count)>(dlsym(library, "rad_kernel_count"));
    row_at = reinterpret_cast<decltype(&rad_kernel_at)>(dlsym(library, "rad_kernel_at"));
    KV_CHECK(row_count && row_at);
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
    /* every (head_dim, gqa) with an int8 row */
    const int64_t instances[][2] = {{256, 2}, {512, 8}, {512, 16}};
    for (const auto& instance : instances) {
        Buffers buffers;
        Caches caches{};
        caches.head_dim = instance[0];
        caches.gqa = instance[1];
        store(caches, buffers, stream);
        for (const int64_t q_len : {1, 5, 80})
            for (const int64_t window : {0, 300}) attend(caches, q_len, window, stream);
    }
    hip_check(hipStreamDestroy(stream));
    dlclose(library);
}
