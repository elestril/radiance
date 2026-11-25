/* r3d_gemm_w4a8_gpu -- gemm_w4a8_nt_m64 by value, on a card.
 *
 * The row reads its weight in the fragment order its layout hooks write, so r3d_selftest cannot
 * build its operands from a shape alone. Here a random symmetric int4 weight goes through the
 * row's own hooks, the row is launched at Gemma 4 31B's projections, and the output is compared
 * with a host computation of the same quantised product from the canonical planes:
 *
 *   y[m, n] = bf16((sum_k a[m, k] * code[n, k] * scale[n, k / 128]) * a_scale[m])
 *
 * with the activation stored as quant_act_i8 writes it: within every sixteen columns the middle
 * two dwords exchanged.
 *
 * Tolerance: none, every compared value is bit for bit. A 128-column group's sum of products is
 * an integer below 2^24, exact in any order and in f32. After it the kernels do the same IEEE
 * operations in one order: acc = fma(16 * sum, scale, acc) over the groups of a K split in
 * ascending order, the splits added from 0 in ascending order, then (v * a_scale) * 0.0625 and
 * one rounding to bf16. The host does exactly that. What it cannot know is the number of K
 * splits, which the row's launch law picks from the shape (r3dh_gemm_args.h, pick_sk), so it
 * computes every split of the law's ladder that divides K and the launch must match one of them
 * in every compared value. Splits differ in the last bits of the f32 sum, which the rounding to
 * bf16 mostly hides, so several splits usually match. The prefill tile (256 rows and more) does
 * not split: it must match at one split.
 *
 * The host product costs M * N * K, so it is taken at a sample: the columns of the first eight
 * and the last eight 16-column tiles and of four tiles between (every position of a tile in a
 * block of either kernel, and the clamped last tile), and all rows up to 256, above that the
 * first and the last 128 and 64 between. Every other value must have been written (the output
 * starts as NaN).
 *
 * K must be whole 128-column scale groups (the row's constraint), so there is no K that is not a
 * multiple of 64 to run.
 */
#include "r3d_w4_fixture.h"
#include "../core/rad_internal.h"
#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <dlfcn.h>
#include <string>

namespace {
using namespace w4_fixture;
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
    ~Buffers() { for (void* data : owned) hip_check(hipFree(data)); }
};
RadTensor stored(void* data, const RadLayout& layout) {
    return matrix(data, layout.dtype, layout.shape[0], layout.shape[1]);
}
/* Column c of an activation row sits at this byte (libref's i8_step_pos). */
int step_pos(int c) { return (c & ~12) | ((c & 4) << 1) | ((c & 8) >> 1); }
uint32_t next(uint32_t& random) { return random = 1664525u * random + 1013904223u; }

/* The splits of K the launch law can pick. */
constexpr int kSplits[]{1, 2, 4, 5, 8, 10, 16, 20};
constexpr int kHeights[]{1, 4, 16, 17, 64, 65, 255, 256, 512, 2048};

std::vector<int> sample_columns(int N) {
    const int tiles = N / 16;
    std::vector<int> picked;
    for (int tile = 0; tile < 8; ++tile) {
        picked.push_back(std::min(tile, tiles - 1));
        picked.push_back(std::max(tiles - 1 - tile, 0));
    }
    for (int between = 1; between <= 4; ++between) picked.push_back(int(int64_t(tiles) * between / 5));
    std::sort(picked.begin(), picked.end());
    picked.erase(std::unique(picked.begin(), picked.end()), picked.end());
    std::vector<int> columns;
    for (int tile : picked)
        for (int column = 0; column < 16; ++column) columns.push_back(tile * 16 + column);
    return columns;
}
std::vector<int> sample_rows(int M) {
    std::vector<int> rows;
    for (int m = 0; m < M; ++m)
        if (M <= 256 || m < 128 || m >= M - 128 || (m - 128) % ((M - 256) / 64) == 0) rows.push_back(m);
    return rows;
}

int run(const RadKernelInfo& kernel, hipStream_t stream, const char* what, int N, int K) {
    const int groups = K / 128;
    auto geometry = params(1, N, K);
    Weight weight(N, K, 17u + (unsigned)N + (unsigned)K);
    const RadLayout codes_layout = describe(geometry, weight, kCodes, kernel.layout);
    const RadLayout scale_layout = describe(geometry, weight, kScale, kernel.layout);
    Buffers weights;
    void* codes_device = weights.upload(pack(geometry, weight, kCodes, kernel.layout, kernel.relayout));
    void* scale_device = weights.upload(pack(geometry, weight, kScale, kernel.layout, kernel.relayout));

    /* the sampled columns' weight rows, as the canonical planes hold them */
    const std::vector<int> columns = sample_columns(N);
    std::vector<int8_t> w(columns.size() * K);
    std::vector<float> w_scale(columns.size() * groups);
    for (size_t c = 0; c < columns.size(); ++c) {
        for (int k = 0; k < K; ++k) w[c * K + k] = (int8_t)weight.code(columns[c], k);
        for (int g = 0; g < groups; ++g)
            w_scale[c * groups + g] = rad_f16_to_f32(weight.scale[size_t(columns[c]) * groups + g]);
    }

    int failures = 0;
    uint32_t random = 99u + (unsigned)N;
    for (int M : kHeights) {
        std::vector<int8_t> a(size_t(M) * K), a_stored(size_t(M) * K);
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k) {
                a[size_t(m) * K + k] = (int8_t)((int)(next(random) >> 24) % 255 - 127);
                a_stored[size_t(m) * K + step_pos(k)] = a[size_t(m) * K + k];
            }
        std::vector<float> a_scale(M);
        for (auto& value : a_scale) value = float(1 + (next(random) >> 22)) / 65536;
        std::vector<uint16_t> y(size_t(M) * N, 0x7fc0);

        Buffers buffers;
        RadTensor tensors[7]{};
        tensors[0] = matrix(buffers.upload(a_stored), RAD_I8, M, K);
        tensors[1] = matrix(buffers.upload(a_scale), RAD_F32, M, 1);
        tensors[2] = stored(codes_device, codes_layout);
        tensors[3] = stored(scale_device, scale_layout);
        tensors[4] = matrix(buffers.upload(y), RAD_BF16, M, N);
        geometry[0] = RAD_INT("M", M);
        RadArgs args{};
        args.t = tensors;
        args.n_t = 7;
        args.p = geometry.data();
        args.n_p = geometry.size();
        W4_CHECK(kernel.launch(&args, stream) == RAD_OK);
        hip_check(hipStreamSynchronize(stream));
        hip_check(hipMemcpy(y.data(), tensors[4].data, y.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));

        long unwritten = 0;
        for (uint16_t value : y) unwritten += (value & 0x7fff) > 0x7f80;

        bool alive[std::size(kSplits)];
        for (size_t s = 0; s < std::size(kSplits); ++s) alive[s] = groups % kSplits[s] == 0;
        long compared = 0, off_at_one = 0;
        std::vector<float> sums(groups);
        for (int m : sample_rows(M)) {
            const int8_t* row = a.data() + size_t(m) * K;
            for (size_t c = 0; c < columns.size(); ++c) {
                const int8_t* column = w.data() + c * K;
                const float* scale = w_scale.data() + c * groups;
                for (int g = 0; g < groups; ++g) {
                    int sum = 0;
                    for (int k = g * 128; k < (g + 1) * 128; ++k) sum += row[k] * column[k];
                    sums[g] = (float)(16 * sum);
                }
                const uint16_t got = y[size_t(m) * N + columns[c]];
                ++compared;
                for (size_t s = 0; s < std::size(kSplits); ++s) {
                    if (!alive[s] && s) continue;
                    const int per = groups / kSplits[s];
                    if (groups % kSplits[s]) continue;
                    float v = 0.0f;
                    for (int split = 0; split < kSplits[s]; ++split) {
                        float acc = 0.0f;
                        for (int g = split * per; g < (split + 1) * per; ++g) acc = std::fma(sums[g], scale[g], acc);
                        v += acc;
                    }
                    const bool same = rad_f32_to_bf16(v * a_scale[m] * 0.0625f) == got;
                    if (!s) off_at_one += !same;
                    if (!same) alive[s] = false;
                }
            }
        }
        std::string matched;
        for (size_t s = 0; s < std::size(kSplits); ++s)
            if (alive[s]) matched += (matched.empty() ? "" : ",") + std::to_string(kSplits[s]);
        const bool ok = !unwritten && !matched.empty() && (M < 256 || alive[0]);
        std::printf("%-9s N %5d K %5d M %4d: %7ld values, bit for bit at the K splits %s (%ld differ at one "
                    "split); %ld never written -> %s\n", what, N, K, M, compared,
                    matched.empty() ? "none" : matched.c_str(), off_at_one, unwritten, ok ? "ok" : "FAIL");
        failures += !ok;
    }
    return failures;
}
}

int main(int argc, char** argv) {
    const bool catalogue = argc == 3 && !std::strcmp(argv[2], "--catalogue-only");
    W4_CHECK(argc == 2 || catalogue);
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) { std::fprintf(stderr, "%s\n", dlerror()); return 1; }
    const auto row_count = reinterpret_cast<decltype(&rad_kernel_count)>(dlsym(library, "rad_kernel_count"));
    const auto row_at = reinterpret_cast<decltype(&rad_kernel_at)>(dlsym(library, "rad_kernel_at"));
    W4_CHECK(row_count && row_at);
    auto match = [&](const w4_fixture::Params& params) {
        rad::Geometry geometry;
        for (const auto& param : params) {
            if (param.kind == RAD_P_INT) geometry.set_i(param.key, param.ival);
            else geometry.set_s(param.key, param.sval);
        }
        const RadKernelInfo* row = nullptr;
        for (int index = 0; index < row_count(); ++index) {
            const auto* candidate = row_at(index);
            if (candidate->domain != RAD_DOMAIN_DEVICE || std::strcmp(candidate->op, "gemm_nt_q")) continue;
            if (std::all_of(candidate->constraints, candidate->constraints + candidate->n_constraints,
                    [&](const RadConstraint& constraint) { return rad::constraint_holds(constraint, geometry); })) {
                W4_CHECK(!row);
                row = candidate;
            }
        }
        return row;
    };
    /* Gemma 4 31B's projections: hidden 5376, MLP 21504, 32 heads of 256 over 16 KV heads on the
     * sliding layers and 32 heads of 512 over 4 KV heads on the full ones. */
    struct Shape { const char* what; int N, K; };
    constexpr Shape shapes[]{{"q", 8192, 5376}, {"k, v", 4096, 5376}, {"o", 5376, 8192},
                             {"q full", 16384, 5376}, {"k, v full", 2048, 5376}, {"o full", 5376, 16384},
                             {"gate_up", 43008, 5376}, {"down", 5376, 21504}};
    const RadKernelInfo* kernel = nullptr;
    for (const Shape& shape : shapes)
        for (int M : kHeights) {
            const auto geometry = params(M, shape.N, shape.K);
            const auto* row = match(geometry);
            W4_CHECK(row && !std::strcmp(row->name, "gemm_w4a8_nt_m64") && row->launch && row->layout && row->relayout
                && row->unrelayout && row->opd_shape && (!kernel || kernel == row));
            kernel = row;
            RadOpdDesc desc{};
            W4_CHECK(row->opd_shape(geometry.data(), geometry.size(), 0, &desc) == RAD_OK && desc.dtype == RAD_I8
                && desc.rank == 2 && desc.shape[0] == M && desc.shape[1] == shape.K);
            W4_CHECK(row->opd_shape(geometry.data(), geometry.size(), kCodes, &desc) == RAD_OK && desc.dtype == RAD_I4
                && desc.shape[0] == shape.N && desc.shape[1] == shape.K);
            W4_CHECK(row->opd_shape(geometry.data(), geometry.size(), kScale, &desc) == RAD_OK && desc.dtype == RAD_F16
                && desc.shape[0] == shape.N && desc.shape[1] == shape.K / 128);
        }
    /* whole scale groups of 128 columns, whole tiles of 16 rows, at most 8192 rows */
    W4_CHECK(!match(params(1, 5376, 5376 + 64)) && !match(params(1, 5376 + 8, 5376)) && !match(params(8193, 5376, 5376)));
    W4_CHECK(match(params(8192, 5376, 5376)) == kernel);
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
    for (const Shape& shape : shapes) failures += run(*kernel, stream, shape.what, shape.N, shape.K);
    hip_check(hipStreamDestroy(stream));
    dlclose(library);
    std::printf("gemm_w4a8_nt_m64: %d launch(es) failed\n", failures);
    return failures ? 1 : 0;
}
