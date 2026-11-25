#include "r3d_moe_w4a8_fixture.h"
#include "../core/rad_internal.h"
#include <hip/hip_runtime_api.h>
#include <cstring>
#include <dlfcn.h>
#include <string>

namespace {
using namespace w4a8_fixture;
constexpr uint16_t sentinel = 0x7f95;
void hip_check(hipError_t status) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "HIP: %s\n", hipGetErrorString(status));
        std::exit(1);
    }
}
struct Buffers {
    std::vector<void*> owned;
    void* upload(const void* data, size_t bytes) {
        void* device = nullptr;
        hip_check(hipMalloc(&device, bytes));
        owned.push_back(device);
        hip_check(hipMemcpy(device, data, bytes, hipMemcpyHostToDevice));
        return device;
    }
    template<class Value> void* upload(const std::vector<Value>& values) {
        return upload(values.data(), values.size() * sizeof(Value));
    }
    ~Buffers() { for (void* data : owned) hip_check(hipFree(data)); }
};
RadTensor matrix(void* data, uint32_t dtype, int rows, int columns, int pitch) {
    RadTensor result{};
    result.data = data;
    result.dtype = dtype;
    result.rank = 2;
    result.shape[0] = rows;
    result.shape[1] = columns;
    result.stride[0] = pitch;
    result.stride[1] = 1;
    return result;
}
RadTensor vector_tensor(void* data, int count) {
    RadTensor result{};
    result.data = data;
    result.dtype = RAD_I32;
    result.rank = 1;
    result.shape[0] = count;
    result.stride[0] = 1;
    return result;
}
/* One parity's experts as a weight table: a device array of one pointer an expert (pitch 0), or
 * one plane with the experts `pitch` elements apart. */
RadTensor table(void* data, uint32_t dtype, int rows, int columns, int64_t pitch) {
    RadTensor result{};
    result.data = data;
    result.dtype = dtype;
    result.rank = 3;
    result.shape[0] = experts / 2;
    result.shape[1] = rows;
    result.shape[2] = result.stride[1] = columns;
    result.stride[0] = pitch;
    result.stride[2] = 1;
    return result;
}
/* The experts on the device, as the four weight operands: the even experts' codes and the odd
 * experts' scales behind pointers, the other two stacked. */
struct DeviceExperts {
    Buffers buffers;
    RadTensor operand[4];
    explicit DeviceExperts(const Experts& weights) {
        const int outputs = weights.shape.outputs, width = weights.shape.width;
        for (int parity = 0; parity < 2; ++parity) {
            std::vector<void*> code_pointers, scale_pointers;
            std::vector<uint8_t> code_plane;
            std::vector<uint16_t> scale_plane;
            for (int expert = parity; expert < experts; expert += 2) {
                const uint8_t* codes = weights.codes.data() + expert * weights.code_bytes();
                const uint16_t* scales = weights.scales.data() + expert * weights.scale_count();
                if (parity) {
                    code_plane.insert(code_plane.end(), codes, codes + weights.code_bytes());
                    scale_pointers.push_back(buffers.upload(scales, weights.scale_count() * sizeof(uint16_t)));
                } else {
                    code_pointers.push_back(buffers.upload(codes, weights.code_bytes()));
                    scale_plane.insert(scale_plane.end(), scales, scales + weights.scale_count());
                }
            }
            operand[2 * parity] = parity
                ? table(buffers.upload(code_plane), RAD_I4, outputs, width, int64_t(outputs) * width)
                : table(buffers.upload(code_pointers), RAD_I4, outputs, width, 0);
            operand[2 * parity + 1] = parity
                ? table(buffers.upload(scale_pointers), RAD_BF16, outputs, width / weights.group, 0)
                : table(buffers.upload(scale_plane), RAD_BF16, outputs, width / weights.group,
                        int64_t(weights.scale_count()));
        }
    }
};
/* The activation at `pitch` elements a row, the bytes between the rows a filler. */
std::vector<uint8_t> spread(const std::vector<uint8_t>& codes, int rows, int width, int pitch) {
    std::vector<uint8_t> result(size_t(rows) * pitch, 0x2a);
    for (int row = 0; row < rows; ++row)
        std::copy_n(codes.begin() + size_t(row) * width, width, result.begin() + size_t(row) * pitch);
    return result;
}
/* Launches the case and returns y with its row pitch, every byte the kernel left as it was a
 * sentinel. */
std::vector<uint16_t> launch(const RadKernelInfo& kernel, const Case& c, const DeviceExperts& weights,
                             const Input& input, const Route& route, int input_pitch, int output_pitch,
                             hipStream_t stream) {
    const int width = c.shape.width, blocks = width / 128, scale_pitch = blocks + 3;
    std::vector<float> factors(size_t(c.input_rows()) * scale_pitch, -1.0f);
    for (int row = 0; row < c.input_rows(); ++row)
        std::copy_n(input.scales.begin() + size_t(row) * blocks, blocks, factors.begin() + size_t(row) * scale_pitch);
    std::vector<uint16_t> actual(size_t(c.rows()) * output_pitch, sentinel);
    Buffers buffers;
    RadTensor tensors[9]{};
    tensors[0] = matrix(buffers.upload(spread(input.codes, c.input_rows(), width, input_pitch)),
                        c.e4m3 ? RAD_F8E4M3 : RAD_I8, c.input_rows(), width, input_pitch);
    tensors[1] = matrix(buffers.upload(factors), RAD_F32, c.input_rows(), blocks, scale_pitch);
    tensors[2] = weights.operand[0];
    tensors[3] = weights.operand[1];
    tensors[4] = vector_tensor(buffers.upload(route.slots), c.rows());
    tensors[5] = vector_tensor(buffers.upload(route.offsets), experts + 1);
    tensors[6] = weights.operand[2];
    tensors[7] = weights.operand[3];
    tensors[8] = matrix(buffers.upload(actual), RAD_BF16, c.rows(), c.shape.outputs, output_pitch);
    const auto geometry = params(c);
    RadArgs args{};
    args.t = tensors;
    args.n_t = 9;
    args.p = geometry.data();
    args.n_p = geometry.size();
    W4A8_CHECK(kernel.launch(&args, stream) == RAD_OK);
    hip_check(hipStreamSynchronize(stream));
    hip_check(hipMemcpy(actual.data(), tensors[8].data, actual.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
    return actual;
}
void compare(const RadKernelInfo& kernel, const Case& c, const Experts& weights, const DeviceExperts& device,
             hipStream_t stream) {
    Random random{0x9e3779b9u + uint32_t(c.shape.width) * 7 + uint32_t(c.tokens) * 131 + uint32_t(c.top_k) * 17
                  + uint32_t(c.routing) * 3 + uint32_t(c.e4m3)};
    const Route route(c, random);
    const Input input(c, random);
    const Expected expected = reference(c, weights, input, route);
    const int outputs = c.shape.outputs, pitch = outputs + 6, width = c.shape.width;
    const auto actual = launch(kernel, c, device, input, route, width, pitch, stream);
    char name[128];
    std::snprintf(name, sizeof(name), "%s %s group %d, %d tokens, top_k %d, %s", c.shape.name,
                  c.e4m3 ? "E4M3" : "int8", c.group, c.tokens, c.top_k, routing_name[int(c.routing)]);
    size_t outside = 0, moved = 0;
    double error = 0, norm = 0;
    for (int row = 0; row < c.rows(); ++row) {
        for (int column = 0; column < pitch; ++column) {
            const uint16_t got = actual[size_t(row) * pitch + column];
            if (column >= outputs) {
                W4A8_CHECK(got == sentinel);
                continue;
            }
            const size_t index = size_t(row) * outputs + column;
            const float value = rad_bf16_to_f32(got), want = rad_bf16_to_f32(expected.value[index]);
            const bool inside = c.e4m3
                ? value >= rad_bf16_to_f32(expected.low[index]) && value <= rad_bf16_to_f32(expected.high[index])
                : got == expected.value[index];
            if (!inside && !outside++)
                std::fprintf(stderr, "%s: sorted row %d col %d: got %04x, expected %04x (%04x to %04x)\n", name, row,
                             column, got, expected.value[index], expected.low[index], expected.high[index]);
            moved += got != expected.value[index];
            error += (double(value) - want) * (double(value) - want);
            norm += double(want) * want;
        }
    }
    const double rel_l2 = std::sqrt(error / norm);
    std::printf("%s: %zu of %zu outputs differ from the reference, rel_l2 %.2e\n", name, moved,
                expected.value.size(), rel_l2);
    if (outside) {
        std::fprintf(stderr, "%s: %zu outputs outside the tolerance\n", name, outside);
        std::exit(1);
    }
    if (!c.e4m3) return;
    /* An E4M3 activation whose rows are not 16 bytes apart is read by another kernel above 32
     * sorted rows (the 16-row block, not the prefill tile), which must store the same bytes. */
    if (launch(kernel, c, device, input, route, width + 8, pitch, stream) != actual) {
        std::fprintf(stderr, "%s: the bytes change with the activation's row pitch\n", name);
        std::exit(1);
    }
}
}

int main(int argc, char** argv) {
    const bool catalogue = argc == 3 && !std::strcmp(argv[2], "--catalogue-only");
    W4A8_CHECK(argc == 2 || catalogue);
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) { std::fprintf(stderr, "%s\n", dlerror()); return 1; }
    const auto row_count = reinterpret_cast<decltype(&rad_kernel_count)>(dlsym(library, "rad_kernel_count"));
    const auto row_at = reinterpret_cast<decltype(&rad_kernel_at)>(dlsym(library, "rad_kernel_at"));
    W4A8_CHECK(row_count && row_at);
    auto match = [&](const Params& params) {
        rad::Geometry geometry;
        for (const auto& param : params) {
            if (param.kind == RAD_P_INT) geometry.set_i(param.key, param.ival);
            else geometry.set_s(param.key, param.sval);
        }
        const RadKernelInfo* row = nullptr;
        for (int index = 0; index < row_count(); ++index) {
            const auto* candidate = row_at(index);
            if (candidate->domain != RAD_DOMAIN_DEVICE || std::strcmp(candidate->op, "moe_gemm_q")) continue;
            if (std::all_of(candidate->constraints, candidate->constraints + candidate->n_constraints,
                    [&](const RadConstraint& constraint) { return rad::constraint_holds(constraint, geometry); })) {
                W4A8_CHECK(!row);
                row = candidate;
            }
        }
        return row;
    };
    /* The heights: one to four tokens are at most 32 sorted rows at top_k 8 (a decode step and a
     * short verify pass), five are the first above; 32 and 33 tokens at top_k 1 are the same
     * boundary in rows; 2048 tokens are a prefill chunk. */
    std::vector<Case> cases;
    for (const Shape& shape : {gate_up, down, other})
        for (int group : {128, 32})
            for (bool e4m3 : {true, false})
                for (Routing routing : {Routing::uniform, Routing::skewed, Routing::sparse}) {
                    for (int tokens : {1, 2, 4, 5, 32, 33, 64, 256, 2048})
                        if (tokens <= 5 || shape.width != other.width)
                            cases.push_back({shape, group, e4m3, tokens, 8, routing});
                    for (int tokens : {32, 33})
                        if (shape.width != other.width) cases.push_back({shape, group, e4m3, tokens, 1, routing});
                }
    const RadKernelInfo* row = nullptr;
    for (const Case& c : cases) {
        const auto* matched = match(params(c));
        W4A8_CHECK(matched && !std::strcmp(matched->name, "moe_gemm_w4a8") && matched->launch);
        W4A8_CHECK(!row || row == matched);
        row = matched;
    }
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
    for (size_t first = 0; first < cases.size();) {
        /* The cases of one shape and group share their experts. */
        const Experts weights(cases[first].shape, cases[first].group);
        const DeviceExperts device(weights);
        size_t last = first;
        for (; last < cases.size() && cases[last].shape.width == weights.shape.width
               && cases[last].group == weights.group; ++last)
            compare(*row, cases[last], weights, device, stream);
        first = last;
    }
    hip_check(hipStreamDestroy(stream));
    dlclose(library);
}
