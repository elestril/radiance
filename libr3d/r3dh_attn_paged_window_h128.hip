/* r3dh_attn_paged_window_h128.hip -- libr3d: libhalo's window attention at Mistral Nemo's geometry
 * (head_dim 128, 4 queries per KV head, bf16 cache): attn_window_h128_gqa4_bf16kv, the WMMA kernel
 * at prefill and the split-KV scalar one at decode and verify. The unit is r3dh_attn_paged_window.hip
 * itself, included without its own entry points, so these kernels sit in a code object of their own
 * (see the note at that unit's entry points). */
#define R3DH_WIN_H128_UNIT 1
#include "r3dh_attn_paged_window.hip"

extern "C" {

int r3dh_rad_attn_window_h128_gqa4_bf16(const RadArgs* a, RadStream s) {
    return r3dh_win::launch_attn<128, 4, 0>(a, s);
}
int64_t r3dh_rad_attn_window_h128_scratch(const RadArgs*) { return r3dh_win::scratch_need<128>(); }

}  /* extern "C" */
