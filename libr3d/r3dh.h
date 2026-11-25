/* r3dh.h -- the entry points of the gated-delta-net kernels ported from libhalo (libhalo/halo.h's
 * gated-delta-net section, the halo_ -> r3dh_ rename): the chunk preamble, the chunked scan and the
 * recurrent update. libhalo's kernels are libr4d's ported to the gfx11 wave matrix and checked
 * against libref on gfx1151 (rad-kbench); they compile for gfx11-generic unchanged, so libr3d
 * carries them for the whole family. Plain C ABI on raw device pointers and a stream; an entry
 * point returns a negative code for a geometry it was not compiled for. */
#pragma once
#include <hip/hip_runtime.h>

// The paged-attention arguments of libhalo's attention kernels (libhalo/halo.h's HaloArgs, the
// same rename), which the windowed paged kernels ported here (r3dh_attn_paged_window.hip) read.
struct R3dhArgs {
    const void*  q;             // (num_seqs*q_len, q_heads, head_dim)  bf16
    const void*  kv;            // (num_blocks, kv_heads, block_size, 2*head_dim)  fp8 e4m3 or bf16
                                //   strides below are in elements of that dtype; K then V per slot
    const int*   block_table;   // (num_seqs, max_blocks)            int32
    const int*   seqused_k;     // (num_seqs,)                       int32
    void*        out;           // (total_q, q_heads, head_dim)      bf16
    const float* k_descale;     // (num_seqs, kv_heads)
    const float* v_descale;     // (num_seqs, kv_heads)
    const float* q_descale;     // unused: the query is bf16
    void*        scratch;       // split-KV partials (decode only), or null
    int num_seqs, q_len, q_heads, kv_heads, head_dim, block_size, max_blocks;
    long kv_block_stride;       // elements between consecutive blocks
    long kv_head_stride;        // elements between kv heads inside a block
    float scale;
    int  splits;                // decode only; 0 = let the split law choose
    int  max_ctx;               // host-visible context bound (seqused_k is device-side)
    int  window;                // sliding window, 0 = none (a full-attention layer)
    int  causal;                // 1 = key <= query; 0 = key < seq_len, window two-sided
    const void* causal_seq;     // (num_seqs,) uint8 per-sequence flag, or null for `causal`

    // ---- Ragged batches --------------------------------------------------------------------
    // A serving step mixes one sequence's prefill chunk with other sequences' single decode rows,
    // so query lengths differ per sequence.
    //
    // `cu_q` is cu_seqlens: [num_seqs + 1] int32 on the device, cu_q[0] == 0, cu_q[num_seqs] ==
    // total_q; sequence s owns query rows [cu_q[s], cu_q[s+1]) of q and out. Null means uniform:
    // row r of sequence `seq` is row `seq * q_len + r`.
    //
    // With cu_q set, `q_len` is the maximum query length in the batch (it sizes the grid and picks
    // the decode kernel's wave count) and `total_q` is the row count of q and out, host-side so the
    // split-KV scratch can be sized without reading device memory. Both are ignored when cu_q is
    // null.
    const int* cu_q;            // (num_seqs+1,) int32 device, or null for a uniform q_len
    int  total_q;               // rows of q/out; ignored unless cu_q is set

    // ---- Views the split-KV decode kernel reads (r3dh_attn_decode_h256_gqa6.hip); every other
    // kernel leaves them at zero and reads q dense with one cache head per query group.
    //
    // `q_ld` is the element distance between q rows, 0 for dense (q_heads * head_dim): the q part
    // of a fused [q|k|v] projection read where it lies (attn_paged_qrow). A multiple of 4.
    //
    // `kv_rep` maps query groups onto cache heads: group g (q heads g*GQA .. g*GQA+GQA-1) reads
    // cache head g / kv_rep, and the descales are per cache head. kv_heads then counts the groups
    // (q_heads / GQA) and the cache holds kv_heads / kv_rep heads. 0 or 1 is one head a group. It
    // runs a gqa of GQA * kv_rep on the kernel compiled for GQA, whose workgroup holds at most 64
    // rows of q_len * GQA: the 31B's DFlash drafter, gqa 8 at a block of 16, on the gqa-4 kernel.
    long q_ld;
    int  kv_rep;
    // The engine's chained launch (RadKernelChainFn, r3dh_chain.h): the decode path launches its
    // chained forms. fill_paged sets it from the resolved `chain` parameter.
    int  chain;
    // The union form of the split-KV decode kernel (O_UNION, r3dh_attn_decode_h256_gqa6.hip): a
    // "sequence" is a group of query rows, `block_table` its pages' union and `umask` one word a
    // union page, four bits a row of the group: how many of the page's tokens that row attends to.
    const uint32_t* umask;
    // A tree verify (attn_paged_qrow_tree, docs/OPS.md): RadBatch::spec_parent, one int32 a query
    // row, -2 for an ordinary causal row, else the row's parent's index within its sequence's rows
    // (-1 for the block's first). Read only by the O_TREE instances; null elsewhere.
    const int* spec_parent;
};

// Rows of q and out. The split-KV scratch is laid out by dense query row, so the host sizing hook
// and the device kernels must agree on this count; hence total_q is carried rather than read off
// the end of cu_q.
__host__ __device__ __forceinline__ int r3dh_q_total(const R3dhArgs& a) {
    return a.cu_q ? a.total_q : a.num_seqs * a.q_len;
}


extern "C" {

// ---- gated delta net -----------------------------------------------------------------------
// One chunked scan over N variable-length sequences: the WY recompute, the recurrent state scan
// and the output in one kernel (FLA's recompute_w_u_fwd + chunk_gated_delta_rule_fwd_h +
// chunk_fwd_o). Layouts, all contiguous and bf16 unless stated:
//   q,k [T, Hg, K]   v,o [T, H, V]   A [T, H, bt]   g,beta [T, H] fp32
//   h0,ht [N, H, V, K] fp32          cu [N+1] int32
// K, V and bt must be the compile-time 128/128/64 in the name; a mismatch returns -1. Prefill and
// chunked prefill run this; decode runs the recurrent update.
int r3dh_gdn_chunk_scan_k128_v128_c64_bf16(
        const void* q, const void* k, const void* v, const void* A,
        const void* g, const void* beta, const void* h0, void* o, void* ht,
        const void* cu, int N, int H, int Hg, int K, int V, int bt,
        float scale, int state_fp16, void* stream);
// The same scan against a slot-indexed state cache: `ssm_state_indices` [N] names each batch
// row's slot, and the two strides are the cache's, in elements. The call above is this one with a
// null index and packed [N, H, V, K] strides, which is correct only while slot == batch row.
int r3dh_gdn_chunk_scan_k128_v128_c64_bf16_sidx(
        const void* q, const void* k, const void* v, const void* A,
        const void* g, const void* beta, const void* h0, void* o, void* ht,
        const void* cu, const void* ssm_state_indices, long indices_stride,
        long state_slot_stride, long state_head_stride,
        int N, int H, int Hg, int K, int V, int bt,
        float scale, int state_fp16, void* stream);
void r3dh_gdn_dims(int* head_k, int* head_v, int* chunk);

// The chunk preamble: A = (I + strict_lower(diag(beta) K K^T e^{g_i-g_j}))^-1 per chunk, i.e.
// FLA's chunk_scaled_dot_kkt_fwd then solve_tril, with the fp32 gram kept out of HBM.
// k [T,Hg,K] bf16; beta, g [T,H] fp32 (g already summed along the chunk); A [T,H,64] bf16 out;
// cu [N+1] int32. Returns -1 for a geometry it was not compiled for.
int r3dh_gdn_kkt_solve_k128_c64_bf16(const void* k, const void* beta, const void* g, void* A,
                                    const void* cu, int N, int T, int H, int Hg, int K, int bt,
                                    void* stream);

// The recurrent delta-rule update decode runs: gating, the qk l2 norm, the state update and the
// output, against the paged state cache. The state is fp32 (per the model config) and one is
// written per candidate token, which dominates the cost. Replaces
// fused_sigmoid_gating_delta_rule_update.
int r3dh_gdn_recurrent_update_k128_v128_bf16_fp32state(
        const void* q, const void* k, const void* v, const void* a, const void* b,
        long ab_stride, int ab_is_bf16, const void* A_log, const void* dt_bias, void* state,
        long state_slot_stride, long state_head_stride, void* o, void* oq, void* os,
        long oq_pitch, long os_ld, const void* cu,
        const void* ssm_state_indices, long indices_stride, const void* num_accepted,
        const void* z_gate, long z_pitch, const void* norm_weight, float norm_eps, int norm_act,
        int N, int H, int Hg, int K, int V, float scale, float softplus_thr, int state_fp16,
        void* stream);

}  // extern "C"
