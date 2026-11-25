/* r3d_chain.h -- the chained forms of libr3d's own kernels (RadKernelChainFn, abi/rad_abi.h).
 *
 * The wait, the completion and the tag types are r3dh_chain.h's (libhalo's, renamed): a kernel
 * whose template argument is `r4dchain` carries the name the backend gives a chain record, and
 * its blocks wait for the queue's counter before their first read and add to it after their last
 * write. This header adds one thing, the bracket as a scope:
 *
 *   template <..., class CH = HaloUnchained> __global__ void k(...) { R3D_CHAIN_SCOPE(CH); ... }
 *
 * The guard's constructor waits and its destructor adds, so every return of the body, an early
 * one included, passes the add exactly once a block. That holds only where a return is the whole
 * block's or comes after the body's last barrier: the add is itself a barrier, and a wave that
 * left early would meet a barrier other waves of its block reach inside the body. Each kernel
 * that takes the guard is checked for that where it takes it.
 *
 * The launch side is r3dh_chain.h's too: a row's entry sets the thread's flag for its launches
 * (HaloChainScope over r3dh_chain_small, decode heights only) and R3D_CHAIN_LAUNCH picks the form.
 */
#pragma once
#include "r3d_launch.h"
#include "r3dh_chain.h"

template <class CH>
struct R3dChainGuard {
    __device__ __forceinline__ R3dChainGuard() {
        if constexpr (CH::on) r3dh_chain_wait();
    }
    __device__ __forceinline__ ~R3dChainGuard() {
        if constexpr (CH::on) r3dh_chain_done();
    }
};
#define R3D_CHAIN_SCOPE(CH) R3dChainGuard<CH> r3d_chain_guard_

/* r3dh_chain_small with the row count stated: where the first operand's leading extent is not the
 * step's rows (a routed kernel's [rows * top_k, ...] slots). */
inline bool r3d_chain_rows(const RadArgs* a, long long rows, long long rows_max = 64) {
    long long on = 0;
    return a && rad_args_geti(a, "chain", &on) && on && rows <= rows_max;
}

/* The chained form of a kernel template whose last argument is the tag, where the launching
 * entry's flag is set. KERNEL is the template's name and ARGS its other arguments in parentheses
 * (empty for a template of the tag alone); the rest is R3D_KLAUNCH's. */
#define R3D_CHAIN_KARGS_(...) __VA_ARGS__
#define R3D_CHAIN_LAUNCH(KERNEL, TARGS, grid, block, stream, ...)                                \
    do {                                                                                          \
        if (r3dh_chain_flag())                                                                    \
            R3D_KLAUNCH(R3D_KNAME(KERNEL<R3D_CHAIN_KARGS_ TARGS r4dchain>), grid, block, stream,  \
                        __VA_ARGS__);                                                             \
        else                                                                                      \
            R3D_KLAUNCH(R3D_KNAME(KERNEL<R3D_CHAIN_KARGS_ TARGS HaloUnchained>), grid, block,     \
                        stream, __VA_ARGS__);                                                     \
    } while (0)
