#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

int prepare_tensor_rpcmem_mapping(const struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#    include <list>
#    include <unordered_map>

struct RpcMemMapper {
    RpcMemMapper() :
        max_active_map_size{ (size_t) -1 },
        max_active_maps{ 0 },
        active_map_size{ 0 },
        defer_unmap{ false } {}

    RpcMemMapper(size_t max_size, bool defer_unmap_ops, size_t max_maps = 0) :
        max_active_map_size{ max_size },
        max_active_maps{ max_maps },
        active_map_size{ 0 },
        defer_unmap{ defer_unmap_ops } {}

    void                    validate(const struct ggml_tensor * dst);
    std::pair<int, ssize_t> get_tensor_mapping(const struct ggml_tensor *) const;  // returns <mapping fd, offset>

    using UnmapRequest = std::tuple<int, void *, size_t>;

    const std::list<UnmapRequest> & get_pending_unmap_reqs() const { return pending_unmap_reqs; }

    void unmap_all_pending_buffers();

    void dump_state() const; // for debugging

  private:
    int n_map_ops = 0; // for debugging only, remove later

    size_t max_active_map_size;
    // Optional cap on simultaneously-mapped rpcmem buffers. Some devices fail fastrpc_mmap once the mapping
    // count exceeds a small limit (e.g. 16). 0 means "no explicit cap".
    size_t max_active_maps;
    size_t active_map_size;

    bool                    defer_unmap;
    std::list<UnmapRequest> pending_unmap_reqs;

    std::unordered_map<void *, std::pair<int, size_t>>      buf_mapping;
    std::unordered_map<void *, std::list<void *>::iterator> buf_iters;      // for LRU replacement
    std::list<void *>                                       accessed_bufs;  // for LRU replacement
};
#endif
