#include "rpcmem_mapper.h"

#include <algorithm>
#include <atomic>
#include <unordered_set>
#include <unistd.h>
#include <vector>

#include "dsprpc_interface.h"
#include "ggml-backend-impl.h"
#include "ggml-htp-impl.h"
#include "ggml-htp.h"
#include "message.h"

namespace {

template <typename T> void write_buf(uint8_t *& p, const T & v) {
    *reinterpret_cast<T *>(p) = v;
    p += sizeof(v);
}

// DSP side caches HAP_mmap_get(fd) results. If we want to fastrpc_munmap a buffer
// (e.g. to stay under a hard mapping-count cap), we must first ask DSP to release
// the mapping via REQUEST_TYPE_RPCMEM_MAP (HAP_mmap_put). Otherwise munmap can
// fail with AEE_EBADSTATE (often seen as return code 1).
int issue_dsp_put_maps(const int32_t * fds, int32_t n_fds) {
    if (n_fds <= 0) {
        return 0;
    }

    auto * ctx = ggml_backend_htp_context::instance();
    if (!ctx || !ctx->ops_backend_initialized || !ctx->ops_msg_chan) {
        return -1;
    }

    auto * msg_hdr = reinterpret_cast<MessageHeader *>(ctx->ops_msg_chan);

    // Reset state (same pattern as htp_ops_issue_request).
    auto * d_ptr = reinterpret_cast<volatile std::atomic<uint64_t> *>(&(msg_hdr->state.d));
    std::atomic_store(d_ptr, static_cast<uint64_t>(0));

    msg_hdr->n_reqs         = 1;
    msg_hdr->req_offsets[0] = message_header_size(msg_hdr);

    const size_t map_req_size = sizeof(RequestHeader) + sizeof(RpcmemMapRequest) + (size_t) n_fds * sizeof(int32_t);
    msg_hdr->req_offsets[1]   = msg_hdr->req_offsets[0] + map_req_size;

    {
        RequestHeader req_hdr{
            .state = 0,
            .type  = REQUEST_TYPE_RPCMEM_MAP,
        };
        RpcmemMapRequest map_req{
            .n_puts = n_fds,
            .n_gets = 0,
        };

        auto * p = reinterpret_cast<uint8_t *>(message_header_get_request_ptr(msg_hdr, 0));
        write_buf(p, req_hdr);
        write_buf(p, map_req);
        for (int i = 0; i < n_fds; ++i) {
            write_buf(p, fds[i]);
        }
    }

    // Compute checksum (DSP side check is currently disabled, but keep consistent).
    {
        uint32_t   sum   = 0;
        uint32_t * begin = ((uint32_t *) msg_hdr) + 3;  // skip state & checksum
        uint32_t * end   = ((uint32_t *) msg_hdr) + ggml_backend_htp_context::MAX_MSG_SIZE / 4;

        for (auto * p = begin; p < end; ++p) {
            sum += *p;
        }
        sum += 0x00000001 + 0x00000000;  // value of `state`
        msg_hdr->checksum = -sum;
    }

    auto * v0_ptr = reinterpret_cast<volatile std::atomic<uint8_t> *>(&(msg_hdr->state.v[0]));
    auto * v1_ptr = reinterpret_cast<volatile std::atomic<uint8_t> *>(&(msg_hdr->state.v[1]));

    std::atomic_store_explicit(v0_ptr, static_cast<uint8_t>(1), std::memory_order_release);

    while (std::atomic_load_explicit(v1_ptr, std::memory_order_acquire) == 0) {
        usleep(1);
    }

    d_ptr->store(0, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);

    return message_header_get_request_ptr(msg_hdr, 0)->state;
}

}  // namespace

void RpcMemMapper::validate(const ggml_tensor * dst) {
    // Collect unique rpcmem buffers required for this op (src + dst). Duplicate buffers can appear
    // through multiple src slots; dedup avoids over-estimating required mapping size.
    std::vector<ggml_backend_buffer *> buffers;
    std::unordered_set<void *>         required_buf_bases;

    auto add_buffer = [&](ggml_backend_buffer * buf) {
        if (ggml_backend_buft_is_rpcmem(buf->buft)) {
            void * base = ggml_backend_buffer_get_base(buf);
            if (required_buf_bases.insert(base).second) {
                buffers.push_back(buf);
            }
        }
    };

    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        auto * src = dst->src[i];
        if (src) {
            add_buffer(src->buffer);
        }
    }
    add_buffer(dst->buffer);

    const auto is_required = [&](void * base) { return required_buf_bases.count(base) != 0; };

    auto try_unmap_lru = [&](bool allow_defer_unmap) -> bool {
        // Pick the least-recently-used buffer that is not required for this op.
        for (auto it = accessed_bufs.rbegin(); it != accessed_bufs.rend(); ++it) {
            void * buf_base = *it;
            if (is_required(buf_base)) {
                continue;
            }
            auto [fd, buf_size] = buf_mapping.at(buf_base);

            if (allow_defer_unmap) {
                // We assume slightly exceeding the planned max_active_map_size is acceptable.
                pending_unmap_reqs.emplace_back(fd, buf_base, buf_size);
            } else {
                // Ensure DSP releases its cached HAP_mmap_get(fd) before host munmap.
                const int put_ret = issue_dsp_put_maps(reinterpret_cast<const int32_t *>(&fd), 1);
                if (put_ret) {
                    fprintf(stderr, "dsp_put_map(fd=%d) failed with return code: %x\n", fd, put_ret);
                }

                int err = fastrpc_munmap(CDSP_DOMAIN_ID, fd, buf_base, buf_size);
                if (err) {
                    // Keep the mapping record; try a different LRU buffer.
                    fprintf(stderr, "fastrpc_munmap failed with return code: %x\n", err);
                    ++n_map_ops;
                    continue;
                }
                ++n_map_ops;
            }

            // Remove bookkeeping only after successful munmap (or after queuing a deferred request).
            auto lru_iter = buf_iters.at(buf_base);
            accessed_bufs.erase(lru_iter);
            buf_iters.erase(buf_base);
            buf_mapping.erase(buf_base);
            active_map_size -= buf_size;
            return true;
        }
        return false;
    };

    size_t required_size = 0;
    size_t required_new_maps = 0;
    for (auto * buf : buffers) {
        void * buf_base = ggml_backend_buffer_get_base(buf);
        if (buf_mapping.count(buf_base)) {
            // buffer already mapped, we put it at the front of the LRU access list
            accessed_bufs.erase(buf_iters.at(buf_base));
            accessed_bufs.push_front(buf_base);
            buf_iters[buf_base] = accessed_bufs.begin();
        } else {
            required_size += buf->size;
            ++required_new_maps;
        }
    }

    GGML_ASSERT(required_size <= max_active_map_size);

    // Some devices fail fastrpc_mmap when the mapping count exceeds a small limit (e.g. 16),
    // even if total active mapping size is within budget. Enforce an optional mapping-count cap.
    if (max_active_maps > 0) {
        while (buf_mapping.size() + required_new_maps > max_active_maps) {
            // For mapping count pressure we must munmap immediately to actually release DSP mappings.
            if (!try_unmap_lru(false /*allow_defer_unmap*/)) {
                GGML_ABORT("rpcmem_mapper: cannot unmap any buffer to satisfy max_active_maps=%zu\n", max_active_maps);
            }
        }
    }

    // If a mapping-count cap is active, disable deferred unmap for size pressure as well; otherwise
    // we can remove bookkeeping without actually releasing DSP mappings and still hit the cap.
    const bool allow_defer_unmap = defer_unmap && (max_active_maps == 0);
    while (active_map_size + required_size > max_active_map_size) {
        if (!try_unmap_lru(allow_defer_unmap)) {
            GGML_ABORT("rpcmem_mapper: cannot unmap any buffer to satisfy max_active_map_size\n");
        }
    }

    for (auto * buf : buffers) {
        void * buf_base = ggml_backend_buffer_get_base(buf);
        size_t buf_size = buf->size;
        if (buf_mapping.count(buf_base)) {
            continue;
        }

        int fd = rpcmem_to_fd(buf_base);
        if (fd < 0) {
            GGML_ABORT("rpcmem_to_fd returns %d, ptr %p, for dst tensor %s, n_ops %d\n", fd, buf_base, dst->name,
                       n_map_ops);
        }

        auto it = std::find_if(pending_unmap_reqs.begin(), pending_unmap_reqs.end(),
                               [fd](const auto & v) { return std::get<0>(v) == fd; });
        if (it == pending_unmap_reqs.end()) {
            int err = 0;
            int retry = 0;
            // NOTE: some devices keep a hard cap on mapping count; if mmap fails, try unmapping one more
            // LRU buffer and retry a few times before aborting.
            while (true) {
                err = fastrpc_mmap(CDSP_DOMAIN_ID, fd, buf_base, 0, buf_size, FASTRPC_MAP_FD);
                if (!err) {
                    break;
                }
                if (++retry > 3) {
                    break;
                }
                // Force an immediate unmap of one extra buffer (best-effort) and retry.
                (void) try_unmap_lru(false /*allow_defer_unmap*/);
                // fastrpc_munmap does not release DSP vm mapping immediately; a tiny delay helps on some builds.
                usleep(1000);
            }
            if (err) {
                dump_state();

                for (int i = 0; i < GGML_MAX_SRC; ++i) {
                    auto src = dst->src[i];
                    if (!src) {
                        continue;
                    }
                    fprintf(stderr, "  src index %d name %s", i, src->name);
                    if (ggml_backend_buft_is_rpcmem(src->buffer->buft)) {
                        auto base = ggml_backend_buffer_get_base(src->buffer);
                        fprintf(stderr, " buf %p\n", base);
                    } else {
                        fprintf(stderr, " (non rpcmem)\n");
                    }
                }

                fprintf(stderr, "dst name: %s, op: %s, n buffers to map in this step: %ld, active size: %.2f MiB\n",
                        dst->name, ggml_op_name(dst->op), buffers.size(), active_map_size / 1048576.0);

                /*
                if (++retry < 3) {
                    // try: unmap all other buffers
                    std::unordered_set<void *> buf_ptrs;
                    for (auto b : buffers) {
                        buf_ptrs.insert(ggml_backend_buffer_get_base(b));
                    }

                    for (auto it = buf_mapping.begin(); it != buf_mapping.end();) {
                        auto ptr = it->first;
                        if (buf_ptrs.count(ptr)) {
                            ++it;
                            continue;  // skip needed buffers
                        }
                        auto [fd, len] = it->second;
                        auto lru_iter  = buf_iters.at(ptr);
                        accessed_bufs.erase(lru_iter);
                        buf_iters.erase(ptr);
                        it = buf_mapping.erase(it);

                        int e = fastrpc_munmap(CDSP_DOMAIN_ID, fd, ptr, len);
                        fprintf(stderr, "try unmap fd %d addr %p len %ld ret %d\n", fd, ptr, len, e);

                        rpcmem_free(ptr);
                        fprintf(stderr, "danger operation: free %p\n", ptr);
                    }

                    goto again;
                }*/

                GGML_ABORT(
                    "fastrpc_mmap failed with return code: 0x%x fd: %d buf_base: %p buf_size: %ld buf usage: %d\n", err,
                    fd, buf_base, buf_size, buf->usage);
            }

            ++n_map_ops;
        } else {
            pending_unmap_reqs.erase(it);
        }

        // fprintf(stderr, "rpcmem_mapper: creating memory mapping for rpcmem buffer %p, size %.2f MiB, fd %d\n", buf_base,
        //         buf_size / 1048576.0, fd);

        accessed_bufs.push_front(buf_base);
        buf_iters[buf_base]   = accessed_bufs.begin();
        buf_mapping[buf_base] = { fd, buf_size };
        active_map_size += buf_size;
    }
}

std::pair<int, ssize_t> RpcMemMapper::get_tensor_mapping(const ggml_tensor * tensor) const {
    GGML_ASSERT(ggml_backend_buft_is_rpcmem(tensor->buffer->buft));

    void * buf_base = ggml_backend_buffer_get_base(tensor->buffer);
    auto [fd, _]    = buf_mapping.at(buf_base);
    auto offset     = (intptr_t) tensor->data - (intptr_t) buf_base;
    return { fd, offset };
}

void RpcMemMapper::unmap_all_pending_buffers() {
    for (auto it = pending_unmap_reqs.begin(); it != pending_unmap_reqs.end();) {
        auto [fd, buf_base, buf_size] = *it;

        // fprintf(stderr, "rpcmem_mapper: removing memory mapping for rpcmem buffer %p, size %.2f MiB, fd %d\n", buf_base,
        //         buf_size / 1048576.0, fd);
        int err = fastrpc_munmap(CDSP_DOMAIN_ID, fd, buf_base, buf_size);
        if (err) {
            fprintf(stderr, "fastrpc_munmap failed with return code: 0x%x\n", err);
        }
        ++n_map_ops;
        it = pending_unmap_reqs.erase(it);
    }
}

void RpcMemMapper::dump_state() const {
    fprintf(stderr, "total %d fastrpc_mmap + fastrpc_munmap ops\n", n_map_ops);

    if (!buf_mapping.empty()) {
        fprintf(stderr, "active mappings:\n");
    }
    for (const auto & [addr, pair] : buf_mapping) {
        fprintf(stderr, "    addr %p -> fd %d, size %.2f MiB\n", addr, pair.first, pair.second / 1048576.0);
    }

    if (!pending_unmap_reqs.empty()) {
        fprintf(stderr, "pending unmap requests:\n");
    }
    for (const auto & [fd, addr, size] : pending_unmap_reqs) {
        fprintf(stderr, "    fd %d, addr %p, size %.2f MiB\n", fd, addr, size / 1048576.0);
    }
}

extern "C" {

int prepare_tensor_rpcmem_mapping(const struct ggml_tensor * dst) {
    ggml_backend_htp_context::instance()->mapper.validate(dst);
    return 0;
}
}
