#define CL_TARGET_OPENCL_VERSION GGML_OPENCL_TARGET_VERSION
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

// suppress warnings in CL headers for GCC and Clang
#pragma GCC diagnostic ignored "-Woverlength-strings"
#ifdef __clang__
#pragma GCC diagnostic ignored "-Wgnu-anonymous-struct"
#endif

#include "ggml-opencl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml.h"
#include "ggml-cpu/ggml-cpu-impl.h"
#include "ggml-cpu/ops.h"

#include <CL/cl.h>

#include <inttypes.h>
#include <string.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <vector>
#include <string>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <map>
#include <memory>
#include <cstdlib>
#include <charconv>
#include <mutex>
#include <atomic>
#include <chrono>

#include "mldrift_attn_h24_m4352_k128.h"
#include "mldrift_attn_h30_m4224_k128.h"

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

#define UNUSED(x) (void)(x)

#define CL_CHECK(err)                                               \
    do {                                                            \
        cl_int err_ = (err);                                        \
        if (err_ != CL_SUCCESS) {                                   \
            GGML_LOG_ERROR("ggml_opencl: %s error %d at %s:%d\n",  \
                #err, err_, __FILE__, __LINE__);                    \
            GGML_ASSERT(0);                                         \
        }                                                           \
    } while (0)

//------------------------------------------------------------------------------
// OpenCL
//------------------------------------------------------------------------------

bool ggml_cl_compute_forward(ggml_backend_t backend, struct ggml_tensor * tensor);

// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
struct fastdiv_vals {
    uint32_t mp;
    uint32_t L;
    uint32_t d;
    uint32_t pad;
};
static_assert(sizeof(fastdiv_vals) == 16, "fastdiv_vals size incorrect");

static fastdiv_vals init_fastdiv_values(uint64_t d_64) {
    GGML_ASSERT(d_64 != 0);
    GGML_ASSERT(d_64 <= std::numeric_limits<uint32_t>::max());

    uint32_t d = (uint32_t)d_64;

    // compute L = ceil(log2(d));
    uint32_t L = 0;
    while (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    // pack divisor as well to reduce error surface
    return { mp, L, d, 0 };
}

struct ggml_backend_opencl_context;

static bool ggml_cl_dump_buffer_region(struct ggml_backend_opencl_context * backend_ctx,
                                       cl_mem buffer,
                                       cl_ulong offset,
                                       size_t bytes,
                                       const std::string & path);

enum GPU_FAMILY {
    ADRENO,
    INTEL,
    UNKNOWN,
};

enum ADRENO_GPU_GEN {
    ADRENO_UNKNOWN,
    A7X,
    A8X,
    X1E,
};

enum ADRENO_CL_COMPILER_TYPE {
    E031,
    DX,
};

struct ggml_cl_version {
    cl_uint major = 0;
    cl_uint minor = 0;
};


struct ggml_cl_compiler_version {
    ADRENO_CL_COMPILER_TYPE type;
    int major = -1;
    int minor = -1;
    int patch = -1;

    bool same(ADRENO_CL_COMPILER_TYPE t, int x, int y, int z) const {
        return major == x && minor == y && patch == z && type == t;
    }
    bool newer_than(ADRENO_CL_COMPILER_TYPE t, int x, int y, int z) const {
        return major*10000 + minor*100 + patch > x*10000 + y*100 + z && type == t;
    }
    bool newer_than_or_same(ADRENO_CL_COMPILER_TYPE t, int x, int y, int z) const {
        return same(t, x, y, z) || newer_than(t, x, y, z);
    }
};

static bool ggml_opencl_log_tensor_offsets_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        enabled = std::getenv("SD_OCL_LOG_TENSOR_OFFSETS") != nullptr ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_opencl_log_tensor_dispatch_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        enabled = std::getenv("SD_OCL_LOG_TENSOR_DISPATCH") != nullptr ? 1 : 0;
    }
    return enabled == 1;
}

static const std::vector<std::string>& ggml_opencl_log_tensor_filters() {
    static std::vector<std::string> filters = []() {
        std::vector<std::string> out;
        const char * env = std::getenv("SD_OCL_LOG_TENSOR_FILTER");
        if (env == nullptr || env[0] == '\0') {
            return out;
        }
        std::string s(env);
        for (char & c : s) {
            if (c == ';' || c == ' ') {
                c = ',';
            }
        }
        size_t start = 0;
        while (start < s.size()) {
            size_t end = s.find(',', start);
            if (end == std::string::npos) {
                end = s.size();
            }
            if (end > start) {
                out.emplace_back(s.substr(start, end - start));
            }
            start = end + 1;
        }
        return out;
    }();
    return filters;
}

static bool ggml_opencl_tensor_name_matches(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return false;
    }
    const char * name = tensor->name;
    if (name == nullptr || name[0] == '\0') {
        return false;
    }
    const auto & filters = ggml_opencl_log_tensor_filters();
    if (filters.empty()) {
        return true;
    }
    for (const auto & filter : filters) {
        if (filter.empty()) {
            continue;
        }
        if (std::strstr(name, filter.c_str()) != nullptr) {
            return true;
        }
    }
    return false;
}

//------------------------------------------------------------------------------
// Tensor extra management
//------------------------------------------------------------------------------
struct ggml_tensor_extra_cl {
    // The buffer object that holds the data.
    cl_mem data_device;
    // The offset into the buffer object. This is primarily for scratch buffer
    // and view operation.
    // NB: this offset no longer includes view offset (view_offs). Whenever this
    // offset is used, view_offs should be considered.
    cl_ulong offset;
    // The actual size of the cl_mem object. This is needed when returning the
    // block to the pool.
    size_t actual_size;

    void reset() {
        data_device = nullptr;
        offset = 0;
        actual_size = 0;
    }
};

static void ggml_opencl_log_tensor_layout(const char * tag,
                                          const ggml_tensor * dst,
                                          const ggml_tensor * src0,
                                          const ggml_tensor * src1) {
    if (!ggml_opencl_log_tensor_dispatch_enabled()) {
        return;
    }
    if (!ggml_opencl_tensor_name_matches(dst)) {
        return;
    }
    const char * dst_name = (dst && dst->name[0] != '\0') ? dst->name : "";
    const char * src0_name = (src0 && src0->name[0] != '\0') ? src0->name : "";
    const char * src1_name = (src1 && src1->name[0] != '\0') ? src1->name : "";
    size_t dst_off = 0;
    size_t src0_off = 0;
    size_t src1_off = 0;
    const void * dst_buf = nullptr;
    const void * src0_buf = nullptr;
    const void * src1_buf = nullptr;
    if (dst && dst->extra) {
        const ggml_tensor_extra_cl * extra = (const ggml_tensor_extra_cl *) dst->extra;
        dst_off = (size_t) extra->offset + (size_t) dst->view_offs;
        dst_buf = (const void *) extra->data_device;
    }
    if (src0 && src0->extra) {
        const ggml_tensor_extra_cl * extra = (const ggml_tensor_extra_cl *) src0->extra;
        src0_off = (size_t) extra->offset + (size_t) src0->view_offs;
        src0_buf = (const void *) extra->data_device;
    }
    if (src1 && src1->extra) {
        const ggml_tensor_extra_cl * extra = (const ggml_tensor_extra_cl *) src1->extra;
        src1_off = (size_t) extra->offset + (size_t) src1->view_offs;
        src1_buf = (const void *) extra->data_device;
    }
    GGML_LOG_INFO("opencl: dispatch %s dst='%s' op=%s type=%s ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] nb=[%zu,%zu,%zu,%zu] off=%zu buf=%p src0='%s' off=%zu buf=%p src1='%s' off=%zu buf=%p\n",
                  tag,
                  dst_name,
                  ggml_op_name(dst->op),
                  ggml_type_name(dst->type),
                  dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                  (size_t) dst->nb[0], (size_t) dst->nb[1], (size_t) dst->nb[2], (size_t) dst->nb[3],
                  dst_off,
                  dst_buf,
                  src0_name,
                  src0_off,
                  src0_buf,
                  src1_name,
                  src1_off,
                  src1_buf);
}

static size_t align_to(size_t value, size_t to_alignment) {
    GGML_ASSERT(to_alignment && "Invalid alignment (must be non-zero)");
    GGML_ASSERT((to_alignment & (to_alignment - 1)) == 0 && "to_alignment must be power-of-two");

    return ((value + to_alignment - 1) / to_alignment) * to_alignment;
}


// Parses a version string of form "XX.YY ". On an error returns ggml_cl_version with all zeroes.
static ggml_cl_version parse_cl_version(std::string_view str) {
    size_t major_str_begin = 0;
    size_t major_str_end   = str.find(".", major_str_begin);
    if (major_str_end == std::string::npos) {
        return {};
    }

    size_t minor_str_begin = major_str_end + 1;
    size_t minor_str_end   = str.find(" ", minor_str_begin);
    if (minor_str_end == std::string::npos) {
        return {};
    }

    cl_uint version_major;
    if (std::from_chars(str.data() + major_str_begin, str.data() + major_str_end, version_major).ec != std::errc{}) {
        return {};
    }

    cl_uint version_minor;
    if (std::from_chars(str.data() + minor_str_begin, str.data() + minor_str_end, version_minor).ec != std::errc{}) {
        return {};
    }
    return { version_major, version_minor };
}

// Returns OpenCL platform's version. On an error returns ggml_cl_version with all zeroes.
static ggml_cl_version get_opencl_platform_version(cl_platform_id platform) {
    size_t param_size;
    CL_CHECK(clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 0, nullptr, &param_size));
    std::unique_ptr<char[]> param_storage(new char[param_size]);
    CL_CHECK(clGetPlatformInfo(platform, CL_PLATFORM_VERSION, param_size, param_storage.get(), nullptr));

    auto              param_value    = std::string_view(param_storage.get(), param_size);
    const std::string version_prefix = "OpenCL ";  // Suffix: "XX.YY <platform-specific-info>"
    if (param_value.find(version_prefix) != 0) {
        return {};
    }
    param_value.remove_prefix(version_prefix.length());
    return parse_cl_version(param_value);
}

// Return a version to use in OpenCL C compilation. On an error returns ggml_cl_version with all zeroes.
static ggml_cl_version get_opencl_c_version(ggml_cl_version platform_version, cl_device_id device) {
    size_t param_size;

#if CL_TARGET_OPENCL_VERSION >= 300
    if (platform_version.major >= 3) {
        CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_ALL_VERSIONS, 0, nullptr, &param_size));
        if (!param_size) {
            return {};
        }

        std::unique_ptr<cl_name_version[]> versions(new cl_name_version[param_size]);
        CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_ALL_VERSIONS, param_size, versions.get(), nullptr));
        unsigned versions_count = param_size / sizeof(cl_name_version);

        cl_version version_max = 0;
        for (unsigned i = 0; i < versions_count; i++) {
            version_max = std::max<cl_version>(versions[i].version, version_max);
        }

        return { CL_VERSION_MAJOR(version_max), CL_VERSION_MINOR(version_max) };
    }
#else
    GGML_UNUSED(platform_version);
#endif  // CL_TARGET_OPENCL_VERSION >= 300

    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, 0, nullptr, &param_size));
    if (!param_size) {
        return {};
    }

    std::unique_ptr<char[]> param_storage(new char[param_size]);
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, param_size, param_storage.get(), nullptr));
    auto param_value = std::string_view(param_storage.get(), param_size);

    const std::string version_prefix = "OpenCL C ";  // Suffix: "XX.YY <platform-specific-info>"
    if (param_value.find(version_prefix) != 0) {
        return {};
    }
    param_value.remove_prefix(version_prefix.length());

    return parse_cl_version(param_value);
}

static ADRENO_GPU_GEN get_adreno_gpu_gen(const char *device_name) {
    if (strstr(device_name, "730") ||
        strstr(device_name, "740") ||
        strstr(device_name, "750")) {
        return ADRENO_GPU_GEN::A7X;
    }

    if (strstr(device_name, "830")) {
        return ADRENO_GPU_GEN::A8X;
    }

    if (strstr(device_name, "X1")) {
        return ADRENO_GPU_GEN::X1E;
    }

    return ADRENO_GPU_GEN::ADRENO_UNKNOWN;
}

static ggml_cl_compiler_version get_adreno_cl_compiler_version(const char *driver_version) {
    std::string driver_ver_str(driver_version);
    ADRENO_CL_COMPILER_TYPE type = ADRENO_CL_COMPILER_TYPE::E031;
    size_t compiler_ver_pos = driver_ver_str.find("E031");
    size_t compiler_ver_len = 13;
    size_t compiler_major_offset = 5;
    size_t compiler_minor_offset = 8;
    size_t compiler_patch_offset = 11;

    if (compiler_ver_pos == std::string::npos) {
        compiler_ver_pos = driver_ver_str.find("DX");
        if (compiler_ver_pos == std::string::npos) {
            return {};
        }
        type = ADRENO_CL_COMPILER_TYPE::DX;
        compiler_ver_len = 11;
        compiler_major_offset = 3;
    }

    std::string compiler_ver_str = driver_ver_str.substr(compiler_ver_pos, compiler_ver_len);
    int major = std::atoi(compiler_ver_str.substr(compiler_major_offset, 2).c_str());
    int minor = std::atoi(compiler_ver_str.substr(compiler_minor_offset, 2).c_str());
    int patch = std::atoi(compiler_ver_str.substr(compiler_patch_offset, 2).c_str());
    return { type, major, minor, patch };
}

// cl buffer wrapper
struct ggml_cl_buffer {
    cl_mem buffer;
    size_t size;

    ggml_cl_buffer()
        : buffer(nullptr), size(0) {}

    ~ggml_cl_buffer() {
        if (buffer) {
            CL_CHECK(clReleaseMemObject(buffer));
        }
    }

    void allocate(cl_context context, size_t new_size) {
        if (new_size > size) {
            size = new_size;
            if (buffer) {
                CL_CHECK(clReleaseMemObject(buffer));
            }
            cl_int err;
            CL_CHECK((buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &err), err));
        }
    }
};

// Profiling
struct ProfilingInfo {
    std::string op_name;
    std::string kernel_name;

    cl_kernel kernel;
    cl_event evt;

    cl_ulong cmd_queued;
    cl_ulong cmd_submit;
    cl_ulong cmd_start;
    cl_ulong cmd_end;
    cl_ulong overhead_start;
    cl_ulong overhead_end;
    // For the times below, see spec for clGetEventProfilingInfo
    // The time kernel spent in cmd queue - SUBMIT - QUEUED
    cl_ulong cmd_queued_duration_ns;
    // The time kernel spent for submission - START - SUBMIT
    cl_ulong cmd_submit_duration_ns;
    // Kernel execution time in nanoseconds - END - START
    cl_ulong cmd_duration_ns;
    // The time for the kernel to complete - COMPLETE - END
    cl_ulong cmd_complete_duration_ns;
    // Total time to finish the kernel - COMPELTE - QUEUED
    cl_ulong cmd_total_duration_ns;
    // Global and local work sizes.
    size_t global_size[3];
    size_t local_size[3];
    // Op output size.
    size_t output_size[4];
};

static void populateProfilingInfo(
        ProfilingInfo& info, cl_event evt, cl_kernel kernel, cl_uint work_dim,
        size_t global_size[3], size_t local_size[3],
        const ggml_tensor * tensor) {
    info.op_name     = tensor->name;
    info.kernel      = kernel;
    info.evt         = evt;

    // 0 means not specified, e.g., 2D workgroup, or NULL for driver to choose
    info.local_size[0] = 0;
    info.local_size[1] = 0;
    info.local_size[2] = 0;

    info.global_size[0] = 0;
    info.global_size[1] = 0;
    info.global_size[2] = 0;

    if (local_size) {
        for (cl_uint i = 0; i < work_dim; ++i) {
            info.local_size[i] = local_size[i];
        }
    }

    for (cl_uint i = 0; i < work_dim; ++i) {
        info.global_size[i] = global_size[i];
    }

    info.output_size[0] = tensor->ne[0];
    info.output_size[1] = tensor->ne[1];
    info.output_size[2] = tensor->ne[2];
    info.output_size[3] = tensor->ne[3];
}

struct ggml_backend_opencl_context;

// backend device context
struct ggml_backend_opencl_device_context {
    cl_platform_id platform;
    std::string platform_name;

    cl_device_id   device;
    std::string    device_name;
    cl_device_type device_type;
    std::string    device_version;

    // Initialized by ggml_cl2_init().
    ggml_backend_opencl_context * backend_ctx = nullptr;

    // Initialized by ggml_backend_opencl_device_get_buffer_type()
    ggml_backend_buffer_type buffer_type;

    cl_context context = nullptr;
};

// backend context
struct ggml_backend_opencl_context {
    int ref_count;

    cl_device_id device;
    std::string device_name;

    std::string driver_version;

    GPU_FAMILY gpu_family;
    ADRENO_GPU_GEN adreno_gen;

    cl_int alignment;
    size_t max_alloc_size;
    size_t max_workgroup_size;
    bool fp16_support;
    bool has_vector_subgroup_broadcast;
    bool disable_fusion;
    ggml_cl_compiler_version adreno_cl_compiler_version;

    int adreno_wave_size;

    cl_bool non_uniform_workgroups;
    size_t  image_max_buffer_size;

    cl_context context;
    cl_command_queue queue;

    // prealloc buffers for transposing weights and activations
    ggml_cl_buffer prealloc_quant_trans;
    ggml_cl_buffer prealloc_scales_trans;
    ggml_cl_buffer prealloc_act_trans;

    // prealloc buffers for src0 and src1
    ggml_cl_buffer prealloc_src0;
    ggml_cl_buffer prealloc_src1;

    struct mldrift_attn_h24_state {
        bool initialized = false;
        int shape_m = 0;
        std::array<cl_mem, ggml_opencl_mldrift_h24_m4352::kNumBuffers> buffers{};
        std::array<cl_mem, ggml_opencl_mldrift_h24_m4352::kNumImages> images{};
        std::array<cl_program, ggml_opencl_mldrift_h24_m4352::kNumPrograms> programs{};
        std::array<cl_kernel, ggml_opencl_mldrift_h24_m4352::kNumKernels> kernels{};
    } mldrift_attn_h24;

    struct mldrift_attn_h30_state {
        bool initialized = false;
        std::array<cl_mem, ggml_opencl_mldrift_h30_m4224::kNumBuffers> buffers{};
        std::array<cl_mem, ggml_opencl_mldrift_h30_m4224::kNumImages> images{};
        std::array<cl_program, ggml_opencl_mldrift_h30_m4224::kNumPrograms> programs{};
        std::array<cl_kernel, ggml_opencl_mldrift_h30_m4224::kNumKernels> kernels{};
    } mldrift_attn_h30;

    cl_program program_add;
    cl_program program_add_id;
    cl_program program_clamp;
    cl_program program_cpy;
    cl_program program_cvt;
    cl_program program_diag_mask_inf;
    cl_program program_gelu;
    cl_program program_gemv_noshuffle_general;
    cl_program program_gemv_noshuffle;
    cl_program program_get_rows;
    cl_program program_set_rows;
    cl_program program_glu;
    cl_program program_im2col_f16;
    cl_program program_im2col_f32;
    cl_program program_mul_mat_Ab_Bi_8x4;
    cl_program program_mul_mv_q4_0_f32;
    cl_program program_mul_mv_q4_0_f32_v;
    cl_program program_mul_mv_q4_0_f32_8x_flat;
    cl_program program_mul_mv_q4_0_f32_1d_8x_flat;
    cl_program program_mul_mv_q4_0_f32_1d_16x_flat;
    cl_program program_mul_mv_q6_K;
    cl_program program_mul_mv_q8_0_f32, program_mul_mv_q8_0_f32_flat;
    cl_program program_mul_mv_mxfp4_f32;
    cl_program program_mul_mv_mxfp4_f32_flat;
    cl_program program_mul_mv_f16_f16;
    cl_program program_mul_mv_f16_f32_1row;
    cl_program program_mul_mv_f16_f32_l4;
    cl_program program_mul_mv_f16_f32;
    cl_program program_mul_mv_f32_f32;
    cl_program program_mul;
    cl_program program_mul_mat_f16_f32_tiled;
    cl_program program_mul_mm_f16_f32_kqv;
    cl_program program_mul_mm_f16_f32_kq;
    cl_program program_div;
    cl_program program_sub;
    cl_program program_norm;
    cl_program program_relu;
    cl_program program_rms_norm;
    cl_program program_group_norm;
    cl_program program_rope;
    cl_program program_scale;
    cl_program program_silu;
    cl_program program_sigmoid;
    cl_program program_softmax_f32;
    cl_program program_softmax_f16;
    cl_program program_softmax_4_f32;
    cl_program program_softmax_4_f16;
    cl_program program_argsort_f32_i32;
    cl_program program_sum_rows_f32;
    cl_program program_repeat;
    cl_program program_pad;
    cl_program program_tanh;
    cl_program program_upscale;
    cl_program program_concat;
    cl_program program_conv_2d_f16;
    cl_program program_conv_2d_f16_vae3x3;
    cl_program program_conv_2d_f32;
    cl_program program_conv_2d_f32_vae3x3;
    cl_program program_conv_2d_f16_f32;
    cl_program program_conv_2d_f16_f32_vae3x3;
    cl_program program_tsembd;
    cl_program program_gemv_moe_mxfp4_f32, program_gemm_moe_mxfp4_f32;
    cl_program program_mul_mv_id_q4_0_f32_8x_flat;
    cl_program program_mul_mv_id_q8_0_f32, program_mul_mv_id_q8_0_f32_flat;
    cl_program program_mul_mv_id_mxfp4_f32;
    cl_program program_mul_mv_id_mxfp4_f32_flat;
    cl_program program_mul_mm_f32_f32_l4_lm;
    cl_program program_mul_mm_f16_f32_l4_lm;
    cl_program program_mul_mm_q8_0_f32_l4_lm;
    cl_program program_convert_f16_f32_linear;

    cl_kernel kernel_add, kernel_add_row, kernel_add_f16, kernel_add_row_f16;
    cl_kernel kernel_mul, kernel_mul_row, kernel_mul_f16, kernel_mul_row_f16;
    cl_kernel kernel_div, kernel_div_row, kernel_div_f16, kernel_div_row_f16;
    cl_kernel kernel_sub, kernel_sub_row, kernel_sub_f16, kernel_sub_row_f16;
    cl_kernel kernel_add_id;
    cl_kernel kernel_scale;
    cl_kernel kernel_sqr_cont_f32, kernel_sqr_cont_f32_4, kernel_sqr_cont_f16, kernel_sqr_cont_f16_4;
    cl_kernel kernel_sqrt_cont_f32, kernel_sqrt_cont_f32_4, kernel_sqrt_cont_f16, kernel_sqrt_cont_f16_4;
    cl_kernel kernel_mean_f32;
    cl_kernel kernel_silu, kernel_silu_4;
    cl_kernel kernel_gelu, kernel_gelu_4;
    cl_kernel kernel_gelu_erf, kernel_gelu_erf_4;
    cl_kernel kernel_gelu_quick, kernel_gelu_quick_4;
    cl_kernel kernel_relu;
    cl_kernel kernel_sigmoid_f32, kernel_sigmoid_f16;
    cl_kernel kernel_tri;
    cl_kernel kernel_fill;
    cl_kernel kernel_clamp;
    cl_kernel kernel_geglu, kernel_reglu, kernel_swiglu, kernel_swiglu_oai, kernel_geglu_erf, kernel_geglu_quick,
              kernel_geglu_f16, kernel_reglu_f16, kernel_swiglu_f16, kernel_geglu_erf_f16, kernel_geglu_quick_f16;
    cl_kernel kernel_norm, kernel_norm_mul_add;
    cl_kernel kernel_rms_norm, kernel_rms_norm_mul;
    cl_kernel kernel_group_norm, kernel_group_norm_mul_add;
    cl_kernel kernel_diag_mask_inf, kernel_diag_mask_inf_8;
    cl_kernel kernel_soft_max, kernel_soft_max_4;
    cl_kernel kernel_soft_max_f16, kernel_soft_max_4_f16;
    std::map<std::pair<int, int>, cl_kernel> kernels_flash_attn_f16;
    std::map<std::pair<int, int>, cl_kernel> kernels_flash_attn_f16_q1;
    std::map<std::pair<int, int>, cl_kernel> kernels_flash_attn_f32;
    std::map<std::pair<int, int>, cl_kernel> kernels_flash_attn_f32_q1;
    std::map<std::pair<int, int>, cl_kernel> kernels_flash_attn_f32_f16;
    std::map<std::pair<int, int>, cl_kernel> kernels_flash_attn_f32_f16_q1;
    std::map<std::pair<int, int>, int>       kernels_flash_attn_bm;
    std::map<std::pair<int, int>, int>       kernels_flash_attn_bn;
    cl_kernel kernel_get_rows_f32, kernel_get_rows_f16, kernel_get_rows_q4_0, kernel_get_rows_q4_0_soa, kernel_get_rows_q4_0_soa_scalar;
    cl_kernel kernel_set_rows_f32_i64, kernel_set_rows_f32_i32, kernel_set_rows_f16_i64, kernel_set_rows_f16_i32;
    cl_kernel kernel_rope_norm_f32, kernel_rope_norm_f16, kernel_rope_neox_f32, kernel_rope_neox_f16;
    cl_kernel kernel_rope_multi_f32, kernel_rope_multi_f16, kernel_rope_vision_f32, kernel_rope_vision_f16;
    cl_kernel kernel_cpy_f16_f16, kernel_cpy_f16_f32, kernel_cpy_f32_f16, kernel_cpy_f32_f32;
    cl_kernel kernel_cpy_f16_f16_1d, kernel_cpy_f32_f32_1d;
    cl_kernel kernel_mul_mat_f32_f32;
    cl_kernel kernel_mul_mat_f16_f16;
    cl_kernel kernel_mul_mat_f16_f32_1row;
    cl_kernel kernel_mul_mat_f16_f32;
    cl_kernel kernel_mul_mat_f16_f32_l4;
    cl_kernel kernel_mul_mat_f16_f32_tiled;
    cl_kernel kernel_mul_mm_f16_f32_kqv;
    cl_kernel kernel_mul_mm_f16_f32_kq;
    cl_kernel kernel_mul_mat_q4_0_f32, kernel_mul_mat_q4_0_f32_v;
    cl_kernel kernel_convert_block_q4_0, kernel_restore_block_q4_0;
    cl_kernel kernel_convert_block_mxfp4, kernel_convert_block_mxfp4_trans, kernel_restore_block_mxfp4, kernel_restore_block_mxfp4_trans;
    cl_kernel kernel_convert_block_q8_0, kernel_restore_block_q8_0;
    cl_kernel kernel_mul_mat_q4_0_f32_8x_flat;
    cl_kernel kernel_mul_mat_q4_0_f32_ref;
    cl_kernel kernel_convert_block_q4_0_noshuffle;
    cl_kernel kernel_restore_block_q4_0_noshuffle;
    cl_kernel kernel_convert_block_q6_K, kernel_restore_block_q6_K;
    cl_kernel kernel_mul_mat_q4_0_f32_1d_8x_flat, kernel_mul_mat_q4_0_f32_1d_16x_flat;
    cl_kernel kernel_mul_mv_q6_K_f32;
    cl_kernel kernel_mul_mv_q6_K_f32_flat;
    cl_kernel kernel_mul_mv_mxfp4_f32, kernel_mul_mv_mxfp4_f32_flat;
    cl_kernel kernel_mul_mv_q8_0_f32, kernel_mul_mv_q8_0_f32_flat;
    cl_kernel kernel_solve_tri_f32;
    cl_kernel kernel_im2col_f32, kernel_im2col_f16;
    cl_kernel kernel_argsort_f32_i32;
    cl_kernel kernel_sum_rows_f32;
    cl_kernel kernel_repeat;
    cl_kernel kernel_pad;
    cl_kernel kernel_tanh_f32_nd;
    cl_kernel kernel_tanh_f16_nd;
    cl_kernel kernel_expm1_f32_nd;
    cl_kernel kernel_expm1_f16_nd;
    cl_kernel kernel_softplus_f32_nd;
    cl_kernel kernel_softplus_f16_nd;
    cl_kernel kernel_upscale;
    cl_kernel kernel_upscale_bilinear;
    cl_kernel kernel_concat_f32_contiguous;
    cl_kernel kernel_concat_f32_non_contiguous;
    cl_kernel kernel_conv_2d_f16;
    cl_kernel kernel_conv_2d_f16_vae3x3;
    cl_kernel kernel_conv_2d_f32;
    cl_kernel kernel_conv_2d_f32_vae3x3;
    cl_kernel kernel_conv_2d_f16_f32;
    cl_kernel kernel_conv_2d_f16_f32_vae3x3;
    cl_kernel kernel_ssm_conv_f32_f32, kernel_ssm_conv_f32_f32_4;
    cl_kernel kernel_timestep_embedding;
    cl_kernel kernel_gemv_moe_mxfp4_f32, kernel_gemm_moe_mxfp4_f32;
    cl_kernel kernel_mul_mv_id_q4_0_f32_8x_flat;
    cl_kernel kernel_mul_mv_id_q8_0_f32, kernel_mul_mv_id_q8_0_f32_flat;
    cl_kernel kernel_mul_mv_id_mxfp4_f32;
    cl_kernel kernel_mul_mv_id_mxfp4_f32_flat;
    cl_kernel kernel_mul_mm_f32_f32_l4_lm;
    cl_kernel kernel_mul_mm_f16_f32_l4_lm;
    cl_kernel kernel_mul_mm_q8_0_f32_l4_lm;
    cl_kernel kernel_f16_to_f32_linear;
    cl_kernel kernel_f16_to_f32_nhd_crop;
    cl_kernel kernel_f16_to_f32_nhd_keep_head_tail;
    cl_kernel kernel_f32_copy_scale_linear;
    cl_kernel kernel_f32_reorder_hqd_to_qhd;

    std::vector<ProfilingInfo> profiling_info;

    void write_profiling_info() {
        FILE * fperf = fopen("cl_profiling.csv", "w");
        if (!fperf) {
            GGML_LOG_ERROR("Failed to open cl_profiling.csv\n");
            return;
        }

        // Populate profiling info
        for (ProfilingInfo & info : profiling_info) {
            cl_ulong cmd_queued;
            cl_ulong cmd_submit;
            cl_ulong cmd_start;
            cl_ulong cmd_end;
            cl_ulong cmd_complete;

            CL_CHECK(clWaitForEvents(1, &info.evt));
            CL_CHECK(clGetEventProfilingInfo(
                info.evt, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &cmd_queued, NULL));
            CL_CHECK(clGetEventProfilingInfo(
                info.evt, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &cmd_submit, NULL));
            CL_CHECK(clGetEventProfilingInfo(
                info.evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &cmd_start, NULL));
            CL_CHECK(clGetEventProfilingInfo(
                info.evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &cmd_end, NULL));
            CL_CHECK(clGetEventProfilingInfo(
                info.evt, CL_PROFILING_COMMAND_COMPLETE, sizeof(cl_ulong), &cmd_complete, NULL));
            CL_CHECK(clReleaseEvent(info.evt));

            char kernel_name[512];
            CL_CHECK(clGetKernelInfo(info.kernel, CL_KERNEL_FUNCTION_NAME,
                sizeof(kernel_name), kernel_name, NULL));
            info.kernel_name = kernel_name;

            info.cmd_queued = cmd_queued;
            info.cmd_submit = cmd_submit;
            info.cmd_start  = cmd_start;
            info.cmd_end    = cmd_end;

            info.cmd_queued_duration_ns     = cmd_submit    - cmd_queued;
            info.cmd_submit_duration_ns     = cmd_start     - cmd_submit;
            info.cmd_duration_ns            = cmd_end       - cmd_start;
            info.cmd_complete_duration_ns   = cmd_complete  - cmd_end;
            info.cmd_total_duration_ns      = cmd_complete  - cmd_queued;
        }

        // Dump a csv
        fprintf(fperf, "op name, kernel name, exec duration (ms), global size, local size, output size\n");
        for (const ProfilingInfo & info : profiling_info) {
            fprintf(fperf, "%s,%s,%f,%zux%zux%zu,%zux%zux%zu,%zux%zux%zux%zu\n",
                info.op_name.c_str(), info.kernel_name.c_str(),
                info.cmd_duration_ns/1.e6f,
                info.global_size[0], info.global_size[1], info.global_size[2],
                info.local_size[0], info.local_size[1], info.local_size[2],
                info.output_size[0], info.output_size[1], info.output_size[2], info.output_size[3]);
        }
        fclose(fperf);

        // Dump a simple chrome trace
        FILE* ftrace = fopen("cl_trace.json", "w");
        if (!ftrace) {
            GGML_LOG_ERROR("Failed to open cl_trace.json\n");
            return;
        }

        fprintf(ftrace, "[\n");
        for (const ProfilingInfo & info : profiling_info) {
            fprintf(ftrace, "{\"name\": \"%s\", \"cat\": \"OpenCL\", \"ph\": \"B\", \"ts\": %" PRIu64 ", \"pid\": \"\", \"tid\": \"Host\"},\n",
                info.kernel_name.c_str(), info.cmd_queued/1000);
            fprintf(ftrace, "{\"name\": \"%s\", \"cat\": \"OpenCL\", \"ph\": \"E\", \"ts\": %" PRIu64 ", \"pid\": \"\", \"tid\": \"Host\"},\n",
                info.kernel_name.c_str(), info.cmd_submit/1000);

            fprintf(ftrace, "{\"name\": \"%s\", \"cat\": \"OpenCL\", \"ph\": \"B\", \"ts\": %" PRIu64 ", \"pid\": \"\", \"tid\": \"Device\"},\n",
                info.kernel_name.c_str(), info.cmd_start/1000);
            fprintf(ftrace, "{\"name\": \"%s\", \"cat\": \"OpenCL\", \"ph\": \"E\", \"ts\": %" PRIu64 ", \"pid\": \"\", \"tid\": \"Device\"},\n",
                info.kernel_name.c_str(), info.cmd_end/1000);
        }
        fclose(ftrace);
    }

    size_t get_kernel_workgroup_size(cl_kernel kernel) const {
        size_t workgroup_size = 0;
        size_t ret_size = 0;
        CL_CHECK(
            clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                sizeof(size_t), &workgroup_size, &ret_size));
        GGML_ASSERT(sizeof(size_t) == ret_size);
        return workgroup_size;
    }

    void enqueue_ndrange_kernel(cl_kernel kernel, cl_uint work_dim, size_t *global_work_size, size_t *local_work_size, const ggml_tensor * tensor) {
#ifdef GGML_OPENCL_PROFILING
        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, &evt));

        profiling_info.emplace_back();
        populateProfilingInfo(profiling_info.back(), evt, kernel, work_dim, global_work_size, local_work_size, tensor);
#else
        GGML_UNUSED(tensor);
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, NULL));
#endif
    }

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    // Transpose kernels
    cl_program program_transpose;

    cl_kernel kernel_transpose_32;
    cl_kernel kernel_transpose_32_16;
    cl_kernel kernel_transpose_32_32;
    cl_kernel kernel_transpose_16;
    cl_kernel kernel_transpose_16_buf;
    cl_kernel kernel_transpose_16_4x1;

    // Gemm and Gemv related programs, kernels, etc
    cl_program program_CL_gemm;
    cl_program program_CL_gemm_kahan;
    cl_program program_CL_gemm_f32act;
    cl_program program_CL_gemm_f32read;
    cl_program program_CL_gemm_f32acc;
    cl_program program_CL_gemm_halfclamp;
    cl_program program_CL_gemm_halfscale;
    cl_program program_CL_gemm_fp16chunk;
    cl_program program_CL_gemv_general;
    cl_program program_CL_gemv_4096_1_11008;
    cl_program program_CL_gemv_4096_1_4096;
    cl_program program_CL_gemv_11008_1_4096;
    cl_program program_CL_gemv_32000_1_4096;
    cl_kernel CL_mul_mat_Ab_Bi_8x4;
    cl_kernel CL_mul_mat_Ab_Bi_8x4_kahan;
    cl_kernel CL_mul_mat_Ab_Bi_8x4_f32act;
    cl_kernel CL_mul_mat_Ab_Bi_8x4_f32read;
    cl_kernel CL_mul_mat_Ab_Bi_8x4_f32acc;
    cl_kernel CL_mul_mat_Ab_Bi_8x4_halfclamp;
    cl_kernel CL_mul_mat_Ab_Bi_8x4_halfscale;
    cl_kernel CL_mul_mat_Ab_Bi_8x4_fp16chunk;
    cl_kernel CL_mul_mat_vec_q4_0_f32_1d_4x_flat_general;
    cl_kernel CL_mul_mat_vec_q4_0_f32_1d_4x_flat_4096_1_11008;
    cl_kernel CL_mul_mat_vec_q4_0_f32_1d_4x_flat_4096_1_4096;
    cl_kernel CL_mul_mat_vec_q4_0_f32_1d_4x_flat_11008_1_4096;
    cl_kernel CL_mul_mat_vec_q4_0_f32_1d_4x_flat_32000_1_4096;
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

    void free() {
        ref_count--;
        if (ref_count == 0) {
#ifdef GGML_OPENCL_PROFILING
            write_profiling_info();
            profiling_info.clear();
#endif
        }
    }
};

static bool ggml_cl_dump_buffer_region(struct ggml_backend_opencl_context * backend_ctx,
                                       cl_mem buffer,
                                       cl_ulong offset,
                                       size_t bytes,
                                       const std::string & path) {
    if (bytes == 0) {
        return false;
    }

    std::vector<uint8_t> host(bytes);
    const cl_int st = clEnqueueReadBuffer(
        backend_ctx->queue,
        buffer,
        CL_TRUE,
        offset,
        bytes,
        host.data(),
        0,
        nullptr,
        nullptr);
    if (st != CL_SUCCESS) {
        GGML_LOG_WARN("ggml_opencl: flash dump read failed (%d) for '%s'\n", (int) st, path.c_str());
        return false;
    }

    FILE * f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        GGML_LOG_WARN("ggml_opencl: flash dump open failed for '%s'\n", path.c_str());
        return false;
    }
    std::fwrite(host.data(), 1, host.size(), f);
    std::fclose(f);
    return true;
}

// All registered devices with a default device in the front.
static std::vector<ggml_backend_device> g_ggml_backend_opencl_devices;

inline std::string read_file(const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs) {
    return "";
  }
  std::string text;
  ifs.seekg(0, std::ios::end);
  text.resize(ifs.tellg());
  ifs.seekg(0, std::ios::beg);
  ifs.read(&text[0], text.size());
  return text;
}

static cl_program build_program_from_source(cl_context ctx, cl_device_id dev, const char* program_buffer, const std::string &compile_opts) {
    cl_program p;
    char *program_log;
    size_t program_size;
    size_t log_size;
    int err;

    program_size = strlen(program_buffer);

    p = clCreateProgramWithSource(ctx, 1, (const char**)&program_buffer, &program_size, &err);
    if(err < 0) {
        GGML_LOG_ERROR("OpenCL error creating program");
        exit(1);
    }

    err = clBuildProgram(p, 0, NULL, compile_opts.c_str(), NULL, NULL);
    if(err < 0) {
        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char*) malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        GGML_LOG_ERROR("ggml_opencl: kernel compile error:\n\n%s\n", program_log);
        free(program_log);
        exit(1);
    }

    return p;
}

static void mldrift_attn_release_h24(ggml_backend_opencl_context * backend_ctx) {
    auto & st = backend_ctx->mldrift_attn_h24;

    for (cl_kernel & kernel : st.kernels) {
        if (kernel != nullptr) {
            CL_CHECK(clReleaseKernel(kernel));
            kernel = nullptr;
        }
    }
    for (cl_program & program : st.programs) {
        if (program != nullptr) {
            CL_CHECK(clReleaseProgram(program));
            program = nullptr;
        }
    }
    for (cl_mem & image : st.images) {
        if (image != nullptr) {
            CL_CHECK(clReleaseMemObject(image));
            image = nullptr;
        }
    }
    for (cl_mem & buffer : st.buffers) {
        if (buffer != nullptr) {
            CL_CHECK(clReleaseMemObject(buffer));
            buffer = nullptr;
        }
    }

    st.initialized = false;
    st.shape_m = 0;
}

static bool mldrift_attn_init_h24(ggml_backend_opencl_context * backend_ctx, int shape_m) {
    auto & st = backend_ctx->mldrift_attn_h24;
    namespace mh = ggml_opencl_mldrift_h24_m4352;

    if (st.initialized) {
        // Reuse preallocated replay resources for all supported M <= kShapeM.
        if (st.shape_m == mh::kShapeM) {
            return true;
        }
        mldrift_attn_release_h24(backend_ctx);
    }

    if (shape_m <= 0 || shape_m > mh::kShapeM) {
        return false;
    }

    cl_int status = CL_SUCCESS;

    for (int i = 0; i < mh::kNumBuffers; ++i) {
        const size_t bytes = mh::kBufferSizes[i];
        if (bytes == 0) {
            continue;
        }
        if (bytes > backend_ctx->max_alloc_size) {
            GGML_LOG_WARN("ggml_opencl: mldrift buffer %d size %zu exceeds max alloc %zu\n",
                          i, bytes, backend_ctx->max_alloc_size);
            mldrift_attn_release_h24(backend_ctx);
            return false;
        }
        st.buffers[i] = clCreateBuffer(backend_ctx->context, mh::kBufferFlags[i], bytes, nullptr, &status);
        if (!mh::mldrift_cl_check(status, "buffer alloc")) {
            GGML_LOG_WARN("ggml_opencl: mldrift buffer %d alloc failed (%d)\n", i, (int) status);
            mldrift_attn_release_h24(backend_ctx);
            return false;
        }
    }

    for (int i = 0; i < mh::kNumImages; ++i) {
        if (mh::kImageWidth[i] == 0) {
            continue;
        }
        cl_image_format fmt = {};
        fmt.image_channel_order = mh::kImageOrder[i];
        fmt.image_channel_data_type = mh::kImageTypeCode[i];

        cl_image_desc desc = {};
        desc.image_type = mh::kImageType[i];
        desc.image_width = mh::kImageWidth[i];
        desc.image_height = mh::kImageHeight[i];
        desc.image_depth = mh::kImageDepth[i];

        int backing = mh::kImageBackingBuffer[i];
        if (std::getenv("GGML_OPENCL_MLDRIFT_INPUT_IMAGES_678") != nullptr) {
            // Debug override: bind first 3 images to q/k/v input buffers 6/7/8.
            if (i == 0) backing = 6;
            if (i == 1) backing = 7;
            if (i == 2) backing = 8;
        }
        if (backing >= 0) {
            desc.buffer = st.buffers[backing];
        }

        st.images[i] = clCreateImage(backend_ctx->context, CL_MEM_READ_WRITE, &fmt, &desc, nullptr, &status);
        if (!mh::mldrift_cl_check(status, "image alloc")) {
            GGML_LOG_WARN("ggml_opencl: mldrift image %d alloc failed (%d)\n", i, (int) status);
            mldrift_attn_release_h24(backend_ctx);
            return false;
        }
    }

    for (int i = 0; i < mh::kNumPrograms; ++i) {
        const char * src = mh::kProgramSources[i];
        if (src == nullptr || src[0] == '\0') {
            continue;
        }
        const char * opts = mh::kProgramOptions[i];
        st.programs[i] = build_program_from_source(
            backend_ctx->context, backend_ctx->device, src, (opts && opts[0]) ? opts : "");
    }

    for (int i = 0; i < mh::kNumKernels; ++i) {
        const int program_id = mh::kKernelProgramId[i];
        st.kernels[i] = clCreateKernel(st.programs[program_id], "main_function", &status);
        if (!mh::mldrift_cl_check(status, "create kernel")) {
            GGML_LOG_WARN("ggml_opencl: mldrift create kernel %d failed (%d)\n", i, (int) status);
            mldrift_attn_release_h24(backend_ctx);
            return false;
        }
    }

    if (!mh::set_kernel_args(st.kernels.data(), st.buffers.data(), st.images.data())) {
        mldrift_attn_release_h24(backend_ctx);
        return false;
    }

    st.initialized = true;
    st.shape_m = mh::kShapeM;
    return true;
}

static inline bool mldrift_set_kernel_arg_int4_h24(cl_kernel kernel, cl_uint arg_index, int x, int y, int z, int w, const char * tag) {
    namespace mh = ggml_opencl_mldrift_h24_m4352;
    const int vals[4] = {x, y, z, w};
    return mh::mldrift_cl_check(clSetKernelArg(kernel, arg_index, sizeof(vals), vals), tag);
}

static bool mldrift_attn_patch_shape_h24(ggml_backend_opencl_context * backend_ctx, int shape_m) {
    namespace mh = ggml_opencl_mldrift_h24_m4352;
    auto & st = backend_ctx->mldrift_attn_h24;

    if (!st.initialized) {
        return false;
    }
    if (shape_m == mh::kShapeM) {
        return true;
    }
    // The kernels are vectorized in groups of 4 channels.
    if (shape_m <= 0 || shape_m > mh::kShapeM || (shape_m % 4) != 0) {
        return false;
    }

    const int m = shape_m;
    const int m4 = m / 4;
    const int m24 = m * 24;
    const int m192 = m * 192;

    if (!mldrift_set_kernel_arg_int4_h24(st.kernels[0], 2, m,   24, m4, 128, "SetKernelArg dyn 0:2")) return false;
    if (!mldrift_set_kernel_arg_int4_h24(st.kernels[1], 2, 8, m192, 32, 24, "SetKernelArg dyn 1:2")) return false;
    if (!mldrift_set_kernel_arg_int4_h24(st.kernels[1], 3, m4, 128, 128, 24, "SetKernelArg dyn 1:3")) return false;
    if (!mldrift_set_kernel_arg_int4_h24(st.kernels[2], 4, 24,  m4,  m, 32, "SetKernelArg dyn 2:4")) return false;
    if (!mldrift_set_kernel_arg_int4_h24(st.kernels[3], 2, 24,   1,  m,  m, "SetKernelArg dyn 3:2")) return false;
    if (!mldrift_set_kernel_arg_int4_h24(st.kernels[3], 3, 24,   m,  0,  0, "SetKernelArg dyn 3:3")) return false;
    if (!mldrift_set_kernel_arg_int4_h24(st.kernels[4], 3, 24,  m4,  m,  1, "SetKernelArg dyn 4:3")) return false;
    if (!mldrift_set_kernel_arg_int4_h24(st.kernels[4], 4, 24,   m,  0,  0, "SetKernelArg dyn 4:4")) return false;
    if (!mldrift_set_kernel_arg_int4_h24(st.kernels[5], 2, 8, m192, m4, 32, "SetKernelArg dyn 5:2")) return false;
    if (!mldrift_set_kernel_arg_int4_h24(st.kernels[5], 3, m,   24, 24,  0, "SetKernelArg dyn 5:3")) return false;
    if (!mldrift_set_kernel_arg_int4_h24(st.kernels[6], 4, 24,  32,  m, 32, "SetKernelArg dyn 6:4")) return false;
    if (!mldrift_set_kernel_arg_int4_h24(st.kernels[6], 5, m4,   0,  0, 24, "SetKernelArg dyn 6:5")) return false;
    if (!mldrift_set_kernel_arg_int4_h24(st.kernels[6], 6, m24, m4,  m,  1, "SetKernelArg dyn 6:6")) return false;
    if (!mldrift_set_kernel_arg_int4_h24(st.kernels[7], 2, 128, 24, 32,  m, "SetKernelArg dyn 7:2")) return false;
    if (!mldrift_set_kernel_arg_int4_h24(st.kernels[8], 2, 128, 24, 32,  m, "SetKernelArg dyn 8:2")) return false;
    if (!mldrift_set_kernel_arg_int4_h24(st.kernels[9], 2, 128, 24, 32,  m, "SetKernelArg dyn 9:2")) return false;
    if (!mldrift_set_kernel_arg_int4_h24(st.kernels[10], 2, 128, 24, 32, m, "SetKernelArg dyn 10:2")) return false;

    return true;
}

static inline void mldrift_attn_patch_gws_h24(size_t gws[][3], int shape_m) {
    namespace mh = ggml_opencl_mldrift_h24_m4352;
    for (int kid = 0; kid < mh::kNumKernels; ++kid) {
        gws[kid][0] = mh::kKernelGws[kid][0];
        gws[kid][1] = mh::kKernelGws[kid][1];
        gws[kid][2] = mh::kKernelGws[kid][2];
    }
    if (shape_m == mh::kShapeM) {
        return;
    }
    const size_t m = (size_t) shape_m;
    const size_t m4 = m / 4;
    gws[0][2] = m4;
    gws[1][0] = m * 192;
    gws[2][0] = m * 8;
    gws[3][0] = m + 768;
    gws[4][0] = m;
    gws[4][2] = m4;
    gws[5][0] = m * 192;
    gws[7][0] = m;
    gws[8][0] = m;
    gws[9][0] = m;
    gws[10][0] = m;
}

static inline bool mldrift_set_kernel_arg_int4_h30(cl_kernel kernel, cl_uint arg_index, int x, int y, int z, int w, const char * tag) {
    namespace mh = ggml_opencl_mldrift_h30_m4224;
    const int vals[4] = {x, y, z, w};
    return mh::mldrift_cl_check(clSetKernelArg(kernel, arg_index, sizeof(vals), vals), tag);
}

static bool mldrift_attn_patch_shape_h30(ggml_backend_opencl_context * backend_ctx, int shape_m, int shape_n) {
    namespace mh = ggml_opencl_mldrift_h30_m4224;
    auto & st = backend_ctx->mldrift_attn_h30;

    if (!st.initialized) {
        GGML_LOG_WARN("ggml_opencl: mldrift(h30) patch requested before init (M=%d, N=%d)\n", shape_m, shape_n);
        return false;
    }
    if (shape_m == mh::kShapeM && shape_n == mh::kShapeN) {
        return true;
    }
    // The kernels are vectorized in groups of 4 channels.
    if (shape_m <= 0 || shape_m > mh::kShapeM || (shape_m % 4) != 0 ||
        shape_n <= 0 || shape_n > mh::kShapeN || (shape_n % 4) != 0) {
        GGML_LOG_WARN("ggml_opencl: mldrift(h30) unsupported dynamic shape M=%d N=%d (maxM=%d maxN=%d mod4=(%d,%d))\n",
                      shape_m, shape_n, mh::kShapeM, mh::kShapeN, shape_m % 4, shape_n % 4);
        return false;
    }

    const int m = shape_m;
    const int m4 = m / 4;
    const int m30 = m * 30;
    const int m240 = m * 240;
    const int n = shape_n;

    if (!mldrift_set_kernel_arg_int4_h30(st.kernels[0], 2, m, 30, m4, 128, "SetKernelArg dyn h30 0:2")) return false;
    if (!mldrift_set_kernel_arg_int4_h30(st.kernels[1], 2, 8, m240, 32, 128, "SetKernelArg dyn h30 1:2")) return false;
    if (!mldrift_set_kernel_arg_int4_h30(st.kernels[1], 3, 30, 30, 30, m4, "SetKernelArg dyn h30 1:3")) return false;
    if (!mldrift_set_kernel_arg_int4_h30(st.kernels[2], 4, 30, m4, m, 32, "SetKernelArg dyn h30 2:4")) return false;
    if (!mldrift_set_kernel_arg_int4_h30(st.kernels[3], 2, 30, 1, m, m, "SetKernelArg dyn h30 3:2")) return false;
    if (!mldrift_set_kernel_arg_int4_h30(st.kernels[3], 3, 30, m, 0, 0, "SetKernelArg dyn h30 3:3")) return false;
    if (!mldrift_set_kernel_arg_int4_h30(st.kernels[4], 3, 30, m4, m, 1, "SetKernelArg dyn h30 4:3")) return false;
    if (!mldrift_set_kernel_arg_int4_h30(st.kernels[4], 4, 30, m, 0, 0, "SetKernelArg dyn h30 4:4")) return false;
    if (!mldrift_set_kernel_arg_int4_h30(st.kernels[5], 2, 8, m240, m4, m, "SetKernelArg dyn h30 5:2")) return false;
    if (!mldrift_set_kernel_arg_int4_h30(st.kernels[6], 4, 30, 32, m, 32, "SetKernelArg dyn h30 6:4")) return false;
    if (!mldrift_set_kernel_arg_int4_h30(st.kernels[6], 5, m4, 0, 0, 30, "SetKernelArg dyn h30 6:5")) return false;
    if (!mldrift_set_kernel_arg_int4_h30(st.kernels[6], 6, m30, m4, m, 1, "SetKernelArg dyn h30 6:6")) return false;
    if (!mldrift_set_kernel_arg_int4_h30(st.kernels[7], 2, 128, 30, 32, m, "SetKernelArg dyn h30 7:2")) return false;
    // K/V upload width tracks N (kv), not M (query/output).
    if (!mldrift_set_kernel_arg_int4_h30(st.kernels[8], 2, 128, 30, 32, n, "SetKernelArg dyn h30 8:2")) return false;
    if (!mldrift_set_kernel_arg_int4_h30(st.kernels[9], 2, 128, 30, 32, n, "SetKernelArg dyn h30 9:2")) return false;
    if (!mldrift_set_kernel_arg_int4_h30(st.kernels[10], 2, 128, 30, 32, m, "SetKernelArg dyn h30 10:2")) return false;

    return true;
}

static inline void mldrift_attn_patch_gws_h30(size_t gws[][3], int shape_m, int shape_n) {
    namespace mh = ggml_opencl_mldrift_h30_m4224;
    for (int kid = 0; kid < mh::kNumKernels; ++kid) {
        gws[kid][0] = mh::kKernelGws[kid][0];
        gws[kid][1] = mh::kKernelGws[kid][1];
        gws[kid][2] = mh::kKernelGws[kid][2];
    }
    if (shape_m == mh::kShapeM && shape_n == mh::kShapeN) {
        return;
    }
    const size_t m = (size_t) shape_m;
    const size_t n = (size_t) shape_n;
    const size_t m4 = m / 4;
    gws[0][2] = m4;
    gws[1][0] = m * 240;
    gws[2][0] = m * 8;
    gws[3][0] = m + 896;
    gws[4][0] = m;
    gws[4][2] = m4;
    gws[5][0] = m * 240;
    gws[7][0] = m;
    gws[8][0] = n;
    gws[9][0] = n;
    gws[10][0] = m;
}

static void mldrift_attn_release_h30(ggml_backend_opencl_context * backend_ctx) {
    auto & st = backend_ctx->mldrift_attn_h30;

    for (cl_kernel & kernel : st.kernels) {
        if (kernel != nullptr) {
            CL_CHECK(clReleaseKernel(kernel));
            kernel = nullptr;
        }
    }
    for (cl_program & program : st.programs) {
        if (program != nullptr) {
            CL_CHECK(clReleaseProgram(program));
            program = nullptr;
        }
    }
    for (cl_mem & image : st.images) {
        if (image != nullptr) {
            CL_CHECK(clReleaseMemObject(image));
            image = nullptr;
        }
    }
    for (cl_mem & buffer : st.buffers) {
        if (buffer != nullptr) {
            CL_CHECK(clReleaseMemObject(buffer));
            buffer = nullptr;
        }
    }

    st.initialized = false;
}

static bool mldrift_attn_init_h30(ggml_backend_opencl_context * backend_ctx) {
    namespace mh = ggml_opencl_mldrift_h30_m4224;
    auto & st = backend_ctx->mldrift_attn_h30;

    if (st.initialized) {
        return true;
    }

    cl_int status = CL_SUCCESS;

    for (int i = 0; i < mh::kNumBuffers; ++i) {
        const size_t bytes = mh::kBufferSizes[i];
        if (bytes == 0) {
            continue;
        }
        if (bytes > backend_ctx->max_alloc_size) {
            GGML_LOG_WARN("ggml_opencl: mldrift(h30) buffer %d size %zu exceeds max alloc %zu\n",
                          i, bytes, backend_ctx->max_alloc_size);
            mldrift_attn_release_h30(backend_ctx);
            return false;
        }
        st.buffers[i] = clCreateBuffer(backend_ctx->context, mh::kBufferFlags[i], bytes, nullptr, &status);
        if (!mh::mldrift_cl_check(status, "buffer alloc")) {
            GGML_LOG_WARN("ggml_opencl: mldrift(h30) buffer %d alloc failed (%d)\n", i, (int) status);
            mldrift_attn_release_h30(backend_ctx);
            return false;
        }
    }

    for (int i = 0; i < mh::kNumImages; ++i) {
        if (mh::kImageWidth[i] == 0) {
            continue;
        }

        cl_image_format fmt = {};
        fmt.image_channel_order = mh::kImageOrder[i];
        fmt.image_channel_data_type = mh::kImageTypeCode[i];

        cl_image_desc desc = {};
        desc.image_type = mh::kImageType[i];
        desc.image_width = mh::kImageWidth[i];
        desc.image_height = mh::kImageHeight[i];
        desc.image_depth = mh::kImageDepth[i];

        const int backing = mh::kImageBackingBuffer[i];
        if (backing >= 0) {
            desc.buffer = st.buffers[backing];
        }

        st.images[i] = clCreateImage(backend_ctx->context, CL_MEM_READ_WRITE, &fmt, &desc, nullptr, &status);
        if (!mh::mldrift_cl_check(status, "image alloc")) {
            GGML_LOG_WARN("ggml_opencl: mldrift(h30) image %d alloc failed (%d)\n", i, (int) status);
            mldrift_attn_release_h30(backend_ctx);
            return false;
        }
    }

    for (int i = 0; i < mh::kNumPrograms; ++i) {
        const char * src = mh::kProgramSources[i];
        if (src == nullptr || src[0] == '\0') {
            continue;
        }
        const char * opts = mh::kProgramOptions[i];
        st.programs[i] = build_program_from_source(
            backend_ctx->context, backend_ctx->device, src, (opts && opts[0]) ? opts : "");
    }

    for (int i = 0; i < mh::kNumKernels; ++i) {
        const int program_id = mh::kKernelProgramId[i];
        st.kernels[i] = clCreateKernel(st.programs[program_id], "main_function", &status);
        if (!mh::mldrift_cl_check(status, "create kernel")) {
            GGML_LOG_WARN("ggml_opencl: mldrift(h30) create kernel %d failed (%d)\n", i, (int) status);
            mldrift_attn_release_h30(backend_ctx);
            return false;
        }
    }

    if (!mh::set_kernel_args(st.kernels.data(), st.buffers.data(), st.images.data())) {
        mldrift_attn_release_h30(backend_ctx);
        return false;
    }

    st.initialized = true;
    return true;
}

static void load_cl_kernels(ggml_backend_opencl_context *backend_ctx, ggml_cl_version opencl_c_version) {
    cl_int err;

    // compiler options for general kernels
    auto opencl_c_std =
        std::string("CL") + std::to_string(opencl_c_version.major) + "." + std::to_string(opencl_c_version.minor);
    const std::string compile_opts_fast = std::string("-cl-std=") + opencl_c_std +
                                          " -cl-mad-enable -cl-unsafe-math-optimizations"
                                          " -cl-finite-math-only -cl-fast-relaxed-math";
    const std::string compile_opts_precise = std::string("-cl-std=") + opencl_c_std +
                                             " -cl-mad-enable";
    const bool use_precise_math = getenv("GGML_OPENCL_PRECISE_MATH") != nullptr;
    if (use_precise_math) {
        GGML_LOG_INFO("ggml_opencl: precise math enabled for rope/softmax/rms_norm\n");
    }
    const bool use_precise_all_math = getenv("GGML_OPENCL_PRECISE_ALL") != nullptr;
    if (use_precise_all_math) {
        GGML_LOG_INFO("ggml_opencl: precise math enabled for all general kernels\n");
    }
    const std::string &compile_opts = use_precise_all_math ? compile_opts_precise : compile_opts_fast;

    GGML_LOG_INFO("ggml_opencl: loading OpenCL kernels");

    // add
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "add.cl.h"
        };
#else
        const std::string kernel_src = read_file("add.cl");
#endif
        backend_ctx->program_add =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_add         = clCreateKernel(backend_ctx->program_add, "kernel_add", &err), err));
        CL_CHECK((backend_ctx->kernel_add_row     = clCreateKernel(backend_ctx->program_add, "kernel_add_row", &err), err));
        CL_CHECK((backend_ctx->kernel_add_f16     = clCreateKernel(backend_ctx->program_add, "kernel_add_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_add_row_f16 = clCreateKernel(backend_ctx->program_add, "kernel_add_row_f16", &err), err));
        GGML_LOG_CONT(".");
    }

    // add_id
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "add_id.cl.h"
        };
#else
        const std::string kernel_src = read_file("add_id.cl");
#endif
        backend_ctx->program_add_id =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_add_id = clCreateKernel(backend_ctx->program_add_id, "kernel_add_id", &err), err));
        GGML_LOG_CONT(".");
    }

    // tri
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "tri.cl.h"
        };
#else
        const std::string kernel_src = read_file("tri.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_tri = clCreateKernel(prog, "kernel_tri_f32", &err), err));
        GGML_LOG_CONT(".");

        CL_CHECK(clReleaseProgram(prog));
    }

    // fill
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "fill.cl.h"
        };
#else
        const std::string kernel_src = read_file("fill.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_fill = clCreateKernel(prog, "kernel_fill_f32", &err), err));
        GGML_LOG_CONT(".");

        CL_CHECK(clReleaseProgram(prog));
    }

    // clamp
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "clamp.cl.h"
        };
#else
        const std::string kernel_src = read_file("clamp.cl");
#endif
        backend_ctx->program_clamp =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_clamp = clCreateKernel(backend_ctx->program_clamp, "kernel_clamp", &err), err));
        GGML_LOG_CONT(".");
    }

    // cpy
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "cpy.cl.h"
        };
#else
        const std::string kernel_src = read_file("cpy.cl");
#endif
        backend_ctx->program_cpy =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_cpy_f16_f16 = clCreateKernel(backend_ctx->program_cpy, "kernel_cpy_f16_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_cpy_f16_f32 = clCreateKernel(backend_ctx->program_cpy, "kernel_cpy_f16_f32", &err), err));
        CL_CHECK((backend_ctx->kernel_cpy_f32_f16 = clCreateKernel(backend_ctx->program_cpy, "kernel_cpy_f32_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_cpy_f32_f32 = clCreateKernel(backend_ctx->program_cpy, "kernel_cpy_f32_f32", &err), err));
        CL_CHECK((backend_ctx->kernel_cpy_f16_f16_1d = clCreateKernel(backend_ctx->program_cpy, "kernel_cpy_f16_f16_1d", &err), err));
        CL_CHECK((backend_ctx->kernel_cpy_f32_f32_1d = clCreateKernel(backend_ctx->program_cpy, "kernel_cpy_f32_f32_1d", &err), err));
        GGML_LOG_CONT(".");
    }

    // cvt
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "cvt.cl.h"
        };
#else
        const std::string kernel_src = read_file("cvt.cl");
#endif
        backend_ctx->program_cvt =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_convert_block_q4_0_noshuffle = clCreateKernel(backend_ctx->program_cvt, "kernel_convert_block_q4_0_noshuffle", &err), err));
        CL_CHECK((backend_ctx->kernel_restore_block_q4_0_noshuffle = clCreateKernel(backend_ctx->program_cvt, "kernel_restore_block_q4_0_noshuffle", &err), err));
        CL_CHECK((backend_ctx->kernel_convert_block_q4_0  = clCreateKernel(backend_ctx->program_cvt, "kernel_convert_block_q4_0", &err), err));
        CL_CHECK((backend_ctx->kernel_restore_block_q4_0  = clCreateKernel(backend_ctx->program_cvt, "kernel_restore_block_q4_0", &err), err));
        CL_CHECK((backend_ctx->kernel_convert_block_mxfp4 = clCreateKernel(backend_ctx->program_cvt, "kernel_convert_block_mxfp4", &err), err));
        CL_CHECK((backend_ctx->kernel_convert_block_mxfp4_trans = clCreateKernel(backend_ctx->program_cvt, "kernel_convert_block_mxfp4_trans", &err), err));
        CL_CHECK((backend_ctx->kernel_restore_block_mxfp4_trans = clCreateKernel(backend_ctx->program_cvt, "kernel_restore_block_mxfp4_trans", &err), err));
        CL_CHECK((backend_ctx->kernel_restore_block_mxfp4 = clCreateKernel(backend_ctx->program_cvt, "kernel_restore_block_mxfp4", &err), err));
        CL_CHECK((backend_ctx->kernel_convert_block_q8_0  = clCreateKernel(backend_ctx->program_cvt, "kernel_convert_block_q8_0", &err), err));
        CL_CHECK((backend_ctx->kernel_restore_block_q8_0  = clCreateKernel(backend_ctx->program_cvt, "kernel_restore_block_q8_0", &err), err));
        CL_CHECK((backend_ctx->kernel_convert_block_q6_K  = clCreateKernel(backend_ctx->program_cvt, "kernel_convert_block_q6_K", &err), err));
        CL_CHECK((backend_ctx->kernel_restore_block_q6_K  = clCreateKernel(backend_ctx->program_cvt, "kernel_restore_block_q6_K", &err), err));
        GGML_LOG_CONT(".");
    }

    // convert_f16_f32_linear
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "convert_f16_f32_linear.cl.h"
        };
#else
        const std::string kernel_src = read_file("convert_f16_f32_linear.cl");
#endif
        backend_ctx->program_convert_f16_f32_linear =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_f16_to_f32_linear = clCreateKernel(backend_ctx->program_convert_f16_f32_linear, "kernel_f16_to_f32_linear", &err), err));
        CL_CHECK((backend_ctx->kernel_f16_to_f32_nhd_crop = clCreateKernel(backend_ctx->program_convert_f16_f32_linear, "kernel_f16_to_f32_nhd_crop", &err), err));
        CL_CHECK((backend_ctx->kernel_f16_to_f32_nhd_keep_head_tail = clCreateKernel(backend_ctx->program_convert_f16_f32_linear, "kernel_f16_to_f32_nhd_keep_head_tail", &err), err));
        CL_CHECK((backend_ctx->kernel_f32_copy_scale_linear = clCreateKernel(backend_ctx->program_convert_f16_f32_linear, "kernel_f32_copy_scale_linear", &err), err));
        CL_CHECK((backend_ctx->kernel_f32_reorder_hqd_to_qhd = clCreateKernel(backend_ctx->program_convert_f16_f32_linear, "kernel_f32_reorder_hqd_to_qhd", &err), err));
        GGML_LOG_CONT(".");
    }

    // diag_mask_inf
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "diag_mask_inf.cl.h"
        };
#else
        const std::string kernel_src = read_file("diag_mask_inf.cl");
#endif
        backend_ctx->program_diag_mask_inf =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_diag_mask_inf_8 = clCreateKernel(backend_ctx->program_diag_mask_inf, "kernel_diag_mask_inf_8", &err), err));
        CL_CHECK((backend_ctx->kernel_diag_mask_inf   = clCreateKernel(backend_ctx->program_diag_mask_inf, "kernel_diag_mask_inf", &err), err));
        GGML_LOG_CONT(".");
    }

    // gelu
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gelu.cl.h"
        };
#else
        const std::string kernel_src = read_file("gelu.cl");
#endif
        backend_ctx->program_gelu =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_gelu         = clCreateKernel(backend_ctx->program_gelu, "kernel_gelu", &err), err));
        CL_CHECK((backend_ctx->kernel_gelu_4       = clCreateKernel(backend_ctx->program_gelu, "kernel_gelu_4", &err), err));
        CL_CHECK((backend_ctx->kernel_gelu_erf     = clCreateKernel(backend_ctx->program_gelu, "kernel_gelu_erf", &err), err));
        CL_CHECK((backend_ctx->kernel_gelu_erf_4   = clCreateKernel(backend_ctx->program_gelu, "kernel_gelu_erf_4", &err), err));
        CL_CHECK((backend_ctx->kernel_gelu_quick   = clCreateKernel(backend_ctx->program_gelu, "kernel_gelu_quick", &err), err));
        CL_CHECK((backend_ctx->kernel_gelu_quick_4 = clCreateKernel(backend_ctx->program_gelu, "kernel_gelu_quick_4", &err), err));
        GGML_LOG_CONT(".");
    }

    // glu
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "glu.cl.h"
        };
#else
        const std::string kernel_src = read_file("glu.cl");
#endif
        backend_ctx->program_glu =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_geglu           = clCreateKernel(backend_ctx->program_glu, "kernel_geglu", &err), err));
        CL_CHECK((backend_ctx->kernel_reglu           = clCreateKernel(backend_ctx->program_glu, "kernel_reglu", &err), err));
        CL_CHECK((backend_ctx->kernel_swiglu          = clCreateKernel(backend_ctx->program_glu, "kernel_swiglu", &err), err));
        CL_CHECK((backend_ctx->kernel_swiglu_oai      = clCreateKernel(backend_ctx->program_glu, "kernel_swiglu_oai", &err), err));
        CL_CHECK((backend_ctx->kernel_geglu_erf       = clCreateKernel(backend_ctx->program_glu, "kernel_geglu_erf", &err), err));
        CL_CHECK((backend_ctx->kernel_geglu_quick     = clCreateKernel(backend_ctx->program_glu, "kernel_geglu_quick", &err), err));
        CL_CHECK((backend_ctx->kernel_geglu_f16       = clCreateKernel(backend_ctx->program_glu, "kernel_geglu_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_reglu_f16       = clCreateKernel(backend_ctx->program_glu, "kernel_reglu_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_swiglu_f16      = clCreateKernel(backend_ctx->program_glu, "kernel_swiglu_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_geglu_erf_f16   = clCreateKernel(backend_ctx->program_glu, "kernel_geglu_erf_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_geglu_quick_f16 = clCreateKernel(backend_ctx->program_glu, "kernel_geglu_quick_f16", &err), err));
        GGML_LOG_CONT(".");
    }

    // get_rows
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "get_rows.cl.h"
        };
#else
        const std::string kernel_src = read_file("get_rows.cl");
#endif
        backend_ctx->program_get_rows =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_get_rows_f32      = clCreateKernel(backend_ctx->program_get_rows, "kernel_get_rows_f32", &err), err));
        CL_CHECK((backend_ctx->kernel_get_rows_f16      = clCreateKernel(backend_ctx->program_get_rows, "kernel_get_rows_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_get_rows_q4_0         = clCreateKernel(backend_ctx->program_get_rows, "kernel_get_rows_q4_0", &err), err));
        CL_CHECK((backend_ctx->kernel_get_rows_q4_0_soa     = clCreateKernel(backend_ctx->program_get_rows, "kernel_get_rows_q4_0_soa", &err), err));
        CL_CHECK((backend_ctx->kernel_get_rows_q4_0_soa_scalar = clCreateKernel(backend_ctx->program_get_rows, "kernel_get_rows_q4_0_soa_scalar", &err), err));
        GGML_LOG_CONT(".");
    }

    // solve_tri_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "solve_tri.cl.h"
        };
#else
        const std::string kernel_src = read_file("solve_tri.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_solve_tri_f32 = clCreateKernel(prog, "kernel_solve_tri_f32", &err), err));
        GGML_LOG_CONT(".");
        CL_CHECK(clReleaseProgram(prog));
    }

    // im2col_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "im2col_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("im2col_f32.cl");
#endif
        backend_ctx->program_im2col_f32 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_im2col_f32 = clCreateKernel(backend_ctx->program_im2col_f32, "kernel_im2col_f32", &err), err));
        GGML_LOG_CONT(".");
    }

    // im2col_f16
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "im2col_f16.cl.h"
        };
#else
        const std::string kernel_src = read_file("im2col_f16.cl");
#endif
        backend_ctx->program_im2col_f16 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_im2col_f16 = clCreateKernel(backend_ctx->program_im2col_f16, "kernel_im2col_f16", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q4_0_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q4_0_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q4_0_f32.cl");
#endif
        backend_ctx->program_mul_mv_q4_0_f32 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mat_q4_0_f32 = clCreateKernel(backend_ctx->program_mul_mv_q4_0_f32, "kernel_mul_mat_q4_0_f32", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q4_0_f32_v
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q4_0_f32_v.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q4_0_f32_v.cl");
#endif
        backend_ctx->program_mul_mv_q4_0_f32_v =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mat_q4_0_f32_v = clCreateKernel(backend_ctx->program_mul_mv_q4_0_f32_v, "kernel_mul_mat_q4_0_f32_v", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q4_0_f32_8x_flat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q4_0_f32_8x_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q4_0_f32_8x_flat.cl");
#endif
        backend_ctx->program_mul_mv_q4_0_f32_8x_flat =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mat_q4_0_f32_8x_flat = clCreateKernel(backend_ctx->program_mul_mv_q4_0_f32_8x_flat, "kernel_mul_mat_q4_0_f32_8x_flat", &err), err));
        CL_CHECK((backend_ctx->kernel_mul_mat_q4_0_f32_ref = clCreateKernel(backend_ctx->program_mul_mv_q4_0_f32_8x_flat, "kernel_mul_mat_q4_0_f32_ref", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q4_0_f32_1d_8x_flat
    // This kernel does not compiler on Adreno cl compiler 38.01. Skip it for
    // those compiler versions since it is anyway not used for Adreno.
    if (backend_ctx->gpu_family != ADRENO ||
        backend_ctx->adreno_cl_compiler_version.newer_than_or_same(E031, 38, 11, 0) ||
        backend_ctx->adreno_cl_compiler_version.type == DX) {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q4_0_f32_1d_8x_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q4_0_f32_1d_8x_flat.cl");
#endif
        backend_ctx->program_mul_mv_q4_0_f32_1d_8x_flat =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mat_q4_0_f32_1d_8x_flat = clCreateKernel(backend_ctx->program_mul_mv_q4_0_f32_1d_8x_flat, "kernel_mul_mat_q4_0_f32_1d_8x_flat", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q4_0_f32_1d_16x_flat
    // This kernel does not compiler on Adreno cl compiler 38.01. Skip it for
    // those compiler versions since it is anyway not used for Adreno.
    if (backend_ctx->gpu_family != ADRENO ||
        backend_ctx->adreno_cl_compiler_version.newer_than_or_same(E031, 38, 11, 0) ||
    backend_ctx->adreno_cl_compiler_version.type == DX) {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q4_0_f32_1d_16x_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q4_0_f32_1d_16x_flat.cl");
#endif
        backend_ctx->program_mul_mv_q4_0_f32_1d_16x_flat =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mat_q4_0_f32_1d_16x_flat = clCreateKernel(backend_ctx->program_mul_mv_q4_0_f32_1d_16x_flat, "kernel_mul_mat_q4_0_f32_1d_16x_flat", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q6_k_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q6_k_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q6_k_f32.cl");
#endif
        backend_ctx->program_mul_mv_q6_K =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mv_q6_K_f32 = clCreateKernel(backend_ctx->program_mul_mv_q6_K, "kernel_mul_mv_q6_K_f32", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q6_k_f32_flat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q6_k_f32_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q6_k_f32_flat.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mv_q6_K_f32_flat = clCreateKernel(prog, "kernel_mul_mv_q6_K_f32_flat", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q8_0_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q8_0_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q8_0_f32.cl");
#endif
        backend_ctx->program_mul_mv_q8_0_f32 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mv_q8_0_f32 = clCreateKernel(backend_ctx->program_mul_mv_q8_0_f32, "kernel_mul_mv_q8_0_f32", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q8_0_f32_flat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q8_0_f32_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q8_0_f32_flat.cl");
#endif
        backend_ctx->program_mul_mv_q8_0_f32_flat =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mv_q8_0_f32_flat = clCreateKernel(backend_ctx->program_mul_mv_q8_0_f32_flat, "kernel_mul_mv_q8_0_f32_flat", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_mxfp4_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_mxfp4_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_mxfp4_f32.cl");
#endif
        backend_ctx->program_mul_mv_mxfp4_f32 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mv_mxfp4_f32 = clCreateKernel(backend_ctx->program_mul_mv_mxfp4_f32, "kernel_mul_mv_mxfp4_f32", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_mxfp4_f32_flat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_mxfp4_f32_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_mxfp4_f32_flat.cl");
#endif
        backend_ctx->program_mul_mv_mxfp4_f32_flat =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mv_mxfp4_f32_flat = clCreateKernel(backend_ctx->program_mul_mv_mxfp4_f32_flat, "kernel_mul_mv_mxfp4_f32_flat", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_f16_f16
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_f16_f16.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_f16_f16.cl");
#endif
        backend_ctx->program_mul_mv_f16_f16 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mat_f16_f16 = clCreateKernel(backend_ctx->program_mul_mv_f16_f16, "kernel_mul_mat_f16_f16", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_f16_f32_1row
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_f16_f32_1row.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_f16_f32_1row.cl");
#endif
        backend_ctx->program_mul_mv_f16_f32_1row =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mat_f16_f32_1row = clCreateKernel(backend_ctx->program_mul_mv_f16_f32_1row, "kernel_mul_mat_f16_f32_1row", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_f16_f32_l4
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_f16_f32_l4.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_f16_f32_l4.cl");
#endif
        backend_ctx->program_mul_mv_f16_f32_l4 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mat_f16_f32_l4   = clCreateKernel(backend_ctx->program_mul_mv_f16_f32_l4, "kernel_mul_mat_f16_f32_l4", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_f16_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_f16_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_f16_f32.cl");
#endif
        backend_ctx->program_mul_mv_f16_f32 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mat_f16_f32 = clCreateKernel(backend_ctx->program_mul_mv_f16_f32, "kernel_mul_mat_f16_f32", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_f32_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_f32_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_f32_f32.cl");
#endif
        backend_ctx->program_mul_mv_f32_f32 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mat_f32_f32 = clCreateKernel(backend_ctx->program_mul_mv_f32_f32, "kernel_mul_mat_f32_f32", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mat_f16_f32_tiled
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mat_f16_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mat_f16_f32.cl");
#endif
        backend_ctx->program_mul_mat_f16_f32_tiled =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mat_f16_f32_tiled = clCreateKernel(backend_ctx->program_mul_mat_f16_f32_tiled, "mul_mat_f16_f32", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mm_f32_f32_l4_lm
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mm_f32_f32_l4_lm.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mm_f32_f32_l4_lm.cl");
#endif
        backend_ctx->program_mul_mm_f32_f32_l4_lm =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mm_f32_f32_l4_lm = clCreateKernel(backend_ctx->program_mul_mm_f32_f32_l4_lm, "kernel_mul_mm_f32_f32_l4_lm", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mm_f16_f32_l4_lm
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mm_f16_f32_l4_lm.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mm_f16_f32_l4_lm.cl");
#endif
        backend_ctx->program_mul_mm_f16_f32_l4_lm =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mm_f16_f32_l4_lm = clCreateKernel(backend_ctx->program_mul_mm_f16_f32_l4_lm, "kernel_mul_mm_f16_f32_l4_lm", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mm_q8_0_f32_l4_lm
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mm_q8_0_f32_l4_lm.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mm_q8_0_f32_l4_lm.cl");
#endif
        backend_ctx->program_mul_mm_q8_0_f32_l4_lm =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mm_q8_0_f32_l4_lm = clCreateKernel(backend_ctx->program_mul_mm_q8_0_f32_l4_lm, "kernel_mul_mm_q8_0_f32_l4_lm", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mm_f16_f32_kq_kqv
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mm_f16_f32_kq_kqv.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mm_f16_f32_kq_kqv.cl");
#endif
        backend_ctx->program_mul_mm_f16_f32_kqv =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts+" -DKQV ");
        backend_ctx->program_mul_mm_f16_f32_kq =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mm_f16_f32_kqv = clCreateKernel(backend_ctx->program_mul_mm_f16_f32_kqv, "mul_mm_f16_f32_kqv", &err), err));
        CL_CHECK((backend_ctx->kernel_mul_mm_f16_f32_kq = clCreateKernel(backend_ctx->program_mul_mm_f16_f32_kq, "mul_mm_f16_f32_kq", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul.cl");
#endif
        backend_ctx->program_mul =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul         = clCreateKernel(backend_ctx->program_mul, "kernel_mul", &err), err));
        CL_CHECK((backend_ctx->kernel_mul_row     = clCreateKernel(backend_ctx->program_mul, "kernel_mul_row", &err), err));
        CL_CHECK((backend_ctx->kernel_mul_f16     = clCreateKernel(backend_ctx->program_mul, "kernel_mul_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_mul_row_f16 = clCreateKernel(backend_ctx->program_mul, "kernel_mul_row_f16", &err), err));
        GGML_LOG_CONT(".");
    }

    // norm
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "norm.cl.h"
        };
#else
        const std::string kernel_src = read_file("norm.cl");
#endif
        backend_ctx->program_norm =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_norm         = clCreateKernel(backend_ctx->program_norm, "kernel_norm", &err), err));
        CL_CHECK((backend_ctx->kernel_norm_mul_add = clCreateKernel(backend_ctx->program_norm, "kernel_norm_mul_add", &err), err));
        GGML_LOG_CONT(".");
    }

    // relu
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "relu.cl.h"
        };
#else
        const std::string kernel_src = read_file("relu.cl");
#endif
        backend_ctx->program_relu =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_relu = clCreateKernel(backend_ctx->program_relu, "kernel_relu", &err), err));
        GGML_LOG_CONT(".");
    }

    // rms_norm
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "rms_norm.cl.h"
        };
#else
        const std::string kernel_src = read_file("rms_norm.cl");
#endif
        backend_ctx->program_rms_norm =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(),
                                      use_precise_math ? compile_opts_precise : compile_opts_fast);

        CL_CHECK((backend_ctx->kernel_rms_norm     = clCreateKernel(backend_ctx->program_rms_norm, "kernel_rms_norm", &err), err));
        CL_CHECK((backend_ctx->kernel_rms_norm_mul = clCreateKernel(backend_ctx->program_rms_norm, "kernel_rms_norm_mul", &err), err));
        GGML_LOG_CONT(".");
    }

    // rope
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "rope.cl.h"
        };
#else
        const std::string kernel_src = read_file("rope.cl");
#endif
        backend_ctx->program_rope =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(),
                                      use_precise_math ? compile_opts_precise : compile_opts_fast);

        CL_CHECK((backend_ctx->kernel_rope_norm_f32   = clCreateKernel(backend_ctx->program_rope, "kernel_rope_norm_f32", &err), err));
        CL_CHECK((backend_ctx->kernel_rope_norm_f16   = clCreateKernel(backend_ctx->program_rope, "kernel_rope_norm_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_rope_neox_f32   = clCreateKernel(backend_ctx->program_rope, "kernel_rope_neox_f32", &err), err));
        CL_CHECK((backend_ctx->kernel_rope_neox_f16   = clCreateKernel(backend_ctx->program_rope, "kernel_rope_neox_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_rope_multi_f32  = clCreateKernel(backend_ctx->program_rope, "kernel_rope_multi_f32", &err), err));
        CL_CHECK((backend_ctx->kernel_rope_multi_f16  = clCreateKernel(backend_ctx->program_rope, "kernel_rope_multi_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_rope_vision_f32 = clCreateKernel(backend_ctx->program_rope, "kernel_rope_vision_f32", &err), err));
        CL_CHECK((backend_ctx->kernel_rope_vision_f16 = clCreateKernel(backend_ctx->program_rope, "kernel_rope_vision_f16", &err), err));
        GGML_LOG_CONT(".");
    }

    // scale
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "scale.cl.h"
        };
#else
        const std::string kernel_src = read_file("scale.cl");
#endif
        backend_ctx->program_scale =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_scale = clCreateKernel(backend_ctx->program_scale, "kernel_scale", &err), err));
        GGML_LOG_CONT(".");
    }

    // silu
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "silu.cl.h"
        };
#else
        const std::string kernel_src = read_file("silu.cl");
#endif
        backend_ctx->program_silu =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_silu   = clCreateKernel(backend_ctx->program_silu, "kernel_silu", &err), err));
        CL_CHECK((backend_ctx->kernel_silu_4 = clCreateKernel(backend_ctx->program_silu, "kernel_silu_4", &err), err));
        GGML_LOG_CONT(".");
    }

    // softmax_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "softmax_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("softmax_f32.cl");
#endif
        backend_ctx->program_softmax_f32 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(),
                                      use_precise_math ? compile_opts_precise : compile_opts_fast);

        CL_CHECK((backend_ctx->kernel_soft_max = clCreateKernel(backend_ctx->program_softmax_f32, "kernel_soft_max", &err), err));
        GGML_LOG_CONT(".");
    }

    // softmax_f16
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "softmax_f16.cl.h"
        };
#else
        const std::string kernel_src = read_file("softmax_f16.cl");
#endif
        backend_ctx->program_softmax_f16 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(),
                                      use_precise_math ? compile_opts_precise : compile_opts_fast);

        CL_CHECK((backend_ctx->kernel_soft_max_f16 = clCreateKernel(backend_ctx->program_softmax_f16, "kernel_soft_max_f16", &err), err));
        GGML_LOG_CONT(".");
    }

    // softmax_4_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "softmax_4_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("softmax_4_f32.cl");
#endif
        backend_ctx->program_softmax_4_f32 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(),
                                      use_precise_math ? compile_opts_precise : compile_opts_fast);

        CL_CHECK((backend_ctx->kernel_soft_max_4 = clCreateKernel(backend_ctx->program_softmax_4_f32, "kernel_soft_max_4", &err), err));
        GGML_LOG_CONT(".");
    }

    // softmax_4_f16
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "softmax_4_f16.cl.h"
        };
#else
        const std::string kernel_src = read_file("softmax_4_f16.cl");
#endif
        backend_ctx->program_softmax_4_f16 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(),
                                      use_precise_math ? compile_opts_precise : compile_opts_fast);

        CL_CHECK((backend_ctx->kernel_soft_max_4_f16 = clCreateKernel(backend_ctx->program_softmax_4_f16, "kernel_soft_max_4_f16", &err), err));
        GGML_LOG_CONT(".");
    }

    // flash_attn
    {
        #ifdef GGML_OPENCL_EMBED_KERNELS
                const std::string kernel_src_f16 {
                    #include "flash_attn_f16.cl.h"
                };
                const std::string kernel_src_f32 {
                    #include "flash_attn_f32.cl.h"
                };
                const std::string kernel_src_f32_f16 {
                    #include "flash_attn_f32_f16.cl.h"
                };
        #else
                const std::string kernel_src_f16 = read_file("flash_attn_f16.cl");
                const std::string kernel_src_f32 = read_file("flash_attn_f32.cl");
                const std::string kernel_src_f32_f16 = read_file("flash_attn_f32_f16.cl");
        #endif

        if (!kernel_src_f16.empty() && !kernel_src_f32.empty() && !kernel_src_f32_f16.empty()) {
            const struct { int dk; int dv; int bm; int bn; } fa_dims[] = {
                { 40,  40, 32, 32}, { 64,  64, 64, 64}, { 80,  80, 64, 32}, { 96,  96, 64, 32},
                {112, 112, 32, 32}, {128, 128, 32, 32}, {192, 128, 16, 16},
                {192, 192, 16, 16}, {256, 256, 16, 16},
            };

            for (size_t i = 0; i < sizeof(fa_dims)/sizeof(fa_dims[0]); ++i) {
                const int dk = fa_dims[i].dk;
                const int dv = fa_dims[i].dv;
                const int bm = fa_dims[i].bm;
                const int bn = fa_dims[i].bn;
                std::string OPTS = compile_opts +
                    " -D DK=" + std::to_string(dk) +
                    " -D DV=" + std::to_string(dv) +
                    " -D BLOCK_M=" + std::to_string(bm) +
                    " -D BLOCK_N=" + std::to_string(bn);

                cl_program prog_f16 = build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src_f16.c_str(), OPTS);
                cl_kernel k_f16, k_f16_q1;
                CL_CHECK((k_f16 = clCreateKernel(prog_f16, "flash_attn_f16", &err), err));
                CL_CHECK((k_f16_q1 = clCreateKernel(prog_f16, "flash_attn_f16_q1", &err), err));
                backend_ctx->kernels_flash_attn_f16[{dk, dv}] = k_f16;
                backend_ctx->kernels_flash_attn_f16_q1[{dk, dv}] = k_f16_q1;
                CL_CHECK(clReleaseProgram(prog_f16));

                cl_program prog_f32 = build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src_f32.c_str(), OPTS);
                cl_kernel k_f32, k_f32_q1;
                CL_CHECK((k_f32 = clCreateKernel(prog_f32, "flash_attn_f32", &err), err));
                CL_CHECK((k_f32_q1 = clCreateKernel(prog_f32, "flash_attn_f32_q1", &err), err));
                backend_ctx->kernels_flash_attn_f32[{dk, dv}] = k_f32;
                backend_ctx->kernels_flash_attn_f32_q1[{dk, dv}] = k_f32_q1;
                CL_CHECK(clReleaseProgram(prog_f32));

                cl_program prog_f32_f16 = build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src_f32_f16.c_str(), OPTS);
                cl_kernel k_f32_f16, k_f32_f16_q1;
                CL_CHECK((k_f32_f16 = clCreateKernel(prog_f32_f16, "flash_attn_f32_f16", &err), err));
                CL_CHECK((k_f32_f16_q1 = clCreateKernel(prog_f32_f16, "flash_attn_f32_f16_q1", &err), err));
                backend_ctx->kernels_flash_attn_f32_f16[{dk, dv}] = k_f32_f16;
                backend_ctx->kernels_flash_attn_f32_f16_q1[{dk, dv}] = k_f32_f16_q1;
                CL_CHECK(clReleaseProgram(prog_f32_f16));

                backend_ctx->kernels_flash_attn_bm[{dk, dv}] = bm;
                backend_ctx->kernels_flash_attn_bn[{dk, dv}] = bn;
            }
            GGML_LOG_CONT(".");
        }
    }

    // argsort
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "argsort.cl.h"
        };
#else
        const std::string kernel_src = read_file("argsort.cl");
#endif
        backend_ctx->program_argsort_f32_i32 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_argsort_f32_i32 = clCreateKernel(backend_ctx->program_argsort_f32_i32, "kernel_argsort_f32_i32", &err), err));
        GGML_LOG_CONT(".");
    }

    // div
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "div.cl.h"
        };
#else
        const std::string kernel_src = read_file("div.cl");
#endif
        std::string compile_opts = std::string("-cl-std=") + opencl_c_std +
                               " -cl-mad-enable -cl-finite-math-only ";

        backend_ctx->program_div =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_div         = clCreateKernel(backend_ctx->program_div, "kernel_div", &err), err));
        CL_CHECK((backend_ctx->kernel_div_row     = clCreateKernel(backend_ctx->program_div, "kernel_div_row", &err), err));
        CL_CHECK((backend_ctx->kernel_div_f16     = clCreateKernel(backend_ctx->program_div, "kernel_div_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_div_row_f16 = clCreateKernel(backend_ctx->program_div, "kernel_div_row_f16", &err), err));
        GGML_LOG_CONT(".");
    }

    // sqr
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "sqr.cl.h"
        };
#else
        const std::string kernel_src = read_file("sqr.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_sqr_cont_f32     = clCreateKernel(prog, "kernel_sqr_cont_f32", &err), err));
        CL_CHECK((backend_ctx->kernel_sqr_cont_f32_4   = clCreateKernel(prog, "kernel_sqr_cont_f32_4", &err), err));
        CL_CHECK((backend_ctx->kernel_sqr_cont_f16     = clCreateKernel(prog, "kernel_sqr_cont_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_sqr_cont_f16_4   = clCreateKernel(prog, "kernel_sqr_cont_f16_4", &err), err));

        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // sqrt
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "sqrt.cl.h"
        };
#else
        const std::string kernel_src = read_file("sqrt.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_sqrt_cont_f32     = clCreateKernel(prog, "kernel_sqrt_cont_f32", &err), err));
        CL_CHECK((backend_ctx->kernel_sqrt_cont_f32_4   = clCreateKernel(prog, "kernel_sqrt_cont_f32_4", &err), err));
        CL_CHECK((backend_ctx->kernel_sqrt_cont_f16     = clCreateKernel(prog, "kernel_sqrt_cont_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_sqrt_cont_f16_4   = clCreateKernel(prog, "kernel_sqrt_cont_f16_4", &err), err));

        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mean
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mean.cl.h"
        };
#else
        const std::string kernel_src = read_file("mean.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mean_f32 = clCreateKernel(prog, "kernel_mean_f32", &err), err));

        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // sub
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "sub.cl.h"
        };
#else
        const std::string kernel_src = read_file("sub.cl");
#endif
        backend_ctx->program_sub =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_sub         = clCreateKernel(backend_ctx->program_sub, "kernel_sub", &err), err));
        CL_CHECK((backend_ctx->kernel_sub_row     = clCreateKernel(backend_ctx->program_sub, "kernel_sub_row", &err), err));
        CL_CHECK((backend_ctx->kernel_sub_f16     = clCreateKernel(backend_ctx->program_sub, "kernel_sub_f16", &err), err));
        CL_CHECK((backend_ctx->kernel_sub_row_f16 = clCreateKernel(backend_ctx->program_sub, "kernel_sub_row_f16", &err), err));
        GGML_LOG_CONT(".");
    }

    // sum_rows
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "sum_rows.cl.h"
        };
#else
        const std::string kernel_src = read_file("sum_rows.cl");
#endif
        backend_ctx->program_sum_rows_f32 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_sum_rows_f32 = clCreateKernel(backend_ctx->program_sum_rows_f32, "kernel_sum_rows_f32", &err), err));
        GGML_LOG_CONT(".");
    }

    // sigmoid
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "sigmoid.cl.h"
        };
#else
        const std::string kernel_src = read_file("sigmoid.cl");
#endif
        backend_ctx->program_sigmoid =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_sigmoid_f32 = clCreateKernel(backend_ctx->program_sigmoid, "kernel_sigmoid_f32", &err), err));
        CL_CHECK((backend_ctx->kernel_sigmoid_f16 = clCreateKernel(backend_ctx->program_sigmoid, "kernel_sigmoid_f16", &err), err));
        GGML_LOG_CONT(".");
    }

    // group_norm
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "group_norm.cl.h"
        };
#else
        const std::string kernel_src = read_file("group_norm.cl");
#endif
        backend_ctx->program_group_norm =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_group_norm         = clCreateKernel(backend_ctx->program_group_norm, "kernel_group_norm", &err), err));
        CL_CHECK((backend_ctx->kernel_group_norm_mul_add = clCreateKernel(backend_ctx->program_group_norm, "kernel_group_norm_mul_add", &err), err));
        GGML_LOG_CONT(".");
    }

    // repeat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "repeat.cl.h"
        };
#else
        const std::string kernel_src = read_file("repeat.cl");
#endif
        if (!kernel_src.empty()) {
            backend_ctx->program_repeat =
                build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);
            CL_CHECK((backend_ctx->kernel_repeat = clCreateKernel(backend_ctx->program_repeat, "kernel_repeat", &err), err));
            GGML_LOG_CONT(".");
        } else {
            GGML_LOG_WARN("ggml_opencl: repeat kernel source not found or empty. Repeat operations will not be available.\n");
            backend_ctx->program_repeat = nullptr;
            backend_ctx->kernel_repeat = nullptr;
        }
    }

    // pad
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "pad.cl.h"
        };
#else
        const std::string kernel_src = read_file("pad.cl");
#endif
        if (!kernel_src.empty()) {
            backend_ctx->program_pad =
                build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);
            CL_CHECK((backend_ctx->kernel_pad = clCreateKernel(backend_ctx->program_pad, "kernel_pad", &err), err));
            GGML_LOG_CONT(".");
        } else {
            GGML_LOG_WARN("ggml_opencl: pad kernel source not found or empty. Pad operations will not be available.\n");
            backend_ctx->program_pad = nullptr;
            backend_ctx->kernel_pad = nullptr;
        }
    }

    // tanh
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "tanh.cl.h"
        };
#else
        const std::string kernel_src = read_file("tanh.cl");
#endif
        if (!kernel_src.empty()) {
            backend_ctx->program_tanh =
                build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);
            CL_CHECK((backend_ctx->kernel_tanh_f32_nd = clCreateKernel(backend_ctx->program_tanh, "kernel_tanh_f32_nd", &err), err));
            CL_CHECK((backend_ctx->kernel_tanh_f16_nd = clCreateKernel(backend_ctx->program_tanh, "kernel_tanh_f16_nd", &err), err));
            GGML_LOG_CONT(".");
        } else {
            GGML_LOG_WARN("ggml_opencl: tanh kernel source not found or empty. Tanh operation will not be available.\n");
            backend_ctx->program_tanh = nullptr;
            backend_ctx->kernel_tanh_f32_nd = nullptr;
            backend_ctx->kernel_tanh_f16_nd = nullptr;
        }
    }

    // expm1
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "expm1.cl.h"
        };
#else
        const std::string kernel_src = read_file("expm1.cl");
#endif
        cl_program prog;
        if (!kernel_src.empty()) {
            prog =
                build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);
            CL_CHECK((backend_ctx->kernel_expm1_f32_nd = clCreateKernel(prog, "kernel_expm1_f32_nd", &err), err));
            CL_CHECK((backend_ctx->kernel_expm1_f16_nd = clCreateKernel(prog, "kernel_expm1_f16_nd", &err), err));
            GGML_LOG_CONT(".");
        } else {
            GGML_LOG_WARN("ggml_opencl: expm1 kernel source not found or empty. Expm1 operation will not be available.\n");
            prog = nullptr;
            backend_ctx->kernel_expm1_f32_nd = nullptr;
            backend_ctx->kernel_expm1_f16_nd = nullptr;
        }
        CL_CHECK(clReleaseProgram(prog));
    }

    // softplus
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "softplus.cl.h"
        };
#else
        const std::string kernel_src = read_file("softplus.cl");
#endif
        cl_program prog;
        if (!kernel_src.empty()) {
            prog =
                build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);
            CL_CHECK((backend_ctx->kernel_softplus_f32_nd = clCreateKernel(prog, "kernel_softplus_f32_nd", &err), err));
            CL_CHECK((backend_ctx->kernel_softplus_f16_nd = clCreateKernel(prog, "kernel_softplus_f16_nd", &err), err));
            GGML_LOG_CONT(".");
        } else {
            GGML_LOG_WARN("ggml_opencl: softplus kernel source not found or empty. Softplus operation will not be available.\n");
            prog = nullptr;
            backend_ctx->kernel_softplus_f32_nd = nullptr;
            backend_ctx->kernel_softplus_f16_nd = nullptr;
        }
        CL_CHECK(clReleaseProgram(prog));
    }

    // upscale
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "upscale.cl.h"
        };
#else
        const std::string kernel_src = read_file("upscale.cl");
#endif
        if (!kernel_src.empty()) {
            backend_ctx->program_upscale =
                build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);
            CL_CHECK((backend_ctx->kernel_upscale = clCreateKernel(backend_ctx->program_upscale, "kernel_upscale", &err), err));
            if (backend_ctx->program_upscale) {
                 cl_int err_bilinear;
                 backend_ctx->kernel_upscale_bilinear = clCreateKernel(backend_ctx->program_upscale, "kernel_upscale_bilinear", &err_bilinear);
                 if (err_bilinear != CL_SUCCESS) {
                    GGML_LOG_WARN("ggml_opencl: kernel_upscale_bilinear not found in upscale.cl. Bilinear upscale will not be available. Error: %d\n", err_bilinear);
                    backend_ctx->kernel_upscale_bilinear = nullptr;
                 }
            } else {
                backend_ctx->kernel_upscale_bilinear = nullptr;
            }
            GGML_LOG_CONT(".");
        } else {
            GGML_LOG_WARN("ggml_opencl: upscale kernel source not found or empty. Upscale operations will not be available.\n");
            backend_ctx->program_upscale = nullptr;
            backend_ctx->kernel_upscale = nullptr;
            backend_ctx->kernel_upscale_bilinear = nullptr;
        }
    }

    // concat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "concat.cl.h"
        };
#else

        const std::string kernel_src = read_file("concat.cl");
#endif
        if (!kernel_src.empty()) {
            backend_ctx->program_concat =
                build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

            CL_CHECK((backend_ctx->kernel_concat_f32_contiguous = clCreateKernel(backend_ctx->program_concat, "kernel_concat_f32_contiguous", &err), err));
            CL_CHECK((backend_ctx->kernel_concat_f32_non_contiguous = clCreateKernel(backend_ctx->program_concat, "kernel_concat_f32_non_contiguous", &err), err));
            GGML_LOG_CONT(".");
        } else {
            GGML_LOG_WARN("ggml_opencl: concat kernel source not found or empty. Concat operations will not be available.\n");
            backend_ctx->program_concat = nullptr;
            backend_ctx->kernel_concat_f32_contiguous = nullptr;
            backend_ctx->kernel_concat_f32_non_contiguous = nullptr;
        }
    }

    // timestep_embedding
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "tsembd.cl.h"
        };
#else

        const std::string kernel_src = read_file("tsembd.cl");
#endif
        if (!kernel_src.empty()) {
            backend_ctx->program_tsembd =
                build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);
            CL_CHECK((backend_ctx->kernel_timestep_embedding = clCreateKernel(backend_ctx->program_tsembd, "kernel_timestep_embedding", &err), err));
            GGML_LOG_CONT(".");
        } else {
            GGML_LOG_WARN("ggml_opencl: timestep_embedding kernel source not found or empty. This op will not be available.\n");
            backend_ctx->program_tsembd = nullptr;
            backend_ctx->kernel_timestep_embedding = nullptr;
        }
    }

    // set_rows
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "set_rows.cl.h"
        };
#else
        const std::string kernel_src = read_file("set_rows.cl");
#endif
        backend_ctx->program_set_rows =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_set_rows_f32_i64 = clCreateKernel(backend_ctx->program_set_rows, "kernel_set_rows_f32_i64", &err), err));
        CL_CHECK((backend_ctx->kernel_set_rows_f32_i32 = clCreateKernel(backend_ctx->program_set_rows, "kernel_set_rows_f32_i32", &err), err));
        CL_CHECK((backend_ctx->kernel_set_rows_f16_i64 = clCreateKernel(backend_ctx->program_set_rows, "kernel_set_rows_f16_i64", &err), err));
        CL_CHECK((backend_ctx->kernel_set_rows_f16_i32 = clCreateKernel(backend_ctx->program_set_rows, "kernel_set_rows_f16_i32", &err), err));
        GGML_LOG_CONT(".");
    }

    // conv2d
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "conv2d.cl.h"
        };
        const std::string kernel_src_f16_f32 {
            #include "conv2d_f16_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("conv2d.cl");
        const std::string kernel_src_f16_f32 = read_file("conv2d_f16_f32.cl");
#endif
        const bool use_conv2d_qcom_accel16 = std::getenv("GGML_OPENCL_CONV2D_QCOM_ACCEL16") != nullptr &&
                                             backend_ctx->gpu_family == GPU_FAMILY::ADRENO;
        if (use_conv2d_qcom_accel16) {
            GGML_LOG_INFO("ggml_opencl: conv2d qcom accel16 enabled\n");
        }

        if (!kernel_src.empty()) {
            std::string conv2d_opts_f16 = std::string(compile_opts) + " -DUSE_FP16=1";
            if (use_conv2d_qcom_accel16) {
                conv2d_opts_f16 += " -qcom-accelerate-16-bit";
            }

            backend_ctx->program_conv_2d_f16 =
                build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), conv2d_opts_f16.c_str());
            CL_CHECK((backend_ctx->kernel_conv_2d_f16 = clCreateKernel(backend_ctx->program_conv_2d_f16, "kernel_conv_2d", &err), err));
            GGML_LOG_CONT(".");

            // Tuned variant for VAE hot path (3x3, stride1, pad1, N=1, Cin=Cout in {128,256,512}).
            std::string conv2d_opts_f16_vae3x3 =
                conv2d_opts_f16 + " -DGGML_CONV2D_BS_CRS=32 -DGGML_CONV2D_BS_NPQ=128";
            backend_ctx->program_conv_2d_f16_vae3x3 =
                build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), conv2d_opts_f16_vae3x3.c_str());
            CL_CHECK((backend_ctx->kernel_conv_2d_f16_vae3x3 = clCreateKernel(backend_ctx->program_conv_2d_f16_vae3x3, "kernel_conv_2d", &err), err));
            GGML_LOG_CONT(".");

            backend_ctx->program_conv_2d_f32 =
                build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);
            CL_CHECK((backend_ctx->kernel_conv_2d_f32 = clCreateKernel(backend_ctx->program_conv_2d_f32, "kernel_conv_2d", &err), err));
            GGML_LOG_CONT(".");

            std::string conv2d_opts_f32_vae3x3 =
                std::string(compile_opts) + " -DGGML_CONV2D_BS_CRS=32 -DGGML_CONV2D_BS_NPQ=128";
            backend_ctx->program_conv_2d_f32_vae3x3 =
                build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), conv2d_opts_f32_vae3x3.c_str());
            CL_CHECK((backend_ctx->kernel_conv_2d_f32_vae3x3 = clCreateKernel(backend_ctx->program_conv_2d_f32_vae3x3, "kernel_conv_2d", &err), err));
            GGML_LOG_CONT(".");
        } else {
            GGML_LOG_WARN("ggml_opencl: conv2d kernel source not found or empty. This op will not be available.\n");
            backend_ctx->program_conv_2d_f16 = nullptr;
            backend_ctx->kernel_conv_2d_f16 = nullptr;
            backend_ctx->program_conv_2d_f16_vae3x3 = nullptr;
            backend_ctx->kernel_conv_2d_f16_vae3x3 = nullptr;
            backend_ctx->program_conv_2d_f32 = nullptr;
            backend_ctx->kernel_conv_2d_f32 = nullptr;
            backend_ctx->program_conv_2d_f32_vae3x3 = nullptr;
            backend_ctx->kernel_conv_2d_f32_vae3x3 = nullptr;
        }
        if (!kernel_src_f16_f32.empty()) {
            std::string conv2d_opts_f16_f32 = std::string(compile_opts);
            if (use_conv2d_qcom_accel16) {
                conv2d_opts_f16_f32 += " -qcom-accelerate-16-bit";
            }
            backend_ctx->program_conv_2d_f16_f32 =
                build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src_f16_f32.c_str(), conv2d_opts_f16_f32.c_str());
            CL_CHECK((backend_ctx->kernel_conv_2d_f16_f32 = clCreateKernel(backend_ctx->program_conv_2d_f16_f32, "kernel_conv_2d", &err), err));
            GGML_LOG_CONT(".");

            std::string conv2d_opts_f16_f32_vae3x3 =
                conv2d_opts_f16_f32 + " -DGGML_CONV2D_BS_CRS=32 -DGGML_CONV2D_BS_NPQ=128";
            backend_ctx->program_conv_2d_f16_f32_vae3x3 =
                build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src_f16_f32.c_str(), conv2d_opts_f16_f32_vae3x3.c_str());
            CL_CHECK((backend_ctx->kernel_conv_2d_f16_f32_vae3x3 = clCreateKernel(backend_ctx->program_conv_2d_f16_f32_vae3x3, "kernel_conv_2d", &err), err));
            GGML_LOG_CONT(".");
        } else {
            GGML_LOG_WARN("ggml_opencl: conv2d_f16_f32 kernel source not found or empty. This op will not be available.\n");
            backend_ctx->program_conv_2d_f16_f32 = nullptr;
            backend_ctx->kernel_conv_2d_f16_f32 = nullptr;
            backend_ctx->program_conv_2d_f16_f32_vae3x3 = nullptr;
            backend_ctx->kernel_conv_2d_f16_f32_vae3x3 = nullptr;
        }
    }

    // ssm_conv
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "ssm_conv.cl.h"
        };
#else
        const std::string kernel_src = read_file("ssm_conv.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_ssm_conv_f32_f32   = clCreateKernel(prog, "kernel_ssm_conv_f32_f32", &err), err));
        CL_CHECK((backend_ctx->kernel_ssm_conv_f32_f32_4 = clCreateKernel(prog, "kernel_ssm_conv_f32_f32_4", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_id_q4_0_f32_8x_flat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_id_q4_0_f32_8x_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_id_q4_0_f32_8x_flat.cl");
#endif
        backend_ctx->program_mul_mv_id_q4_0_f32_8x_flat =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mv_id_q4_0_f32_8x_flat = clCreateKernel(backend_ctx->program_mul_mv_id_q4_0_f32_8x_flat, "kernel_mul_mv_id_q4_0_f32_8x_flat", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_id_q8_0_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_id_q8_0_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_id_q8_0_f32.cl");
#endif
        backend_ctx->program_mul_mv_id_q8_0_f32 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mv_id_q8_0_f32 = clCreateKernel(backend_ctx->program_mul_mv_id_q8_0_f32, "kernel_mul_mv_id_q8_0_f32", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_id_q8_0_f32_flat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_id_q8_0_f32_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_id_q8_0_f32_flat.cl");
#endif
        backend_ctx->program_mul_mv_id_q8_0_f32_flat =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mv_id_q8_0_f32_flat = clCreateKernel(backend_ctx->program_mul_mv_id_q8_0_f32_flat, "kernel_mul_mv_id_q8_0_f32_flat", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_id_mxfp4_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_id_mxfp4_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_id_mxfp4_f32.cl");
#endif
        backend_ctx->program_mul_mv_id_mxfp4_f32 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mv_id_mxfp4_f32 = clCreateKernel(backend_ctx->program_mul_mv_id_mxfp4_f32, "kernel_mul_mv_id_mxfp4_f32", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mv_id_mxfp4_f32_flat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_id_mxfp4_f32_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_id_mxfp4_f32_flat.cl");
#endif
        backend_ctx->program_mul_mv_id_mxfp4_f32_flat =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_mul_mv_id_mxfp4_f32_flat = clCreateKernel(backend_ctx->program_mul_mv_id_mxfp4_f32_flat, "kernel_mul_mv_id_mxfp4_f32_flat", &err), err));
        GGML_LOG_CONT(".");
    }

    // Adreno kernels
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    // transpose
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "transpose.cl.h"
        };
#else
        const std::string kernel_src = read_file("transpose.cl");
#endif
        backend_ctx->program_transpose =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->kernel_transpose_32_16 = clCreateKernel(backend_ctx->program_transpose, "kernel_transpose_32_16", &err), err));
        CL_CHECK((backend_ctx->kernel_transpose_32_32 = clCreateKernel(backend_ctx->program_transpose, "kernel_transpose_32_32", &err), err));
        CL_CHECK((backend_ctx->kernel_transpose_32    = clCreateKernel(backend_ctx->program_transpose, "kernel_transpose_32", &err), err));
        CL_CHECK((backend_ctx->kernel_transpose_16    = clCreateKernel(backend_ctx->program_transpose, "kernel_transpose_16", &err), err));
        CL_CHECK((backend_ctx->kernel_transpose_16_buf = clCreateKernel(backend_ctx->program_transpose, "kernel_transpose_16_buf", &err), err));
        CL_CHECK((backend_ctx->kernel_transpose_16_4x1 = clCreateKernel(backend_ctx->program_transpose, "kernel_transpose_16_4x1", &err), err));
        GGML_LOG_CONT(".");
    }

    // gemv_noshuffle_general
    {
        std::string CL_gemv_compile_opts = std::string("-cl-std=") + opencl_c_std +
                                       " -cl-mad-enable "
                                       " -DSIMDGROUP_WIDTH=" +
                                       std::to_string(backend_ctx->adreno_wave_size);
        if (backend_ctx->has_vector_subgroup_broadcast) {
            CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAT ";
        }

#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src_CL_gemv_general {
            #include "gemv_noshuffle_general.cl.h"
        };
#else
        const std::string kernel_src_CL_gemv_general = read_file("gemv_noshuffle_general.cl");
#endif

        backend_ctx->program_CL_gemv_general = build_program_from_source(
            backend_ctx->context, backend_ctx->device, kernel_src_CL_gemv_general.c_str(), CL_gemv_compile_opts);

        CL_CHECK((backend_ctx->CL_mul_mat_vec_q4_0_f32_1d_4x_flat_general = clCreateKernel(backend_ctx->program_CL_gemv_general, "kernel_gemv_noshuffle", &err), err));
        GGML_LOG_CONT(".");
    }

    // gemv_noshuffle
    {
        // Gemv 2048, 16384
        std::string CL_gemv_compile_opts = std::string("-cl-std=") + opencl_c_std +
            " -cl-mad-enable "
            " -DLINE_STRIDE_A=2048 "
            " -DBLOCK_STRIDE_A=16384 "
            " -DSIMDGROUP_WIDTH=" +
            std::to_string(backend_ctx->adreno_wave_size);
        if (backend_ctx->has_vector_subgroup_broadcast) {
            CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAT ";
        }

#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src_CL_gemv {
            #include "gemv_noshuffle.cl.h"
        };
#else
        const std::string kernel_src_CL_gemv = read_file("gemv_noshuffle.cl");
#endif

        backend_ctx->program_CL_gemv_4096_1_4096 = build_program_from_source(
            backend_ctx->context, backend_ctx->device, kernel_src_CL_gemv.c_str(), CL_gemv_compile_opts);
        CL_CHECK((backend_ctx->CL_mul_mat_vec_q4_0_f32_1d_4x_flat_4096_1_4096 = clCreateKernel(backend_ctx->program_CL_gemv_4096_1_4096, "kernel_gemv_noshuffle", &err), err));
        GGML_LOG_CONT(".");

        // Gemv 2048, 16384
        CL_gemv_compile_opts = std::string("-cl-std=") + opencl_c_std +
            " -cl-mad-enable "
            " -DLINE_STRIDE_A=2048 "
            " -DBLOCK_STRIDE_A=16384 "
            " -DSIMDGROUP_WIDTH=" +
            std::to_string(backend_ctx->adreno_wave_size);
        if (backend_ctx->has_vector_subgroup_broadcast) {
            CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAT ";
        }

        backend_ctx->program_CL_gemv_4096_1_11008 = build_program_from_source(
            backend_ctx->context, backend_ctx->device, kernel_src_CL_gemv.c_str(), CL_gemv_compile_opts);
        CL_CHECK((backend_ctx->CL_mul_mat_vec_q4_0_f32_1d_4x_flat_4096_1_11008 = clCreateKernel(backend_ctx->program_CL_gemv_4096_1_11008, "kernel_gemv_noshuffle", &err), err));
        GGML_LOG_CONT(".");

        // Gemv 5504, 44032
        CL_gemv_compile_opts = std::string("-cl-std=") + opencl_c_std +
            " -cl-mad-enable "
            " -DLINE_STRIDE_A=5504 "
            " -DBLOCK_STRIDE_A=44032 "
            " -DSIMDGROUP_WIDTH=" +
            std::to_string(backend_ctx->adreno_wave_size);
        if (backend_ctx->has_vector_subgroup_broadcast) {
            CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAT ";
        }

        backend_ctx->program_CL_gemv_11008_1_4096 = build_program_from_source(
            backend_ctx->context, backend_ctx->device, kernel_src_CL_gemv.c_str(), CL_gemv_compile_opts);
        CL_CHECK((backend_ctx->CL_mul_mat_vec_q4_0_f32_1d_4x_flat_11008_1_4096 = clCreateKernel(backend_ctx->program_CL_gemv_11008_1_4096, "kernel_gemv_noshuffle", &err), err));
        GGML_LOG_CONT(".");

        // Gemv 16000, 128000
        CL_gemv_compile_opts = std::string("-cl-std=") + opencl_c_std +
            " -cl-mad-enable "
            " -DLINE_STRIDE_A=16000 "
            " -DBLOCK_STRIDE_A=128000 "
            " -DSIMDGROUP_WIDTH=" +
            std::to_string(backend_ctx->adreno_wave_size);

        if (backend_ctx->has_vector_subgroup_broadcast) {
            CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAT ";
        }

        backend_ctx->program_CL_gemv_32000_1_4096 = build_program_from_source(
            backend_ctx->context, backend_ctx->device, kernel_src_CL_gemv.c_str(), CL_gemv_compile_opts);
        CL_CHECK((backend_ctx->CL_mul_mat_vec_q4_0_f32_1d_4x_flat_32000_1_4096 = clCreateKernel(backend_ctx->program_CL_gemv_32000_1_4096, "kernel_gemv_noshuffle", &err), err));
        GGML_LOG_CONT(".");
    }

    // mul_mat_Ab_Bi_8x4
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src_CL_gemm {
            #include "mul_mat_Ab_Bi_8x4.cl.h"
        };
#else
        const std::string kernel_src_CL_gemm = read_file("mul_mat_Ab_Bi_8x4.cl");
#endif
        const bool precise_q4_gemm = std::getenv("GGML_OPENCL_PRECISE_Q4_GEMM") != nullptr;
        if (precise_q4_gemm) {
            GGML_LOG_INFO("ggml_opencl: precise compile opts enabled for q4 gemm kernel\n");
        }
        const bool global_kahan_q4_gemm = std::getenv("GGML_OPENCL_Q4_GEMM_KAHAN") != nullptr;
        const bool selective_kahan_q4_gemm =
            !global_kahan_q4_gemm && (std::getenv("SD_OCL_Q4_GEMM_KAHAN_SUBSTR") != nullptr);
        const bool global_f32_act_q4_gemm = std::getenv("SD_OCL_Q4_GEMM_F32_ACT") != nullptr;
        const bool selective_f32_act_q4_gemm = !global_f32_act_q4_gemm;
        const bool global_f32_acc_q4_gemm = std::getenv("SD_OCL_Q4_GEMM_F32_ACC") != nullptr;
        const bool selective_f32_acc_q4_gemm =
            !global_f32_acc_q4_gemm && (std::getenv("SD_OCL_Q4_GEMM_F32_ACC_SUBSTR") != nullptr);
        const bool global_f32_read_q4_gemm = std::getenv("SD_OCL_Q4_GEMM_F32_READ") != nullptr;
        const bool selective_f32_read_q4_gemm =
            !global_f32_read_q4_gemm && (std::getenv("SD_OCL_Q4_GEMM_F32_READ_SUBSTR") != nullptr);
        const bool global_half_acc_clamp_q4_gemm = std::getenv("SD_OCL_Q4_GEMM_HALF_ACC_CLAMP") != nullptr;
        const bool selective_half_acc_clamp_q4_gemm =
            !global_half_acc_clamp_q4_gemm && (std::getenv("SD_OCL_Q4_GEMM_HALF_ACC_CLAMP_SUBSTR") != nullptr);
        const bool global_half_scale_q4_gemm = std::getenv("SD_OCL_Q4_GEMM_HALF_SCALE") != nullptr;
        const bool selective_half_scale_q4_gemm =
            !global_half_scale_q4_gemm && (std::getenv("SD_OCL_Q4_GEMM_HALF_SCALE_SUBSTR") != nullptr);
        const bool global_fp16_chunk_acc_q4_gemm = std::getenv("SD_OCL_Q4_GEMM_FP16_CHUNK_ACC") != nullptr;
        const bool selective_fp16_chunk_acc_q4_gemm =
            !global_fp16_chunk_acc_q4_gemm && (std::getenv("SD_OCL_Q4_GEMM_FP16_CHUNK_ACC_SUBSTR") != nullptr);
        const auto append_half_scale_opts = [](std::string & opts) {
            opts += " -DUSE_HALF_SCALE_ACC=1";
            float scale = 0.5f;
            if (const char * scale_env = std::getenv("SD_OCL_Q4_GEMM_HALF_SCALE_VALUE")) {
                const float parsed = std::atof(scale_env);
                if (parsed > 0.0f && parsed < 1.0f) {
                    scale = parsed;
                }
            }
            opts += " -DHALF_SCALE_ACC_F=" + std::to_string(scale);
        };

        std::string gemm_compile_opts = precise_q4_gemm ? compile_opts_precise : compile_opts;
        if (global_kahan_q4_gemm) {
            GGML_LOG_INFO("ggml_opencl: Kahan accumulation enabled for q4 gemm kernel\n");
            gemm_compile_opts += " -DUSE_KAHAN_ACC=1";
        }
        if (global_f32_act_q4_gemm) {
            GGML_LOG_INFO("ggml_opencl: full-f32 activation reads enabled for q4 gemm kernel\n");
            gemm_compile_opts += " -DUSE_F32_ACT=1";
        }
        if (global_f32_acc_q4_gemm) {
            GGML_LOG_INFO("ggml_opencl: fp32 accumulation enabled for q4 gemm kernel\n");
            gemm_compile_opts += " -DUSE_F32_ACC=1";
        }
        if (global_f32_read_q4_gemm) {
            GGML_LOG_INFO("ggml_opencl: f32 activation-read enabled for q4 gemm kernel\n");
            gemm_compile_opts += " -DUSE_F32_READ=1";
        }
        if (global_half_acc_clamp_q4_gemm) {
            GGML_LOG_INFO("ggml_opencl: half-acc clamp enabled for q4 gemm kernel\n");
            gemm_compile_opts += " -DUSE_HALF_ACC_CLAMP=1";
        }
        if (global_half_scale_q4_gemm) {
            GGML_LOG_INFO("ggml_opencl: half-acc contribution scaling enabled for q4 gemm kernel\n");
            append_half_scale_opts(gemm_compile_opts);
        }
        if (global_fp16_chunk_acc_q4_gemm) {
            GGML_LOG_INFO("ggml_opencl: fp16 chunk-acc enabled for q4 gemm kernel\n");
            gemm_compile_opts += " -DUSE_FP16_CHUNK_ACC=1";
        }

        backend_ctx->program_CL_gemm =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src_CL_gemm.c_str(), gemm_compile_opts);
        CL_CHECK((backend_ctx->CL_mul_mat_Ab_Bi_8x4 =
            clCreateKernel(backend_ctx->program_CL_gemm, "kernel_mul_mat_Ab_Bi_8x4", &err), err));

        backend_ctx->program_CL_gemm_kahan = nullptr;
        backend_ctx->CL_mul_mat_Ab_Bi_8x4_kahan = nullptr;
        if (selective_kahan_q4_gemm) {
            GGML_LOG_INFO("ggml_opencl: selective Kahan accumulation enabled for q4 gemm kernel\n");
            std::string gemm_compile_opts_kahan = gemm_compile_opts + " -DUSE_KAHAN_ACC=1";
            backend_ctx->program_CL_gemm_kahan =
                build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src_CL_gemm.c_str(), gemm_compile_opts_kahan);
            CL_CHECK((backend_ctx->CL_mul_mat_Ab_Bi_8x4_kahan =
                clCreateKernel(backend_ctx->program_CL_gemm_kahan, "kernel_mul_mat_Ab_Bi_8x4", &err), err));
        }

        backend_ctx->program_CL_gemm_f32act = nullptr;
        backend_ctx->CL_mul_mat_Ab_Bi_8x4_f32act = nullptr;
        if (selective_f32_act_q4_gemm) {
            GGML_LOG_INFO("ggml_opencl: selective full-f32 activation reads enabled for q4 gemm kernel\n");
            std::string gemm_compile_opts_f32act = gemm_compile_opts + " -DUSE_F32_ACT=1";
            backend_ctx->program_CL_gemm_f32act =
                build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src_CL_gemm.c_str(), gemm_compile_opts_f32act);
            CL_CHECK((backend_ctx->CL_mul_mat_Ab_Bi_8x4_f32act =
                clCreateKernel(backend_ctx->program_CL_gemm_f32act, "kernel_mul_mat_Ab_Bi_8x4", &err), err));
        }

        backend_ctx->program_CL_gemm_f32acc = nullptr;
        backend_ctx->CL_mul_mat_Ab_Bi_8x4_f32acc = nullptr;
        if (selective_f32_acc_q4_gemm) {
            GGML_LOG_INFO("ggml_opencl: selective fp32 accumulation enabled for q4 gemm kernel\n");
            std::string gemm_compile_opts_f32acc = gemm_compile_opts + " -DUSE_F32_ACC=1";
            backend_ctx->program_CL_gemm_f32acc =
                build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src_CL_gemm.c_str(), gemm_compile_opts_f32acc);
            CL_CHECK((backend_ctx->CL_mul_mat_Ab_Bi_8x4_f32acc =
                clCreateKernel(backend_ctx->program_CL_gemm_f32acc, "kernel_mul_mat_Ab_Bi_8x4", &err), err));
        }

        backend_ctx->program_CL_gemm_f32read = nullptr;
        backend_ctx->CL_mul_mat_Ab_Bi_8x4_f32read = nullptr;
        if (selective_f32_read_q4_gemm) {
            GGML_LOG_INFO("ggml_opencl: selective f32 activation-read enabled for q4 gemm kernel\n");
            std::string gemm_compile_opts_f32read = gemm_compile_opts + " -DUSE_F32_READ=1";
            backend_ctx->program_CL_gemm_f32read =
                build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src_CL_gemm.c_str(), gemm_compile_opts_f32read);
            CL_CHECK((backend_ctx->CL_mul_mat_Ab_Bi_8x4_f32read =
                clCreateKernel(backend_ctx->program_CL_gemm_f32read, "kernel_mul_mat_Ab_Bi_8x4", &err), err));
        }

        backend_ctx->program_CL_gemm_halfclamp = nullptr;
        backend_ctx->CL_mul_mat_Ab_Bi_8x4_halfclamp = nullptr;
        if (selective_half_acc_clamp_q4_gemm) {
            GGML_LOG_INFO("ggml_opencl: selective half-acc clamp enabled for q4 gemm kernel\n");
            std::string gemm_compile_opts_halfclamp = gemm_compile_opts + " -DUSE_HALF_ACC_CLAMP=1";
            backend_ctx->program_CL_gemm_halfclamp =
                build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src_CL_gemm.c_str(), gemm_compile_opts_halfclamp);
            CL_CHECK((backend_ctx->CL_mul_mat_Ab_Bi_8x4_halfclamp =
                clCreateKernel(backend_ctx->program_CL_gemm_halfclamp, "kernel_mul_mat_Ab_Bi_8x4", &err), err));
        }

        backend_ctx->program_CL_gemm_halfscale = nullptr;
        backend_ctx->CL_mul_mat_Ab_Bi_8x4_halfscale = nullptr;
        if (selective_half_scale_q4_gemm) {
            GGML_LOG_INFO("ggml_opencl: selective half-acc contribution scaling enabled for q4 gemm kernel\n");
            std::string gemm_compile_opts_halfscale = gemm_compile_opts;
            append_half_scale_opts(gemm_compile_opts_halfscale);
            backend_ctx->program_CL_gemm_halfscale =
                build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src_CL_gemm.c_str(), gemm_compile_opts_halfscale);
            CL_CHECK((backend_ctx->CL_mul_mat_Ab_Bi_8x4_halfscale =
                clCreateKernel(backend_ctx->program_CL_gemm_halfscale, "kernel_mul_mat_Ab_Bi_8x4", &err), err));
        }

        backend_ctx->program_CL_gemm_fp16chunk = nullptr;
        backend_ctx->CL_mul_mat_Ab_Bi_8x4_fp16chunk = nullptr;
        if (selective_fp16_chunk_acc_q4_gemm) {
            GGML_LOG_INFO("ggml_opencl: selective fp16 chunk-acc enabled for q4 gemm kernel\n");
            std::string gemm_compile_opts_fp16chunk = gemm_compile_opts + " -DUSE_FP16_CHUNK_ACC=1";
            if (const char * chunk_iters_env = std::getenv("SD_OCL_Q4_GEMM_FP16_CHUNK_ITERS")) {
                int chunk_iters = std::atoi(chunk_iters_env);
                if (chunk_iters > 0) {
                    gemm_compile_opts_fp16chunk += " -DCHUNK_ACC_ITERS=" + std::to_string(chunk_iters);
                }
            }
            backend_ctx->program_CL_gemm_fp16chunk =
                build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src_CL_gemm.c_str(), gemm_compile_opts_fp16chunk);
            CL_CHECK((backend_ctx->CL_mul_mat_Ab_Bi_8x4_fp16chunk =
                clCreateKernel(backend_ctx->program_CL_gemm_fp16chunk, "kernel_mul_mat_Ab_Bi_8x4", &err), err));
        }
        GGML_LOG_CONT(".");
    }

    std::string CL_moe_compile_opts = std::string("-cl-std=") + opencl_c_std +
            " -cl-mad-enable "
            " -cl-fast-relaxed-math";

    // gemv_moe_mxfp4_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemv_moe_mxfp4_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemv_moe_mxfp4_f32.cl");
#endif
        backend_ctx->program_gemv_moe_mxfp4_f32 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), CL_moe_compile_opts);

        CL_CHECK((backend_ctx->kernel_gemv_moe_mxfp4_f32 = clCreateKernel(backend_ctx->program_gemv_moe_mxfp4_f32, "kernel_gemv_moe_mxfp4_f32", &err), err));
        GGML_LOG_CONT(".");
    }

    // gemm_moe_mxfp4_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_moe_mxfp4_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_moe_mxfp4_f32.cl");
#endif
        backend_ctx->program_gemm_moe_mxfp4_f32 =
            build_program_from_source(backend_ctx->context, backend_ctx->device, kernel_src.c_str(), CL_moe_compile_opts);

        CL_CHECK((backend_ctx->kernel_gemm_moe_mxfp4_f32 = clCreateKernel(backend_ctx->program_gemm_moe_mxfp4_f32, "kernel_gemm_moe_mxfp4_f32", &err), err));
        GGML_LOG_CONT(".");
    }
#endif // GGML_OPENCL_USE_ADRENO_KERNELS
    GGML_LOG_CONT("\n");
}

// XXX static ggml_backend_opencl_context * ggml_cl2_init(ggml_backend_dev_t dev) {
// XXX    static bool initialized = false;
// XXX    static ggml_backend_opencl_context *backend_ctx = nullptr;

static ggml_backend_opencl_context * ggml_cl2_init(ggml_backend_dev_t dev);

namespace /* anonymous */ {
extern struct ggml_backend_device_i ggml_backend_opencl_device_i;
}

// Look for available and suitable devices.
static std::vector<ggml_backend_device> ggml_opencl_probe_devices(ggml_backend_reg * reg) {
    std::vector<ggml_backend_device> found_devices;

#ifdef GGML_OPENCL_PROFILING
    GGML_LOG_INFO("ggml_opencl: OpenCL profiling enabled\n");
#endif

    struct cl_device;
    struct cl_platform {
        cl_platform_id id;
        unsigned number;
        char name[128];
        char vendor[128];
        struct cl_device * devices;
        unsigned n_devices;
        struct cl_device * default_device;
    };

    struct cl_device {
        struct cl_platform * platform;
        cl_device_id id;
        unsigned number;
        cl_device_type type;
        char name[128];
        char version[128];
    };

    enum { NPLAT = 16, NDEV = 16 };

    struct cl_platform platforms[NPLAT];
    unsigned n_platforms = 0;
    struct cl_device devices[NDEV];
    unsigned n_devices = 0;
    struct cl_device * default_device = NULL;
    unsigned           default_platform_number = 0;

    cl_platform_id platform_ids[NPLAT];
    if (clGetPlatformIDs(NPLAT, platform_ids, &n_platforms) != CL_SUCCESS) {
        GGML_LOG_ERROR("ggml_opencl: plaform IDs not available.\n");
        return found_devices;
    }

    for (unsigned i = 0; i < n_platforms; i++) {
        struct cl_platform * p = &platforms[i];
        p->number = i;
        p->id = platform_ids[i];
        CL_CHECK(clGetPlatformInfo(p->id, CL_PLATFORM_NAME, sizeof(p->name), &p->name, NULL));
        CL_CHECK(clGetPlatformInfo(p->id, CL_PLATFORM_VENDOR, sizeof(p->vendor), &p->vendor, NULL));

        cl_device_id device_ids[NDEV];
        cl_int clGetDeviceIDsError = clGetDeviceIDs(p->id, CL_DEVICE_TYPE_ALL, NDEV, device_ids, &p->n_devices);
        if (clGetDeviceIDsError == CL_DEVICE_NOT_FOUND) {
            p->n_devices = 0;
        } else {
            CL_CHECK(clGetDeviceIDsError);
        }
        p->devices = p->n_devices > 0 ? &devices[n_devices] : NULL;
        p->default_device = NULL;

        for (unsigned j = 0; j < p->n_devices; j++) {
            struct cl_device * d = &devices[n_devices];
            d->number = n_devices++;
            d->id = device_ids[j];
            d->platform = p;
            CL_CHECK(clGetDeviceInfo(d->id, CL_DEVICE_NAME, sizeof(d->name), &d->name, NULL));
            CL_CHECK(clGetDeviceInfo(d->id, CL_DEVICE_TYPE, sizeof(d->type), &d->type, NULL));
            CL_CHECK(clGetDeviceInfo(d->id, CL_DEVICE_VERSION, sizeof(d->version), &d->version, NULL));

            if (p->default_device == NULL && d->type == CL_DEVICE_TYPE_GPU) {
                p->default_device = d;
            }
        }

        if (default_device == NULL && p->default_device != NULL) {
            default_device          = p->default_device;
            default_platform_number = i;
        }
    }

    if (n_devices == 0) {
        GGML_LOG_ERROR("ggml_opencl: could find any OpenCL devices.\n");
        return found_devices;
    }

    char *      user_platform_string = getenv("GGML_OPENCL_PLATFORM");
    char *      user_device_string   = getenv("GGML_OPENCL_DEVICE");
    int         user_platform_number = -1;
    int         user_device_number   = -1;
    cl_device * candidate_devices    = nullptr;
    unsigned    n_candidate_devices  = 0;

    unsigned n;
    if (user_platform_string != NULL && sscanf(user_platform_string, " %u", &n) == 1 && n < n_platforms) {
        user_platform_number = (int)n;
    }
    if (user_device_string != NULL && sscanf(user_device_string, " %u", &n) == 1 && n < n_devices) {
        user_device_number = (int)n;
    }
    if (user_platform_number != -1 && user_device_number != -1) {
        cl_platform* platform = &platforms[user_platform_number];
        if ((unsigned)user_device_number >= platform->n_devices) {
            GGML_LOG_ERROR("ggml_opencl: invalid device number %d\n", user_device_number);
            exit(1);
        }
        default_device      = &platform->devices[user_device_number];
        candidate_devices   = platform->devices;
        n_candidate_devices = platform->n_devices;
    } else {
        // Choose a platform by matching a substring.
        if (user_platform_number == -1 && user_platform_string != NULL && user_platform_string[0] != 0) {
            for (unsigned i = 0; i < n_platforms; i++) {
                struct cl_platform * p = &platforms[i];
                if (strstr(p->name, user_platform_string) != NULL ||
                    strstr(p->vendor, user_platform_string) != NULL) {
                    user_platform_number = (int)i;
                    break;
                }
            }
            if (user_platform_number == -1) {
                GGML_LOG_ERROR("ggml_opencl: no platform matching '%s' was found.\n", user_platform_string);
                exit(1);
            }
        }

        int                  platform_idx = user_platform_number != -1 ? user_platform_number : default_platform_number;
        struct cl_platform * p            = &platforms[platform_idx];
        candidate_devices                 = p->devices;
        n_candidate_devices               = p->n_devices;
        default_device                    = p->default_device;
        if (n_candidate_devices == 0) {
            GGML_LOG_ERROR("ggml_opencl: selected platform '%s' does not have any devices.\n", p->name);
            exit(1);
        }

        if (user_device_number == -1 && user_device_string != NULL && user_device_string[0] != 0) {
            for (unsigned i = 0; i < n_candidate_devices; i++) {
                struct cl_device * d = &candidate_devices[i];
                if (strstr(d->name, user_device_string) != NULL) {
                    user_device_number = d->number;
                    break;
                }
            }
            if (user_device_number == -1) {
                GGML_LOG_ERROR("ggml_opencl: no device matching '%s' was found.\n", user_device_string);
                exit(1);
            }
        }
        if (user_device_number != -1) {
            candidate_devices   = &devices[user_device_number];
            n_candidate_devices = 1;
            default_device      = &candidate_devices[0];
        }

        GGML_ASSERT(n_candidate_devices > 0);

        if (default_device == NULL) {
            default_device = &candidate_devices[0];
        }
    }

    GGML_ASSERT(n_candidate_devices != 0 && candidate_devices);

    // Put the default device in front.
    for (unsigned i = 1; i < n_candidate_devices; i++) {
        if (&candidate_devices[i] == default_device) {
            std::swap(candidate_devices[0], candidate_devices[i]);
            default_device = &candidate_devices[0];
            break;
        }
    }

    GGML_LOG_INFO("ggml_opencl: selected platform: '%s'\n", default_device->platform->name);

    std::vector<cl_device_id> device_ids;
    for (auto dev = candidate_devices, dev_end = candidate_devices + n_candidate_devices; dev != dev_end; dev++) {
        device_ids.push_back(dev->id);
    }

    cl_int                err;
    cl_context            shared_context;
    cl_context_properties properties[] = { (intptr_t) CL_CONTEXT_PLATFORM, (intptr_t) default_device->platform->id, 0 };

    CL_CHECK(
        (shared_context = clCreateContext(properties, device_ids.size(), device_ids.data(), NULL, NULL, &err), err));

    for (auto dev = candidate_devices, dev_end = candidate_devices + n_candidate_devices; dev != dev_end; dev++) {
        GGML_LOG_INFO("\nggml_opencl: device: '%s (%s)'\n", dev->name, dev->version);

        auto dev_ctx = std::unique_ptr<ggml_backend_opencl_device_context>(new ggml_backend_opencl_device_context{
            /*.platform         =*/dev->platform->id,
            /*.platform_nane    =*/dev->platform->name,
            /*.device           =*/dev->id,
            /*.device_name      =*/dev->name,
            /*.device_type      =*/dev->type,
            /*.device_version   =*/dev->version,
            /*.backend_ctx      =*/nullptr,
            /*.buffer_type      =*/{},
            /*.context          =*/shared_context,
        });

        found_devices.push_back(ggml_backend_device{
            /* .iface   = */ ggml_backend_opencl_device_i,
            /* .reg     = */ reg,
            /* .context = */ dev_ctx.get(),
        });

        if (!ggml_cl2_init(&found_devices.back())) {
            found_devices.pop_back();
            GGML_LOG_INFO("ggml_opencl: drop unsupported device.\n");
            continue;
        }

        dev_ctx.release();
    }

    if (found_devices.size()) {
        auto * dev_ctx = static_cast<ggml_backend_opencl_device_context *>(found_devices.front().context);
        GGML_LOG_INFO("ggml_opencl: default device: '%s (%s)'\n", dev_ctx->device_name.c_str(),
                      dev_ctx->device_version.c_str());

        if (dev_ctx->device_type != CL_DEVICE_TYPE_GPU) {
            GGML_LOG_WARN("ggml_opencl: warning, the default device is not a GPU: '%s'.\n",
                          dev_ctx->device_name.c_str());
        }
    }

    return found_devices;
}

// Initialize device if it is supported (returns nullptr if it is not).
static ggml_backend_opencl_context * ggml_cl2_init(ggml_backend_dev_t dev) {
    GGML_ASSERT(dev);
    GGML_ASSERT(dev->context);

    ggml_backend_opencl_device_context * dev_ctx = (ggml_backend_opencl_device_context *) dev->context;
    GGML_ASSERT(dev_ctx->platform);
    GGML_ASSERT(dev_ctx->device);

    if (dev_ctx->backend_ctx) {
        return dev_ctx->backend_ctx;
    }

    auto backend_ctx        = std::make_unique<ggml_backend_opencl_context>();
    backend_ctx->device     = dev_ctx->device;
    backend_ctx->gpu_family = GPU_FAMILY::UNKNOWN;

    // ref_count get increased in ggml_backend_opencl_device_init
    // This function is also used to retrieve backend context, so we don't want
    // to increase ref_count for each call. We only want to increase ref_count
    // when the associated device is initialized
    backend_ctx->ref_count  = 0;

    if (strstr(dev_ctx->device_name.c_str(), "Adreno") ||
        strstr(dev_ctx->device_name.c_str(), "Qualcomm") ||
        strstr(dev_ctx->device_version.c_str(), "Adreno")) {
        backend_ctx->gpu_family = GPU_FAMILY::ADRENO;
        // Usually device version contains the detailed device name
        backend_ctx->adreno_gen = get_adreno_gpu_gen(dev_ctx->device_version.c_str());
        if (backend_ctx->adreno_gen == ADRENO_GPU_GEN::ADRENO_UNKNOWN) {
            backend_ctx->adreno_gen = get_adreno_gpu_gen(dev_ctx->device_name.c_str());
        }

        // Use wave size of 64 for all Adreno GPUs.
        backend_ctx->adreno_wave_size = 64;
    } else if (strstr(dev_ctx->device_name.c_str(), "Intel")) {
        backend_ctx->gpu_family = GPU_FAMILY::INTEL;
    } else {
        GGML_LOG_ERROR("Unsupported GPU: %s\n", dev_ctx->device_name.c_str());
        backend_ctx->gpu_family = GPU_FAMILY::UNKNOWN;
        return nullptr;
    }

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    if (backend_ctx->gpu_family != GPU_FAMILY::ADRENO) {
        GGML_LOG_ERROR("ggml_opencl: Adreno-specific kernels should not be enabled for non-Adreno GPUs; "
            "run on an Adreno GPU or recompile with CMake option `-DGGML_OPENCL_USE_ADRENO_KERNELS=OFF`\n");
        return nullptr;
    }
#endif

    // Populate backend device name
    backend_ctx->device_name = dev_ctx->device_name;

    // A local ref of cl_device_id for convenience
    cl_device_id device = backend_ctx->device;

    ggml_cl_version platform_version = get_opencl_platform_version(dev_ctx->platform);

    // Check device OpenCL version, OpenCL 2.0 or above is required
    ggml_cl_version opencl_c_version = get_opencl_c_version(platform_version, device);
    if (opencl_c_version.major < 2) {
        GGML_LOG_ERROR("ggml_opencl: OpenCL 2.0 or above is required\n");
        return nullptr;
    }

    // Check driver version
    size_t driver_version_str_size;
    clGetDeviceInfo(device, CL_DRIVER_VERSION, 0, NULL, &driver_version_str_size);
    char *driver_version = (char *)alloca(driver_version_str_size + 1);
    clGetDeviceInfo(device, CL_DRIVER_VERSION, driver_version_str_size, driver_version, NULL);
    driver_version[driver_version_str_size] = '\0';
    GGML_LOG_INFO("ggml_opencl: OpenCL driver: %s\n", driver_version);
    backend_ctx->driver_version = driver_version;

    backend_ctx->adreno_cl_compiler_version = get_adreno_cl_compiler_version(driver_version);
    backend_ctx->has_vector_subgroup_broadcast =
        (backend_ctx->adreno_cl_compiler_version.type == E031 && backend_ctx->adreno_cl_compiler_version.major >= 47) ||
        (backend_ctx->adreno_cl_compiler_version.type == DX   && backend_ctx->adreno_cl_compiler_version.major >= 17);
    GGML_LOG_INFO("ggml_opencl: vector subgroup broadcast support: %s\n",
        backend_ctx->has_vector_subgroup_broadcast ? "true" : "false");

    size_t ext_str_size;
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, NULL, &ext_str_size);
    char *ext_buffer = (char *)alloca(ext_str_size + 1);
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, ext_str_size, ext_buffer, NULL);
    ext_buffer[ext_str_size] = '\0'; // ensure it is null terminated
    // Check if ext_buffer contains cl_khr_fp16
    backend_ctx->fp16_support = strstr(ext_buffer, "cl_khr_fp16") != NULL;
    GGML_LOG_INFO("ggml_opencl: device FP16 support: %s\n", backend_ctx->fp16_support ? "true" : "false");

    // fp16 is required
    if (!backend_ctx->fp16_support) {
        GGML_LOG_ERROR("ggml_opencl: device does not support FP16\n");
        return nullptr;
    }

    // If OpenCL 3.0 is supported, then check for cl_khr_subgroups, which becomes
    // optional in OpenCL 3.0 (cl_khr_subgroup is mandatory in OpenCL 2.x)
    if (opencl_c_version.major == 3 && strstr(ext_buffer, "cl_khr_subgroups") == NULL &&
        strstr(ext_buffer, "cl_intel_subgroups") == NULL) {
        GGML_LOG_ERROR("ggml_opencl: device does not support subgroups (cl_khr_subgroups or cl_intel_subgroups) "
            "(note that subgroups is an optional feature in OpenCL 3.0)\n");
        return nullptr;
    }

    cl_uint base_align_in_bits;
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), &base_align_in_bits, NULL));
    GGML_ASSERT(base_align_in_bits % 8u == 0);
    backend_ctx->alignment = base_align_in_bits / 8u;
    GGML_LOG_INFO("ggml_opencl: mem base addr align: %u\n", backend_ctx->alignment);

    clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(size_t), &backend_ctx->max_alloc_size, NULL);
    GGML_LOG_INFO("ggml_opencl: max mem alloc size: %zu MB\n", backend_ctx->max_alloc_size/1024/1024);

    clGetDeviceInfo(device, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, sizeof(size_t), &backend_ctx->image_max_buffer_size, NULL);
    GGML_LOG_INFO("ggml_opencl: device max image buffer size (pixels): %lu\n", backend_ctx->image_max_buffer_size);

    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &backend_ctx->max_workgroup_size, NULL);
    GGML_LOG_INFO("ggml_opencl: device max workgroup size: %lu\n", backend_ctx->max_workgroup_size);

    // Check SVM.
    cl_device_svm_capabilities svm_caps;
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES, sizeof(cl_device_svm_capabilities), &svm_caps, 0));
    GGML_LOG_INFO("ggml_opencl: SVM coarse grain buffer support: %s\n",
        svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER ? "true" : "false");
    GGML_LOG_INFO("ggml_opencl: SVM fine grain buffer support: %s\n",
        svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER ? "true" : "false");
    GGML_LOG_INFO("ggml_opencl: SVM fine grain system support: %s\n",
        svm_caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM ? "true" : "false");
    GGML_LOG_INFO("ggml_opencl: SVM atomics support: %s\n",
        svm_caps & CL_DEVICE_SVM_ATOMICS ? "true" : "false");

    if (opencl_c_version.major >= 3) {
        // Assume it is not available for 3.0, since it is optional in 3.0.
        // If compiling against 3.0, then we can query.
        backend_ctx->non_uniform_workgroups = false;
#if CL_TARGET_OPENCL_VERSION >= 300
        CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT, sizeof(cl_bool),
                                 &backend_ctx->non_uniform_workgroups, 0));
#endif
    } else {
        GGML_ASSERT(opencl_c_version.major == 2);
        // Non-uniform workgroup sizes is mandatory feature in v2.x.
        backend_ctx->non_uniform_workgroups = true;
    }

    // Print out configurations
#ifdef GGML_OPENCL_SOA_Q
    GGML_LOG_INFO("ggml_opencl: flattening quantized weights representation as struct of arrays (GGML_OPENCL_SOA_Q)\n");
#endif // GGML_OPENCL_SOA_Q

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    GGML_LOG_INFO("ggml_opencl: using kernels optimized for Adreno (GGML_OPENCL_USE_ADRENO_KERNELS)\n");
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

    cl_int err;

    // A local ref of cl_context for convenience
    cl_context context = backend_ctx->context = dev_ctx->context;

    //CL_CHECK((queue = clCreateCommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err),
    //    (err != CL_INVALID_QUEUE_PROPERTIES && err != CL_INVALID_VALUE ? err :
    //    (queue = clCreateCommandQueue(context, device, 0, &err), err)
    //)));
    cl_command_queue_properties command_queue_props = 0;
#ifdef GGML_OPENCL_PROFILING
    command_queue_props |= CL_QUEUE_PROFILING_ENABLE;
#endif
    CL_CHECK((backend_ctx->queue = clCreateCommandQueue(context, device, command_queue_props, &err), err));

    // Load kernels
    load_cl_kernels(backend_ctx.get(), opencl_c_version);

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    // Allocate intermediate buffers and images
    size_t required_A_q_d_bytes = 311164928;
    size_t required_A_s_d_bytes = 38895616;
    size_t required_B_d_bytes = 45088768;

    // Ensure buffer sizes do not exceed the maximum allocation size
    size_t max_A_q_d_bytes = MIN(required_A_q_d_bytes, backend_ctx->max_alloc_size);
    size_t max_A_s_d_bytes = MIN(required_A_s_d_bytes, backend_ctx->max_alloc_size);
    size_t max_B_d_bytes   = MIN(required_B_d_bytes, backend_ctx->max_alloc_size);
    if (required_A_q_d_bytes > backend_ctx->max_alloc_size) {
        GGML_LOG_WARN("ggml_opencl: A_q_d buffer size reduced from %zu to %zu due to device limitations.\n",
                      required_A_q_d_bytes, max_A_q_d_bytes);
    }
    if (required_A_s_d_bytes > backend_ctx->max_alloc_size) {
        GGML_LOG_WARN("ggml_opencl: A_s_d buffer size reduced from %zu to %zu due to device limitations.\n",
                      required_A_s_d_bytes, max_A_s_d_bytes);
    }
    if (required_B_d_bytes > backend_ctx->max_alloc_size) {
        GGML_LOG_WARN("ggml_opencl: B_d buffer size reduced from %zu to %zu due to device limitations.\n",
                      required_B_d_bytes, max_B_d_bytes);
    }

    backend_ctx->prealloc_quant_trans.allocate(context, max_A_q_d_bytes);
    backend_ctx->prealloc_scales_trans.allocate(context, max_A_s_d_bytes);
    backend_ctx->prealloc_act_trans.allocate(context, max_B_d_bytes);
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

    backend_ctx->disable_fusion = getenv("GGML_OPENCL_DISABLE_FUSION") != nullptr;

    dev_ctx->backend_ctx = backend_ctx.release();
    return dev_ctx->backend_ctx;
}

static void ggml_cl2_free(ggml_backend_t backend) {
    ggml_backend_opencl_context * ctx = (ggml_backend_opencl_context *) backend->context;
    mldrift_attn_release_h24(ctx);
    mldrift_attn_release_h30(ctx);
    ctx->free();

    // The CL context is shared by all backends, release it if all backends have been released
    bool should_release_opencl = true;
    for (auto device : g_ggml_backend_opencl_devices) {
        ggml_backend_opencl_device_context * ctx_dev = (ggml_backend_opencl_device_context *) device.context;
        if (ctx_dev->backend_ctx->ref_count > 0) {
            should_release_opencl = false;
        }
    }

    if (should_release_opencl) {
        CL_CHECK(clReleaseContext(ctx->context));
    }
}

static void ggml_opencl_log_tensor_alloc(const ggml_tensor * tensor, const ggml_tensor_extra_cl * extra) {
    if (!ggml_opencl_log_tensor_offsets_enabled()) {
        return;
    }
    if (tensor == nullptr || extra == nullptr) {
        return;
    }
    if (!ggml_opencl_tensor_name_matches(tensor)) {
        return;
    }
    const char * name = tensor->name;
    if (name == nullptr || name[0] == '\0') {
        name = "";
    }
    const char * view_src_name = "";
    if (tensor->view_src != nullptr) {
        const char * view_name = tensor->view_src->name;
        if (view_name != nullptr && view_name[0] != '\0') {
            view_src_name = view_name;
        }
    }
    size_t offset = extra->offset + tensor->view_offs;
    GGML_LOG_INFO("opencl: tensor alloc name='%s' op=%s type=%s nbytes=%zu offset=%zu view_offs=%zu view_src='%s' data_device=%p\n",
                  name,
                  ggml_op_name(tensor->op),
                  ggml_type_name(tensor->type),
                  ggml_nbytes(tensor),
                  offset,
                  (size_t) tensor->view_offs,
                  view_src_name,
                  (void *) extra->data_device);
}

// Additional tensor extra structs for quantized tensors.
// These tensors are loaded from files and should not be allocated in scratch --
// they should always be allocated from the pool. Hence, they do not have an
// `offset`, which indicate their locations in the scratch buffer.
struct ggml_tensor_extra_cl_q4_0 {
    // Quantized values.
    cl_mem q = nullptr;
    // Optional AOS buffer (block_q4_0) for get_rows correctness.
    cl_mem aos = nullptr;
    // Quantized values in image1d_buffer_t.
    cl_mem q_img = nullptr;
    // Scales.
    cl_mem d = nullptr;
    // Scales in image1d_buffer_t.
    cl_mem d_img = nullptr;
    // Size of quantized values.
    size_t size_q = 0;
    // Size of scales.
    size_t size_d = 0;

    ~ggml_tensor_extra_cl_q4_0() {
        reset();
    }

    void reset() {
        // q and d are subbuffers into the bigger buffer allocated in ggml_backend_buffer.
        // They must be properly released so that the original buffer can be
        // properly released to avoid memory leak.
        if (q != nullptr) {
            CL_CHECK(clReleaseMemObject(q));
            q = nullptr;
        }
        if (aos != nullptr) {
            CL_CHECK(clReleaseMemObject(aos));
            aos = nullptr;
        }
        if (d != nullptr) {
            CL_CHECK(clReleaseMemObject(d));
            d = nullptr;
        }
        // Currently, q_img and d_img are only initialized when SMALL_ALLOC is
        // enabled. They point to the images in ggml_backend_opencl_buffer_context.
        // So, there is no need to release them here.
        // TODO: initialize them for non SMALL_PATH path, or remove them.
        q_img = nullptr;
        d_img = nullptr;
        size_q = 0;
        size_d = 0;
    }
};

struct ggml_tensor_extra_cl_mxfp4 {
    // Quantized values.
    cl_mem q = nullptr;
    // Quantized values in image1d_buffer_t.
    cl_mem q_img = nullptr;
    // Scales in E8M0.
    cl_mem e = nullptr;
    // Scales in image1d_buffer_t.
    cl_mem e_img = nullptr;
    // Size of quantized values.
    size_t size_q = 0;
    // Size of scales.
    size_t size_e = 0;

    ~ggml_tensor_extra_cl_mxfp4() {
        reset();
    }

    void reset() {
        // q and d are subbuffers into the bigger buffer allocated in ggml_backend_buffer.
        // They must be properly released so that the original buffer can be
        // properly released to avoid memory leak.
        if (q != nullptr) {
            CL_CHECK(clReleaseMemObject(q));
            q = nullptr;
        }
        if (e != nullptr) {
            CL_CHECK(clReleaseMemObject(e));
            e = nullptr;
        }
        if (q != nullptr) {
            CL_CHECK(clReleaseMemObject(q_img));
            q = nullptr;
        }
        // Currently, q_img and d_img are not used. They can be image1d_buffer_t
        // that wraps around q and d to utilize image access path.
        q_img = nullptr;
        e_img = nullptr;
        size_q = 0;
        size_e = 0;
    }
};

struct ggml_tensor_extra_cl_q8_0 {
    cl_mem q = nullptr;
    cl_mem q_img = nullptr;

    cl_mem d = nullptr;
    cl_mem d_img = nullptr;

    size_t size_q = 0;
    size_t size_d = 0;

    ~ggml_tensor_extra_cl_q8_0() {
        reset();
    }

    void reset() {
        // q and d are subbuffers into the bigger buffer allocated in ggml_backend_buffer.
        // They must be properly released so that the original buffer can be
        // properly released to avoid memory leak.
        if (q != nullptr) {
            CL_CHECK(clReleaseMemObject(q));
            q = nullptr;
        }
        if (d != nullptr) {
            CL_CHECK(clReleaseMemObject(d));
            d = nullptr;
        }
        // Currently, q_img and d_img are not used. They can be image1d_buffer_t
        // that wraps around q and d to utilize image access path.
        q_img = nullptr;
        d_img = nullptr;
        size_q = 0;
        size_d = 0;
    }
};

struct ggml_tensor_extra_cl_q6_K {
    // Lower 4 bits of quantized weights.
    cl_mem ql = nullptr;
    // Upper 2 bits of quantized weights.
    cl_mem qh = nullptr;
    // Scales for each block.
    cl_mem s  = nullptr;
    // Scales for each super block.
    cl_mem d  = nullptr;

    size_t size_ql = 0;
    size_t size_qh = 0;
    size_t size_s  = 0;
    size_t size_d  = 0;

    ~ggml_tensor_extra_cl_q6_K() {
        reset();
    }

    void reset() {
        if (ql != nullptr) {
            CL_CHECK(clReleaseMemObject(ql));
            ql = nullptr;
        }
        if (qh != nullptr) {
            CL_CHECK(clReleaseMemObject(qh));
            qh = nullptr;
        }
        if (s != nullptr) {
            CL_CHECK(clReleaseMemObject(s));
            s = nullptr;
        }
        if (d != nullptr) {
            CL_CHECK(clReleaseMemObject(d));
            d = nullptr;
        }

        size_ql = 0;
        size_qh = 0;
        size_s  = 0;
        size_d  = 0;
    }
};

//------------------------------------------------------------------------------
// Backend API
//------------------------------------------------------------------------------

//
// backend
//
static const char * ggml_backend_opencl_name(ggml_backend_t backend) {
    return "OpenCL";

    UNUSED(backend);
}

static void ggml_backend_opencl_free(ggml_backend_t backend) {
    ggml_cl2_free(backend);
}

static void ggml_backend_opencl_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_UNUSED(backend);
    GGML_UNUSED(tensor);
    GGML_UNUSED(data);
    GGML_UNUSED(offset);
    GGML_UNUSED(size);
}

static void ggml_backend_opencl_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_UNUSED(backend);
    GGML_UNUSED(tensor);
    GGML_UNUSED(data);
    GGML_UNUSED(offset);
    GGML_UNUSED(size);
}

static bool ggml_backend_opencl_cpy_tensor_async(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_UNUSED(backend);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    return false;
}

static void ggml_backend_opencl_synchronize(ggml_backend_t backend) {
    auto * backend_ctx = static_cast<ggml_backend_opencl_context *>(backend->context);

    cl_event evt;
    CL_CHECK(clEnqueueBarrierWithWaitList(backend_ctx->queue, 0, nullptr, &evt));
    CL_CHECK(clWaitForEvents(1, &evt));
    CL_CHECK(clReleaseEvent(evt));
}

// Syncronizes the 'backend_ctx's device with others so that commands
// enqueued to it won't start until commands in the other devices have
// completed.
static void sync_with_other_backends(ggml_backend_opencl_context * backend_ctx) {
    if (g_ggml_backend_opencl_devices.size() < 2)
      return; // No other devices to synchronize with.

    std::vector<cl_event> events;
    events.reserve(g_ggml_backend_opencl_devices.size());

    for (ggml_backend_device & backend_dev : g_ggml_backend_opencl_devices) {
        auto * other_backend_ctx = ggml_cl2_init(&backend_dev);
        if (backend_ctx != other_backend_ctx) {
            cl_event ev;
            CL_CHECK(clEnqueueMarkerWithWaitList(other_backend_ctx->queue, 0, nullptr, &ev));
            CL_CHECK(clFlush(other_backend_ctx->queue));
            events.push_back(ev);
        }
    }

    CL_CHECK(clEnqueueBarrierWithWaitList(backend_ctx->queue, events.size(), events.data(), nullptr));
    for (auto ev : events) {
        CL_CHECK(clReleaseEvent(ev));
    }
}

static void sync_with_other_backends(ggml_backend_t backend) {
    auto * backend_ctx = static_cast<ggml_backend_opencl_context *>(backend->context);
    sync_with_other_backends(backend_ctx);
}

static bool ggml_opencl_can_fuse(const struct ggml_cgraph * cgraph, int node_idx, std::initializer_list<enum ggml_op> ops) {
    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul      = cgraph->nodes[node_idx+1];

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        // rms_norm only supports f32
        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        // if rms_norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] &&
            !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        // rms_norm assumes contiguous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }
    } else if (ops.size() == 3 && ops.begin()[0] == GGML_OP_NORM && ops.begin()[1] == GGML_OP_MUL && ops.begin()[2] == GGML_OP_ADD) {
        const ggml_tensor *norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul  = cgraph->nodes[node_idx+1];
        const ggml_tensor *add  = cgraph->nodes[node_idx+2];
        const ggml_tensor *w    = mul->src[0] == norm ? mul->src[1] : mul->src[0];
        const ggml_tensor *b    = add->src[0] == mul  ? add->src[1] : add->src[0];

        // norm fusion only supports F32
        if (norm->src[0]->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
            return false;
        }

        if (norm->src[0]->ne[0] % 4 != 0) {
            return false;
        }

        if (!ggml_is_contiguous(norm->src[0]) || !ggml_is_contiguous(w) || !ggml_is_contiguous(b)) {
            return false;
        }
    } else if (ops.size() == 3 && ops.begin()[0] == GGML_OP_GROUP_NORM && ops.begin()[1] == GGML_OP_MUL && ops.begin()[2] == GGML_OP_ADD) {
        const ggml_tensor *gn = cgraph->nodes[node_idx];
        const ggml_tensor *mul = cgraph->nodes[node_idx+1];
        const ggml_tensor *add = cgraph->nodes[node_idx+2];
        const ggml_tensor *w   = mul->src[0] == gn ? mul->src[1] : mul->src[0];
        const ggml_tensor *b   = add->src[0] == mul ? add->src[1] : add->src[0];

        if (gn->src[0]->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
            return false;
        }

        if (!ggml_is_contiguous(gn->src[0]) || !ggml_is_contiguous(w) || !ggml_is_contiguous(b)) {
            return false;
        }
    }

    return true;
}

static void ggml_opencl_op_rms_norm_fused(ggml_backend_t backend, ggml_tensor * rms_norm_tensor, ggml_tensor * mul_tensor);
static void ggml_opencl_op_norm_fused(ggml_backend_t backend, ggml_tensor * norm_tensor, ggml_tensor * mul_tensor, ggml_tensor * add_tensor);
static void ggml_opencl_op_group_norm_fused(ggml_backend_t backend, ggml_tensor * gn_tensor, ggml_tensor * mul_tensor, ggml_tensor * add_tensor);

static ggml_status ggml_backend_opencl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    const bool op_timing = std::getenv("GGML_OPENCL_OP_TIMING") != nullptr;
    const bool op_timing_detail = std::getenv("GGML_OPENCL_OP_TIMING_DETAIL") != nullptr;
    std::map<std::string, double> op_ms;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        // NOTE: this may oversynchronize by synchronizing with
        //       backends/devices which don't compute 'cgraph's
        //       dependencies.
        sync_with_other_backends(backend);

        if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            continue;
        }

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        if (!backend_ctx->disable_fusion && ggml_opencl_can_fuse(cgraph, i, { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD })) {
            int64_t t0_us = 0;
            if (op_timing) {
                CL_CHECK(clFinish(backend_ctx->queue));
                t0_us = ggml_time_us();
            }
            ggml_opencl_op_norm_fused(backend, node, cgraph->nodes[i+1], cgraph->nodes[i+2]);
            if (op_timing) {
                CL_CHECK(clFinish(backend_ctx->queue));
                const double dt_ms = (ggml_time_us() - t0_us) * 1e-3;
                op_ms["fused_norm_mul_add"] += dt_ms;
                if (op_timing_detail) {
                    const ggml_tensor * src0 = node->src[0];
                    const ggml_tensor * src1 = node->src[1];
                    GGML_LOG_INFO("ggml_opencl: op timing detail ms=%.3f op=fused_norm_mul_add name='%s' ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] src0_ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] src1_ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                                  dt_ms, node->name, node->ne[0], node->ne[1], node->ne[2], node->ne[3],
                                  src0 ? src0->ne[0] : -1, src0 ? src0->ne[1] : -1, src0 ? src0->ne[2] : -1, src0 ? src0->ne[3] : -1,
                                  src1 ? src1->ne[0] : -1, src1 ? src1->ne[1] : -1, src1 ? src1->ne[2] : -1, src1 ? src1->ne[3] : -1);
                }
            }
            i += 2;
            continue;
        }
        if (!backend_ctx->disable_fusion && ggml_opencl_can_fuse(cgraph, i, { GGML_OP_GROUP_NORM, GGML_OP_MUL, GGML_OP_ADD })) {
            int64_t t0_us = 0;
            if (op_timing) {
                CL_CHECK(clFinish(backend_ctx->queue));
                t0_us = ggml_time_us();
            }
            ggml_opencl_op_group_norm_fused(backend, node, cgraph->nodes[i+1], cgraph->nodes[i+2]);
            if (op_timing) {
                CL_CHECK(clFinish(backend_ctx->queue));
                const double dt_ms = (ggml_time_us() - t0_us) * 1e-3;
                op_ms["fused_group_norm_mul_add"] += dt_ms;
                if (op_timing_detail) {
                    const ggml_tensor * src0 = node->src[0];
                    const ggml_tensor * src1 = node->src[1];
                    GGML_LOG_INFO("ggml_opencl: op timing detail ms=%.3f op=fused_group_norm_mul_add name='%s' ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] src0_ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] src1_ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                                  dt_ms, node->name, node->ne[0], node->ne[1], node->ne[2], node->ne[3],
                                  src0 ? src0->ne[0] : -1, src0 ? src0->ne[1] : -1, src0 ? src0->ne[2] : -1, src0 ? src0->ne[3] : -1,
                                  src1 ? src1->ne[0] : -1, src1 ? src1->ne[1] : -1, src1 ? src1->ne[2] : -1, src1 ? src1->ne[3] : -1);
                }
            }
            i += 2;
            continue;
        }
        if (!backend_ctx->disable_fusion && ggml_opencl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
            int64_t t0_us = 0;
            if (op_timing) {
                CL_CHECK(clFinish(backend_ctx->queue));
                t0_us = ggml_time_us();
            }
            ggml_opencl_op_rms_norm_fused(backend, node, cgraph->nodes[i+1]);
            if (op_timing) {
                CL_CHECK(clFinish(backend_ctx->queue));
                const double dt_ms = (ggml_time_us() - t0_us) * 1e-3;
                op_ms["fused_rms_norm_mul"] += dt_ms;
                if (op_timing_detail) {
                    const ggml_tensor * src0 = node->src[0];
                    const ggml_tensor * src1 = node->src[1];
                    GGML_LOG_INFO("ggml_opencl: op timing detail ms=%.3f op=fused_rms_norm_mul name='%s' ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] src0_ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] src1_ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                                  dt_ms, node->name, node->ne[0], node->ne[1], node->ne[2], node->ne[3],
                                  src0 ? src0->ne[0] : -1, src0 ? src0->ne[1] : -1, src0 ? src0->ne[2] : -1, src0 ? src0->ne[3] : -1,
                                  src1 ? src1->ne[0] : -1, src1 ? src1->ne[1] : -1, src1 ? src1->ne[2] : -1, src1 ? src1->ne[3] : -1);
                }
            }
            i++;
            continue;
        }

        int64_t t0_us = 0;
        if (op_timing) {
            CL_CHECK(clFinish(backend_ctx->queue));
            t0_us = ggml_time_us();
        }
        bool ok = ggml_cl_compute_forward(backend, node);
        if (op_timing) {
            CL_CHECK(clFinish(backend_ctx->queue));
            const double dt_ms = (ggml_time_us() - t0_us) * 1e-3;
            op_ms[ggml_op_name(node->op)] += dt_ms;
            if (op_timing_detail) {
                const ggml_tensor * src0 = node->src[0];
                const ggml_tensor * src1 = node->src[1];
                GGML_LOG_INFO("ggml_opencl: op timing detail ms=%.3f op=%s name='%s' dst_type=%s src0_type=%s src1_type=%s ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] src0_ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] src1_ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                              dt_ms, ggml_op_name(node->op), node->name,
                              ggml_type_name(node->type),
                              src0 ? ggml_type_name(src0->type) : "nil",
                              src1 ? ggml_type_name(src1->type) : "nil",
                              node->ne[0], node->ne[1], node->ne[2], node->ne[3],
                              src0 ? src0->ne[0] : -1, src0 ? src0->ne[1] : -1, src0 ? src0->ne[2] : -1, src0 ? src0->ne[3] : -1,
                              src1 ? src1->ne[0] : -1, src1 ? src1->ne[1] : -1, src1 ? src1->ne[2] : -1, src1 ? src1->ne[3] : -1);
            }
        }
        if (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
        }
        GGML_ASSERT(ok);
    }

    if (op_timing && !op_ms.empty()) {
        std::vector<std::pair<std::string, double>> op_vec(op_ms.begin(), op_ms.end());
        std::sort(op_vec.begin(), op_vec.end(), [](const auto & a, const auto & b) { return a.second > b.second; });
        GGML_LOG_INFO("ggml_opencl: op timing summary (ms, top %zu)\n", std::min<size_t>(12, op_vec.size()));
        for (size_t i = 0; i < op_vec.size() && i < 12; ++i) {
            GGML_LOG_INFO("  %2zu. %-24s %.3f\n", i + 1, op_vec[i].first.c_str(), op_vec[i].second);
        }
    }

    return GGML_STATUS_SUCCESS;
}

static bool ggml_opencl_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    ggml_backend_opencl_device_context * dev_ctx     = (ggml_backend_opencl_device_context *)dev->context;
    ggml_backend_opencl_context *        backend_ctx = dev_ctx->backend_ctx;
    const bool disable_softmax = getenv("GGML_OPENCL_DISABLE_SOFTMAX") != nullptr;
    const bool disable_rope    = getenv("GGML_OPENCL_DISABLE_ROPE") != nullptr;
    const bool disable_rmsnorm = getenv("GGML_OPENCL_DISABLE_RMSNORM") != nullptr;
    const bool disable_concat  = getenv("GGML_OPENCL_DISABLE_CONCAT") != nullptr;

    switch (op->op) {
        case GGML_OP_NONE:
            return true;
        case GGML_OP_GET_ROWS:
            switch (op->src[0]->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                    return true;
                case GGML_TYPE_Q4_0:
#ifdef GGML_OPENCL_SOA_Q
                    // We do not support flattened Q4_0 (and possibly other Q's)
                    return false;
#else // GGML_OPENCL_SOA_Q
                    return true;
#endif // GGML_OPENCL_SOA_Q
                default:
                    return false;
            }
        case GGML_OP_SET_ROWS:
            {
                // TODO: add support
                // ref: https://github.com/ggml-org/llama.cpp/pull/14274
#pragma message("TODO: implement BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL support (https://github.com/ggml-org/llama.cpp/pull/14661)")
                if (op->src[0]->type != GGML_TYPE_F32) {
                    return false;
                }
                switch (op->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                        return (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
                    default:
                        return false;
                }
            }
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
            switch (op->src[0]->type) {
                case GGML_TYPE_F32:
                    switch (op->type) {
                        case GGML_TYPE_F16:
                        case GGML_TYPE_F32:
                            return true;
                        default:
                            return false;
                    }
                case GGML_TYPE_F16:
                    switch (op->type) {
                        case GGML_TYPE_F16:
                        case GGML_TYPE_F32:
                            return true;
                        default:
                            return false;
                    }
                default:
                    return false;
            }
        case GGML_OP_SCALE:
            return op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]);
        case GGML_OP_ADD:
            if (op->type == GGML_TYPE_F16) {
                const bool src0_ok = op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32;
                const bool src1_ok = op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32;
                if (src0_ok && src1_ok) {
                    return true;
                }
            }
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SUB:
            return (op->src[0]->type == op->src[1]->type) &&
                   (op->src[0]->type == op->type) &&
                   (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
        case GGML_OP_ADD_ID:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                    ggml_is_contiguous(op->src[0]);
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                   return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
                case GGML_UNARY_OP_SIGMOID:
                    return ggml_is_contiguous(op->src[0]);
                case GGML_UNARY_OP_TANH:
                   return (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) ||
                          (op->src[0]->type == GGML_TYPE_F16 && op->type == GGML_TYPE_F16);
                case GGML_UNARY_OP_EXPM1:
                   return (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) ||
                          (op->src[0]->type == GGML_TYPE_F16 && op->type == GGML_TYPE_F16);
                case GGML_UNARY_OP_SOFTPLUS:
                   return (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) ||
                          (op->src[0]->type == GGML_TYPE_F16 && op->type == GGML_TYPE_F16);
                default:
                    return false;
            }
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    return ggml_is_contiguous_1(op->src[0]) && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
                default:
                    return false;
            }
        case GGML_OP_TRI:
            return op->type == GGML_TYPE_F32 && ggml_is_contiguous(op);
        case GGML_OP_FILL:
            return op->type == GGML_TYPE_F32 && ggml_is_contiguous(op);
        case GGML_OP_CLAMP:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_SOFT_MAX:
            if (disable_softmax) {
                return false;
            }
        case GGML_OP_NORM:
            return true;
        case GGML_OP_RMS_NORM:
            if (disable_rmsnorm) {
                return false;
            }
            return op->ne[0] % 4 == 0 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_REPEAT:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32; // Assuming F32 for now, can be expanded
        case GGML_OP_PAD:
            // TODO: add circular padding support for opencl, see https://github.com/ggml-org/llama.cpp/pull/16985
            if (ggml_get_op_params_i32(op, 8) != 0) {
                return false;
            }
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_UPSCALE: {
            ggml_scale_mode mode = (ggml_scale_mode)(ggml_get_op_params_i32(op, 0) & 0xFF);
            const bool antialias = (ggml_scale_mode)(ggml_get_op_params_i32(op, 0) & GGML_SCALE_FLAG_ANTIALIAS);
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   (mode == GGML_SCALE_MODE_NEAREST || mode == GGML_SCALE_MODE_BILINEAR) && !antialias;
        }
        case GGML_OP_CONV_2D:
            return (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F16 && op->type == GGML_TYPE_F16) ||
                   (op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) ||
                   (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32);
        case GGML_OP_SSM_CONV:
            return (op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32);
        case GGML_OP_CONCAT:
            if (disable_concat) {
                return false;
            }
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_TIMESTEP_EMBEDDING:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_GROUP_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_MUL_MAT:
            if (op->src[0]->type == GGML_TYPE_F16) {
                return true;
            } else if (op->src[0]->type == GGML_TYPE_F32) {
                return op->src[1]->type == GGML_TYPE_F32;
            } else if (op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_MXFP4 ||
                       op->src[0]->type == GGML_TYPE_Q6_K) {
                return op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
            } else if (op->src[0]->type == GGML_TYPE_Q8_0) {
                return op->src[1]->type == GGML_TYPE_F32;
            }
            return false;
        case GGML_OP_MUL_MAT_ID:
            if (op->src[0]->type == GGML_TYPE_Q4_0 ||
                op->src[0]->type == GGML_TYPE_Q8_0 ||
                op->src[0]->type == GGML_TYPE_MXFP4) {
                if (op->src[1]->type == GGML_TYPE_F32) {
                    return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
                }
            }
            return false;
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_DIAG_MASK_INF:
            return op->ne[3] == 1;
        case GGML_OP_ROPE: {
            if (disable_rope) {
                return false;
            }
            const int mode = ((const int32_t *) op->op_params)[2];
            const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
            const bool is_vision = mode == GGML_ROPE_TYPE_VISION;
            if (is_mrope && !is_vision) {
                if (op->src[0]->type == GGML_TYPE_F32 ||
                    op->src[0]->type == GGML_TYPE_F16) {
                    return true;
                }
                return false;
            }
            if (is_vision) {
                if (op->src[0]->type == GGML_TYPE_F32 ||
                    op->src[0]->type == GGML_TYPE_F16) {
                    return true;
                }
                return false;
            }
            return true;
        }
        case GGML_OP_SOLVE_TRI:
            return op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]);
        case GGML_OP_IM2COL:
            return true;
        case GGML_OP_ARGSORT: {
            cl_kernel kernel = backend_ctx->kernel_argsort_f32_i32;
            int max_workgroup_size = backend_ctx->get_kernel_workgroup_size(kernel);

            int cols = 1;
            while (cols < op->ne[0]) {
                cols *= 2;
            }

            return cols <= max_workgroup_size && op->src[0]->type == GGML_TYPE_F32;
        }
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
            return op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]);
        case GGML_OP_FLASH_ATTN_EXT:
            {
                const ggml_tensor * q = op->src[0];
                const ggml_tensor * k = op->src[1];
                const ggml_tensor * v = op->src[2];

                const int dk = q->ne[0];
                const int dv = v->ne[0];

                const struct { int dk; int dv; } supported_dims[] = {
                    { 40,  40}, { 64,  64}, { 80,  80}, { 96,  96},
                    {112, 112}, {128, 128}, {192, 128},
                    {192, 192}, {256, 256},
                };

                bool dims_supported = false;
                for (size_t i = 0; i < sizeof(supported_dims)/sizeof(supported_dims[0]); ++i) {
                    if (supported_dims[i].dk == dk && supported_dims[i].dv == dv) {
                        dims_supported = true;
                        break;
                    }
                }
                if (!dims_supported) {
                    return false;
                }

                const bool is_f32_f32 = q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32 &&
                                        v->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
                const bool is_f16_f16 = q->type == GGML_TYPE_F16 && k->type == GGML_TYPE_F16 &&
                                        v->type == GGML_TYPE_F16 && op->type == GGML_TYPE_F16;
                const bool is_f32_f16 = q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 &&
                                        v->type == GGML_TYPE_F16 && op->type == GGML_TYPE_F32;

                return is_f32_f32 || is_f16_f16 || is_f32_f16;
            }
        default:
            return false;
    }
}

// Forward declaration - implementation appears later in the file.
static const char * ggml_backend_opencl_buffer_type_get_name(ggml_backend_buffer_type_t buffer_type);

static ggml_guid_t ggml_backend_opencl_guid() {
    static ggml_guid guid = { 0xde, 0xe0, 0x70, 0xa2, 0x73, 0x4e, 0x4d, 0xbc, 0xb0, 0xc7, 0x4f, 0xd4, 0x6d, 0x4e, 0x90, 0xfe };
    return &guid;
}

static ggml_backend_i ggml_backend_opencl_i = {
    /* .get_name                = */ ggml_backend_opencl_name,
    /* .free                    = */ ggml_backend_opencl_free,
    /* .set_tensor_async        = */ NULL,  /* ggml_backend_opencl_set_tensor_async */
    /* .get_tensor_async        = */ NULL,  /* ggml_backend_opencl_get_tensor_async */
    /* .cpy_tensor_async        = */ NULL,  /* ggml_backend_opencl_cpy_tensor_async */
    /* .synchronize             = */ ggml_backend_opencl_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_opencl_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

ggml_backend_t ggml_backend_opencl_init(void) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_opencl_reg(), 0);
    ggml_backend_opencl_context *backend_ctx = ggml_cl2_init(dev);

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_opencl_guid(),
        /* .iface   = */ ggml_backend_opencl_i,
        /* .device  = */ dev,
        /* .context = */ backend_ctx
    };

    return backend;
}

bool ggml_backend_is_opencl(ggml_backend_t backend) {
    return backend && backend->iface.get_name == ggml_backend_opencl_name;
}

//
// buffer
//
struct ggml_backend_opencl_buffer_context {
    // A buffer context can hold multiple cl_mem objects. This is for flattening
    // quantized weights and should be used with GGML_OPENCL_SMALL_ALLOC where
    // each tensor is allocated a separate buffer. When flattening is enabled
    // with small allocation, each tensor is backed by two cl_mem objects (for
    // quants and scales) packed into a backend_opencl_buffer.
    ggml_backend_opencl_buffer_context(cl_mem buf)
        : name("OpenCL") {
        buffer.push_back(buf);
    }

    ~ggml_backend_opencl_buffer_context() {
        // Extras can be allocated from multiple loader threads while tensors are uploaded.
        // Guard pool teardown and recycle paths with the same mutex to avoid vector races.
        std::lock_guard<std::mutex> lock(temp_tensor_extras_mutex);
        for (cl_mem buf : buffer) {
            CL_CHECK(clReleaseMemObject(buf));
        }
        for (cl_mem im : img) {
            CL_CHECK(clReleaseMemObject(im));
        }

        // Delete all extras to trigger their destructors
        for (ggml_tensor_extra_cl * e : temp_tensor_extras) {
            delete e;
        }
        for (ggml_tensor_extra_cl * e : temp_tensor_extras_in_use) {
            delete e;
        }
        for (ggml_tensor_extra_cl_q4_0 * e : temp_tensor_extras_q4_0) {
            delete e;
        }
        for (ggml_tensor_extra_cl_q4_0 * e : temp_tensor_extras_q4_0_in_use) {
            delete e;
        }
        for (ggml_tensor_extra_cl_mxfp4 * e : temp_tensor_extras_mxfp4) {
            delete e;
        }
        for (ggml_tensor_extra_cl_mxfp4 * e : temp_tensor_extras_mxfp4_in_use) {
            delete e;
        }
        for (ggml_tensor_extra_cl_q8_0 * e : temp_tensor_extras_q8_0) {
            delete e;
        }
        for (ggml_tensor_extra_cl_q8_0 * e : temp_tensor_extras_q8_0_in_use) {
            delete e;
        }
        for (ggml_tensor_extra_cl_q6_K * e : temp_tensor_extras_q6_K) {
            delete e;
        }
        for (ggml_tensor_extra_cl_q6_K * e : temp_tensor_extras_q6_K_in_use) {
            delete e;
        }
    }

    ggml_tensor_extra_cl * ggml_opencl_alloc_temp_tensor_extra() {
        std::lock_guard<std::mutex> lock(temp_tensor_extras_mutex);
        ggml_tensor_extra_cl * extra;
        if (temp_tensor_extras.empty()) {
            extra = new ggml_tensor_extra_cl();
        } else {
            extra = temp_tensor_extras.back();
            temp_tensor_extras.pop_back();
        }

        temp_tensor_extras_in_use.push_back(extra);

        extra->reset();
        return extra;
    }

    ggml_tensor_extra_cl_q4_0 * ggml_opencl_alloc_temp_tensor_extra_q4_0() {
        std::lock_guard<std::mutex> lock(temp_tensor_extras_mutex);
        ggml_tensor_extra_cl_q4_0 * extra;
        if (temp_tensor_extras_q4_0.empty()) {
            extra = new ggml_tensor_extra_cl_q4_0();
        } else {
            extra = temp_tensor_extras_q4_0.back();
            temp_tensor_extras_q4_0.pop_back();
        }

        temp_tensor_extras_q4_0_in_use.push_back(extra);

        extra->reset();
        return extra;
    }

    ggml_tensor_extra_cl_mxfp4 * ggml_opencl_alloc_temp_tensor_extra_mxfp4() {
        std::lock_guard<std::mutex> lock(temp_tensor_extras_mutex);
        ggml_tensor_extra_cl_mxfp4 * extra;
        if (temp_tensor_extras_mxfp4.empty()) {
            extra = new ggml_tensor_extra_cl_mxfp4();
        } else {
            extra = temp_tensor_extras_mxfp4.back();
            temp_tensor_extras_mxfp4.pop_back();
        }

        temp_tensor_extras_mxfp4_in_use.push_back(extra);

        extra->reset();
        return extra;
    }

    ggml_tensor_extra_cl_q8_0 * ggml_opencl_alloc_temp_tensor_extra_q8_0() {
        std::lock_guard<std::mutex> lock(temp_tensor_extras_mutex);
        ggml_tensor_extra_cl_q8_0 * extra;
        if (temp_tensor_extras_q8_0.empty()) {
            extra = new ggml_tensor_extra_cl_q8_0();
        } else {
            extra = temp_tensor_extras_q8_0.back();
            temp_tensor_extras_q8_0.pop_back();
        }

        temp_tensor_extras_q8_0_in_use.push_back(extra);

        extra->reset();
        return extra;
    }

    ggml_tensor_extra_cl_q6_K * ggml_opencl_alloc_temp_tensor_extra_q6_K() {
        std::lock_guard<std::mutex> lock(temp_tensor_extras_mutex);
        ggml_tensor_extra_cl_q6_K * extra;
        if (temp_tensor_extras_q6_K.empty()) {
            extra = new ggml_tensor_extra_cl_q6_K();
        } else {
            extra = temp_tensor_extras_q6_K.back();
            temp_tensor_extras_q6_K.pop_back();
        }

        temp_tensor_extras_q6_K_in_use.push_back(extra);

        extra->reset();
        return extra;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(temp_tensor_extras_mutex);
        for (ggml_tensor_extra_cl * e : temp_tensor_extras_in_use) {
            temp_tensor_extras.push_back(e);
        }
        temp_tensor_extras_in_use.clear();

        for (ggml_tensor_extra_cl_q4_0 * e : temp_tensor_extras_q4_0_in_use) {
            temp_tensor_extras_q4_0.push_back(e);
        }
        temp_tensor_extras_q4_0_in_use.clear();

        for (ggml_tensor_extra_cl_mxfp4 * e : temp_tensor_extras_mxfp4_in_use) {
            temp_tensor_extras_mxfp4.push_back(e);
        }
        temp_tensor_extras_mxfp4_in_use.clear();

        for (ggml_tensor_extra_cl_q8_0 * e : temp_tensor_extras_q8_0_in_use) {
            temp_tensor_extras_q8_0.push_back(e);
        }
        temp_tensor_extras_q8_0_in_use.clear();

        for (ggml_tensor_extra_cl_q6_K * e : temp_tensor_extras_q6_K_in_use) {
            temp_tensor_extras_q6_K.push_back(e);
        }
        temp_tensor_extras_q6_K_in_use.clear();
    }

    // Pools for extras. Available extras are in `temp_tensor_extras`. Extras
    // being used are in `temp_tensor_extras_in_use`. At the first run, new
    // extras get created and put in `in_use`. When the buffer is reset via
    // the `reset` callback, all extras in `in_use` get moved to available extras
    // for reuse.
    std::vector<ggml_tensor_extra_cl *> temp_tensor_extras;
    std::vector<ggml_tensor_extra_cl *> temp_tensor_extras_in_use;
    std::vector<ggml_tensor_extra_cl_q4_0 *> temp_tensor_extras_q4_0;
    std::vector<ggml_tensor_extra_cl_q4_0 *> temp_tensor_extras_q4_0_in_use;
    std::vector<ggml_tensor_extra_cl_mxfp4 *> temp_tensor_extras_mxfp4;
    std::vector<ggml_tensor_extra_cl_mxfp4 *> temp_tensor_extras_mxfp4_in_use;
    std::vector<ggml_tensor_extra_cl_q8_0 *> temp_tensor_extras_q8_0;
    std::vector<ggml_tensor_extra_cl_q8_0 *> temp_tensor_extras_q8_0_in_use;
    std::vector<ggml_tensor_extra_cl_q6_K *> temp_tensor_extras_q6_K;
    std::vector<ggml_tensor_extra_cl_q6_K *> temp_tensor_extras_q6_K_in_use;
    std::mutex temp_tensor_extras_mutex;

    // The buffer_context is initially created by ggml_backend_buft_alloc_buffer
    // before any tensor is initialized (at the beginning of alloc_tensor_range).
    // Hence, there is alway a buffer object in this vector. When each tensor is
    // being initialized, this original buffer object will be released if both
    // flattening and small allocation are enabled, and additional buffer
    // objects will be created in init_tensor to represent flattened quantized
    // weights.
    std::vector<cl_mem> buffer;
    // These are image1d_buffer_t objects that wrap around the quants and scales.
    // For Q4_0 quantization, there should be two of them - one for quants and
    // one for scales. They should be populated only when flattening and small
    // allocation are enabled.
    std::vector<cl_mem> img;
    std::string name;
};

static void ggml_backend_opencl_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
    delete ctx;
}

static void * ggml_backend_opencl_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_opencl_context * backend_ctx = ggml_cl2_init(buffer->buft->device);
    return (void *) (uintptr_t) backend_ctx->alignment;
}

static enum ggml_status ggml_backend_opencl_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;

    ggml_cl2_init(buffer->buft->device);

    if (tensor->view_src != nullptr) {
        GGML_ASSERT(tensor->view_src->buffer->buft == buffer->buft);

        ggml_tensor_extra_cl * view_extra = (ggml_tensor_extra_cl *) tensor->view_src->extra;
        GGML_ASSERT(view_extra && "view_extra is nullptr?");

        // Reuse extra of the parent tensor. The offset of this view tensor
        // becomes `extra->offset + view_offs` and needs to be calculated when
        // it is used. This changes is needed because of the change to
        // ggml_alloc.c in https://github.com/ggerganov/llama.cpp/pull/7640.
        // `buffer` passed in here will always be `tensor->buffer`. It is OK
        // to allocate extras from the same buffer context for ordinary
        // intermediate tensors. But for views into kv cache tensors, doing so
        // would mess up the extras used by kv cache.
        // Before #7640, `buffer` is for intermediate tensors, which is always
        // different from that of kv cache tensors.
        //
        // NB: now extra->offset no longer accounts for view_offs.
        // NB: this should not apply to weight tensors (for end-to-end runs, but
        //     may apply for test-backend-ops).
        // FIXME: if any unexpected results are seen, double check the offset -
        // there could be other places that need fix.
        tensor->extra = view_extra;
    } else {
        {
            size_t offset = (char *) tensor->data - (char *) ggml_backend_opencl_buffer_get_base(buffer);

            ggml_tensor_extra_cl * extra = ctx->ggml_opencl_alloc_temp_tensor_extra();
            extra->offset = offset;
            extra->data_device = ctx->buffer[0];
            extra->actual_size = ggml_nbytes(tensor);

            tensor->extra = extra;
        }
    }
    ggml_opencl_log_tensor_alloc(tensor, (ggml_tensor_extra_cl *) tensor->extra);
    return GGML_STATUS_SUCCESS;
}

// The optimized gemm and gemv kernels are used for large matrices without batch.
// tensor is the quantized weights matrix.
static inline bool ggml_opencl_disable_adreno_q4_substr(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_Q4_0) {
        return false;
    }
    const char * name = tensor->name;
    if (name == nullptr || name[0] == '\0') {
        return false;
    }
    const char * pats = std::getenv("SD_OCL_DISABLE_ADRENO_Q4_SUBSTR");
    if (pats == nullptr || pats[0] == '\0') {
        return false;
    }
    std::string s(pats);
    size_t start = 0;
    while (start < s.size()) {
        size_t end = s.find(',', start);
        if (end == std::string::npos) {
            end = s.size();
        }
        size_t l = start;
        size_t r = end;
        while (l < r && std::isspace(static_cast<unsigned char>(s[l]))) {
            ++l;
        }
        while (r > l && std::isspace(static_cast<unsigned char>(s[r - 1]))) {
            --r;
        }
        if (r > l) {
            std::string tok = s.substr(l, r - l);
            if (std::strstr(name, tok.c_str()) != nullptr) {
                return true;
            }
        }
        start = end + 1;
    }
    return false;
}

inline bool use_adreno_kernels(const ggml_backend_opencl_context *backend_ctx, const ggml_tensor *tensor) {
    if (tensor != nullptr &&
        tensor->type == GGML_TYPE_Q4_0 &&
        (std::getenv("GGML_OPENCL_DISABLE_ADRENO_Q4") != nullptr || ggml_opencl_disable_adreno_q4_substr(tensor))) {
        return false;
    }
    int64_t threshold_ne0 = 512;
    int64_t threshold_ne1 = 512;
    if (!backend_ctx->adreno_cl_compiler_version.newer_than_or_same(E031, 38, 11, 0) &&
         backend_ctx->adreno_cl_compiler_version.type != DX) {
        threshold_ne0 = 128;
        threshold_ne1 = 128;
    }
    return tensor->ne[0] >= threshold_ne0 && tensor->ne[1] >= threshold_ne1 &&
            tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

static inline bool ggml_opencl_is_embedding_tensor(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return false;
    }
    const char * name = tensor->name;
    if (name == nullptr || name[0] == '\0') {
        // Fall through to shape heuristic below.
    }
    if (name != nullptr && name[0] != '\0') {
        if (strstr(name, "embed_tokens.weight") != nullptr) {
            return true;
        }
        if (strstr(name, "tok_embeddings") != nullptr) {
            return true;
        }
        if (strstr(name, "wte.weight") != nullptr) {
            return true;
        }
        if (strstr(name, "token_embedding") != nullptr) {
            return true;
        }
    }

    // Fallback: large vocab-sized 2D weights are almost always embeddings.
    // Use a conservative threshold to avoid disabling reorder for MLP/attn.
    int64_t vocab_threshold = 32000;
    if (const char * env = std::getenv("GGML_OPENCL_EMBED_VOCAB_MIN")) {
        char * end = nullptr;
        long val = std::strtol(env, &end, 10);
        if (end != env && val > 0) {
            vocab_threshold = val;
        }
    }
    if (tensor->ne[2] == 1 && tensor->ne[3] == 1 &&
        tensor->ne[1] >= vocab_threshold && tensor->ne[0] >= 256) {
        return true;
    }

    return false;
}

static inline bool ggml_opencl_force_no_reorder(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return false;
    }
    const char * name = tensor->name;
    if (name == nullptr || name[0] == '\0') {
        return false;
    }
    if (std::getenv("SD_OCL_NO_REORDER_VPROJ") != nullptr) {
        if (std::strstr(name, "v_proj.weight") != nullptr) {
            return true;
        }
    }
    if (std::getenv("SD_OCL_NO_REORDER_LLM_MLP") != nullptr) {
        if (std::strstr(name, "mlp.gate_proj.weight") != nullptr ||
            std::strstr(name, "mlp.up_proj.weight") != nullptr ||
            std::strstr(name, "mlp.down_proj.weight") != nullptr) {
            return true;
        }
    }
    if (const char * pats = std::getenv("SD_OCL_NO_REORDER_SUBSTR")) {
        std::string s(pats);
        size_t start = 0;
        while (start < s.size()) {
            size_t end = s.find(',', start);
            if (end == std::string::npos) {
                end = s.size();
            }
            size_t l = start;
            size_t r = end;
            while (l < r && std::isspace(static_cast<unsigned char>(s[l]))) {
                ++l;
            }
            while (r > l && std::isspace(static_cast<unsigned char>(s[r - 1]))) {
                --r;
            }
            if (r > l) {
                std::string tok = s.substr(l, r - l);
                if (std::strstr(name, tok.c_str()) != nullptr) {
                    return true;
                }
            }
            start = end + 1;
        }
    }
    return false;
}

static inline bool ggml_opencl_force_kahan_gemm(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return false;
    }
    const char * name = tensor->name;
    if (name == nullptr || name[0] == '\0') {
        return false;
    }
    const char * pats = std::getenv("SD_OCL_Q4_GEMM_KAHAN_SUBSTR");
    if (pats == nullptr || pats[0] == '\0') {
        return false;
    }
    std::string s(pats);
    size_t start = 0;
    while (start < s.size()) {
        size_t end = s.find(',', start);
        if (end == std::string::npos) {
            end = s.size();
        }
        size_t l = start;
        size_t r = end;
        while (l < r && std::isspace(static_cast<unsigned char>(s[l]))) {
            ++l;
        }
        while (r > l && std::isspace(static_cast<unsigned char>(s[r - 1]))) {
            --r;
        }
        if (r > l) {
            std::string tok = s.substr(l, r - l);
            if (std::strstr(name, tok.c_str()) != nullptr) {
                return true;
            }
        }
        start = end + 1;
    }
    return false;
}

static inline bool ggml_opencl_force_silu_cpu(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return false;
    }
    const char * name = tensor->name;
    if (name == nullptr || name[0] == '\0') {
        return false;
    }
    const char * pats = std::getenv("GGML_OPENCL_SILU_CPU_SUBSTR");
    if (pats == nullptr || pats[0] == '\0') {
        return false;
    }
    std::string s(pats);
    size_t start = 0;
    while (start < s.size()) {
        size_t end = s.find(',', start);
        if (end == std::string::npos) {
            end = s.size();
        }
        size_t l = start;
        size_t r = end;
        while (l < r && std::isspace(static_cast<unsigned char>(s[l]))) {
            ++l;
        }
        while (r > l && std::isspace(static_cast<unsigned char>(s[r - 1]))) {
            --r;
        }
        if (r > l) {
            std::string tok = s.substr(l, r - l);
            if (std::strstr(name, tok.c_str()) != nullptr) {
                return true;
            }
        }
        start = end + 1;
    }
    return false;
}

static inline bool ggml_opencl_force_f32_act_gemm(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return false;
    }
    const char * name = tensor->name;
    if (name == nullptr || name[0] == '\0') {
        return false;
    }
    const char * pats = std::getenv("SD_OCL_Q4_GEMM_F32_ACT_SUBSTR");
    if (pats != nullptr && pats[0] != '\0') {
        std::string s(pats);
        size_t start = 0;
        while (start < s.size()) {
            size_t end = s.find(',', start);
            if (end == std::string::npos) {
                end = s.size();
            }
            size_t l = start;
            size_t r = end;
            while (l < r && std::isspace(static_cast<unsigned char>(s[l]))) {
                ++l;
            }
            while (r > l && std::isspace(static_cast<unsigned char>(s[r - 1]))) {
                --r;
            }
            if (r > l) {
                std::string tok = s.substr(l, r - l);
                if (std::strstr(name, tok.c_str()) != nullptr) {
                    return true;
                }
            }
            start = end + 1;
        }
    }

    // Auto route for Z-Image: the FP16-activation Q4 GEMM path can produce
    // NaNs on attention projections on Adreno. Keep this narrow and opt-out-able.
    if (std::getenv("SD_OCL_Q4_GEMM_F32_ACT_NO_AUTO") == nullptr) {
        if (std::strstr(name, "attention.qkv.weight") != nullptr ||
            std::strstr(name, "attention.out.weight") != nullptr) {
            return true;
        }
    }

    return false;
}

static inline bool ggml_opencl_force_f32_acc_gemm(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return false;
    }
    const char * name = tensor->name;
    if (name == nullptr || name[0] == '\0') {
        return false;
    }
    const char * pats = std::getenv("SD_OCL_Q4_GEMM_F32_ACC_SUBSTR");
    if (pats == nullptr || pats[0] == '\0') {
        return false;
    }
    std::string s(pats);
    size_t start = 0;
    while (start < s.size()) {
        size_t end = s.find(',', start);
        if (end == std::string::npos) {
            end = s.size();
        }
        size_t l = start;
        size_t r = end;
        while (l < r && std::isspace(static_cast<unsigned char>(s[l]))) {
            ++l;
        }
        while (r > l && std::isspace(static_cast<unsigned char>(s[r - 1]))) {
            --r;
        }
        if (r > l) {
            std::string tok = s.substr(l, r - l);
            if (std::strstr(name, tok.c_str()) != nullptr) {
                return true;
            }
        }
        start = end + 1;
    }
    return false;
}

static inline bool ggml_opencl_force_f32_read_gemm(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return false;
    }
    const char * name = tensor->name;
    if (name == nullptr || name[0] == '\0') {
        return false;
    }
    const char * pats = std::getenv("SD_OCL_Q4_GEMM_F32_READ_SUBSTR");
    if (pats == nullptr || pats[0] == '\0') {
        return false;
    }
    std::string s(pats);
    size_t start = 0;
    while (start < s.size()) {
        size_t end = s.find(',', start);
        if (end == std::string::npos) {
            end = s.size();
        }
        size_t l = start;
        size_t r = end;
        while (l < r && std::isspace(static_cast<unsigned char>(s[l]))) {
            ++l;
        }
        while (r > l && std::isspace(static_cast<unsigned char>(s[r - 1]))) {
            --r;
        }
        if (r > l) {
            std::string tok = s.substr(l, r - l);
            if (std::strstr(name, tok.c_str()) != nullptr) {
                return true;
            }
        }
        start = end + 1;
    }
    return false;
}

static inline bool ggml_opencl_force_half_acc_clamp_gemm(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return false;
    }
    const char * name = tensor->name;
    if (name == nullptr || name[0] == '\0') {
        return false;
    }
    const char * pats = std::getenv("SD_OCL_Q4_GEMM_HALF_ACC_CLAMP_SUBSTR");
    if (pats == nullptr || pats[0] == '\0') {
        return false;
    }
    std::string s(pats);
    size_t start = 0;
    while (start < s.size()) {
        size_t end = s.find(',', start);
        if (end == std::string::npos) {
            end = s.size();
        }
        size_t l = start;
        size_t r = end;
        while (l < r && std::isspace(static_cast<unsigned char>(s[l]))) {
            ++l;
        }
        while (r > l && std::isspace(static_cast<unsigned char>(s[r - 1]))) {
            --r;
        }
        if (r > l) {
            std::string tok = s.substr(l, r - l);
            if (std::strstr(name, tok.c_str()) != nullptr) {
                return true;
            }
        }
        start = end + 1;
    }
    return false;
}

static inline bool ggml_opencl_force_half_scale_gemm(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return false;
    }
    const char * name = tensor->name;
    if (name == nullptr || name[0] == '\0') {
        return false;
    }
    const char * pats = std::getenv("SD_OCL_Q4_GEMM_HALF_SCALE_SUBSTR");
    if (pats == nullptr || pats[0] == '\0') {
        return false;
    }
    std::string s(pats);
    size_t start = 0;
    while (start < s.size()) {
        size_t end = s.find(',', start);
        if (end == std::string::npos) {
            end = s.size();
        }
        size_t l = start;
        size_t r = end;
        while (l < r && std::isspace(static_cast<unsigned char>(s[l]))) {
            ++l;
        }
        while (r > l && std::isspace(static_cast<unsigned char>(s[r - 1]))) {
            --r;
        }
        if (r > l) {
            std::string tok = s.substr(l, r - l);
            if (std::strstr(name, tok.c_str()) != nullptr) {
                return true;
            }
        }
        start = end + 1;
    }
    return false;
}

static inline bool ggml_opencl_force_fp16_chunk_acc_gemm(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return false;
    }
    const char * name = tensor->name;
    if (name == nullptr || name[0] == '\0') {
        return false;
    }
    const char * pats = std::getenv("SD_OCL_Q4_GEMM_FP16_CHUNK_ACC_SUBSTR");
    if (pats == nullptr || pats[0] == '\0') {
        return false;
    }
    std::string s(pats);
    size_t start = 0;
    while (start < s.size()) {
        size_t end = s.find(',', start);
        if (end == std::string::npos) {
            end = s.size();
        }
        size_t l = start;
        size_t r = end;
        while (l < r && std::isspace(static_cast<unsigned char>(s[l]))) {
            ++l;
        }
        while (r > l && std::isspace(static_cast<unsigned char>(s[r - 1]))) {
            --r;
        }
        if (r > l) {
            std::string tok = s.substr(l, r - l);
            if (std::strstr(name, tok.c_str()) != nullptr) {
                return true;
            }
        }
        start = end + 1;
    }
    return false;
}

inline bool use_adreno_moe_kernels(const ggml_backend_opencl_context *backend_ctx, const ggml_tensor *tensor) {
    GGML_UNUSED(backend_ctx);
    int ne01 = tensor->ne[1];
    return ((strstr(tensor->name, "ffn") != NULL) || (strstr(tensor->name, "as") != NULL)) && (ne01 % 64 == 0);
}

static void ggml_backend_opencl_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_opencl_context *backend_ctx = ggml_cl2_init(buffer->buft->device);

    cl_context context = backend_ctx->context;
    cl_command_queue queue = backend_ctx->queue;

#ifdef GGML_OPENCL_SOA_Q
    // We separate the quantized bits and scale from block_q4_0 by using an
    // additional kernel, where each thread handles a block. We first read the
    // original weights into a temporary buffer, then create two separate
    // buffers for quantized bits and scales, which are then populated by the
    // conversion kernel.
    if (tensor->type == GGML_TYPE_Q4_0) {
        static std::mutex q4_upload_mutex;
        std::lock_guard<std::mutex> q4_upload_lock(q4_upload_mutex);

        // Tensors should have been preallocated, therefore they should
        // already have ggml_tensor_extra_cl as extra.
        ggml_tensor_extra_cl * extra_orig = (ggml_tensor_extra_cl *)tensor->extra;
        GGML_ASSERT(extra_orig && "Tesnors in OpenCL backend should have been allocated and initialized");

        // Allocate the new extra and create aliases from the original.
        ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
        ggml_tensor_extra_cl_q4_0 * extra = ctx->ggml_opencl_alloc_temp_tensor_extra_q4_0();
        bool is_embed = ggml_opencl_is_embedding_tensor(tensor);
        bool force_no_reorder = std::getenv("GGML_OPENCL_Q4_0_NO_REORDER") != nullptr || ggml_opencl_force_no_reorder(tensor);
        // Keep an AOS copy for embedding get_rows, but still transpose SOA q/d for
        // Adreno matmul/gemv so lm_head (embed_tokens.weight) follows the same layout.
        bool use_reorder = use_adreno_kernels(backend_ctx, tensor) && !force_no_reorder;
        if (std::getenv("GGML_OPENCL_DEBUG_EMBED") != nullptr) {
            GGML_LOG_INFO("opencl: q4_0 tensor '%s' ne0=%lld ne1=%lld ne2=%lld ne3=%lld embed=%d reorder=%d\n",
                          tensor->name,
                          (long long)tensor->ne[0], (long long)tensor->ne[1],
                          (long long)tensor->ne[2], (long long)tensor->ne[3],
                          is_embed ? 1 : 0, use_reorder ? 1 : 0);
        }

        size_t size_d = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(ggml_fp16_t);
        size_t size_q = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*ggml_blck_size(tensor->type)/2;
        GGML_ASSERT(size_d + size_q == ggml_nbytes(tensor) && "Incorrect tensor size");

        cl_int err;
        cl_mem data_device = clCreateBuffer(context, CL_MEM_READ_WRITE,
            ggml_nbytes(tensor), NULL, &err);
        CL_CHECK(err);
        CL_CHECK(clEnqueueWriteBuffer(
            queue, data_device, CL_TRUE, 0,
            ggml_nbytes(tensor), data, 0, NULL, NULL));

        // We consider the specified offset arg as always, although For weights
        // the offset arg should be 0 (we do not assert this).
        //GGML_ASSERT(offset == 0);

        // We create subbuffers from the original tensor buffer for scales and
        // quants - i.e., scales and quants are aliases into the buffer obejct
        // that backs the original tensor. This is a cleaner way to adapt to the
        // new memory management.
        // In the old code, we allocate new buffers for scales and quants
        // respectively, which could still be done but would result in double
        // allocation; properly deallocating the preallocated buffer that backs
        // the tensors is tricky and would leak the backend specific information
        // into the general backend code.
        // Does this create misaligned subbuffers (alignment is 1024) in certain
        // cases ?
        cl_buffer_region region;

        // The original tensor memory is divided into scales and quants, i.e.,
        // we first store scales, then quants.
        // Create subbuffer for scales.
        region.origin = align_to(extra_orig->offset + tensor->view_offs + offset, backend_ctx->alignment);
        region.size = size_d;
        extra->d = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);
        auto previous_origin = region.origin;

        // Create subbuffer for quants.
        region.origin = align_to(previous_origin + size_d, backend_ctx->alignment);
        region.size = size_q;
        extra->q = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);

        //cl_kernel kernel = backend_ctx->kernel_convert_block_q4_0;
    #ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        cl_kernel kernel = backend_ctx->kernel_convert_block_q4_0;

        // The optimized kernels need weights in natural order, so unshuffle.
        if (use_reorder) {
            kernel = backend_ctx->kernel_convert_block_q4_0_noshuffle;
        }
    #else
        cl_kernel kernel = backend_ctx->kernel_convert_block_q4_0;
    #endif // GGML_OPENCL_USE_ADRENO_KERNELS
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->q));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->d));

        size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[] = {64, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));

        if (is_embed) {
            // Keep the original AOS buffer for get_rows on Adreno.
            extra->aos = data_device;
        } else {
            CL_CHECK(clReleaseMemObject(data_device));
        }

        if (std::getenv("GGML_OPENCL_DEBUG_Q4_0_BLOCK") != nullptr && ggml_opencl_is_embedding_tensor(tensor)) {
            ggml_fp16_t d_host = 0;
            uint8_t q_host[16];
            CL_CHECK(clEnqueueReadBuffer(queue, extra->d, CL_TRUE, 0, sizeof(d_host), &d_host, 0, NULL, NULL));
            CL_CHECK(clEnqueueReadBuffer(queue, extra->q, CL_TRUE, 0, sizeof(q_host), q_host, 0, NULL, NULL));

            const uint8_t * src = reinterpret_cast<const uint8_t *>(data);
            uint16_t d_src = 0;
            memcpy(&d_src, src, sizeof(d_src));
            GGML_LOG_INFO("opencl: q4_0 debug d src=0x%04x dev=0x%04x\n", (unsigned)d_src, (unsigned)d_host);
            GGML_LOG_INFO("opencl: q4_0 debug q src0=%02x dev0=%02x src1=%02x dev1=%02x\n",
                          (unsigned)src[2], (unsigned)q_host[0], (unsigned)src[3], (unsigned)q_host[1]);
        }
        if (const char * row_env = std::getenv("GGML_OPENCL_DEBUG_Q4_0_ROW"); row_env != nullptr && ggml_opencl_is_embedding_tensor(tensor)) {
            char * end = nullptr;
            long row = std::strtol(row_env, &end, 10);
            if (end != row_env && row >= 0) {
                const int64_t ne0 = tensor->ne[0];
                const int64_t blocks_per_row = ne0 / ggml_blck_size(tensor->type);
                const size_t q_row_bytes = (size_t)blocks_per_row * 16;
                const size_t d_row_bytes = (size_t)blocks_per_row * sizeof(ggml_fp16_t);
                std::vector<uint8_t> q_row(q_row_bytes);
                std::vector<ggml_fp16_t> d_row(blocks_per_row);
                const size_t q_off = (size_t)row * q_row_bytes;
                const size_t d_off = (size_t)row * d_row_bytes;
                CL_CHECK(clEnqueueReadBuffer(queue, extra->q, CL_TRUE, q_off, q_row_bytes, q_row.data(), 0, NULL, NULL));
                CL_CHECK(clEnqueueReadBuffer(queue, extra->d, CL_TRUE, d_off, d_row_bytes, d_row.data(), 0, NULL, NULL));

                const uint8_t * src = reinterpret_cast<const uint8_t *>(data);
                const size_t block_bytes = sizeof(ggml_fp16_t) + 16;
                const uint8_t * row_src = src + (size_t)row * (size_t)blocks_per_row * block_bytes;
                if (blocks_per_row > 0) {
                    uint16_t d_src = 0;
                    memcpy(&d_src, row_src, sizeof(d_src));
                    GGML_LOG_INFO("opencl: q4_0 row %ld d src=0x%04x dev=0x%04x\n",
                                  row, (unsigned)d_src, (unsigned)d_row[0]);
                    GGML_LOG_INFO("opencl: q4_0 row %ld q src0=%02x dev0=%02x src1=%02x dev1=%02x\n",
                                  row, (unsigned)row_src[2], (unsigned)q_row[0], (unsigned)row_src[3], (unsigned)q_row[1]);
                }
            }
        }

        tensor->extra = extra;

        // transpose the weights and scales
    #ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        // Only do transpose for large, non batched matrix
        // TODO: use preallocated images instead of sub-buffer then image
        if (use_reorder) {
            static std::mutex q4_reorder_mutex;
            std::lock_guard<std::mutex> lock(q4_reorder_mutex);
        // <----------------------------------------------------------------------------------> //
        // start transpose
        // <----------------------------------------------------------------------------------> //
        int M = tensor->ne[1];   // ne01
        int K = tensor->ne[0];   // ne00

        //For matrix-vector multiplication kernel, we assume K is a multiple of 32
        GGML_ASSERT(K % 32 == 0);
        //For transpose kernels, we assume K is a multiple of 4 (satisfied by prior assert), and M is a multiple of 4
        GGML_ASSERT(M % 4 == 0);

        // transpose is out of place, so we need to allocate transposed buffers
        // <----------------------------------------------------------------------------------> //
        // use sub_buffer of max buffer size instead

        size_t q_size_bytes = K * M / 8 * sizeof(float);
        backend_ctx->prealloc_quant_trans.allocate(context, q_size_bytes);

        cl_buffer_region region;
        region.origin = 0;
        region.size = q_size_bytes;
        cl_mem qT_d = clCreateSubBuffer(
            backend_ctx->prealloc_quant_trans.buffer,
            0,
            CL_BUFFER_CREATE_TYPE_REGION,
            &region,
            &err);
        CL_CHECK(err);

        bool K_tile_trans = true;
        if ((K / 32) % 4 != 0){
            K_tile_trans =false;
        }

        size_t d_size_bytes = M * (K / 32) * 2;
        backend_ctx->prealloc_scales_trans.allocate(context, d_size_bytes);

        region.origin = 0;
        region.size = d_size_bytes;
        cl_mem dT_d = clCreateSubBuffer(
            backend_ctx->prealloc_scales_trans.buffer,
            0,
            CL_BUFFER_CREATE_TYPE_REGION,
            &region,
            &err);
        CL_CHECK(err);

        // <----------------------------------------------------------------------------------> //


        bool use_buf_transpose = true;
        if (std::getenv("GGML_OPENCL_Q4_0_TRANSPOSE_IMAGE") != nullptr) {
            use_buf_transpose = false;
        }

        if (use_buf_transpose) {
            // buffer-based transpose (16-bit), avoids image1d path on Adreno
            // <----------------------------------------------------------------------------------> //
            int rows_q = M;
            int cols_q = K / 4;
            kernel = backend_ctx->kernel_transpose_16_buf;

            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->q));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &qT_d));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int),    &cols_q));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int),    &rows_q));

            size_t global_size_q[2] = {static_cast<size_t>(cols_q), static_cast<size_t>(rows_q)};
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size_q, NULL, 0, NULL, &evt));
            CL_CHECK(clWaitForEvents(1, &evt));

            int rows_s = M;
            int cols_s = K / 32;
            kernel = backend_ctx->kernel_transpose_16_buf;

            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->d));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &dT_d));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int),    &cols_s));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int),    &rows_s));

            size_t global_size_s[2] = {static_cast<size_t>(cols_s), static_cast<size_t>(rows_s)};
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size_s, NULL, 0, NULL, &evt));
            CL_CHECK(clWaitForEvents(1, &evt));
            // <----------------------------------------------------------------------------------> //
        } else {
            // create images from the buffers
            // <----------------------------------------------------------------------------------> //
            cl_mem q_d_image1D;
            cl_mem d_d_image1D;
            cl_mem qT_d_image1D;
            cl_mem dT_d_image1D;

            cl_image_format img_fmt_1d = { CL_RGBA, CL_HALF_FLOAT };
            cl_image_desc img_desc_1d;

            memset(&img_desc_1d, 0, sizeof(img_desc_1d));
            img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            img_desc_1d.image_width = M * K / 4 / 4;
            img_desc_1d.buffer = extra->q;
            q_d_image1D = clCreateImage(context, 0, &img_fmt_1d, &img_desc_1d, NULL, &err);
            CL_CHECK(err);

            img_fmt_1d = { CL_RGBA, CL_HALF_FLOAT };
            memset(&img_desc_1d, 0, sizeof(img_desc_1d));
            img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            img_desc_1d.image_width = M * K / 4 / 4;
            img_desc_1d.buffer = qT_d;
            qT_d_image1D = clCreateImage(context, 0, &img_fmt_1d, &img_desc_1d, NULL, &err);
            CL_CHECK(err);

            memset(&img_desc_1d, 0, sizeof(img_desc_1d));
            if (K_tile_trans) {
                img_fmt_1d = { CL_RGBA, CL_HALF_FLOAT };
                img_desc_1d.image_width = M * K / 32 / 4;
            } else {
                img_fmt_1d = { CL_R, CL_HALF_FLOAT };
                img_desc_1d.image_width = M * K / 32;
            }
            img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            img_desc_1d.buffer = extra->d;
            d_d_image1D = clCreateImage(context, 0, &img_fmt_1d, &img_desc_1d, NULL, &err);
            CL_CHECK(err);

            img_fmt_1d = { CL_RGBA, CL_HALF_FLOAT };
            memset(&img_desc_1d, 0, sizeof(img_desc_1d));
            img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            img_desc_1d.image_width = M * K / 32 / 4;
            img_desc_1d.buffer = dT_d;
            dT_d_image1D = clCreateImage(context, 0, &img_fmt_1d, &img_desc_1d, NULL, &err);
            CL_CHECK(err);
            // <----------------------------------------------------------------------------------> //

            // set up and call the transpose kernels
            // <----------------------------------------------------------------------------------> //
            // weights
            int height_q = M / 4;
            int width_q = K / 4 / 4;
            kernel = backend_ctx->kernel_transpose_16;

            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &q_d_image1D));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &qT_d_image1D));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int),    &height_q));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int),    &width_q));

            size_t local_size_q[3] = {4, 16, 1};
            size_t global_size_q[3] = {static_cast<size_t>(width_q), static_cast<size_t>(height_q), 1};
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_size_q, local_size_q, 0, NULL, &evt));
            CL_CHECK(clWaitForEvents(1, &evt));

            // scales
            int height_s = M / 4;
            int width_s = K / 32 / 4;

            kernel = backend_ctx->kernel_transpose_16;
            if (!K_tile_trans) {
                kernel = backend_ctx->kernel_transpose_16_4x1;
                width_s = K / 32;
            }
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_d_image1D));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &dT_d_image1D));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), &height_s));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &width_s));

            size_t local_size_s[3] = {4, 16, 1};
            size_t global_size_s[3] = {static_cast<size_t>(width_s), static_cast<size_t>(height_s), 1};
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_size_s, local_size_s, 0, NULL, &evt));
            CL_CHECK(clWaitForEvents(1, &evt));
            // <----------------------------------------------------------------------------------> //

            // deallocate temporary images
            CL_CHECK(clReleaseMemObject(q_d_image1D));
            CL_CHECK(clReleaseMemObject(d_d_image1D));
            CL_CHECK(clReleaseMemObject(qT_d_image1D));
            CL_CHECK(clReleaseMemObject(dT_d_image1D));
            // <----------------------------------------------------------------------------------> //
        }

        // copy transposed buffer contents to original buffers
        // <----------------------------------------------------------------------------------> //
        // weights
        CL_CHECK(clEnqueueCopyBuffer(queue, qT_d, extra->q, 0, 0, q_size_bytes, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));

        // scales
        CL_CHECK(clEnqueueCopyBuffer(queue, dT_d, extra->d, 0, 0, d_size_bytes, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        // <----------------------------------------------------------------------------------> //

        // deallocate transpose buffers
        // <----------------------------------------------------------------------------------> //
        CL_CHECK(clReleaseMemObject(qT_d));
        CL_CHECK(clReleaseMemObject(dT_d));
        // <----------------------------------------------------------------------------------> //
        // end transpose
        // <----------------------------------------------------------------------------------> //
        }
    #endif // GGML_OPENCL_USE_ADRENO_KERNELS

        return;

    }
    if (tensor->type == GGML_TYPE_MXFP4) {
        ggml_tensor_extra_cl * extra_orig = (ggml_tensor_extra_cl *)tensor->extra;
        GGML_ASSERT(extra_orig && "Tesnors in OpenCL backend should have been allocated and initialized");

        // Allocate the new extra and create aliases from the original.
        ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
        ggml_tensor_extra_cl_mxfp4 * extra = ctx->ggml_opencl_alloc_temp_tensor_extra_mxfp4();

        size_t size_e = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(char);
        size_t size_q = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*ggml_blck_size(tensor->type)/2;
        GGML_ASSERT(size_e + size_q == ggml_nbytes(tensor) && "Incorrect tensor size");

        cl_int err;
        cl_mem data_device = clCreateBuffer(context, CL_MEM_READ_WRITE,
            ggml_nbytes(tensor), NULL, &err);
        CL_CHECK(err);
        CL_CHECK(clEnqueueWriteBuffer(
            queue, data_device, CL_TRUE, 0,
            ggml_nbytes(tensor), data, 0, NULL, NULL));

        // The original tensor memory is divided into scales and quants, i.e.,
        // we first store scales, then quants.
        cl_buffer_region region;

        // Create subbuffer for scales.
        region.origin = align_to(extra_orig->offset + tensor->view_offs + offset, backend_ctx->alignment);
        region.size = size_e;
        extra->e = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);
        auto previous_origin = region.origin;

        // Create subbuffer for quants.
        region.origin = align_to(previous_origin + size_e, backend_ctx->alignment);
        region.size = size_q;
        extra->q = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        if (use_adreno_moe_kernels(backend_ctx, tensor)) {
            cl_kernel kernel = backend_ctx->kernel_convert_block_mxfp4_trans;

            int ne00 = tensor->ne[0];
            int ne01 = tensor->ne[1];
            int ne02 = tensor->ne[2];
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->q));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->e));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &ne00));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), &ne01));

            size_t global_work_size[3] = {static_cast<size_t>(((ne01 + 63) / 64) * 64), static_cast<size_t>(ne00 / 32), static_cast<size_t>(ne02)};
            size_t local_work_size[3] = {64, 2, 1};

            cl_event evt;
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
            CL_CHECK(clWaitForEvents(1, &evt));
            CL_CHECK(clReleaseMemObject(data_device));
            tensor->extra = extra;

            return;
        }
#endif
        cl_kernel kernel = backend_ctx->kernel_convert_block_mxfp4;

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->q));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->e));

        size_t global_work_size[3] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[3] = {64, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clReleaseMemObject(data_device));

        // Create image for Q
        cl_image_format img_format_q = {CL_RG, CL_UNSIGNED_INT32};
        cl_image_desc img_desc_q = {
            CL_MEM_OBJECT_IMAGE1D_BUFFER,
            static_cast<size_t>(ggml_nelements(tensor)/32*2),
            0, 0, 0, 0, 0, 0, 0,
            { extra->q }
        };
        extra->q_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_format_q, &img_desc_q, NULL, &err);
        tensor->extra = extra;

        return;
    }
    if (tensor->type == GGML_TYPE_Q8_0) {
        ggml_tensor_extra_cl * extra_orig = (ggml_tensor_extra_cl *)tensor->extra;
        GGML_ASSERT(extra_orig && "Tesnors in OpenCL backend should have been allocated and initialized");

        // Allocate the new extra and create aliases from the original.
        ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
        ggml_tensor_extra_cl_q8_0 * extra = ctx->ggml_opencl_alloc_temp_tensor_extra_q8_0();

        size_t size_d = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(ggml_fp16_t);
        size_t size_q = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*(ggml_blck_size(tensor->type)*sizeof(char));
        GGML_ASSERT(size_d + size_q == ggml_nbytes(tensor) && "Incorrect tensor size");

        cl_int err;
        cl_mem data_device = clCreateBuffer(context, CL_MEM_READ_WRITE,
            ggml_nbytes(tensor), NULL, &err);
        CL_CHECK(err);
        CL_CHECK(clEnqueueWriteBuffer(
            queue, data_device, CL_TRUE, 0,
            ggml_nbytes(tensor), data, 0, NULL, NULL));

        // The original tensor memory is divided into scales and quants, i.e.,
        // we first store scales, then quants.
        cl_buffer_region region;

        // Create subbuffer for scales.
        region.origin = align_to(extra_orig->offset + tensor->view_offs + offset, backend_ctx->alignment);
        region.size = size_d;
        extra->d = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);
        auto previous_origin = region.origin;

        // Create subbuffer for quants.
        region.origin = align_to(previous_origin + size_d, backend_ctx->alignment);
        region.size = size_q;
        extra->q = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);

        cl_kernel kernel = backend_ctx->kernel_convert_block_q8_0;

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->q));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->d));

        size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[] = {64, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clReleaseMemObject(data_device));

        tensor->extra = extra;

        return;
    }
    if (tensor->type == GGML_TYPE_Q6_K) {
        ggml_tensor_extra_cl * extra_orig = (ggml_tensor_extra_cl *)tensor->extra;
        GGML_ASSERT(extra_orig && "Tesnors in OpenCL backend should have been allocated and initialized");

        // Allocate the new extra and create aliases from the original.
        ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
        ggml_tensor_extra_cl_q6_K * extra = ctx->ggml_opencl_alloc_temp_tensor_extra_q6_K();

        size_t size_ql = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*ggml_blck_size(tensor->type)/2;
        size_t size_qh = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*ggml_blck_size(tensor->type)/4;
        size_t size_s  = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*ggml_blck_size(tensor->type)/16;
        size_t size_d  = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(ggml_fp16_t);
        GGML_ASSERT(size_ql + size_qh + size_s + size_d == ggml_nbytes(tensor) &&
            "Incorrect tensor size");

        cl_int err;
        cl_mem data_device = clCreateBuffer(context, CL_MEM_READ_WRITE,
            ggml_nbytes(tensor), NULL, &err);
        CL_CHECK(err);
        CL_CHECK(clEnqueueWriteBuffer(
            queue, data_device, CL_TRUE, 0,
            ggml_nbytes(tensor), data, 0, NULL, NULL));

        cl_buffer_region region;

        // Subbuffer for ql
        region.origin = align_to(extra_orig->offset + tensor->view_offs + offset, backend_ctx->alignment);
        region.size = size_ql;
        extra->ql = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);
        auto previous_origin = region.origin;

        // Subbuffer for qh
        region.origin = align_to(previous_origin + size_ql, backend_ctx->alignment);
        region.size = size_qh;
        extra->qh = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);
        previous_origin = region.origin;

        // Subbuffer for scales
        region.origin = align_to(previous_origin + size_qh, backend_ctx->alignment);
        region.size = size_s;
        extra->s = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);
        previous_origin = region.origin;

        // Create subbuffer for d.
        region.origin = align_to(previous_origin + size_s, backend_ctx->alignment);
        region.size = size_d;
        extra->d = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);
        previous_origin = region.origin;

        // Flatten the weights
        cl_kernel kernel = backend_ctx->kernel_convert_block_q6_K;

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->ql));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->qh));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &extra->s));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &extra->d));

        size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[] = {64, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clReleaseMemObject(data_device));

        extra->size_ql = size_ql;
        extra->size_qh = size_qh;
        extra->size_s  = size_s;
        extra->size_d  = size_d;

        tensor->extra  = extra;
        return;
    }
#endif // GGML_OPENCL_SOA_Q

    ggml_tensor_extra_cl * extra = (ggml_tensor_extra_cl *) tensor->extra;
    GGML_ASSERT(extra);

    CL_CHECK(clEnqueueWriteBuffer(
        queue, extra->data_device, CL_TRUE, extra->offset + offset,
        size, data, 0, NULL, NULL));

    GGML_UNUSED(buffer);
}

static void ggml_backend_opencl_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor->extra);

    ggml_backend_opencl_context *backend_ctx = ggml_cl2_init(buffer->buft->device);

    cl_context context = backend_ctx->context;
    cl_command_queue queue = backend_ctx->queue;

    // Make sure all previously submitted commands in other devices are finished.
    sync_with_other_backends(backend_ctx);

#ifdef GGML_OPENCL_SOA_Q
    // In end-to-end runs, get_tensor is usually used to get back the logits,
    // where we can simply do clEnqueueReadBuffer since they are f32.
    // However, in test-backend-ops, the GPU graph is copied to the CPU backend,
    // which requires reading back quantized weight tensors.
    // To properly support this, we need to restore block_q4_0 struct arrays
    // from the flattened buffers.
    if (tensor->type == GGML_TYPE_Q4_0) {
        ggml_tensor_extra_cl_q4_0 * extra = (ggml_tensor_extra_cl_q4_0 *)tensor->extra;

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        if (use_adreno_kernels(backend_ctx, tensor)) {
            cl_int err;
            cl_kernel kernel;

            cl_int M = tensor->ne[1];   // ne01
            cl_int K = tensor->ne[0];   // ne00

            GGML_ASSERT(K % 32 == 0);
            GGML_ASSERT(M % 4 == 0);

            size_t size_q = (ggml_nelements(tensor)/ggml_blck_size(tensor->type))*ggml_blck_size(tensor->type)/2;
            size_t size_d = (ggml_nelements(tensor)/ggml_blck_size(tensor->type))*sizeof(ggml_fp16_t);
            GGML_ASSERT(size_d + size_q == ggml_nbytes(tensor) && "Incorrect tensor size");

            cl_mem buf_trans_q;
            cl_mem buf_trans_d;

            CL_CHECK((buf_trans_q = clCreateBuffer(context, CL_MEM_READ_WRITE,
                size_q, NULL, &err), err));
            CL_CHECK((buf_trans_d = clCreateBuffer(context, CL_MEM_READ_WRITE,
                size_d, NULL, &err), err));

            kernel = backend_ctx->kernel_transpose_16_buf;

            // transpose q back
            cl_int stride_k_q = K/4;
            size_t local_size_q[3] = {64, 1, 1};
            size_t global_size_q[3] = {(size_t)M, (size_t)stride_k_q, 1};

            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->q));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_trans_q));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), &M));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), &stride_k_q));

            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
                global_size_q, local_size_q, 0, NULL, NULL));

            // transpose scales back
            cl_int stride_k_d = K/32;
            size_t local_size_d[3] = {64, 1, 1};
            size_t global_size_d[3] = {(size_t)M, (size_t)stride_k_d, 1};

            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->d));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_trans_d));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), &M));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), &stride_k_d));

            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
                global_size_d, local_size_d, 0, NULL, NULL));

            // unpack
            cl_mem data_device = clCreateBuffer(context, CL_MEM_READ_WRITE,
                ggml_nbytes(tensor), NULL, &err);
            CL_CHECK(err);

            cl_uchar mask_0F = 0x0F;
            cl_uchar mask_F0 = 0xF0;

            size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
            size_t local_work_size[] = {1, 1, 1};

            kernel = backend_ctx->kernel_restore_block_q4_0_noshuffle;
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &buf_trans_q));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &buf_trans_d));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &data_device));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_uchar), &mask_0F));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_uchar), &mask_F0));

            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
                global_work_size, local_work_size, 0, NULL, NULL));

            // read back to host
            CL_CHECK(clEnqueueReadBuffer(
                queue, data_device, CL_TRUE, offset,
                size, data, 0, NULL, NULL));

            CL_CHECK(clReleaseMemObject(data_device));
            CL_CHECK(clReleaseMemObject(buf_trans_q));
            CL_CHECK(clReleaseMemObject(buf_trans_d));

            return;
        }
#endif

        cl_int err;
        cl_mem data_device = clCreateBuffer(context, CL_MEM_READ_WRITE,
            ggml_nbytes(tensor), NULL, &err);
        CL_CHECK(err);

        cl_kernel kernel = backend_ctx->kernel_restore_block_q4_0;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->q));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->d));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &data_device));

        size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[] = {1, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
            global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clEnqueueReadBuffer(
            queue, data_device, CL_TRUE, offset,
            size, data, 0, NULL, NULL));
        CL_CHECK(clReleaseMemObject(data_device));
        return;
    } else if (tensor->type == GGML_TYPE_MXFP4) {
        ggml_tensor_extra_cl_mxfp4 * extra = (ggml_tensor_extra_cl_mxfp4 *)tensor->extra;

        cl_int err;
        cl_mem data_device = clCreateBuffer(context, CL_MEM_READ_WRITE,
            ggml_nbytes(tensor), NULL, &err);
        CL_CHECK(err);

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        if (use_adreno_moe_kernels(backend_ctx, tensor)) {
            cl_kernel kernel = backend_ctx->kernel_restore_block_mxfp4_trans;

            int ne00 = tensor->ne[0];
            int ne01 = tensor->ne[1];
            int ne02 = tensor->ne[2];
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->q));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->e));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &data_device));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), &ne00));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_int), &ne01));

            size_t global_work_size[3] = {static_cast<size_t>(((ne01 + 63) / 64) * 64), static_cast<size_t>(ne00 / 32), static_cast<size_t>(ne02)};
            size_t local_work_size[3] = {64, 2, 1};

            cl_event evt;
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
                global_work_size, local_work_size, 0, NULL, &evt));
            CL_CHECK(clWaitForEvents(1, &evt));
            CL_CHECK(clEnqueueReadBuffer(
                queue, data_device, CL_TRUE, offset,
                size, data, 0, NULL, NULL));
            CL_CHECK(clReleaseMemObject(data_device));
            return;
        }
#endif
        cl_kernel kernel = backend_ctx->kernel_restore_block_mxfp4;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->q));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->e));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &data_device));

        size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[] = {1, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
            global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clEnqueueReadBuffer(
            queue, data_device, CL_TRUE, offset,
            size, data, 0, NULL, NULL));
        CL_CHECK(clReleaseMemObject(data_device));
        return;
    }
    if (tensor->type == GGML_TYPE_Q8_0) {
        ggml_tensor_extra_cl_q8_0 * extra = (ggml_tensor_extra_cl_q8_0 *)tensor->extra;

        cl_int err;
        cl_mem data_device = clCreateBuffer(context, CL_MEM_READ_WRITE,
            ggml_nbytes(tensor), NULL, &err);
        CL_CHECK(err);

        cl_kernel kernel = backend_ctx->kernel_restore_block_q8_0;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->q));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->d));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &data_device));

        size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[] = {1, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
            global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clEnqueueReadBuffer(
            queue, data_device, CL_TRUE, offset,
            size, data, 0, NULL, NULL));
        CL_CHECK(clReleaseMemObject(data_device));
        return;
    }
    if (tensor->type == GGML_TYPE_Q6_K) {
        ggml_tensor_extra_cl_q6_K * extra = (ggml_tensor_extra_cl_q6_K *)tensor->extra;

        cl_int err;
        cl_mem data_device = clCreateBuffer(context, CL_MEM_READ_WRITE,
            ggml_nbytes(tensor), NULL, &err);
        CL_CHECK(err);

        cl_kernel kernel = backend_ctx->kernel_restore_block_q6_K;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->ql));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->qh));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->s));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &extra->d));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &data_device));

        size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[] = {1, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
            global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clEnqueueReadBuffer(
            queue, data_device, CL_TRUE, offset,
            size, data, 0, NULL, NULL));
        CL_CHECK(clReleaseMemObject(data_device));
        return;
    }
#endif // GGML_OPENCL_SOA_Q

    ggml_tensor_extra_cl * extra = (ggml_tensor_extra_cl *) tensor->extra;

    CL_CHECK(clEnqueueReadBuffer(
        queue, extra->data_device, CL_TRUE, extra->offset + tensor->view_offs + offset,
        size, data, 0, NULL, NULL));

    GGML_UNUSED(buffer);
}

static void ggml_backend_opencl_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_dev_t dev = buffer->buft->device;
    ggml_backend_opencl_context *backend_ctx = ggml_cl2_init(dev);
    cl_command_queue queue = backend_ctx->queue;

    ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
    for (cl_mem buf : ctx->buffer) {
        CL_CHECK(clEnqueueFillBuffer(queue, buf, &value, sizeof(value), 0, buffer->size, 0, NULL, NULL));
    }
    CL_CHECK(clFinish(queue));
}

static void ggml_backend_opencl_buffer_reset(ggml_backend_buffer_t buffer) {
    ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
    ctx->reset();
}

static ggml_backend_buffer_i ggml_backend_opencl_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_opencl_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_opencl_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_opencl_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_opencl_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_opencl_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_opencl_buffer_clear,
    /* .reset           = */ ggml_backend_opencl_buffer_reset,
};

//
// buffer type
//

static const char * ggml_backend_opencl_buffer_type_get_name(ggml_backend_buffer_type_t buffer_type) {
    return "OpenCL";

    GGML_UNUSED(buffer_type);
}

static ggml_backend_buffer_t ggml_backend_opencl_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buffer_type, size_t size) {
    ggml_backend_opencl_context *backend_ctx = ggml_cl2_init(buffer_type->device);

    // clCreateBuffer returns -61 for size 0
    size = std::max(size, (size_t)1);

    cl_int err;
    cl_mem mem = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE, size, NULL, &err);
    if (err != CL_SUCCESS) {
        GGML_LOG_INFO("%s: failed to allocate %.2f MiB\n", __func__, size / 1024.0 / 1024.0);
        return nullptr;
    }

    ggml_backend_opencl_buffer_context * ctx = new ggml_backend_opencl_buffer_context(mem);

    return ggml_backend_buffer_init(buffer_type, ggml_backend_opencl_buffer_interface, ctx, size);
}

static size_t ggml_backend_opencl_buffer_type_get_alignment(ggml_backend_buffer_type_t buffer_type) {
    ggml_backend_opencl_context * backend_ctx = ggml_cl2_init(buffer_type->device);
    return backend_ctx->alignment;
}

static size_t ggml_backend_opencl_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buffer_type, const struct ggml_tensor * tensor) {
    size_t size = ggml_nbytes(tensor);
#ifdef GGML_OPENCL_SOA_Q
    if (tensor->type == GGML_TYPE_Q4_0) {
        ggml_backend_opencl_context * backend_ctx = ggml_cl2_init(buffer_type->device);
        const size_t alignment = backend_ctx->alignment;
        const size_t nblocks   = ggml_nelements(tensor) / ggml_blck_size(tensor->type);
        const size_t size_d    = nblocks * sizeof(ggml_fp16_t);
        const size_t size_q    = nblocks * ggml_blck_size(tensor->type) / 2;
        const size_t size_d_aligned = align_to(size_d, alignment);
        size = size_d_aligned + size_q;
    }
#endif
    return size;
}

static size_t ggml_backend_opencl_buffer_type_get_max_size(ggml_backend_buffer_type_t buffer_type) {
    static size_t max_size = -1;
    if (max_size == (size_t)-1) {
        ggml_backend_opencl_context * backend_ctx = ggml_cl2_init(buffer_type->device);
        max_size = backend_ctx->max_alloc_size;
    }
    return max_size;
}

static bool ggml_backend_opencl_buffer_type_supports_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend) {
    return ggml_backend_is_opencl(backend);

    UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_opencl_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_opencl_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_opencl_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_opencl_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_opencl_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_opencl_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

//
// backend device
//

static const char * ggml_backend_opencl_device_get_name(ggml_backend_dev_t dev) {
    return "GPUOpenCL";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_opencl_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_opencl_device_context *dev_ctx = (ggml_backend_opencl_device_context *) dev->context;
    return dev_ctx->device_name.c_str();
}

static void ggml_backend_opencl_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_opencl_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_opencl_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_opencl_device_get_name(dev);
    props->description = ggml_backend_opencl_device_get_description(dev);
    props->type        = ggml_backend_opencl_device_get_type(dev);
    ggml_backend_opencl_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = ggml_backend_dev_caps {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_opencl_device_init(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_opencl_context * backend_ctx = ggml_cl2_init(dev);
    // Getting a new reference to the backend, increase ref_count
    backend_ctx->ref_count++;

    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_opencl_guid(),
        /* .interface = */ ggml_backend_opencl_i,
        /* .device    = */ dev,
        /* .context   = */ backend_ctx,
    };

    return backend;

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_opencl_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto * dev_ctx = static_cast<ggml_backend_opencl_device_context *>(dev->context);

    dev_ctx->buffer_type = ggml_backend_buffer_type{
        /* .iface   = */ ggml_backend_opencl_buffer_type_interface,
        /* .device  = */ dev,
        /* .context = */ nullptr,
    };

    return &dev_ctx->buffer_type;
}

static ggml_backend_buffer_t ggml_backend_opencl_device_buffer_from_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(ptr);
    GGML_UNUSED(size);
    GGML_UNUSED(max_tensor_size);
    return nullptr;
}

static bool ggml_backend_opencl_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    return ggml_opencl_supports_op(dev, op);
}

static bool ggml_backend_opencl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    // Check 'dev' and 'buffer_type' are not objects belonging to this backend.
    if (dev->iface.get_name != ggml_backend_opencl_device_get_name ||
        buft->iface.get_name != ggml_backend_opencl_buffer_type_get_name) {
        return false;
    }

    // Check cl_context is the same. clEnqueue* commands may not use
    // buffers from another cl_context.
    ggml_backend_opencl_context * backend_ctx0 = ggml_cl2_init(dev);
    ggml_backend_opencl_context * backend_ctx1 = ggml_cl2_init(buft->device);
    return backend_ctx0->context == backend_ctx1->context;
}

namespace /* anonymous */ {
struct ggml_backend_device_i ggml_backend_opencl_device_i = {
    /* .get_name             = */ ggml_backend_opencl_device_get_name,
    /* .get_description      = */ ggml_backend_opencl_device_get_description,
    /* .get_memory           = */ ggml_backend_opencl_device_get_memory,
    /* .get_type             = */ ggml_backend_opencl_device_get_type,
    /* .get_props            = */ ggml_backend_opencl_device_get_props,
    /* .init_backend         = */ ggml_backend_opencl_device_init,
    /* .get_buffer_type      = */ ggml_backend_opencl_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_opencl_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_opencl_device_supports_op,
    /* .supports_buft        = */ ggml_backend_opencl_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};
}

// Backend registry

static const char * ggml_backend_opencl_reg_get_name(ggml_backend_reg_t reg) {
    return "OpenCL";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_opencl_reg_device_count(ggml_backend_reg_t reg) {
    return g_ggml_backend_opencl_devices.size();

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_opencl_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index < ggml_backend_opencl_reg_device_count(reg));

    return &g_ggml_backend_opencl_devices[index];

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static struct ggml_backend_reg_i ggml_backend_opencl_reg_i = {
    /* .get_name         = */ ggml_backend_opencl_reg_get_name,
    /* .device_count     = */ ggml_backend_opencl_reg_device_count,
    /* .device_get       = */ ggml_backend_opencl_reg_device_get,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_opencl_reg(void) {
    static std::mutex mutex;
    static ggml_backend_reg reg;
    static bool initialized = false;
    std::lock_guard<std::mutex> lock(mutex);

    if (initialized) {
        return &reg;
    }
    initialized = true;

    g_ggml_backend_opencl_devices = ggml_opencl_probe_devices(&reg);

    reg = ggml_backend_reg{
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_opencl_reg_i,
        /* .context     = */ NULL,
    };

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_opencl_reg)

//------------------------------------------------------------------------------
// Debugging utils
//------------------------------------------------------------------------------
#if 0
#define QK4_0 32
typedef struct {
    ggml_fp16_t d;          // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_fp16_t) + QK4_0 / 2,
    "wrong q4_0 block size/padding");

#include <math.h>
#ifdef __cplusplus
#include "half.hpp"
#endif

static void dump_tensor(ggml_backend_t backend, const struct ggml_tensor * tensor) {
    void * buf = malloc(ggml_nbytes(tensor));

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    cl_command_queue queue = backend_ctx->queue;
#ifdef GGML_OPENCL_SOA_Q
    void * buf_q;
    void * buf_d;
#endif

    // Make sure everything is done.
    CL_CHECK(clFinish(queue));

#ifdef GGML_OPENCL_SOA_Q
    if (tensor->type == GGML_TYPE_Q4_0) {
        ggml_tensor_extra_cl_q4_0 * extra = (ggml_tensor_extra_cl_q4_0 *) tensor->extra;
        GGML_ASSERT(extra);

        size_t size_q = ggml_nelements(tensor)/QK4_0 * QK4_0/2;
        size_t size_d = ggml_nelements(tensor)/QK4_0 * sizeof(ggml_fp16_t);
        GGML_ASSERT(size_q + size_d == ggml_nbytes(tensor));
        buf_q = malloc(size_q);
        buf_d = malloc(size_d);

        CL_CHECK(clEnqueueReadBuffer(queue, extra->q, CL_TRUE, 0, size_q, buf_q, 0, NULL, NULL));
        CL_CHECK(clEnqueueReadBuffer(queue, extra->d, CL_TRUE, 0, size_d, buf_d, 0, NULL, NULL));
        CL_CHECK(clFinish(queue));
    } else if (tensor->type == GGML_TYPE_MXFP4) {
        ggml_tensor_extra_cl_mxfp4 * extra = (ggml_tensor_extra_cl_mxfp4 *) tensor->extra;
        GGML_ASSERT(extra);

        size_t size_q = ggml_nelements(tensor)/QK_MXFP4 * QK_MXFP4/2;
        size_t size_e = ggml_nelements(tensor)/QK_MXFP4 * sizeof(char);
        GGML_ASSERT(size_q + size_e == ggml_nbytes(tensor));
        buf_q = malloc(size_q);
        buf_d = malloc(size_e);

        CL_CHECK(clEnqueueReadBuffer(queue, extra->q, CL_TRUE, 0, size_q, buf_q, 0, NULL, NULL));
        CL_CHECK(clEnqueueReadBuffer(queue, extra->d, CL_TRUE, 0, size_e, buf_d, 0, NULL, NULL));
        CL_CHECK(clFinish(queue));
    } else {
        // Read out the tensor from GPU memory.
        ggml_tensor_extra_cl * extra = (ggml_tensor_extra_cl *) tensor->extra;
        GGML_ASSERT(extra);

        CL_CHECK(clEnqueueReadBuffer(queue, extra->data_device, CL_TRUE,
        extra->offset, ggml_nbytes(tensor), buf, 0, NULL, NULL));
        CL_CHECK(clFinish(queue));
    }
#else
    // Read out the tensor from GPU memory.
    ggml_tensor_extra_cl * extra = (ggml_tensor_extra_cl *) tensor->extra;
    GGML_ASSERT(extra);

    CL_CHECK(clEnqueueReadBuffer(queue, extra->data_device, CL_TRUE,
        extra->offset, ggml_nbytes(tensor), buf, 0, NULL, NULL));
    CL_CHECK(clFinish(queue));
#endif // GGML_OPENCL_SOA_Q

    // Open file and dump.
    char fname[512];
    snprintf(fname, sizeof(fname), "./tensor-dumps/%s.txt", tensor->name);
    FILE * f = fopen(fname, "w");
    if (!f) {
        printf("Failed to open %s\n", fname);
        return;
    }

    if (tensor->type == GGML_TYPE_F32) {
        float * data = (float *) buf;
        for (int i = 0; i < ggml_nelements(tensor); ++i) {
            if (isnan(data[i])) {
                printf("NaN found: %s\n", tensor->name);
                break;
            }
            fprintf(f, "%f\n", data[i]);
        }
    } else if (tensor->type == GGML_TYPE_I32) {
        int * data = (int *) buf;
        for (int i = 0; i < ggml_nelements(tensor); ++i) {
            if (isnan(data[i])) {
                printf("NaN found: %s\n", tensor->name);
                break;
            }
            fprintf(f, "%d\n", data[i]);
        }
    } else if (tensor->type == GGML_TYPE_F16) {
#ifdef __cplusplus
        half_float::half * data = (half_float::half *) buf;
        for (int i = 0; i < ggml_nelements(tensor); ++i) {
            if (std::isnan(data[i])) {
                printf("NaN found: %s\n", tensor->name);
                break;
            }
            fprintf(f, "%f\n", float(data[i]));
        }
#endif
    } else if (tensor->type == GGML_TYPE_Q4_0) {
#ifdef GGML_OPENCL_SOA_Q
        ggml_fp16_t * data_d = (ggml_fp16_t *)buf_d;
        unsigned char * data_q = (unsigned char *)buf_q;

        for (int i = 0; i < ggml_nelements(tensor)/QK4_0; ++i) {
            fprintf(f, "%04x, ", data_d[i]);
            for (int k = 0; k < QK4_0/2; ++k) {
                fprintf(f, "%02x, ", data_q[k]);
            }
            fprintf(f, "\n");
            data_q += QK4_0/2;
        }
        free(buf_d);
        free(buf_q);
#else
        block_q4_0 * data = (block_q4_0 *) buf;
        for (int i = 0; i < ggml_nelements(tensor)/QK4_0; ++i) {
            fprintf(f, "%04x, ", data[i].d);
            for (int k = 0; k < QK4_0/2; ++k) {
                fprintf(f, "%02x, ", data[i].qs[k]);
            }
            fprintf(f, "\n");
        }
#endif // GGML_OPENCL_SOA_Q
    }
    free(buf);
    fflush(f);
    fclose(f);
}
#else
#define dump_tensor(tensor)
#endif

//------------------------------------------------------------------------------
// Ops
//------------------------------------------------------------------------------

static bool ggml_cl_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {
    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    // TODO: find the optimal values for these
    return (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) &&
            src1->type == GGML_TYPE_F32 &&
             dst->type == GGML_TYPE_F32 &&
            (ne0 >= 32 && ne1 >= 32 && ne10 >= 32);
}

// Copy a noncontiguous tensor to contiguous tensor. ne[] remains the same but
// nb[] is recalculated such that tensor is contiguous.
static void ggml_cl_copy_to_contiguous(ggml_backend_t backend, const ggml_tensor * src, cl_mem dst,
                                       cl_ulong &nb0, cl_ulong &nb1, cl_ulong &nb2, cl_ulong &nb3) {
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    const int tensor_type_size = ggml_type_size(src->type);

    const int ne00 = src->ne[0];
    const int ne01 = src->ne[1];
    const int ne02 = src->ne[2];
    const int ne03 = src->ne[3];

    const cl_ulong nb00 = src->nb[0];
    const cl_ulong nb01 = src->nb[1];
    const cl_ulong nb02 = src->nb[2];
    const cl_ulong nb03 = src->nb[3];

    const int ne0 = src->ne[0];
    const int ne1 = src->ne[1];
    const int ne2 = src->ne[2];
    const int ne3 = src->ne[3];

    nb0 = tensor_type_size;
    nb1 = tensor_type_size*ne00;
    nb2 = tensor_type_size*ne00*ne01;
    nb3 = tensor_type_size*ne00*ne01*ne02;

    ggml_tensor_extra_cl * extra = (ggml_tensor_extra_cl *)src->extra;

    cl_ulong offset0 = extra->offset + src->view_offs;
    cl_ulong offsetd = 0;

    cl_kernel kernel;

    switch (src->type) {
        case GGML_TYPE_F32:
            kernel = backend_ctx->kernel_cpy_f32_f32;
            break;
        case GGML_TYPE_F16:
            kernel = backend_ctx->kernel_cpy_f16_f16;
            break;
        default:
            GGML_ASSERT(false && "not implemented");
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &dst));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne03));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb00));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne1));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne2));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne3));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb0));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb1));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb2));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb3));

    const int nth = MIN(64, ne00);

    size_t global_work_size[] = {(size_t)ne01*nth, (size_t)ne02, (size_t)ne03};
    size_t local_work_size[] = {(size_t)nth, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, src);
}

static void ggml_cl_nop(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    UNUSED(backend);
    UNUSED(src0);
    UNUSED(src1);
    UNUSED(dst);
}

static void ggml_cl_get_rows(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    if (getenv("GGML_OPENCL_GET_ROWS_CPU") != nullptr) {
        const size_t nbytes_src0 = ggml_nbytes(src0);
        const size_t nbytes_src1 = ggml_nbytes(src1);
        const size_t nbytes_dst  = ggml_nbytes(dst);

        std::vector<char> h_src0(nbytes_src0);
        std::vector<char> h_src1(nbytes_src1);
        std::vector<char> h_dst(nbytes_dst);

        ggml_backend_tensor_get(src0, h_src0.data(), 0, nbytes_src0);
        ggml_backend_tensor_get(src1, h_src1.data(), 0, nbytes_src1);

        ggml_tensor host_src0 = *src0;
        ggml_tensor host_src1 = *src1;
        ggml_tensor host_dst  = *dst;

        host_src0.data = h_src0.data();
        host_src0.buffer = nullptr;
        host_src0.extra = nullptr;
        host_src0.view_src = nullptr;
        host_src0.view_offs = 0;

        host_src1.data = h_src1.data();
        host_src1.buffer = nullptr;
        host_src1.extra = nullptr;
        host_src1.view_src = nullptr;
        host_src1.view_offs = 0;

        host_dst.data = h_dst.data();
        host_dst.buffer = nullptr;
        host_dst.extra = nullptr;
        host_dst.view_src = nullptr;
        host_dst.view_offs = 0;
        host_dst.src[0] = &host_src0;
        host_dst.src[1] = &host_src1;

        ggml_compute_params params;
        params.ith = 0;
        params.nth = 1;
        params.threadpool = nullptr;
        params.wdata = nullptr;
        params.wsize = 0;

        ggml_compute_forward_get_rows(&params, &host_dst);
        ggml_backend_tensor_set(dst, h_dst.data(), 0, nbytes_dst);
        return;
    }

    const int      ne00 = src0->ne[0];
    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];
    const int      ne10 = src1->ne[0];
    const cl_ulong nb10 = src1->nb[0];
    const int      ne11 = src1->ne[1];
    const int      ne12 = src1->ne[2];
    const cl_ulong nb11 = src1->nb[1];
    const cl_ulong nb12 = src1->nb[2];
    const cl_ulong nb1  = dst->nb[1];
    const cl_ulong nb2  = dst->nb[2];
    const cl_ulong nb3  = dst->nb[3];

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = 0;
#ifndef GGML_OPENCL_SOA_Q
    offset0 = extra0->offset + src0->view_offs;
#else
    if (src0->type != GGML_TYPE_Q4_0) {
        offset0 = extra0->offset + src0->view_offs;
    }
#endif
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel = nullptr;
    bool args_set = false;

#ifdef GGML_OPENCL_SOA_Q
    if (src0->type == GGML_TYPE_Q4_0) {
        ggml_tensor_extra_cl_q4_0 * extra0_q4_0 = (ggml_tensor_extra_cl_q4_0 *)src0->extra;
        GGML_ASSERT(extra0_q4_0 && extra0_q4_0->q && extra0_q4_0->d);
        bool force_scalar = std::getenv("GGML_OPENCL_GET_ROWS_SCALAR") != nullptr;
        bool is_embed = ggml_opencl_is_embedding_tensor(src0);
        bool use_aos = extra0_q4_0->aos != nullptr;
        if (std::getenv("GGML_OPENCL_NO_AOS") != nullptr) {
            use_aos = false;
        }
        if (use_aos) {
            // Use AOS buffer with the original kernel to avoid SOA issues.
            kernel = backend_ctx->kernel_get_rows_q4_0;
            offset0 = 0;
        } else {
            kernel = backend_ctx->kernel_get_rows_q4_0_soa;
            if (force_scalar || is_embed) {
                kernel = backend_ctx->kernel_get_rows_q4_0_soa_scalar;
            }
        }
        if (std::getenv("GGML_OPENCL_DEBUG_GET_ROWS") != nullptr) {
            GGML_LOG_INFO("opencl: get_rows q4_0 ne0=%lld ne1=%lld embed=%d scalar=%d aos=%d\n",
                          (long long)src0->ne[0], (long long)src0->ne[1],
                          is_embed ? 1 : 0,
                          (kernel == backend_ctx->kernel_get_rows_q4_0_soa_scalar) ? 1 : 0,
                          (kernel == backend_ctx->kernel_get_rows_q4_0) ? 1 : 0);
        }

        if (use_aos) {
            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q4_0->aos));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb03));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne10));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb10));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb1));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb2));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb3));
        } else {
            cl_ulong offset_q = 0;
            cl_ulong offset_d = 0;
            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q4_0->q));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset_q));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra0_q4_0->d));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset_d));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
            const int ne01 = src0->ne[1];
            const int ne02 = src0->ne[2];
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne02));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne10));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb10));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb1));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb2));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb3));
        }
        args_set = true;
    } else
#endif
    switch (src0->type) {
        case GGML_TYPE_F32:
            kernel = backend_ctx->kernel_get_rows_f32;
            break;
        case GGML_TYPE_F16:
            kernel = backend_ctx->kernel_get_rows_f16;
            break;
        case GGML_TYPE_Q4_0:
            kernel = backend_ctx->kernel_get_rows_q4_0;
            break;
        default:
            GGML_ASSERT(false && "not implemented");
    }

#ifndef GGML_OPENCL_SOA_Q
    if (kernel == backend_ctx->kernel_get_rows_q4_0 || kernel == backend_ctx->kernel_get_rows_f16 || kernel == backend_ctx->kernel_get_rows_f32) {
#endif
        if (!args_set && (kernel == backend_ctx->kernel_get_rows_f32 || kernel == backend_ctx->kernel_get_rows_f16 || kernel == backend_ctx->kernel_get_rows_q4_0)) {
            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb03));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne10));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb10));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb1));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb2));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb3));
        }
#ifndef GGML_OPENCL_SOA_Q
    }
#endif

    size_t global_work_size[] = {(size_t)ne10*64, (size_t)ne11, (size_t)ne12};
    size_t local_work_size[] = {64, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

    if (std::getenv("GGML_OPENCL_DEBUG_GET_ROWS_REF") != nullptr && src0->type == GGML_TYPE_Q4_0) {
        static bool ref_done = false;
        if (!ref_done) {
            ref_done = true;
            int32_t r = -1;
            ggml_backend_tensor_get(src1, &r, 0, sizeof(r));
            if (r >= 0 && r < src0->ne[1]) {
                ggml_tensor_extra_cl_q4_0 * extra0_q4_0 = (ggml_tensor_extra_cl_q4_0 *)src0->extra;
                const int qk4_0 = 32;
                const size_t blocks_per_row = (size_t)ne00 / qk4_0;
                const size_t q_row_bytes = blocks_per_row * (qk4_0/2);
                const size_t d_row_bytes = blocks_per_row * sizeof(ggml_fp16_t);
                std::vector<uint8_t> q_row(q_row_bytes);
                std::vector<ggml_fp16_t> d_row(blocks_per_row);
                CL_CHECK(clEnqueueReadBuffer(backend_ctx->queue, extra0_q4_0->q, CL_TRUE,
                                             (size_t)r * q_row_bytes, q_row_bytes, q_row.data(), 0, NULL, NULL));
                CL_CHECK(clEnqueueReadBuffer(backend_ctx->queue, extra0_q4_0->d, CL_TRUE,
                                             (size_t)r * d_row_bytes, d_row_bytes, d_row.data(), 0, NULL, NULL));

                std::vector<float> ref(ne00);
                size_t out_idx = 0;
                for (size_t bi = 0; bi < blocks_per_row; ++bi) {
                    const float d = ggml_fp16_to_fp32(d_row[bi]);
                    const uint8_t * qs = q_row.data() + bi * (qk4_0/2);
                    for (int j = 0; j < qk4_0/2; ++j) {
                        const int q0 = (qs[j] & 0x0F) - 8;
                        const int q1 = (qs[j] >> 4) - 8;
                        ref[out_idx + j] = q0 * d;
                        ref[out_idx + j + qk4_0/2] = q1 * d;
                    }
                    out_idx += qk4_0;
                }

                std::vector<float> out(ne00);
                CL_CHECK(clEnqueueReadBuffer(backend_ctx->queue, extrad->data_device, CL_TRUE,
                                             offsetd, ne00 * sizeof(float), out.data(), 0, NULL, NULL));
                float max_diff = 0.0f;
                float max_abs_out = 0.0f;
                int nan_count = 0;
                int inf_count = 0;
                for (int i = 0; i < ne00; ++i) {
                    const float v = out[i];
                    if (std::isnan(v)) {
                        nan_count++;
                        continue;
                    }
                    if (std::isinf(v)) {
                        inf_count++;
                        continue;
                    }
                    const float av = std::fabs(v);
                    if (av > max_abs_out) {
                        max_abs_out = av;
                    }
                    const float diff = std::fabs(v - ref[i]);
                    if (diff > max_diff) {
                        max_diff = diff;
                    }
                }
                GGML_LOG_INFO("opencl: get_rows ref row=%d max_diff=%g max_abs_out=%g nan=%d inf=%d\n",
                              r, max_diff, max_abs_out, nan_count, inf_count);
            }
        }
    }
}

static void ggml_cl_set_rows(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    // ne0 = ne00
    // ne2 = ne02
    // ne3 = ne03

    const int      ne01 = src0->ne[1];
    const int      ne02 = src0->ne[2];
    const int      ne03 = src0->ne[3];

    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];

    const int      ne11 = src1->ne[1];
    const int      ne12 = src1->ne[2];

    const cl_ulong nb10 = src1->nb[0];
    const cl_ulong nb11 = src1->nb[1];
    const cl_ulong nb12 = src1->nb[2];

    const int      ne0  = dst->ne[0];

    const cl_ulong nb1  = dst->nb[1];
    const cl_ulong nb2  = dst->nb[2];
    const cl_ulong nb3  = dst->nb[3];

    const int nblk0 = ne0/ggml_blck_size(dst->type);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel;

    switch (dst->type) {
        case GGML_TYPE_F32:
            if (src1->type == GGML_TYPE_I64) {
                kernel = backend_ctx->kernel_set_rows_f32_i64;
            } else {
                kernel = backend_ctx->kernel_set_rows_f32_i32;
            }
            break;
        case GGML_TYPE_F16:
            if (src1->type == GGML_TYPE_I64) {
                kernel = backend_ctx->kernel_set_rows_f16_i64;
            } else {
                kernel = backend_ctx->kernel_set_rows_f16_i32;
            }
            break;
        default:
            GGML_ABORT("not implemented");
    }

    fastdiv_vals ne11_ = init_fastdiv_values(ne11);
    fastdiv_vals ne12_ = init_fastdiv_values(ne12);

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(fastdiv_vals), &ne11_));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(fastdiv_vals), &ne12_));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb10));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb12));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &nblk0));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb1));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb2));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb3));

    int nth0 = 64;
    if (backend_ctx->gpu_family == INTEL) {
        nth0 = 32;
    } else if (backend_ctx->gpu_family == ADRENO) {
        nth0 = 64;
    }

    int max_workgroup_size = backend_ctx->get_kernel_workgroup_size(kernel);
    while (nth0 < nblk0 && nth0 < max_workgroup_size) {
        nth0 *= 2;
    }

    int rows_per_workgroup = 1;
    if (nth0 > nblk0) {
        rows_per_workgroup = nth0 / nblk0;
        nth0 = nblk0;
    }

    size_t global_work_size[] = {
        (size_t)(ne01 + rows_per_workgroup - 1)/rows_per_workgroup*nth0,
        (size_t)ne02*rows_per_workgroup,
        (size_t)ne03};
    size_t local_work_size[] = {(size_t)nth0, (size_t)rows_per_workgroup, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_add(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const cl_ulong nb00 = src0->nb[0];
    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3];

    const cl_ulong nb10 = src1->nb[0];
    const cl_ulong nb11 = src1->nb[1];
    const cl_ulong nb12 = src1->nb[2];
    const cl_ulong nb13 = src1->nb[3];

    const int ne0  = dst->ne[0];
    const int ne1  = dst->ne[1];
    const int ne2  = dst->ne[2];
    const int ne3  = dst->ne[3];

    const cl_ulong nb0  = dst->nb[0];
    const cl_ulong nb1  = dst->nb[1];
    const cl_ulong nb2  = dst->nb[2];
    const cl_ulong nb3  = dst->nb[3];

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel;

    const bool bcast_row = ggml_nelements(src1) == ne10 && ggml_is_contiguous(src1) && ne00 % 4 == 0 && ne10 % 4 == 0;

    if (bcast_row) {
        GGML_ASSERT(ggml_is_contiguous(src0));
        GGML_ASSERT(ne11 == 1);
    }

    if (dst->type == GGML_TYPE_F32) {
        GGML_ASSERT(src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32);
        if (bcast_row) {
            kernel = backend_ctx->kernel_add_row;
            const int ne = ne00 / 4;
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &ne));
        } else {
            kernel = backend_ctx->kernel_add;
            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne03));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb00));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb03));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne10));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne11));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &ne13));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb10));
            CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_ulong), &nb13));
            CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 24, sizeof(int),      &ne2));
            CL_CHECK(clSetKernelArg(kernel, 25, sizeof(int),      &ne3));
            CL_CHECK(clSetKernelArg(kernel, 26, sizeof(cl_ulong), &nb0));
            CL_CHECK(clSetKernelArg(kernel, 27, sizeof(cl_ulong), &nb1));
            CL_CHECK(clSetKernelArg(kernel, 28, sizeof(cl_ulong), &nb2));
            CL_CHECK(clSetKernelArg(kernel, 29, sizeof(cl_ulong), &nb3));
        }
    } else if (dst->type == GGML_TYPE_F16) {
        GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
        GGML_ASSERT(src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);
        const int type_src0 = (src0->type == GGML_TYPE_F32);
        const int type_src1 = (src1->type == GGML_TYPE_F32);
        if (bcast_row) {
            kernel = backend_ctx->kernel_add_row_f16;
            const int ne = ne00 / 4;
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &ne));
            CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int),      &type_src0));
            CL_CHECK(clSetKernelArg(kernel, 8, sizeof(int),      &type_src1));
        } else {
            kernel = backend_ctx->kernel_add_f16;
            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne03));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb00));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb03));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne10));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne11));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &ne13));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb10));
            CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_ulong), &nb13));
            CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 24, sizeof(int),      &ne2));
            CL_CHECK(clSetKernelArg(kernel, 25, sizeof(int),      &ne3));
            CL_CHECK(clSetKernelArg(kernel, 26, sizeof(cl_ulong), &nb0));
            CL_CHECK(clSetKernelArg(kernel, 27, sizeof(cl_ulong), &nb1));
            CL_CHECK(clSetKernelArg(kernel, 28, sizeof(cl_ulong), &nb2));
            CL_CHECK(clSetKernelArg(kernel, 29, sizeof(cl_ulong), &nb3));
            CL_CHECK(clSetKernelArg(kernel, 30, sizeof(int),      &type_src0));
            CL_CHECK(clSetKernelArg(kernel, 31, sizeof(int),      &type_src1));
        }
    } else {
        GGML_ASSERT(false && "unsupported data types for add");
    }

    if (bcast_row) {
        int n = ggml_nelements(dst)/4;
        size_t global_work_size[] = {(size_t)n, 1, 1};
        size_t local_work_size[] = {64, 1, 1};

        size_t * local_work_size_ptr = local_work_size;
        if (n % 64 != 0 && !backend_ctx->non_uniform_workgroups) {
            local_work_size_ptr = nullptr;
        }

        backend_ctx->enqueue_ndrange_kernel(kernel, 1, global_work_size, local_work_size_ptr, dst);
    } else {
        unsigned int nth = MIN(64, ne0);
        size_t global_work_size[] = {(size_t)ne01*nth, (size_t)ne02, (size_t)ne03};
        size_t local_work_size[] = {nth, 1, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    }
}

static void ggml_cl_add_id(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    const ggml_tensor * src2 = dst->src[2];
    GGML_ASSERT(src2);
    GGML_ASSERT(src2->extra);

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous_rows(src0));

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];

    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];

    const cl_ulong nb11 = src1->nb[1];

    const cl_ulong nb21 = src2->nb[1];

    const int ne0 = dst->ne[0];
    const int ne1 = dst->ne[1];

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extra2 = (ggml_tensor_extra_cl *)src2->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offset2 = extra2->offset + src2->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel = backend_ctx->kernel_add_id;

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra2->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset2));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb21));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne1));

    int nth = MIN(ne00, (int) backend_ctx->get_kernel_workgroup_size(kernel));
    size_t global_work_size[] = { (size_t)ne01*nth, (size_t)ne02, 1 };
    size_t local_work_size[] = { (size_t)nth, 1, 1 };

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_mul(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const cl_ulong nb00 = src0->nb[0];
    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3]; UNUSED(ne13);

    const cl_ulong nb10 = src1->nb[0];
    const cl_ulong nb11 = src1->nb[1];
    const cl_ulong nb12 = src1->nb[2];
    const cl_ulong nb13 = src1->nb[3]; UNUSED(nb13);

    const int ne0  = dst->ne[0];
    const int ne1  = dst->ne[1];
    const int ne2  = dst->ne[2];
    const int ne3  = dst->ne[3];

    const cl_ulong nb0  = dst->nb[0];
    const cl_ulong nb1  = dst->nb[1];
    const cl_ulong nb2  = dst->nb[2];
    const cl_ulong nb3  = dst->nb[3];

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    bool bcast_row = false;
    cl_kernel kernel;

    if (ggml_nelements(src1) == ne10 && ggml_is_contiguous(src1) && ne00 % 4 == 0 && ne10 % 4 == 0) {
        GGML_ASSERT(ggml_is_contiguous(src0));

        // src1 is a row
        GGML_ASSERT(ne11 == 1);

        bcast_row = true;
        int ne = ne00 / 4;

        if (src0->type == GGML_TYPE_F32) {
            kernel = backend_ctx->kernel_mul_row;
        } else {
            kernel = backend_ctx->kernel_mul_row_f16;
        }

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra1->data_device));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offset1));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &ne));
    } else {
        if (src0->type == GGML_TYPE_F32) {
            kernel = backend_ctx->kernel_mul;
        } else {
            kernel = backend_ctx->kernel_mul_f16;
        }

        CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
        CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
        CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
        CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
        CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
        CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
        CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
        CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne03));
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb00));
        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb01));
        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb02));
        CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb03));
        CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne10));
        CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne11));
        CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &ne12));
        CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &ne13));
        CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb10));
        CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb11));
        CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong), &nb12));
        CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_ulong), &nb13));
        CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),      &ne0));
        CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),      &ne1));
        CL_CHECK(clSetKernelArg(kernel, 24, sizeof(int),      &ne2));
        CL_CHECK(clSetKernelArg(kernel, 25, sizeof(int),      &ne3));
        CL_CHECK(clSetKernelArg(kernel, 26, sizeof(cl_ulong), &nb0));
        CL_CHECK(clSetKernelArg(kernel, 27, sizeof(cl_ulong), &nb1));
        CL_CHECK(clSetKernelArg(kernel, 28, sizeof(cl_ulong), &nb2));
        CL_CHECK(clSetKernelArg(kernel, 29, sizeof(cl_ulong), &nb3));
    }

    if (bcast_row) {
        int n = ggml_nelements(dst)/4;
        size_t global_work_size[] = {(size_t)n, 1, 1};
        size_t local_work_size[] = {64, 1, 1};

        size_t * local_work_size_ptr = local_work_size;
        if (n % 64 != 0 && !backend_ctx->non_uniform_workgroups) {
            local_work_size_ptr = nullptr;  // Let driver choose the work-group sizes.
        }

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
    } else {
        unsigned int nth = MIN(64, ne0);
        size_t global_work_size[] = {ne01*nth, (size_t)ne02, (size_t)ne03};
        size_t local_work_size[] = {nth, 1, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    }
}

static void ggml_cl_div(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const cl_ulong nb00 = src0->nb[0];
    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3];

    const cl_ulong nb10 = src1->nb[0];
    const cl_ulong nb11 = src1->nb[1];
    const cl_ulong nb12 = src1->nb[2];
    const cl_ulong nb13 = src1->nb[3];

    const int ne0  = dst->ne[0];

    const cl_ulong nb0  = dst->nb[0];
    const cl_ulong nb1  = dst->nb[1];
    const cl_ulong nb2  = dst->nb[2];
    const cl_ulong nb3  = dst->nb[3];

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    bool bcast_row = false;
    cl_kernel kernel;

    if (ggml_nelements(src1) == ne10 && ggml_is_contiguous(src1) && ne00 % 4 == 0 && ne10 % 4 == 0) {
        GGML_ASSERT(ggml_is_contiguous(src0));

        // src1 is a row
        GGML_ASSERT(ne11 == 1);

        bcast_row = true;
        int ne = ne00 / 4;

        if (src0->type == GGML_TYPE_F32) {
            kernel = backend_ctx->kernel_div_row;
        } else {
            kernel = backend_ctx->kernel_div_row_f16;
        }

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra1->data_device));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offset1));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &ne));
    } else {
        if (src0->type == GGML_TYPE_F32) {
            kernel = backend_ctx->kernel_div;
        } else {
            kernel = backend_ctx->kernel_div_f16;
        }

        CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
        CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
        CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
        CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
        CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_ulong), &nb00));
        CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &nb01));
        CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb02));
        CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb03));
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne10));
        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne11));
        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne12));
        CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne13));
        CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb10));
        CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb11));
        CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb12));
        CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb13));
        CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &ne0));
        CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb0));
        CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong), &nb1));
        CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_ulong), &nb2));
        CL_CHECK(clSetKernelArg(kernel, 22, sizeof(cl_ulong), &nb3));
    }

    if (bcast_row) {
        int n = ggml_nelements(dst)/4;
        size_t global_work_size[] = {(size_t)n, 1, 1};
        size_t local_work_size[] = {64, 1, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    } else {
        unsigned int nth = MIN(64, ne0);
        size_t global_work_size[] = {ne01*nth, (size_t)ne02, (size_t)ne03};
        size_t local_work_size[] = {nth, 1, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    }
}

static void ggml_cl_sub(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const cl_ulong nb00 = src0->nb[0];
    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3];

    const cl_ulong nb10 = src1->nb[0];
    const cl_ulong nb11 = src1->nb[1];
    const cl_ulong nb12 = src1->nb[2];
    const cl_ulong nb13 = src1->nb[3];

    const int ne0  = dst->ne[0];

    const cl_ulong nb0  = dst->nb[0];
    const cl_ulong nb1  = dst->nb[1];
    const cl_ulong nb2  = dst->nb[2];
    const cl_ulong nb3  = dst->nb[3];

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    bool bcast_row = false;
    cl_kernel kernel;

    if (ggml_nelements(src1) == ne10 && ggml_is_contiguous(src1) && ne00 % 4 == 0 && ne10 % 4 == 0) {
        GGML_ASSERT(ggml_is_contiguous(src0));

        // src1 is a row
        GGML_ASSERT(ne11 == 1);

        bcast_row = true;
        int ne = ne00 / 4;

        if (src0->type == GGML_TYPE_F32) {
            kernel = backend_ctx->kernel_sub_row;
        } else {
            kernel = backend_ctx->kernel_sub_row_f16;
        }

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra1->data_device));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offset1));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &ne));
    } else {
        if (src0->type == GGML_TYPE_F32) {
            kernel = backend_ctx->kernel_sub;
        } else {
            kernel = backend_ctx->kernel_sub_f16;
        }

        CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
        CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
        CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
        CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
        CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_ulong), &nb00));
        CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &nb01));
        CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb02));
        CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb03));
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne10));
        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne11));
        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne12));
        CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne13));
        CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb10));
        CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb11));
        CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb12));
        CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb13));
        CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &ne0));
        CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb0));
        CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong), &nb1));
        CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_ulong), &nb2));
        CL_CHECK(clSetKernelArg(kernel, 22, sizeof(cl_ulong), &nb3));
    }

    if (bcast_row) {
        int n = ggml_nelements(dst)/4;
        size_t global_work_size[] = {(size_t)n, 1, 1};
        size_t local_work_size[] = {64, 1, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    } else {
        unsigned int nth = MIN(64, ne0);
        size_t global_work_size[] = {ne01*nth, (size_t)ne02, (size_t)ne03};
        size_t local_work_size[] = {nth, 1, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    }
}

static void ggml_cl_sqr(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel;

    // Currently assumes src0 is contiguous
    int n = ggml_nelements(dst);
    if (n % 4 == 0) {
        if (src0->type == GGML_TYPE_F32) {
            kernel = backend_ctx->kernel_sqr_cont_f32_4;
        } else {
            kernel = backend_ctx->kernel_sqr_cont_f16_4;
        }
        n /= 4;
    } else {
        if (src0->type == GGML_TYPE_F32) {
            kernel = backend_ctx->kernel_sqr_cont_f32;
        } else {
            kernel = backend_ctx->kernel_sqr_cont_f16;
        }
    }

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));

    size_t global_work_size[] = {(size_t)n, 1, 1};
    size_t local_work_size[] = {64, 1, 1};

    size_t * local_work_size_ptr = local_work_size;
    if (n % 64 != 0 && !backend_ctx->non_uniform_workgroups) {
        local_work_size_ptr = nullptr;
    }

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
}

static void ggml_cl_sqrt(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel;

    // Currently assumes src0 is contiguous
    int n = ggml_nelements(dst);
    if (n % 4 == 0) {
        if (src0->type == GGML_TYPE_F32) {
            kernel = backend_ctx->kernel_sqrt_cont_f32_4;
        } else {
            kernel = backend_ctx->kernel_sqrt_cont_f16_4;
        }
        n /= 4;
    } else {
        if (src0->type == GGML_TYPE_F32) {
            kernel = backend_ctx->kernel_sqrt_cont_f32;
        } else {
            kernel = backend_ctx->kernel_sqrt_cont_f16;
        }
    }

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));

    size_t global_work_size[] = {(size_t)n, 1, 1};
    size_t local_work_size[] = {64, 1, 1};

    size_t * local_work_size_ptr = local_work_size;
    if (n % 64 != 0 && !backend_ctx->non_uniform_workgroups) {
        local_work_size_ptr = nullptr;
    }

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
}

static void ggml_cl_mean(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    GGML_UNUSED(src1);

    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(ggml_is_contiguous(src0));

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];

    const cl_ulong nb1  = dst->nb[1];
    const cl_ulong nb2  = dst->nb[2];
    const cl_ulong nb3  = dst->nb[3];

    cl_kernel kernel = backend_ctx->kernel_mean_f32;

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne03));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb1));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb2));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb3));

    size_t global_work_size[] = {(size_t)ne01, (size_t)ne02, (size_t)ne03};
    size_t local_work_size[] = {(size_t)64, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_ssm_conv(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    int ne01 = src0->ne[1];
    cl_ulong nb00 = src0->nb[0];
    cl_ulong nb01 = src0->nb[1];
    cl_ulong nb02 = src0->nb[2];

    int ne10 = src1->ne[0];
    cl_ulong nb11 = src1->nb[1];

    int ne1  = dst->ne[1];
    int ne2  = dst->ne[2];
    cl_ulong nb0 = dst->nb[0];
    cl_ulong nb1 = dst->nb[1];
    cl_ulong nb2 = dst->nb[2];

    cl_kernel kernel = backend_ctx->kernel_ssm_conv_f32_f32;

    if (ne10 % 4 == 0) {
        kernel = backend_ctx->kernel_ssm_conv_f32_f32_4;
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_ulong), &nb00));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne10));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb0));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb1));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb2));

    size_t global_work_size[] = {(size_t)ne01, (size_t)ne1, (size_t)ne2};
    size_t local_work_size[]  = {64, 1, 1};

    size_t * local_work_size_ptr = local_work_size;
    if (ne01 % 64 != 0 && !backend_ctx->non_uniform_workgroups) {
        local_work_size_ptr = nullptr;
    }

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
}

static void ggml_cl_gelu(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel;

    int n = ggml_nelements(dst);

    if (n % 4 == 0) {
        kernel = backend_ctx->kernel_gelu_4;
        n /= 4;
    } else {
        kernel = backend_ctx->kernel_gelu;
    }

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));

    size_t global_work_size[] = {(size_t)n, 1, 1};
    size_t local_work_size[] = {64, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_gelu_erf(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel;

    int n = ggml_nelements(dst);

    if (n % 4 == 0) {
        kernel = backend_ctx->kernel_gelu_erf_4;
        n /= 4;
    } else {
        kernel = backend_ctx->kernel_gelu_erf;
    }

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));

    size_t global_work_size[] = {(size_t)n, 1, 1};
    size_t local_work_size[] = {64, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_gelu_quick(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel;

    int n = ggml_nelements(dst);

    if (n % 4 == 0) {
        kernel = backend_ctx->kernel_gelu_quick_4;
        n /= 4;
    } else {
        kernel = backend_ctx->kernel_gelu_quick;
    }

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));

    size_t global_work_size[] = {(size_t)n, 1, 1};
    size_t local_work_size[] = {64, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_silu(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    if (getenv("GGML_OPENCL_SILU_CPU") != nullptr || ggml_opencl_force_silu_cpu(dst)) {
        const size_t nbytes_src0 = ggml_nbytes(src0);
        const size_t nbytes_dst  = ggml_nbytes(dst);

        std::vector<char> h_src0(nbytes_src0);
        std::vector<char> h_dst(nbytes_dst);

        ggml_backend_tensor_get(src0, h_src0.data(), 0, nbytes_src0);

        ggml_tensor host_src0 = *src0;
        ggml_tensor host_dst  = *dst;

        host_src0.data = h_src0.data();
        host_src0.buffer = nullptr;
        host_src0.extra = nullptr;
        host_src0.view_src = nullptr;
        host_src0.view_offs = 0;

        host_dst.data = h_dst.data();
        host_dst.buffer = nullptr;
        host_dst.extra = nullptr;
        host_dst.view_src = nullptr;
        host_dst.view_offs = 0;
        host_dst.src[0] = &host_src0;

        ggml_compute_params params;
        params.ith = 0;
        params.nth = 1;
        params.threadpool = nullptr;
        params.wdata = nullptr;
        params.wsize = 0;

        ggml_compute_forward_unary(&params, &host_dst);
        ggml_backend_tensor_set(dst, h_dst.data(), 0, nbytes_dst);
        return;
    }

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel;

    int n = ggml_nelements(dst);
    const bool force_scalar_silu = getenv("GGML_OPENCL_SILU_SCALAR") != nullptr;

    if (!force_scalar_silu && n % 4 == 0) {
        kernel = backend_ctx->kernel_silu_4;
        n /= 4;
    } else {
        kernel = backend_ctx->kernel_silu;
    }

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));

    size_t global_work_size[] = {(size_t)n, 1, 1};
    size_t local_work_size[] = {64, 1, 1};

    size_t * local_work_size_ptr = local_work_size;
    if (n % 64 != 0 && !backend_ctx->non_uniform_workgroups) {
        local_work_size_ptr = nullptr;  // Let driver choose the work-group sizes.
    }

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
}

static void ggml_cl_relu(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel = backend_ctx->kernel_relu;

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));

    const int64_t n = ggml_nelements(dst);

    size_t global_work_size[] = {(size_t)n, 1, 1};
    size_t local_work_size[] = {64, 1, 1};

    size_t * local_work_size_ptr = local_work_size;
    if (n % 64 != 0 && !backend_ctx->non_uniform_workgroups) {
        local_work_size_ptr = nullptr;  // Let driver choose the work-group sizes.
    }

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
}

static void ggml_cl_sigmoid(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel;
    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        kernel = backend_ctx->kernel_sigmoid_f32;
    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        kernel = backend_ctx->kernel_sigmoid_f16;
    } else {
        GGML_ASSERT(false && "Unsupported data types for sigmoid (input and output must be both f32 or f16)");
    }

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));

    const int64_t n = ggml_nelements(dst);

    size_t global_work_size[] = {(size_t)n, 1, 1};
    size_t local_work_size[] = {64, 1, 1};

    size_t * local_work_size_ptr = local_work_size;
    if (n % 64 != 0 && !backend_ctx->non_uniform_workgroups) {
        local_work_size_ptr = nullptr;  // Let driver choose the work-group sizes.
    }

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
}

static void ggml_cl_tri(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const int tri_type = ggml_get_op_params_i32(dst, 0);
    const int64_t n = ggml_nelements(dst);
    const int     ne0  = dst->ne[0];
    const int     ne1  = dst->ne[1];

    cl_kernel kernel = backend_ctx->kernel_tri;

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),      &n));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &ne1));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int),      &tri_type));

    size_t local_work_size[1] = { 256 };
    size_t global_work_size[1] = { ((size_t)n + local_work_size[0] - 1) / local_work_size[0] * local_work_size[0] };

    backend_ctx->enqueue_ndrange_kernel(kernel, 1, global_work_size, local_work_size, dst);
}

static void ggml_cl_fill(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src0);
    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    float v = 0.0f;
    memcpy(&v, ((int32_t *) dst->op_params), sizeof(float));

    const int64_t n = ggml_nelements(dst);

    cl_kernel kernel = backend_ctx->kernel_fill;

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(float),    &v));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(float),    &n));

    size_t local_work_size[1] = { 256 };
    size_t global_work_size[1] = { ((size_t)n + local_work_size[0] - 1) / local_work_size[0] * local_work_size[0] };

    backend_ctx->enqueue_ndrange_kernel(kernel, 1, global_work_size, local_work_size, dst);
}

static void ggml_cl_clamp(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    float min;
    float max;
    memcpy(&min, ((int32_t *) dst->op_params) + 0, sizeof(float));
    memcpy(&max, ((int32_t *) dst->op_params) + 1, sizeof(float));

    cl_kernel kernel = backend_ctx->kernel_clamp;

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(float),    &min));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(float),    &max));

    const int64_t n = ggml_nelements(dst);

    size_t global_work_size[] = {(size_t)n, 1, 1};
    size_t local_work_size[] = {64, 1, 1};

    size_t * local_work_size_ptr = local_work_size;
    if (n % 64 != 0 && !backend_ctx->non_uniform_workgroups) {
        local_work_size_ptr = nullptr;  // Let driver choose the work-group sizes.
    }

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
}

static void ggml_cl_norm(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    const int ne00 = src0 ? src0->ne[0] : 0;
    const int ne01 = src0 ? src0->ne[1] : 0;
    const int ne02 = src0 ? src0->ne[2] : 0;
    const int ne03 = src0 ? src0->ne[3] : 0;

    const cl_ulong nb01 = src0 ? src0->nb[1] : 0;
    const cl_ulong nb02 = src0 ? src0->nb[2] : 0;
    const cl_ulong nb03 = src0 ? src0->nb[3] : 0;

    const int nth = MIN(64, ne00);

    cl_kernel kernel = backend_ctx->kernel_norm;

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),    &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong),  &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),    &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong),  &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(int),       &ne00));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(int),       &ne01));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),       &ne02));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),       &ne03));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong),  &nb01));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong),  &nb02));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong),  &nb03));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(float),     &eps));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(float)*nth, NULL));

    size_t global_work_size[] = {(size_t)ne01*nth, (size_t)ne02, (size_t)ne03};
    size_t local_work_size[] = {(size_t)nth, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_rms_norm(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    //ggml_backend_opencl_device_context * dev_ctx =
    //    (ggml_backend_opencl_device_context *)backend->device->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    const int ne00 = src0 ? src0->ne[0] : 0;
    const int ne01 = src0 ? src0->ne[1] : 0;
    const int ne02 = src0 ? src0->ne[2] : 0;
    const int ne03 = src0 ? src0->ne[3] : 0;

    const cl_ulong nb01 = src0 ? src0->nb[1] : 0;
    const cl_ulong nb02 = src0 ? src0->nb[2] : 0;
    const cl_ulong nb03 = src0 ? src0->nb[3] : 0;

    GGML_ASSERT(ne00 % 4 == 0);

    if (getenv("GGML_OPENCL_RMSNORM_CPU") != nullptr) {
        const size_t nbytes_src0 = ggml_nbytes(src0);
        const size_t nbytes_dst  = ggml_nbytes(dst);

        std::vector<char> h_src0(nbytes_src0);
        std::vector<char> h_dst(nbytes_dst);

        ggml_backend_tensor_get(src0, h_src0.data(), 0, nbytes_src0);

        auto read_f32 = [](const char * base, const ggml_tensor * t, int64_t i0, int64_t i1, int64_t i2, int64_t i3) -> float {
            const size_t off = (size_t)i0 * t->nb[0] + (size_t)i1 * t->nb[1] + (size_t)i2 * t->nb[2] + (size_t)i3 * t->nb[3];
            return *(const float *)(base + off);
        };
        auto write_f32 = [](char * base, const ggml_tensor * t, int64_t i0, int64_t i1, int64_t i2, int64_t i3, float v) {
            const size_t off = (size_t)i0 * t->nb[0] + (size_t)i1 * t->nb[1] + (size_t)i2 * t->nb[2] + (size_t)i3 * t->nb[3];
            *(float *)(base + off) = v;
        };

        for (int64_t i3 = 0; i3 < ne03; ++i3) {
            for (int64_t i2 = 0; i2 < ne02; ++i2) {
                for (int64_t i1 = 0; i1 < ne01; ++i1) {
                    float sum = 0.0f;
                    for (int64_t i0 = 0; i0 < ne00; ++i0) {
                        const float v = read_f32(h_src0.data(), src0, i0, i1, i2, i3);
                        sum += v * v;
                    }
                    const float mean  = sum / (float)ne00;
                    const float scale = 1.0f / sqrtf(mean + eps);
                    for (int64_t i0 = 0; i0 < ne00; ++i0) {
                        const float v = read_f32(h_src0.data(), src0, i0, i1, i2, i3);
                        write_f32(h_dst.data(), dst, i0, i1, i2, i3, v * scale);
                    }
                }
            }
        }

        ggml_backend_tensor_set(dst, h_dst.data(), 0, nbytes_dst);
        return;
    }

    const int nth = MIN(64, ne00);

    size_t global_work_size[] = {(size_t)ne01*nth, (size_t)ne02, (size_t)ne03};
    size_t local_work_size[] = {(size_t)nth, 1, 1};

    cl_kernel kernel = backend_ctx->kernel_rms_norm;

    // Note, this kernel declares local memory in kernel args and the size
    // depends on subgroup size.
    // Note, this requires OpenCL 2.1 and above
    // For now we use fixed subgroup size to simplify support for OpenCL 2.0.
    size_t sgs;
    //CL_CHECK(clGetKernelSubGroupInfo(kernel, dev_ctx->device,
    //    CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE,
    //    sizeof(local_work_size), local_work_size,
    //    sizeof(size_t), &sgs, NULL));
    if (backend_ctx->gpu_family == ADRENO) {
        sgs = 64;
    } else if (backend_ctx->gpu_family == INTEL) {
        sgs = 32;
    } else {
        GGML_ASSERT(false && "Unsupported GPU");
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),    &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong),  &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),    &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong),  &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(int),       &ne00));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(int),       &ne01));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),       &ne02));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),       &ne03));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong),  &nb01));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong),  &nb02));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong),  &nb03));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(float),     &eps));
    // This is local memory - the size depends on subgroup size.
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(float)*nth/sgs,  NULL));

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_opencl_op_rms_norm_fused(ggml_backend_t backend, ggml_tensor * rms_norm_tensor, ggml_tensor * mul_tensor) {
    GGML_ASSERT(mul_tensor);
    GGML_ASSERT(rms_norm_tensor);

    // src0 is the src of rms_norm, src1 is the other src of mul (one being rms_norm)
    const ggml_tensor * src0 = rms_norm_tensor->src[0];
    const ggml_tensor * src1;
    if (mul_tensor->src[0] == rms_norm_tensor) {
        src1 = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == rms_norm_tensor) {
        src1 = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false && "Invalid args for rms_norm and mul");
    }
    ggml_tensor * dst = mul_tensor;

    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    float eps;
    memcpy(&eps, rms_norm_tensor->op_params, sizeof(float));

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3];

    const cl_ulong nb11 = src1->nb[1];
    const cl_ulong nb12 = src1->nb[2];
    const cl_ulong nb13 = src1->nb[3];

    const cl_ulong nb1 = dst->nb[1];
    const cl_ulong nb2 = dst->nb[2];
    const cl_ulong nb3 = dst->nb[3];

    GGML_ASSERT(ne00 % 4 == 0);

    if (getenv("GGML_OPENCL_RMSNORM_CPU") != nullptr) {
        const size_t nbytes_src0 = ggml_nbytes(src0);
        const size_t nbytes_src1 = ggml_nbytes(src1);
        const size_t nbytes_dst  = ggml_nbytes(dst);

        std::vector<char> h_src0(nbytes_src0);
        std::vector<char> h_src1(nbytes_src1);
        std::vector<char> h_dst(nbytes_dst);

        ggml_backend_tensor_get(src0, h_src0.data(), 0, nbytes_src0);
        ggml_backend_tensor_get(src1, h_src1.data(), 0, nbytes_src1);

        auto read_f32 = [](const char * base, const ggml_tensor * t, int64_t i0, int64_t i1, int64_t i2, int64_t i3) -> float {
            const size_t off = (size_t)i0 * t->nb[0] + (size_t)i1 * t->nb[1] + (size_t)i2 * t->nb[2] + (size_t)i3 * t->nb[3];
            return *(const float *)(base + off);
        };
        auto write_f32 = [](char * base, const ggml_tensor * t, int64_t i0, int64_t i1, int64_t i2, int64_t i3, float v) {
            const size_t off = (size_t)i0 * t->nb[0] + (size_t)i1 * t->nb[1] + (size_t)i2 * t->nb[2] + (size_t)i3 * t->nb[3];
            *(float *)(base + off) = v;
        };

        for (int64_t i3 = 0; i3 < ne03; ++i3) {
            for (int64_t i2 = 0; i2 < ne02; ++i2) {
                for (int64_t i1 = 0; i1 < ne01; ++i1) {
                    float sum = 0.0f;
                    for (int64_t i0 = 0; i0 < ne00; ++i0) {
                        const float v = read_f32(h_src0.data(), src0, i0, i1, i2, i3);
                        sum += v * v;
                    }
                    const float mean  = sum / (float)ne00;
                    const float scale = 1.0f / sqrtf(mean + eps);
                    for (int64_t i0 = 0; i0 < ne00; ++i0) {
                        const float v = read_f32(h_src0.data(), src0, i0, i1, i2, i3);
                        const int64_t i10 = i0 % ne10;
                        const int64_t i11 = i1 % ne11;
                        const int64_t i12 = i2 % ne12;
                        const int64_t i13 = i3 % ne13;
                        const float w = read_f32(h_src1.data(), src1, i10, i11, i12, i13);
                        write_f32(h_dst.data(), dst, i0, i1, i2, i3, (v * scale) * w);
                    }
                }
            }
        }

        ggml_backend_tensor_set(dst, h_dst.data(), 0, nbytes_dst);
        return;
    }

    size_t sgs;
    if (backend_ctx->gpu_family == ADRENO) {
        sgs = 64;
    } else if (backend_ctx->gpu_family == INTEL) {
        sgs = 32;
    } else {
        GGML_ASSERT(false && "Unsupported GPU");
    }

    cl_kernel kernel = backend_ctx->kernel_rms_norm_mul;

    int nth = sgs;
    int max_workgroup_size = backend_ctx->get_kernel_workgroup_size(kernel);
    while (nth < ne00 && nth < max_workgroup_size) {
        nth *= 2;
    }
    nth = MIN(nth, max_workgroup_size);
    nth = MIN(nth, ne00);

    size_t global_work_size[] = {(size_t)ne01*nth, (size_t)ne02, (size_t)ne03};
    size_t local_work_size[] = {(size_t)nth, 1, 1};

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),        &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong),      &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),        &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong),      &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),        &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong),      &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),           &ne00));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),           &ne01));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),           &ne02));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),           &ne03));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong),      &nb01));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong),      &nb02));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong),      &nb03));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),           &ne10));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),           &ne11));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),           &ne12));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),           &ne13));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong),      &nb11));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong),      &nb12));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong),      &nb13));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong),      &nb1));
    CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_ulong),      &nb2));
    CL_CHECK(clSetKernelArg(kernel, 22, sizeof(cl_ulong),      &nb3));
    CL_CHECK(clSetKernelArg(kernel, 23, sizeof(float),         &eps));
    CL_CHECK(clSetKernelArg(kernel, 24, sizeof(float)*sgs,     NULL));

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_opencl_op_norm_fused(ggml_backend_t backend, ggml_tensor * norm_tensor, ggml_tensor * mul_tensor, ggml_tensor * add_tensor) {
    GGML_ASSERT(norm_tensor && mul_tensor && add_tensor);

    const ggml_tensor * src0 = norm_tensor->src[0];
    const ggml_tensor * src1 = mul_tensor->src[0] == norm_tensor ? mul_tensor->src[1] : mul_tensor->src[0];
    const ggml_tensor * src2 = add_tensor->src[0] == mul_tensor ? add_tensor->src[1] : add_tensor->src[0];
    const ggml_tensor * dst = add_tensor;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extra2 = (ggml_tensor_extra_cl *)src2->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offset2 = extra2->offset + src2->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    float eps;
    memcpy(&eps, norm_tensor->op_params, sizeof(float));

    const int ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    const cl_ulong nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const int ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];
    const cl_ulong nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    const int ne20 = src2->ne[0], ne21 = src2->ne[1], ne22 = src2->ne[2], ne23 = src2->ne[3];
    const cl_ulong nb21 = src2->nb[1], nb22 = src2->nb[2], nb23 = src2->nb[3];
    const cl_ulong nbd1 = dst->nb[1], nbd2 = dst->nb[2], nbd3 = dst->nb[3];

    size_t sgs;
    if (backend_ctx->gpu_family == ADRENO) sgs = 64;
    else if (backend_ctx->gpu_family == INTEL) sgs = 32;
    else GGML_ASSERT(false && "Unsupported GPU");

    cl_kernel kernel = backend_ctx->kernel_norm_mul_add;

    int nth = sgs;
    int max_workgroup_size = backend_ctx->get_kernel_workgroup_size(kernel);
    while (nth < ne00/4 && nth < max_workgroup_size) nth *= 2;
    nth = MIN(nth, max_workgroup_size);
    nth = MIN(nth, ne00/4);

    size_t gws[] = {(size_t)ne01*nth, (size_t)ne02, (size_t)ne03};
    size_t lws[] = {(size_t)nth, 1, 1};
    size_t num_subgroups = (nth + sgs - 1) / sgs;

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &extra2->data_device));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong), &offset2));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem), &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(int), &ne00));
    CL_CHECK(clSetKernelArg(kernel, 9, sizeof(int), &ne01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int), &ne02));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int), &ne03));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int), &ne10));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int), &ne11));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int), &ne12));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int), &ne13));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong), &nb12));
    CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_ulong), &nb13));
    CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int), &ne20));
    CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int), &ne21));
    CL_CHECK(clSetKernelArg(kernel, 24, sizeof(int), &ne22));
    CL_CHECK(clSetKernelArg(kernel, 25, sizeof(int), &ne23));
    CL_CHECK(clSetKernelArg(kernel, 26, sizeof(cl_ulong), &nb21));
    CL_CHECK(clSetKernelArg(kernel, 27, sizeof(cl_ulong), &nb22));
    CL_CHECK(clSetKernelArg(kernel, 28, sizeof(cl_ulong), &nb23));
    CL_CHECK(clSetKernelArg(kernel, 29, sizeof(cl_ulong), &nbd1));
    CL_CHECK(clSetKernelArg(kernel, 30, sizeof(cl_ulong), &nbd2));
    CL_CHECK(clSetKernelArg(kernel, 31, sizeof(cl_ulong), &nbd3));
    CL_CHECK(clSetKernelArg(kernel, 32, sizeof(float), &eps));
    CL_CHECK(clSetKernelArg(kernel, 33, sizeof(cl_float2) * num_subgroups, NULL));

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, gws, lws, dst);
}

static void ggml_opencl_op_group_norm_fused(ggml_backend_t backend, ggml_tensor * gn_tensor, ggml_tensor * mul_tensor, ggml_tensor * add_tensor) {
    GGML_ASSERT(gn_tensor && mul_tensor && add_tensor);

    const ggml_tensor * src0 = gn_tensor->src[0];
    const ggml_tensor * src1 = mul_tensor->src[0] == gn_tensor ? mul_tensor->src[1] : mul_tensor->src[0];
    const ggml_tensor * src2 = add_tensor->src[0] == mul_tensor ? add_tensor->src[1] : add_tensor->src[0];
    const ggml_tensor * dst = add_tensor;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extra2 = (ggml_tensor_extra_cl *)src2->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offset2 = extra2->offset + src2->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    int groups;
    float eps;
    memcpy(&groups, gn_tensor->op_params, sizeof(int));
    memcpy(&eps, (char *)gn_tensor->op_params + sizeof(int), sizeof(float));

    cl_kernel kernel = backend_ctx->kernel_group_norm_mul_add;
    int max_workgroup_size = backend_ctx->get_kernel_workgroup_size(kernel);
    int ne = ggml_nelements(src0);
    int group_size = ne / groups;

    size_t lws[] = { (size_t)MIN(max_workgroup_size, group_size) };
    size_t gws[] = { (size_t)groups * lws[0] };

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &extra2->data_device));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong), &offset2));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem), &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(int), &ne));
    CL_CHECK(clSetKernelArg(kernel, 9, sizeof(int), &group_size));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(float), &eps));

    backend_ctx->enqueue_ndrange_kernel(kernel, 1, gws, lws, dst);
}

static void ggml_cl_group_norm(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    int32_t n_groups   = ((const int32_t *) dst->op_params)[0];
    int32_t group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + n_groups - 1) / n_groups);
    float   eps        = ((const float *) dst->op_params)[1];

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne = ne00*ne01*ne02;

    cl_kernel kernel = backend_ctx->kernel_group_norm;

    size_t sgs = 64;
    if (backend_ctx->gpu_family == ADRENO) {
        sgs = 64;
    } else if (backend_ctx->gpu_family == INTEL) {
        sgs = 32;
    } else {
        GGML_ASSERT(false && "Unsupported GPU");
    }

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),      &ne));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),      &group_size));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(float),    &eps));

    size_t global_work_size[] = {(size_t)n_groups*sgs, 1, 1};
    size_t local_work_size[] = {(size_t)sgs, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_tanh(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0_abs = extra0->offset + src0->view_offs;
    cl_ulong offsetd_abs = extrad->offset + dst->view_offs;

    cl_kernel kernel;
    if (dst->type == GGML_TYPE_F32) {
        kernel = backend_ctx->kernel_tanh_f32_nd;
    } else if (dst->type == GGML_TYPE_F16) {
        kernel = backend_ctx->kernel_tanh_f16_nd;
    } else {
        GGML_ASSERT(false && "Unsupported type for ggml_cl_tanh");
    }
    GGML_ASSERT(kernel != nullptr);

    const int ne00 = src0->ne[0]; const int ne01 = src0->ne[1]; const int ne02 = src0->ne[2]; const int ne03 = src0->ne[3];
    const cl_ulong nb00 = src0->nb[0]; const cl_ulong nb01 = src0->nb[1]; const cl_ulong nb02 = src0->nb[2]; const cl_ulong nb03 = src0->nb[3];

    const int ne10 = dst->ne[0]; const int ne11 = dst->ne[1]; const int ne12 = dst->ne[2]; const int ne13 = dst->ne[3];
    const cl_ulong nb10 = dst->nb[0]; const cl_ulong nb11 = dst->nb[1]; const cl_ulong nb12 = dst->nb[2]; const cl_ulong nb13 = dst->nb[3];

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0_abs));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd_abs));

    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int),      &ne03));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_ulong), &nb00));
    CL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong),&nb02));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong),&nb03));

    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),     &ne10));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),     &ne11));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),     &ne12));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),     &ne13));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong),&nb10));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong),&nb11));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong),&nb12));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong),&nb13));

    size_t global_work_size[3];
    if (ne10 == 0 || ne11 == 0 || ne12 == 0 || ne13 == 0) { // Handle case of 0 elements
        return;
    }
    global_work_size[0] = (size_t)ne10;
    global_work_size[1] = (size_t)ne11;
    global_work_size[2] = (size_t)ne12;

    size_t lws0 = 16, lws1 = 4, lws2 = 1;
    if (ne10 < 16) lws0 = ne10;
    if (ne11 < 4) lws1 = ne11;
    if (ne12 < 1) lws2 = ne12 > 0 ? ne12 : 1;

    while (lws0 * lws1 * lws2 > 256 && lws0 > 1) lws0 /= 2;
    while (lws0 * lws1 * lws2 > 256 && lws1 > 1) lws1 /= 2;
    while (lws0 * lws1 * lws2 > 256 && lws2 > 1) lws2 /= 2;


    size_t local_work_size[] = {lws0, lws1, lws2};

    size_t* local_work_size_ptr = local_work_size;
    if (!backend_ctx->non_uniform_workgroups) {
        if (global_work_size[0] % local_work_size[0] != 0 ||
            global_work_size[1] % local_work_size[1] != 0 ||
            global_work_size[2] % local_work_size[2] != 0) {
            local_work_size_ptr = NULL;
        }
    }
    if (global_work_size[0] == 0 || global_work_size[1] == 0 || global_work_size[2] == 0) return;

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
}

static void ggml_cl_expm1(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0_abs = extra0->offset + src0->view_offs;
    cl_ulong offsetd_abs = extrad->offset + dst->view_offs;

    cl_kernel kernel;
    if (dst->type == GGML_TYPE_F32) {
        kernel = backend_ctx->kernel_expm1_f32_nd;
    } else if (dst->type == GGML_TYPE_F16) {
        kernel = backend_ctx->kernel_expm1_f16_nd;
    } else {
        GGML_ASSERT(false && "Unsupported type for ggml_cl_expm1");
    }
    GGML_ASSERT(kernel != nullptr);

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const cl_ulong nb00 = src0->nb[0];
    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];

    const int ne10 = dst->ne[0];
    const int ne11 = dst->ne[1];
    const int ne12 = dst->ne[2];
    const int ne13 = dst->ne[3];

    const cl_ulong nb10 = dst->nb[0];
    const cl_ulong nb11 = dst->nb[1];
    const cl_ulong nb12 = dst->nb[2];
    const cl_ulong nb13 = dst->nb[3];

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0_abs));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd_abs));

    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int),      &ne03));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_ulong), &nb00));
    CL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong),&nb02));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong),&nb03));

    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),     &ne10));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),     &ne11));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),     &ne12));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),     &ne13));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong),&nb10));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong),&nb11));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong),&nb12));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong),&nb13));

    size_t global_work_size[3];
    if (ne10 == 0 || ne11 == 0 || ne12 == 0 || ne13 == 0) { // Handle case of 0 elements
        return;
    }
    global_work_size[0] = (size_t)ne10;
    global_work_size[1] = (size_t)ne11;
    global_work_size[2] = (size_t)ne12;

    size_t lws0 = 16, lws1 = 4, lws2 = 1;
    if (ne10 < 16) lws0 = ne10;
    if (ne11 < 4) lws1 = ne11;
    if (ne12 < 1) lws2 = ne12 > 0 ? ne12 : 1;

    while (lws0 * lws1 * lws2 > 256 && lws0 > 1) lws0 /= 2;
    while (lws0 * lws1 * lws2 > 256 && lws1 > 1) lws1 /= 2;
    while (lws0 * lws1 * lws2 > 256 && lws2 > 1) lws2 /= 2;


    size_t local_work_size[] = {lws0, lws1, lws2};

    size_t* local_work_size_ptr = local_work_size;
    if (!backend_ctx->non_uniform_workgroups) {
        if (global_work_size[0] % local_work_size[0] != 0 ||
            global_work_size[1] % local_work_size[1] != 0 ||
            global_work_size[2] % local_work_size[2] != 0) {
            local_work_size_ptr = NULL;
        }
    }
    if (global_work_size[0] == 0 || global_work_size[1] == 0 || global_work_size[2] == 0) return;

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
}

static void ggml_cl_softplus(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0_abs = extra0->offset + src0->view_offs;
    cl_ulong offsetd_abs = extrad->offset + dst->view_offs;

    cl_kernel kernel;
    if (dst->type == GGML_TYPE_F32) {
        kernel = backend_ctx->kernel_softplus_f32_nd;
    } else if (dst->type == GGML_TYPE_F16) {
        kernel = backend_ctx->kernel_softplus_f16_nd;
    } else {
        GGML_ASSERT(false && "Unsupported type for ggml_cl_softplus");
    }
    GGML_ASSERT(kernel != nullptr);

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const cl_ulong nb00 = src0->nb[0];
    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];

    const int ne10 = dst->ne[0];
    const int ne11 = dst->ne[1];
    const int ne12 = dst->ne[2];
    const int ne13 = dst->ne[3];

    const cl_ulong nb10 = dst->nb[0];
    const cl_ulong nb11 = dst->nb[1];
    const cl_ulong nb12 = dst->nb[2];
    const cl_ulong nb13 = dst->nb[3];

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0_abs));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd_abs));

    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int),      &ne03));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_ulong), &nb00));
    CL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong),&nb02));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong),&nb03));

    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),     &ne10));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),     &ne11));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),     &ne12));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),     &ne13));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong),&nb10));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong),&nb11));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong),&nb12));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong),&nb13));

    size_t global_work_size[3];
    if (ne10 == 0 || ne11 == 0 || ne12 == 0 || ne13 == 0) { // Handle case of 0 elements
        return;
    }
    global_work_size[0] = (size_t)ne10;
    global_work_size[1] = (size_t)ne11;
    global_work_size[2] = (size_t)ne12;

    size_t lws0 = 16, lws1 = 4, lws2 = 1;
    if (ne10 < 16) lws0 = ne10;
    if (ne11 < 4) lws1 = ne11;
    if (ne12 < 1) lws2 = ne12 > 0 ? ne12 : 1;

    while (lws0 * lws1 * lws2 > 256 && lws0 > 1) lws0 /= 2;
    while (lws0 * lws1 * lws2 > 256 && lws1 > 1) lws1 /= 2;
    while (lws0 * lws1 * lws2 > 256 && lws2 > 1) lws2 /= 2;


    size_t local_work_size[] = {lws0, lws1, lws2};

    size_t* local_work_size_ptr = local_work_size;
    if (!backend_ctx->non_uniform_workgroups) {
        if (global_work_size[0] % local_work_size[0] != 0 ||
            global_work_size[1] % local_work_size[1] != 0 ||
            global_work_size[2] % local_work_size[2] != 0) {
            local_work_size_ptr = NULL;
        }
    }
    if (global_work_size[0] == 0 || global_work_size[1] == 0 || global_work_size[2] == 0) return;

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
}

static void ggml_cl_repeat(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1_shape_def, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    GGML_ASSERT(dst->type == src0->type);

    UNUSED(src1_shape_def);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    if (backend_ctx->kernel_repeat == nullptr) {
        GGML_LOG_WARN("%s: repeat kernel not available, skipping OpenCL execution.\n", __func__);
        return;
    }

    ggml_tensor_extra_cl * extra_src0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra_dst  = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong off_src0 = extra_src0->offset + src0->view_offs;
    cl_ulong off_dst  = extra_dst->offset  + dst->view_offs;

    const int src0_ne0 = src0->ne[0]; const int src0_ne1 = src0->ne[1]; const int src0_ne2 = src0->ne[2]; const int src0_ne3 = src0->ne[3];
    const cl_ulong src0_nb0 = src0->nb[0]; const cl_ulong src0_nb1 = src0->nb[1]; const cl_ulong src0_nb2 = src0->nb[2]; const cl_ulong src0_nb3 = src0->nb[3];

    const int dst_ne0 = dst->ne[0]; const int dst_ne1 = dst->ne[1]; const int dst_ne2 = dst->ne[2]; const int dst_ne3 = dst->ne[3];
    const cl_ulong dst_nb0 = dst->nb[0]; const cl_ulong dst_nb1 = dst->nb[1]; const cl_ulong dst_nb2 = dst->nb[2]; const cl_ulong dst_nb3 = dst->nb[3];

    cl_kernel kernel = backend_ctx->kernel_repeat;

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),    &extra_src0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),    &extra_dst->data_device));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_ulong),  &off_src0));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong),  &off_dst));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),       &src0_ne0));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),       &src0_ne1));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),       &src0_ne2));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int),       &src0_ne3));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_ulong),  &src0_nb0));
    CL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_ulong),  &src0_nb1));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &src0_nb2));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &src0_nb3));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &dst_ne0));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &dst_ne1));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &dst_ne2));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &dst_ne3));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &dst_nb0));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &dst_nb1));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &dst_nb2));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &dst_nb3));

    size_t gws0 = dst_ne1 > 0 ? (size_t)dst_ne1 : 1;
    size_t gws1 = dst_ne2 > 0 ? (size_t)dst_ne2 : 1;
    size_t gws2 = dst_ne3 > 0 ? (size_t)dst_ne3 : 1;

    size_t global_work_size[] = { gws0, gws1, gws2 };

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, NULL, dst);
}

static void ggml_cl_pad(ggml_backend_t backend, const ggml_tensor * src0, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    if (backend_ctx->kernel_pad == nullptr) {
        GGML_LOG_WARN("%s: pad kernel not available, skipping OpenCL execution.\n", __func__);
        return;
    }

    ggml_tensor_extra_cl * extra_src0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra_dst  = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong off_src0 = extra_src0->offset + src0->view_offs;
    cl_ulong off_dst  = extra_dst->offset  + dst->view_offs;

    const int s_ne0 = src0->ne[0];
    const int s_ne1 = src0->ne[1];
    const int s_ne2 = src0->ne[2];
    const int s_ne3 = src0->ne[3];

    const int s_nb0 = src0->nb[0];
    const int s_nb1 = src0->nb[1];
    const int s_nb2 = src0->nb[2];
    const int s_nb3 = src0->nb[3];

    const int d_ne0 = dst->ne[0];
    const int d_ne1 = dst->ne[1];
    const int d_ne2 = dst->ne[2];
    const int d_ne3 = dst->ne[3];

    const int d_nb0 = dst->nb[0];
    const int d_nb1 = dst->nb[1];
    const int d_nb2 = dst->nb[2];
    const int d_nb3 = dst->nb[3];

    const int lp0 = ((const int*)(dst->op_params))[0];
    const int rp0 = ((const int*)(dst->op_params))[1];
    const int lp1 = ((const int*)(dst->op_params))[2];
    const int rp1 = ((const int*)(dst->op_params))[3];
    const int lp2 = ((const int*)(dst->op_params))[4];
    const int rp2 = ((const int*)(dst->op_params))[5];
    const int lp3 = ((const int*)(dst->op_params))[6];
    const int rp3 = ((const int*)(dst->op_params))[7];

    cl_kernel kernel = backend_ctx->kernel_pad;

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),    &extra_src0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong),  &off_src0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),    &extra_dst->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong),  &off_dst));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(int),       &s_ne0));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(int),       &s_ne1));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),       &s_ne2));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),       &s_ne3));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong),  &s_nb0));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong),  &s_nb1));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong),  &s_nb2));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong),  &s_nb3));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),       &d_ne0));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),       &d_ne1));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),       &d_ne2));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),       &d_ne3));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong),  &d_nb0));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong),  &d_nb1));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong),  &d_nb2));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong),  &d_nb3));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),       &lp0));
    CL_CHECK(clSetKernelArg(kernel, 21, sizeof(int),       &rp0));
    CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),       &lp1));
    CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),       &rp1));
    CL_CHECK(clSetKernelArg(kernel, 24, sizeof(int),       &lp2));
    CL_CHECK(clSetKernelArg(kernel, 25, sizeof(int),       &rp2));
    CL_CHECK(clSetKernelArg(kernel, 26, sizeof(int),       &lp3));
    CL_CHECK(clSetKernelArg(kernel, 27, sizeof(int),       &rp3));

    size_t lws0 = 64;
    size_t gws0 = (( (size_t)d_ne0 + lws0 - 1 ) / lws0) * lws0;

    size_t global_work_size[] = { gws0, (size_t)d_ne1, (size_t)d_ne2*d_ne3 };
    size_t local_work_size[]  = { lws0, 1, 1 };

    size_t * local_work_size_ptr = local_work_size;
     if (d_ne0 % lws0 != 0 && !backend_ctx->non_uniform_workgroups) {
        local_work_size_ptr = nullptr;
    }

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
}

static void ggml_cl_upscale(ggml_backend_t backend, const ggml_tensor * src0, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    const int mode_flags        = (ggml_scale_mode) ggml_get_op_params_i32(dst, 0);
    const ggml_scale_mode mode  = (ggml_scale_mode) (mode_flags & 0xFF);
    cl_kernel kernel = nullptr;

    if (mode == GGML_SCALE_MODE_NEAREST) {
        kernel = backend_ctx->kernel_upscale;
        if (kernel == nullptr) {
            GGML_LOG_WARN("%s: nearest upscale kernel not available, skipping OpenCL execution.\n", __func__);
            return;
        }
    } else if (mode == GGML_SCALE_MODE_BILINEAR) {
        kernel = backend_ctx->kernel_upscale_bilinear;
        if (kernel == nullptr) {
            GGML_LOG_WARN("%s: bilinear upscale kernel not available, skipping OpenCL execution.\n", __func__);
            return;
        }
    } else {
        GGML_LOG_WARN("%s: unsupported upscale mode %d, skipping OpenCL execution.\n", __func__, mode);
        return;
    }

    ggml_tensor_extra_cl * extra_src0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra_dst  = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong off_src0 = extra_src0->offset + src0->view_offs;
    cl_ulong off_dst  = extra_dst->offset  + dst->view_offs;

    const cl_ulong nb00 = src0->nb[0];
    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const int ne0 = dst->ne[0];
    const int ne1 = dst->ne[1];
    const int ne2 = dst->ne[2];
    const int ne3 = dst->ne[3];

    float sf0 = (float)ne0 / ne00;
    float sf1 = (float)ne1 / ne01;
    float sf2 = (float)ne2 / ne02;
    float sf3 = (float)ne3 / ne03;

    float pixel_offset = 0.5f;

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),    &extra_src0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong),  &off_src0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),    &extra_dst->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong),  &off_dst));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_ulong),  &nb00));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong),  &nb01));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_ulong),  &nb02));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_ulong),  &nb03));

    if (mode == GGML_SCALE_MODE_NEAREST) {
        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(int),       &ne0));
        CL_CHECK(clSetKernelArg(kernel, 9, sizeof(int),       &ne1));
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne2));
        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne3));
        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(float),    &sf0));
        CL_CHECK(clSetKernelArg(kernel, 13, sizeof(float),    &sf1));
        CL_CHECK(clSetKernelArg(kernel, 14, sizeof(float),    &sf2));
        CL_CHECK(clSetKernelArg(kernel, 15, sizeof(float),    &sf3));
    } else if (mode == GGML_SCALE_MODE_BILINEAR) {
        if (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
            sf0 = ne0 > 1 && ne00 > 1 ? (float)(ne0 - 1) / (ne00 - 1) : sf0;
            sf1 = ne1 > 1 && ne01 > 1 ? (float)(ne1 - 1) / (ne01 - 1) : sf1;
            pixel_offset = 0.0f;
        }

        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(int),       &ne00));
        CL_CHECK(clSetKernelArg(kernel, 9, sizeof(int),       &ne01));
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne0));
        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne1));
        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne2));
        CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne3));
        CL_CHECK(clSetKernelArg(kernel, 14, sizeof(float),    &sf0));
        CL_CHECK(clSetKernelArg(kernel, 15, sizeof(float),    &sf1));
        CL_CHECK(clSetKernelArg(kernel, 16, sizeof(float),    &sf2));
        CL_CHECK(clSetKernelArg(kernel, 17, sizeof(float),    &sf3));
        CL_CHECK(clSetKernelArg(kernel, 18, sizeof(float),    &pixel_offset));
    }


    size_t dst_total_elements = (size_t)ne0 * ne1 * ne2 * ne3;
    if (dst_total_elements == 0) {
        return;
    }
    size_t global_work_size[] = { dst_total_elements, 1, 1 };
    size_t local_work_size_pref = 256;
    size_t local_work_size[] = { MIN(local_work_size_pref, dst_total_elements), 1, 1};

    size_t * local_work_size_ptr = local_work_size;
    if (dst_total_elements % local_work_size[0] != 0 && !backend_ctx->non_uniform_workgroups) {
        local_work_size_ptr = nullptr;
    }

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
}

static void ggml_cl_concat(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    cl_command_queue queue = backend_ctx->queue;

    if (backend_ctx->kernel_concat_f32_contiguous == nullptr || backend_ctx->kernel_concat_f32_non_contiguous == nullptr) {
        GGML_LOG_WARN("%s: concat kernels not available, skipping OpenCL execution.\n", __func__);
        return;
    }

    ggml_tensor_extra_cl * extra0_cl = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1_cl = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad_cl = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong off_src0 = extra0_cl->offset + src0->view_offs;
    cl_ulong off_src1 = extra1_cl->offset + src1->view_offs;
    cl_ulong off_dst  = extrad_cl->offset + dst->view_offs;

    const int32_t dim = ((const int32_t *) dst->op_params)[0];
    GGML_ASSERT(dim >= 0 && dim <= 3);

    const bool disable_concat = getenv("GGML_OPENCL_DISABLE_CONCAT") != nullptr;
    const bool allow_dim0_concat = getenv("GGML_OPENCL_ALLOW_DIM0_CONCAT") != nullptr;
    const bool force_cpu_concat = (backend_ctx->gpu_family == ADRENO && dim == 0 && !allow_dim0_concat);
    if (disable_concat || force_cpu_concat) {
        const size_t nbytes_src0 = ggml_nbytes(src0);
        const size_t nbytes_src1 = ggml_nbytes(src1);
        const size_t nbytes_dst  = ggml_nbytes(dst);

        std::vector<char> h_src0(nbytes_src0);
        std::vector<char> h_src1(nbytes_src1);
        std::vector<char> h_dst(nbytes_dst);

        ggml_backend_tensor_get(src0, h_src0.data(), 0, nbytes_src0);
        ggml_backend_tensor_get(src1, h_src1.data(), 0, nbytes_src1);

        const int64_t d0 = dst->ne[0], d1 = dst->ne[1], d2 = dst->ne[2], d3 = dst->ne[3];
        const int64_t s0_0 = src0->ne[0], s1_0 = src0->ne[1], s2_0 = src0->ne[2], s3_0 = src0->ne[3];

        for (int64_t i3 = 0; i3 < d3; ++i3) {
            for (int64_t i2 = 0; i2 < d2; ++i2) {
                for (int64_t i1 = 0; i1 < d1; ++i1) {
                    for (int64_t i0 = 0; i0 < d0; ++i0) {
                        bool use_src0 = true;
                        int64_t si0 = i0, si1 = i1, si2 = i2, si3 = i3;

                        if (dim == 0) {
                            use_src0 = (i0 < s0_0);
                            if (!use_src0) { si0 = i0 - s0_0; }
                        } else if (dim == 1) {
                            use_src0 = (i1 < s1_0);
                            if (!use_src0) { si1 = i1 - s1_0; }
                        } else if (dim == 2) {
                            use_src0 = (i2 < s2_0);
                            if (!use_src0) { si2 = i2 - s2_0; }
                        } else { // dim == 3
                            use_src0 = (i3 < s3_0);
                            if (!use_src0) { si3 = i3 - s3_0; }
                        }

                        const ggml_tensor * src = use_src0 ? src0 : src1;
                        const char * h_src = use_src0 ? h_src0.data() : h_src1.data();

                        const size_t off_src = (size_t)si0 * src->nb[0] + (size_t)si1 * src->nb[1] +
                                               (size_t)si2 * src->nb[2] + (size_t)si3 * src->nb[3];
                        const size_t off_dst = (size_t)i0 * dst->nb[0] + (size_t)i1 * dst->nb[1] +
                                               (size_t)i2 * dst->nb[2] + (size_t)i3 * dst->nb[3];

                        *(float *)(h_dst.data() + off_dst) = *(const float *)(h_src + off_src);
                    }
                }
            }
        }

        ggml_backend_tensor_set(dst, h_dst.data(), 0, nbytes_dst);
        return;
    }

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)) {
        if (dim == 3) {

            size_t nbytes_src0 = ggml_nbytes(src0);
            size_t nbytes_src1 = ggml_nbytes(src1);

            CL_CHECK(clEnqueueCopyBuffer(queue, extra0_cl->data_device, extrad_cl->data_device,
                                         off_src0, off_dst, nbytes_src0, 0, NULL, NULL));
            CL_CHECK(clEnqueueCopyBuffer(queue, extra1_cl->data_device, extrad_cl->data_device,
                                         off_src1, off_dst + nbytes_src0, nbytes_src1, 0, NULL, NULL));
        } else {

            cl_kernel kernel = backend_ctx->kernel_concat_f32_contiguous;
            size_t global_work_size[3];

            for (int i3 = 0; i3 < dst->ne[3]; ++i3) {
                cl_ulong current_off_src0 = off_src0 + (i3 * src0->nb[3]);
                cl_ulong current_off_src1 = off_src1 + (i3 * src1->nb[3]);
                cl_ulong current_off_dst  = off_dst  + (i3 * dst->nb[3]);

                int d_ne00 = src0->ne[0]; int d_ne01 = src0->ne[1]; int d_ne02 = src0->ne[2];
                int d_ne10 = src1->ne[0]; int d_ne11 = src1->ne[1]; int d_ne12 = src1->ne[2];
                int d_ne0  = dst->ne[0];  int d_ne1  = dst->ne[1];  int d_ne2  = dst->ne[2];

                CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),    &extra0_cl->data_device));
                CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong),  &current_off_src0));
                CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),    &extra1_cl->data_device));
                CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong),  &current_off_src1));
                CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),    &extrad_cl->data_device));
                CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong),  &current_off_dst));
                CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),       &d_ne00));
                CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int),       &d_ne01));
                CL_CHECK(clSetKernelArg(kernel, 8, sizeof(int),       &d_ne02));
                CL_CHECK(clSetKernelArg(kernel, 9, sizeof(int),       &d_ne10));
                CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &d_ne11));
                CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &d_ne12));
                CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &d_ne0));
                CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &d_ne1));
                CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &d_ne2));
                CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &dim));

                global_work_size[0] = d_ne0;
                global_work_size[1] = d_ne1;
                global_work_size[2] = d_ne2;

                backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, NULL, dst);
            }
        }
    } else {
        cl_kernel kernel = backend_ctx->kernel_concat_f32_non_contiguous;

        cl_long ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
        cl_ulong nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];

        cl_ulong nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];

        cl_long d_ne0 = dst->ne[0], d_ne1 = dst->ne[1], d_ne2 = dst->ne[2], d_ne3 = dst->ne[3];
        cl_ulong d_nb0 = dst->nb[0], d_nb1 = dst->nb[1], d_nb2 = dst->nb[2], d_nb3 = dst->nb[3];


        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),    &extra0_cl->data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong),  &off_src0));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),    &extra1_cl->data_device));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong),  &off_src1));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),    &extrad_cl->data_device));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong),  &off_dst));

        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_long),      &ne00));
        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_long),      &ne01));
        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_long),      &ne02));
        CL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_long),      &ne03));
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong),    &nb00));
        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong),    &nb01));
        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong),    &nb02));
        CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong),    &nb03));

        CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong),    &nb10));
        CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong),    &nb11));
        CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong),    &nb12));
        CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong),    &nb13));

        CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_long),     &d_ne0));
        CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_long),     &d_ne1));
        CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_long),     &d_ne2));
        CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_long),     &d_ne3));
        CL_CHECK(clSetKernelArg(kernel, 22, sizeof(cl_ulong),    &d_nb0));
        CL_CHECK(clSetKernelArg(kernel, 23, sizeof(cl_ulong),    &d_nb1));
        CL_CHECK(clSetKernelArg(kernel, 24, sizeof(cl_ulong),    &d_nb2));
        CL_CHECK(clSetKernelArg(kernel, 25, sizeof(cl_ulong),    &d_nb3));
        CL_CHECK(clSetKernelArg(kernel, 26, sizeof(int),      &dim));

        size_t global_work_size_nc[] = { d_ne1 > 0 ? (size_t)d_ne1 : 1,
                                         d_ne2 > 0 ? (size_t)d_ne2 : 1,
                                         d_ne3 > 0 ? (size_t)d_ne3 : 1 };

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size_nc, NULL, dst);
    }
}

static void ggml_cl_timestep_embedding(ggml_backend_t backend, const ggml_tensor * src0, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    if (backend_ctx->kernel_timestep_embedding == nullptr) {
        GGML_LOG_WARN("%s: timestep_embedding kernel not available, skipping OpenCL execution.\n", __func__);
        return;
    }

    ggml_tensor_extra_cl * extra_src0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra_dst  = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong off_src0 = extra_src0->offset + src0->view_offs;
    cl_ulong off_dst  = extra_dst->offset  + dst->view_offs;

    const int logical_dim = dst->op_params[0];
    const int max_period  = dst->op_params[1];
    const int dst_nb1_bytes = dst->nb[1];

    cl_kernel kernel = backend_ctx->kernel_timestep_embedding;

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),    &extra_src0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong),  &off_src0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),    &extra_dst->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong),  &off_dst));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),       &dst_nb1_bytes));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),       &logical_dim));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),       &max_period));

    size_t gws0 = (size_t)(((logical_dim + 1) / 2) + 1);

    size_t gws1 = (size_t)src0->ne[0];

    size_t global_work_size[] = {gws0, gws1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, NULL, dst);
}

static void ggml_cl_flash_attn(ggml_backend_t backend, const ggml_tensor * q, const ggml_tensor * k, ggml_tensor * dst) {
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    GGML_ASSERT(q->extra);
    GGML_ASSERT(k->extra);
    GGML_ASSERT(v->extra);
    GGML_ASSERT(dst->extra);
    if (mask) {
        GGML_ASSERT(mask->extra);
    }
    if (sinks) {
        GGML_ASSERT(sinks->extra);
    }

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    const int n_q = q->ne[1];
    const int n_kv = k->ne[1];
    const int d_head_q = q->ne[0];
    const int d_head_v = v->ne[0];
    const int n_head = q->ne[2];
    const int n_head_kv = k->ne[2];
    const int n_batch = q->ne[3];

    cl_kernel kernel = NULL;

    const bool is_f16 = q->type == GGML_TYPE_F16;
    const bool is_mixed = q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16;
    const bool force_flash_f32 = std::getenv("GGML_OPENCL_FLASH_FORCE_F32") != nullptr;
    const bool use_mixed_kernel = is_mixed && !force_flash_f32;
    const std::pair<int, int> dk_dv = {d_head_q, d_head_v};

    if (n_q == 1) {
        if (use_mixed_kernel) {
            kernel = backend_ctx->kernels_flash_attn_f32_f16_q1.at(dk_dv);
        } else if (is_f16) {
            kernel = backend_ctx->kernels_flash_attn_f16_q1.at(dk_dv);
        } else {
            kernel = backend_ctx->kernels_flash_attn_f32_q1.at(dk_dv);
        }
    } else {
        if (use_mixed_kernel) {
            kernel = backend_ctx->kernels_flash_attn_f32_f16.at(dk_dv);
        } else if (is_f16) {
            kernel = backend_ctx->kernels_flash_attn_f16.at(dk_dv);
        } else {
            kernel = backend_ctx->kernels_flash_attn_f32.at(dk_dv);
        }
    }
    GGML_ASSERT(kernel != NULL);

    ggml_tensor_extra_cl * extra_q = (ggml_tensor_extra_cl *)q->extra;
    ggml_tensor_extra_cl * extra_k = (ggml_tensor_extra_cl *)k->extra;
    ggml_tensor_extra_cl * extra_v = (ggml_tensor_extra_cl *)v->extra;
    ggml_tensor_extra_cl * extra_o = (ggml_tensor_extra_cl *)dst->extra;
    ggml_tensor_extra_cl * extra_mask = mask ? (ggml_tensor_extra_cl *)mask->extra : NULL;
    ggml_tensor_extra_cl * extra_sinks = sinks ? (ggml_tensor_extra_cl *)sinks->extra : NULL;

    cl_ulong offset_q = extra_q->offset + q->view_offs;
    cl_ulong offset_k = extra_k->offset + k->view_offs;
    cl_ulong offset_v = extra_v->offset + v->view_offs;
    cl_ulong offset_o = extra_o->offset + dst->view_offs;
    cl_mem   mask_buffer = extra_mask ? extra_mask->data_device : NULL;
    cl_ulong offset_mask = extra_mask ? extra_mask->offset + mask->view_offs : 0;
    cl_mem   sinks_buffer = extra_sinks ? extra_sinks->data_device : NULL;
    cl_ulong offset_sinks = extra_sinks ? extra_sinks->offset + sinks->view_offs : 0;

    const cl_ulong q_nb1 = q->nb[1], q_nb2 = q->nb[2], q_nb3 = q->nb[3];
    const cl_ulong k_nb1 = k->nb[1], k_nb2 = k->nb[2], k_nb3 = k->nb[3];
    const cl_ulong v_nb1 = v->nb[1], v_nb2 = v->nb[2], v_nb3 = v->nb[3];
    const cl_ulong o_nb1 = dst->nb[1], o_nb2 = dst->nb[2], o_nb3 = dst->nb[3];
    const cl_ulong mask_nb1 = mask ? mask->nb[1] : 0;
    const cl_ulong mask_nb2 = mask ? mask->nb[2] : 0;
    const cl_ulong mask_nb3 = mask ? mask->nb[3] : 0;
    const int mask_ne2 = mask ? mask->ne[2] : 0;
    const int mask_ne3 = mask ? mask->ne[3] : 0;

    float scale, max_bias, logit_softcap;
    const float * params = (const float *)dst->op_params;
    scale         = params[0];
    max_bias      = params[1];
    logit_softcap = params[2];

    const char * flash_dump_dir = std::getenv("GGML_OPENCL_FLASH_DUMP_DIR");
    static std::atomic<int> flash_dump_call_counter {0};
    static std::atomic<int> flash_call_counter {0};
    const int flash_call_id = flash_call_counter.fetch_add(1, std::memory_order_relaxed);
    bool flash_dump_this_call = false;
    int flash_dump_call = -1;
    const bool flash_dump_match_h24 =
        is_mixed && v->type == GGML_TYPE_F16 &&
        mask == nullptr && sinks == nullptr &&
        n_batch == 1 && n_q == n_kv &&
        n_head == 24 && d_head_q == 128 && d_head_v == 128;
    const bool flash_dump_match_h30 =
        is_mixed && v->type == GGML_TYPE_F16 &&
        sinks == nullptr &&
        n_batch == 1 &&
        n_head == 30 && d_head_q == 128 && d_head_v == 128 &&
        n_q > 0 && n_q <= ggml_opencl_mldrift_h30_m4224::kShapeM &&
        (n_kv == ggml_opencl_mldrift_h30_m4224::kShapeN || n_kv == ggml_opencl_mldrift_h30_m4224::kShapeN + 128);
    if (flash_dump_dir != nullptr && (flash_dump_match_h24 || flash_dump_match_h30)) {
        flash_dump_call = flash_dump_call_counter.fetch_add(1, std::memory_order_relaxed);
        int target_call = 0;
        if (const char * env = std::getenv("GGML_OPENCL_FLASH_DUMP_CALL")) {
            target_call = std::atoi(env);
            if (target_call < 0) {
                target_call = 0;
            }
        }
        flash_dump_this_call = flash_dump_call == target_call;
    }
    if (flash_dump_this_call) {
        ggml_cl_dump_buffer_region(backend_ctx, extra_q->data_device, offset_q, ggml_nbytes(q),
                                   std::string(flash_dump_dir) + "/flash_call" + std::to_string(flash_dump_call) + "_q_f32.bin");
        ggml_cl_dump_buffer_region(backend_ctx, extra_k->data_device, offset_k, ggml_nbytes(k),
                                   std::string(flash_dump_dir) + "/flash_call" + std::to_string(flash_dump_call) + "_k_f16.bin");
        ggml_cl_dump_buffer_region(backend_ctx, extra_v->data_device, offset_v, ggml_nbytes(v),
                                   std::string(flash_dump_dir) + "/flash_call" + std::to_string(flash_dump_call) + "_v_f16.bin");
    }

    const bool use_mldrift = std::getenv("GGML_OPENCL_MLDRIFT") != nullptr;
    const bool mldrift_dynamic_4352 = std::getenv("GGML_OPENCL_MLDRIFT_DYNAMIC_4352") != nullptr;
    const bool mldrift_debug = std::getenv("GGML_OPENCL_MLDRIFT_DEBUG") != nullptr;
    const bool mldrift_debug_all = std::getenv("GGML_OPENCL_MLDRIFT_DEBUG_ALL") != nullptr;
    if (use_mldrift && mldrift_debug) {
        static bool printed = false;
        if (!printed) {
            printed = true;
            GGML_LOG_INFO("ggml_opencl: mldrift check q=(%d,%d,%d,%d,%d) k=(%d,%d,%d,%d,%d) v_type=%d mask=%d sinks=%d contig(q,k,v,o)=(%d,%d,%d,%d) max_bias=%g logit_softcap=%g\n",
                          (int) q->type, n_q, d_head_q, n_head, n_batch,
                          (int) k->type, n_kv, (int) k->ne[0], n_head_kv, (int) k->ne[3],
                          (int) v->type, mask != nullptr, sinks != nullptr,
                          ggml_is_contiguous(q), ggml_is_contiguous(k), ggml_is_contiguous(v), ggml_is_contiguous(dst),
                          (double) max_bias, (double) logit_softcap);
        }
    }
    const bool mldrift_shape_supported_h24 =
        (n_q == ggml_opencl_mldrift_h24_m4352::kShapeM) ||
        (mldrift_dynamic_4352 &&
         n_q > 0 &&
         n_q < ggml_opencl_mldrift_h24_m4352::kShapeM &&
         (n_q % 128) == 0);

    const bool mldrift_shape_match_h24 =
        backend_ctx->gpu_family == ADRENO &&
        is_mixed && v->type == GGML_TYPE_F16 &&
        mask == nullptr && sinks == nullptr &&
        n_batch == 1 &&
        n_head == ggml_opencl_mldrift_h24_m4352::kShapeH &&
        n_head_kv == n_head &&
        mldrift_shape_supported_h24 &&
        n_kv == n_q &&
        d_head_q == ggml_opencl_mldrift_h24_m4352::kShapeK &&
        d_head_v == d_head_q &&
        ggml_is_contiguous(q) &&
        ggml_is_contiguous(k) &&
        ggml_is_contiguous(v) &&
        ggml_is_contiguous(dst) &&
        std::abs(max_bias) <= 1e-6f &&
        std::abs(logit_softcap) <= 1e-6f;
    const bool mldrift_shape_supported_h30 =
        n_q > 0 &&
        n_q <= ggml_opencl_mldrift_h30_m4224::kShapeM &&
        (n_q % 4) == 0;
    const bool mldrift_h30_mask_compatible =
        (mask == nullptr) ||
        (n_kv == ggml_opencl_mldrift_h30_m4224::kShapeN + 128);
    const bool mldrift_h30_kv_supported =
        // Native replay shape.
        (n_kv == ggml_opencl_mldrift_h30_m4224::kShapeN) ||
        // Common text-extended mask case (N + 128).
        (n_kv == ggml_opencl_mldrift_h30_m4224::kShapeN + 128) ||
        // Unmasked dynamic case (e.g. 4096x4096 in z-image).
        (mask == nullptr &&
         n_kv > 0 &&
         n_kv <= ggml_opencl_mldrift_h30_m4224::kShapeN &&
         (n_kv % 4) == 0);
    const bool mldrift_shape_match_h30 =
        backend_ctx->gpu_family == ADRENO &&
        is_mixed && v->type == GGML_TYPE_F16 &&
        mldrift_h30_mask_compatible && sinks == nullptr &&
        n_batch == 1 &&
        n_head == ggml_opencl_mldrift_h30_m4224::kShapeH &&
        n_head_kv == n_head &&
        mldrift_shape_supported_h30 &&
        mldrift_h30_kv_supported &&
        d_head_q == ggml_opencl_mldrift_h30_m4224::kShapeK &&
        d_head_v == d_head_q &&
        ggml_is_contiguous(q) &&
        ggml_is_contiguous(k) &&
        ggml_is_contiguous(v) &&
        ggml_is_contiguous(dst) &&
        std::abs(max_bias) <= 1e-6f &&
        std::abs(logit_softcap) <= 1e-6f;
    const bool mldrift_shape_match = mldrift_shape_match_h24 || mldrift_shape_match_h30;
    if (use_mldrift && mldrift_debug_all) {
        GGML_LOG_INFO("ggml_opencl: mldrift call id=%d qtype=%s ktype=%s vtype=%s mixed=%d n_q=%d n_kv=%d d=%d h=%d h_kv=%d b=%d mask=%d sinks=%d contig(q,k,v,o)=(%d,%d,%d,%d) bias=%g softcap=%g match_h24=%d match_h30=%d\n",
                      flash_call_id,
                      ggml_type_name(q->type), ggml_type_name(k->type), ggml_type_name(v->type), is_mixed ? 1 : 0,
                      n_q, n_kv, d_head_q, n_head, n_head_kv, n_batch,
                      mask != nullptr ? 1 : 0, sinks != nullptr ? 1 : 0,
                      ggml_is_contiguous(q) ? 1 : 0,
                      ggml_is_contiguous(k) ? 1 : 0,
                      ggml_is_contiguous(v) ? 1 : 0,
                      ggml_is_contiguous(dst) ? 1 : 0,
                      (double) max_bias, (double) logit_softcap,
                      mldrift_shape_match_h24 ? 1 : 0,
                      mldrift_shape_match_h30 ? 1 : 0);
    }

    if (use_mldrift && mldrift_shape_match) {
        if (mldrift_shape_match_h24) {
            namespace mh = ggml_opencl_mldrift_h24_m4352;
            if (mldrift_attn_init_h24(backend_ctx, n_q)) {
                auto & st = backend_ctx->mldrift_attn_h24;

            const size_t q_bytes = ggml_nbytes(q);
            const size_t k_bytes = ggml_nbytes(k);
            const size_t v_bytes = ggml_nbytes(v);
            const size_t out_bytes = ggml_nbytes(dst);

            if (q_bytes <= mh::kBufferSizes[mh::kInputBufIds[0]] &&
                out_bytes <= mh::kBufferSizes[mh::kOutputBufId] &&
                k_bytes * 2 <= mh::kBufferSizes[mh::kInputBufIds[1]] &&
                v_bytes * 2 <= mh::kBufferSizes[mh::kInputBufIds[2]]) {
                if (!mldrift_attn_patch_shape_h24(backend_ctx, n_q)) {
                    if (mldrift_debug_all) {
                        GGML_LOG_WARN("ggml_opencl: mldrift dynamic shape patch failed for M=%d, fallback to native flash\n", n_q);
                    }
                    goto mldrift_fallback;
                }

                int q_dst_id = mh::kInputBufIds[0];
                int k_dst_id = mh::kInputBufIds[1];
                int v_dst_id = mh::kInputBufIds[2];
                if (const char * map = std::getenv("GGML_OPENCL_MLDRIFT_INPUT_MAP")) {
                    // Optional debug override, e.g. "687" means q->6, k->8, v->7.
                    if (std::strlen(map) >= 3) {
                        const int a = map[0] - '0';
                        const int b = map[1] - '0';
                        const int c = map[2] - '0';
                        const bool in_range = (a >= 0 && a < mh::kNumBuffers) &&
                                              (b >= 0 && b < mh::kNumBuffers) &&
                                              (c >= 0 && c < mh::kNumBuffers);
                        if (in_range && a != b && a != c && b != c) {
                            q_dst_id = a;
                            k_dst_id = b;
                            v_dst_id = c;
                        }
                    }
                }
                const int q_elems = (int) (q_bytes / sizeof(float));
                const int kv_elems = (int) (k_bytes / sizeof(uint16_t));
                // Prefer the stable replay-compatible defaults; legacy behavior stays opt-in.
                const bool force_no_scale = std::getenv("GGML_OPENCL_MLDRIFT_NO_SCALE") != nullptr;
                float q_scale_mul = 1.0f;
                if (const char * q_scale_mul_env = std::getenv("GGML_OPENCL_MLDRIFT_Q_SCALE_MUL")) {
                    q_scale_mul = strtof(q_scale_mul_env, nullptr);
                    if (!std::isfinite(q_scale_mul) || q_scale_mul <= 0.0f) {
                        q_scale_mul = 1.0f;
                    }
                }
                const float q_scale = force_no_scale ? 1.0f : (scale * q_scale_mul);
                const cl_ulong zero_offset = 0;
                const bool mldrift_phase_timing = std::getenv("GGML_OPENCL_MLDRIFT_PHASE_TIMING") != nullptr;
                const auto now_ms = []() -> double {
                    using clock = std::chrono::steady_clock;
                    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
                };
                if (mldrift_phase_timing) {
                    // Isolate per-call phase timing from previous queued work.
                    CL_CHECK(clFinish(backend_ctx->queue));
                }
                const double t_phase_start = mldrift_phase_timing ? now_ms() : 0.0;

                // Fast path: when scale is exactly 1, avoid the extra scale kernel and use a raw buffer copy.
                if (q_scale == 1.0f) {
                    CL_CHECK(clEnqueueCopyBuffer(
                        backend_ctx->queue,
                        extra_q->data_device,
                        st.buffers[q_dst_id],
                        offset_q,
                        0,
                        q_bytes,
                        0,
                        nullptr,
                        nullptr));
                } else {
                    cl_kernel copy_scale = backend_ctx->kernel_f32_copy_scale_linear;
                    CL_CHECK(clSetKernelArg(copy_scale, 0, sizeof(cl_mem), &st.buffers[q_dst_id]));
                    CL_CHECK(clSetKernelArg(copy_scale, 1, sizeof(cl_ulong), &zero_offset));
                    CL_CHECK(clSetKernelArg(copy_scale, 2, sizeof(cl_mem), &extra_q->data_device));
                    CL_CHECK(clSetKernelArg(copy_scale, 3, sizeof(cl_ulong), &offset_q));
                    CL_CHECK(clSetKernelArg(copy_scale, 4, sizeof(float), &q_scale));
                    CL_CHECK(clSetKernelArg(copy_scale, 5, sizeof(int), &q_elems));
                    const size_t lws = 256;
                    const size_t gws = (size_t) CEIL_DIV(q_elems, lws) * lws;
                    size_t global_work_size[] = { gws, 1, 1 };
                    size_t local_work_size[] = { lws, 1, 1 };
                    backend_ctx->enqueue_ndrange_kernel(copy_scale, 1, global_work_size, local_work_size, dst);
                }
                double t_after_q = 0.0;
                if (mldrift_phase_timing) {
                    CL_CHECK(clFinish(backend_ctx->queue));
                    t_after_q = now_ms();
                }

                {
                    cl_kernel convert = backend_ctx->kernel_f16_to_f32_linear;

                    CL_CHECK(clSetKernelArg(convert, 0, sizeof(cl_mem), &st.buffers[k_dst_id]));
                    CL_CHECK(clSetKernelArg(convert, 1, sizeof(cl_ulong), &zero_offset));
                    CL_CHECK(clSetKernelArg(convert, 2, sizeof(cl_mem), &extra_k->data_device));
                    CL_CHECK(clSetKernelArg(convert, 3, sizeof(cl_ulong), &offset_k));
                    CL_CHECK(clSetKernelArg(convert, 4, sizeof(int), &kv_elems));
                    const size_t lws = 256;
                    const size_t gws = (size_t) CEIL_DIV(kv_elems, lws) * lws;
                    size_t global_work_size[] = { gws, 1, 1 };
                    size_t local_work_size[] = { lws, 1, 1 };
                    backend_ctx->enqueue_ndrange_kernel(convert, 1, global_work_size, local_work_size, dst);

                    CL_CHECK(clSetKernelArg(convert, 0, sizeof(cl_mem), &st.buffers[v_dst_id]));
                    CL_CHECK(clSetKernelArg(convert, 1, sizeof(cl_ulong), &zero_offset));
                    CL_CHECK(clSetKernelArg(convert, 2, sizeof(cl_mem), &extra_v->data_device));
                    CL_CHECK(clSetKernelArg(convert, 3, sizeof(cl_ulong), &offset_v));
                    CL_CHECK(clSetKernelArg(convert, 4, sizeof(int), &kv_elems));
                    backend_ctx->enqueue_ndrange_kernel(convert, 1, global_work_size, local_work_size, dst);
                }
                double t_after_prep = 0.0;
                if (mldrift_phase_timing) {
                    CL_CHECK(clFinish(backend_ctx->queue));
                    t_after_prep = now_ms();
                }

                static const int alt_run_order[] = {7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 10};
                const bool legacy_run_order = std::getenv("GGML_OPENCL_MLDRIFT_LEGACY_RUN_ORDER") != nullptr;
                const int * run_order = legacy_run_order ? mh::kRunOrder : alt_run_order;
                const int run_order_n = legacy_run_order ? (int) (sizeof(mh::kRunOrder) / sizeof(mh::kRunOrder[0]))
                                                         : (int) (sizeof(alt_run_order) / sizeof(alt_run_order[0]));
                size_t kernel_gws[mh::kNumKernels][3];
                mldrift_attn_patch_gws_h24(kernel_gws, n_q);

                for (int rid = 0; rid < run_order_n; ++rid) {
                    const int kid = run_order[rid];
                    size_t global_work_size[] = {
                        kernel_gws[kid][0],
                        kernel_gws[kid][1],
                        kernel_gws[kid][2],
                    };
                    size_t local_work_size[] = {
                        mh::kKernelLws[kid][0],
                        mh::kKernelLws[kid][1],
                        mh::kKernelLws[kid][2],
                    };
                    backend_ctx->enqueue_ndrange_kernel(st.kernels[kid], 3, global_work_size, local_work_size, dst);
                }
                double t_after_main = 0.0;
                if (mldrift_phase_timing) {
                    CL_CHECK(clFinish(backend_ctx->queue));
                    t_after_main = now_ms();
                }

                int out_src_id = mh::kOutputBufId;
                if (const char * out_env = std::getenv("GGML_OPENCL_MLDRIFT_OUTPUT_BUF")) {
                    const int v = out_env[0] - '0';
                    if (v >= 0 && v < mh::kNumBuffers) {
                        out_src_id = v;
                    }
                }

                const bool disable_reorder = std::getenv("GGML_OPENCL_MLDRIFT_NO_REORDER_OUT") != nullptr;
                const bool apply_reorder = (out_src_id == mh::kOutputBufId) && !disable_reorder;
                if (apply_reorder) {
                    cl_kernel reorder = backend_ctx->kernel_f32_reorder_hqd_to_qhd;
                    CL_CHECK(clSetKernelArg(reorder, 0, sizeof(cl_mem), &extra_o->data_device));
                    CL_CHECK(clSetKernelArg(reorder, 1, sizeof(cl_ulong), &offset_o));
                    CL_CHECK(clSetKernelArg(reorder, 2, sizeof(cl_mem), &st.buffers[out_src_id]));
                    CL_CHECK(clSetKernelArg(reorder, 3, sizeof(cl_ulong), &zero_offset));
                    CL_CHECK(clSetKernelArg(reorder, 4, sizeof(int), &n_head));
                    CL_CHECK(clSetKernelArg(reorder, 5, sizeof(int), &n_q));
                    CL_CHECK(clSetKernelArg(reorder, 6, sizeof(int), &d_head_v));
                    const int out_elems = (int) (out_bytes / sizeof(float));
                    const size_t lws = 256;
                    const size_t gws = (size_t) CEIL_DIV(out_elems, lws) * lws;
                    size_t global_work_size[] = { gws, 1, 1 };
                    size_t local_work_size[] = { lws, 1, 1 };
                    backend_ctx->enqueue_ndrange_kernel(reorder, 1, global_work_size, local_work_size, dst);
                } else {
                    CL_CHECK(clEnqueueCopyBuffer(
                        backend_ctx->queue,
                        st.buffers[out_src_id],
                        extra_o->data_device,
                        0,
                        offset_o,
                        out_bytes,
                        0,
                        nullptr,
                        nullptr));
                }
                if (mldrift_phase_timing) {
                    CL_CHECK(clFinish(backend_ctx->queue));
                    const double t_after_out = now_ms();
                    static std::mutex timing_mu;
                    static int call_counter = 0;
                    static double q_ms_sum = 0.0;
                    static double kv_ms_sum = 0.0;
                    static double prep_ms_sum = 0.0;
                    static double main_ms_sum = 0.0;
                    static double out_ms_sum = 0.0;
                    const double q_ms = t_after_q - t_phase_start;
                    const double kv_ms = t_after_prep - t_after_q;
                    const double prep_ms = t_after_prep - t_phase_start;
                    const double main_ms = t_after_main - t_after_prep;
                    const double out_ms = t_after_out - t_after_main;
                    int call_id = 0;
                    double q_avg = 0.0;
                    double kv_avg = 0.0;
                    double prep_avg = 0.0;
                    double main_avg = 0.0;
                    double out_avg = 0.0;
                    {
                        std::lock_guard<std::mutex> lock(timing_mu);
                        ++call_counter;
                        call_id = call_counter;
                        q_ms_sum += q_ms;
                        kv_ms_sum += kv_ms;
                        prep_ms_sum += prep_ms;
                        main_ms_sum += main_ms;
                        out_ms_sum += out_ms;
                        q_avg = q_ms_sum / call_id;
                        kv_avg = kv_ms_sum / call_id;
                        prep_avg = prep_ms_sum / call_id;
                        main_avg = main_ms_sum / call_id;
                        out_avg = out_ms_sum / call_id;
                    }
                    if (call_id % 25 == 0) {
                        GGML_LOG_INFO("ggml_opencl: mldrift phase timing calls=%d q=%.3fms kv=%.3fms prep=%.3fms main=%.3fms out=%.3fms total=%.3fms\n",
                                      call_id, q_avg, kv_avg, prep_avg, main_avg, out_avg, prep_avg + main_avg + out_avg);
                    }
                }
                if (flash_dump_this_call) {
                    CL_CHECK(clFinish(backend_ctx->queue));
                    ggml_cl_dump_buffer_region(backend_ctx, extra_o->data_device, offset_o, out_bytes,
                                               std::string(flash_dump_dir) + "/flash_call" + std::to_string(flash_dump_call) + "_out_mldrift_f32.bin");
                }
                return;
            }
            } else if (mldrift_debug_all) {
                GGML_LOG_WARN("ggml_opencl: mldrift(h24) init failed, fallback to native flash\n");
            }
        } else if (mldrift_shape_match_h30) {
            namespace mh = ggml_opencl_mldrift_h30_m4224;
            if (mldrift_attn_init_h30(backend_ctx)) {
                auto & st = backend_ctx->mldrift_attn_h30;
                const size_t q_bytes = ggml_nbytes(q);
                const size_t k_bytes = ggml_nbytes(k);
                const size_t v_bytes = ggml_nbytes(v);
                const size_t out_bytes = ggml_nbytes(dst);
                const int kv_n_src = n_kv;
                const int kv_n_dst = (kv_n_src == mh::kShapeN + 128) ? mh::kShapeN : kv_n_src;
                const bool need_kv_crop = kv_n_src != kv_n_dst;
                if (!mldrift_attn_patch_shape_h30(backend_ctx, n_q, kv_n_dst)) {
                    if (mldrift_debug_all) {
                        GGML_LOG_WARN("ggml_opencl: mldrift(h30) dynamic shape patch failed for M=%d N=%d, fallback to native flash\n", n_q, kv_n_dst);
                    }
                    goto mldrift_fallback;
                }
                if (mask != nullptr && mldrift_debug_all) {
                    GGML_LOG_INFO("ggml_opencl: mldrift(h30) using kv-crop fast path with mask present (n_q=%d n_kv=%d)\n", n_q, n_kv);
                }
                const size_t kv_dst_bytes = (size_t) d_head_q * (size_t) n_head * (size_t) kv_n_dst * sizeof(float);
                const size_t k_work_bytes = need_kv_crop ? kv_dst_bytes : (k_bytes * 2);
                const size_t v_work_bytes = need_kv_crop ? kv_dst_bytes : (v_bytes * 2);
                if (q_bytes <= mh::kBufferSizes[mh::kInputBufIds[0]] &&
                    out_bytes <= mh::kBufferSizes[mh::kOutputBufId] &&
                    k_work_bytes <= mh::kBufferSizes[mh::kInputBufIds[1]] &&
                    v_work_bytes <= mh::kBufferSizes[mh::kInputBufIds[2]]) {
                    int q_dst_id = mh::kInputBufIds[0];
                    int k_dst_id = mh::kInputBufIds[1];
                    int v_dst_id = mh::kInputBufIds[2];
                    if (const char * map = std::getenv("GGML_OPENCL_MLDRIFT_INPUT_MAP")) {
                        // Optional debug override, e.g. "687" means q->6, k->8, v->7.
                        if (std::strlen(map) >= 3) {
                            const int a = map[0] - '0';
                            const int b = map[1] - '0';
                            const int c = map[2] - '0';
                            const bool in_range = (a >= 0 && a < mh::kNumBuffers) &&
                                                  (b >= 0 && b < mh::kNumBuffers) &&
                                                  (c >= 0 && c < mh::kNumBuffers);
                            if (in_range && a != b && a != c && b != c) {
                                q_dst_id = a;
                                k_dst_id = b;
                                v_dst_id = c;
                            }
                        }
                    }
                    const bool force_no_scale = std::getenv("GGML_OPENCL_MLDRIFT_NO_SCALE") != nullptr;
                    float q_scale_mul = 1.0f;
                    if (const char * q_scale_mul_env = std::getenv("GGML_OPENCL_MLDRIFT_Q_SCALE_MUL")) {
                        q_scale_mul = strtof(q_scale_mul_env, nullptr);
                        if (!std::isfinite(q_scale_mul) || q_scale_mul <= 0.0f) {
                            q_scale_mul = 1.0f;
                        }
                    }
                    const float q_scale = force_no_scale ? 1.0f : (scale * q_scale_mul);
                    if (mldrift_debug_all && flash_call_id < 16) {
                        GGML_LOG_INFO("ggml_opencl: mldrift(h30) scales call=%d scale=%g q_scale_mul=%g force_no_scale=%d q_scale=%g\n",
                                      flash_call_id, (double) scale, (double) q_scale_mul, force_no_scale ? 1 : 0, (double) q_scale);
                    }
                    const cl_ulong zero_offset = 0;
                    const int q_elems = (int) (q_bytes / sizeof(float));
                    const int kv_elems_src = (int) (k_bytes / sizeof(uint16_t));
                    const int kv_elems_dst = d_head_q * n_head * kv_n_dst;
                    const bool kv_crop_tail = std::getenv("GGML_OPENCL_MLDRIFT_KV_CROP_TAIL") != nullptr;
                    const int kv_src_start = (need_kv_crop && kv_crop_tail) ? (kv_n_src - kv_n_dst) : 0;
                    const bool has_kv_keep_head = std::getenv("GGML_OPENCL_MLDRIFT_KV_KEEP_HEAD") != nullptr;
                    const bool has_kv_keep_tail = std::getenv("GGML_OPENCL_MLDRIFT_KV_KEEP_TAIL") != nullptr;
                    int kv_keep_head = kv_n_dst;
                    if (has_kv_keep_tail) {
                        int kv_keep_tail = std::atoi(std::getenv("GGML_OPENCL_MLDRIFT_KV_KEEP_TAIL"));
                        if (kv_keep_tail < 0) {
                            kv_keep_tail = 0;
                        } else if (kv_keep_tail > kv_n_dst) {
                            kv_keep_tail = kv_n_dst;
                        }
                        kv_keep_head = kv_n_dst - kv_keep_tail;
                    }
                    if (has_kv_keep_head) {
                        kv_keep_head = std::atoi(std::getenv("GGML_OPENCL_MLDRIFT_KV_KEEP_HEAD"));
                        if (kv_keep_head < 0) {
                            kv_keep_head = 0;
                        } else if (kv_keep_head > kv_n_dst) {
                            kv_keep_head = kv_n_dst;
                        }
                    }
                    const bool use_kv_head_tail_crop = need_kv_crop && (has_kv_keep_head || has_kv_keep_tail);
                    if (mldrift_debug_all && need_kv_crop) {
                        GGML_LOG_INFO("ggml_opencl: mldrift(h30) kv crop mode src=%d dst=%d src_start=%d keep_head=%d keep_tail=%d\n",
                                      kv_n_src, kv_n_dst, kv_src_start, kv_keep_head, kv_n_dst - kv_keep_head);
                    }

                    const size_t q_full_bytes = mh::kBufferSizes[q_dst_id];
                    // The intercepted h30 op schedule is numerically unstable in ggml migration.
                    // Default to simple order (upload q/k/v first, then run kernels 0..10).
                    const bool force_h30_simple_order = std::getenv("GGML_OPENCL_MLDRIFT_H30_SIMPLE_ORDER") != nullptr;
                    const bool force_h30_op_schedule  = std::getenv("GGML_OPENCL_MLDRIFT_H30_OP_SCHEDULE") != nullptr;
                    const bool force_h30_io_first     = std::getenv("GGML_OPENCL_MLDRIFT_H30_IO_FIRST") != nullptr;
                    const bool use_h30_op_schedule    = force_h30_op_schedule && !force_h30_simple_order;

                    auto enqueue_q_upload = [&]() {
                        if (q_scale == 1.0f) {
                            CL_CHECK(clEnqueueCopyBuffer(
                                backend_ctx->queue,
                                extra_q->data_device,
                                st.buffers[q_dst_id],
                                offset_q,
                                0,
                                q_bytes,
                                0,
                                nullptr,
                                nullptr));
                        } else {
                            cl_kernel copy_scale = backend_ctx->kernel_f32_copy_scale_linear;
                            CL_CHECK(clSetKernelArg(copy_scale, 0, sizeof(cl_mem), &st.buffers[q_dst_id]));
                            CL_CHECK(clSetKernelArg(copy_scale, 1, sizeof(cl_ulong), &zero_offset));
                            CL_CHECK(clSetKernelArg(copy_scale, 2, sizeof(cl_mem), &extra_q->data_device));
                            CL_CHECK(clSetKernelArg(copy_scale, 3, sizeof(cl_ulong), &offset_q));
                            CL_CHECK(clSetKernelArg(copy_scale, 4, sizeof(float), &q_scale));
                            CL_CHECK(clSetKernelArg(copy_scale, 5, sizeof(int), &q_elems));
                            const size_t lws = 256;
                            const size_t gws = (size_t) CEIL_DIV(q_elems, lws) * lws;
                            size_t global_work_size[] = { gws, 1, 1 };
                            size_t local_work_size[] = { lws, 1, 1 };
                            backend_ctx->enqueue_ndrange_kernel(copy_scale, 1, global_work_size, local_work_size, dst);
                        }
                        // Dynamic M may leave tail elements unwritten in static replay buffers.
                        if (q_bytes < q_full_bytes) {
                            const float zero_f32 = 0.0f;
                            CL_CHECK(clEnqueueFillBuffer(
                                backend_ctx->queue,
                                st.buffers[q_dst_id],
                                &zero_f32,
                                sizeof(zero_f32),
                                q_bytes,
                                q_full_bytes - q_bytes,
                                0,
                                nullptr,
                                nullptr));
                        }
                    };

                    auto enqueue_k_upload = [&]() {
                        if (!need_kv_crop) {
                            cl_kernel convert = backend_ctx->kernel_f16_to_f32_linear;
                            CL_CHECK(clSetKernelArg(convert, 0, sizeof(cl_mem), &st.buffers[k_dst_id]));
                            CL_CHECK(clSetKernelArg(convert, 1, sizeof(cl_ulong), &zero_offset));
                            CL_CHECK(clSetKernelArg(convert, 2, sizeof(cl_mem), &extra_k->data_device));
                            CL_CHECK(clSetKernelArg(convert, 3, sizeof(cl_ulong), &offset_k));
                            CL_CHECK(clSetKernelArg(convert, 4, sizeof(int), &kv_elems_src));
                            const size_t lws = 256;
                            const size_t gws = (size_t) CEIL_DIV(kv_elems_src, lws) * lws;
                            size_t global_work_size[] = { gws, 1, 1 };
                            size_t local_work_size[] = { lws, 1, 1 };
                            backend_ctx->enqueue_ndrange_kernel(convert, 1, global_work_size, local_work_size, dst);
                        } else {
                            if (mldrift_debug_all) {
                                GGML_LOG_INFO("ggml_opencl: mldrift(h30) cropping kv from %d to %d tokens\n", kv_n_src, kv_n_dst);
                            }
                            cl_kernel convert_crop = use_kv_head_tail_crop
                                ? backend_ctx->kernel_f16_to_f32_nhd_keep_head_tail
                                : backend_ctx->kernel_f16_to_f32_nhd_crop;
                            CL_CHECK(clSetKernelArg(convert_crop, 0, sizeof(cl_mem), &st.buffers[k_dst_id]));
                            CL_CHECK(clSetKernelArg(convert_crop, 1, sizeof(cl_ulong), &zero_offset));
                            CL_CHECK(clSetKernelArg(convert_crop, 2, sizeof(cl_mem), &extra_k->data_device));
                            CL_CHECK(clSetKernelArg(convert_crop, 3, sizeof(cl_ulong), &offset_k));
                            CL_CHECK(clSetKernelArg(convert_crop, 4, sizeof(int), &d_head_q));
                            CL_CHECK(clSetKernelArg(convert_crop, 5, sizeof(int), &n_head));
                            CL_CHECK(clSetKernelArg(convert_crop, 6, sizeof(int), &kv_n_src));
                            CL_CHECK(clSetKernelArg(convert_crop, 7, sizeof(int), &kv_n_dst));
                            CL_CHECK(clSetKernelArg(convert_crop, 8, sizeof(int), use_kv_head_tail_crop ? &kv_keep_head : &kv_src_start));
                            const size_t lws = 256;
                            const size_t gws = (size_t) CEIL_DIV(kv_elems_dst, lws) * lws;
                            size_t global_work_size[] = { gws, 1, 1 };
                            size_t local_work_size[] = { lws, 1, 1 };
                            backend_ctx->enqueue_ndrange_kernel(convert_crop, 1, global_work_size, local_work_size, dst);
                        }
                    };

                    auto enqueue_v_upload = [&]() {
                        if (!need_kv_crop) {
                            cl_kernel convert = backend_ctx->kernel_f16_to_f32_linear;
                            CL_CHECK(clSetKernelArg(convert, 0, sizeof(cl_mem), &st.buffers[v_dst_id]));
                            CL_CHECK(clSetKernelArg(convert, 1, sizeof(cl_ulong), &zero_offset));
                            CL_CHECK(clSetKernelArg(convert, 2, sizeof(cl_mem), &extra_v->data_device));
                            CL_CHECK(clSetKernelArg(convert, 3, sizeof(cl_ulong), &offset_v));
                            CL_CHECK(clSetKernelArg(convert, 4, sizeof(int), &kv_elems_src));
                            const size_t lws = 256;
                            const size_t gws = (size_t) CEIL_DIV(kv_elems_src, lws) * lws;
                            size_t global_work_size[] = { gws, 1, 1 };
                            size_t local_work_size[] = { lws, 1, 1 };
                            backend_ctx->enqueue_ndrange_kernel(convert, 1, global_work_size, local_work_size, dst);
                        } else {
                            cl_kernel convert_crop = use_kv_head_tail_crop
                                ? backend_ctx->kernel_f16_to_f32_nhd_keep_head_tail
                                : backend_ctx->kernel_f16_to_f32_nhd_crop;
                            CL_CHECK(clSetKernelArg(convert_crop, 0, sizeof(cl_mem), &st.buffers[v_dst_id]));
                            CL_CHECK(clSetKernelArg(convert_crop, 1, sizeof(cl_ulong), &zero_offset));
                            CL_CHECK(clSetKernelArg(convert_crop, 2, sizeof(cl_mem), &extra_v->data_device));
                            CL_CHECK(clSetKernelArg(convert_crop, 3, sizeof(cl_ulong), &offset_v));
                            CL_CHECK(clSetKernelArg(convert_crop, 4, sizeof(int), &d_head_q));
                            CL_CHECK(clSetKernelArg(convert_crop, 5, sizeof(int), &n_head));
                            CL_CHECK(clSetKernelArg(convert_crop, 6, sizeof(int), &kv_n_src));
                            CL_CHECK(clSetKernelArg(convert_crop, 7, sizeof(int), &kv_n_dst));
                            CL_CHECK(clSetKernelArg(convert_crop, 8, sizeof(int), use_kv_head_tail_crop ? &kv_keep_head : &kv_src_start));
                            const size_t lws = 256;
                            const size_t gws = (size_t) CEIL_DIV(kv_elems_dst, lws) * lws;
                            size_t global_work_size[] = { gws, 1, 1 };
                            size_t local_work_size[] = { lws, 1, 1 };
                            backend_ctx->enqueue_ndrange_kernel(convert_crop, 1, global_work_size, local_work_size, dst);
                        }
                    };

                    size_t kernel_gws[mh::kNumKernels][3];
                    mldrift_attn_patch_gws_h30(kernel_gws, n_q, kv_n_dst);
                    auto enqueue_kernel_h30 = [&](int kid) {
                        size_t global_work_size[] = {
                            kernel_gws[kid][0],
                            kernel_gws[kid][1],
                            kernel_gws[kid][2],
                        };
                        size_t local_work_size[] = {
                            mh::kKernelLws[kid][0],
                            mh::kKernelLws[kid][1],
                            mh::kKernelLws[kid][2],
                        };
                        backend_ctx->enqueue_ndrange_kernel(st.kernels[kid], 3, global_work_size, local_work_size, dst);
                    };

                    if (force_h30_io_first) {
                        // Ensure image0/1/2 are refreshed from q/k/v before kernels 0..6 consume them.
                        enqueue_q_upload();
                        enqueue_kernel_h30(7);
                        enqueue_k_upload();
                        enqueue_kernel_h30(8);
                        enqueue_v_upload();
                        enqueue_kernel_h30(9);
                        for (int kid = 0; kid <= 6; ++kid) {
                            enqueue_kernel_h30(kid);
                        }
                        enqueue_kernel_h30(10);
                    } else if (use_h30_op_schedule) {
                        // Match replay op order for h30 kernels: [0..6], q-upload, 7, k-upload, 8, v-upload, 9, 10.
                        for (int kid = 0; kid <= 6; ++kid) {
                            enqueue_kernel_h30(kid);
                        }
                        enqueue_q_upload();
                        enqueue_kernel_h30(7);
                        enqueue_k_upload();
                        enqueue_kernel_h30(8);
                        enqueue_v_upload();
                        enqueue_kernel_h30(9);
                        enqueue_kernel_h30(10);
                    } else {
                        enqueue_q_upload();
                        enqueue_k_upload();
                        enqueue_v_upload();
                        for (int kid = 0; kid < mh::kNumKernels; ++kid) {
                            enqueue_kernel_h30(kid);
                        }
                    }

                    const bool disable_reorder = std::getenv("GGML_OPENCL_MLDRIFT_NO_REORDER_OUT") != nullptr;
                    if (!disable_reorder) {
                        cl_kernel reorder = backend_ctx->kernel_f32_reorder_hqd_to_qhd;
                        CL_CHECK(clSetKernelArg(reorder, 0, sizeof(cl_mem), &extra_o->data_device));
                        CL_CHECK(clSetKernelArg(reorder, 1, sizeof(cl_ulong), &offset_o));
                        CL_CHECK(clSetKernelArg(reorder, 2, sizeof(cl_mem), &st.buffers[mh::kOutputBufId]));
                        CL_CHECK(clSetKernelArg(reorder, 3, sizeof(cl_ulong), &zero_offset));
                        CL_CHECK(clSetKernelArg(reorder, 4, sizeof(int), &n_head));
                        CL_CHECK(clSetKernelArg(reorder, 5, sizeof(int), &n_q));
                        CL_CHECK(clSetKernelArg(reorder, 6, sizeof(int), &d_head_v));
                        const int out_elems = (int) (out_bytes / sizeof(float));
                        const size_t lws = 256;
                        const size_t gws = (size_t) CEIL_DIV(out_elems, lws) * lws;
                        size_t global_work_size[] = { gws, 1, 1 };
                        size_t local_work_size[] = { lws, 1, 1 };
                        backend_ctx->enqueue_ndrange_kernel(reorder, 1, global_work_size, local_work_size, dst);
                    } else {
                        CL_CHECK(clEnqueueCopyBuffer(
                            backend_ctx->queue,
                            st.buffers[mh::kOutputBufId],
                            extra_o->data_device,
                            0,
                            offset_o,
                            out_bytes,
                            0,
                            nullptr,
                            nullptr));
                    }

                    if (flash_dump_this_call) {
                        CL_CHECK(clFinish(backend_ctx->queue));
                        ggml_cl_dump_buffer_region(backend_ctx, extra_o->data_device, offset_o, out_bytes,
                                                   std::string(flash_dump_dir) + "/flash_call" + std::to_string(flash_dump_call) + "_out_mldrift_h30_f32.bin");
                    }
                    if (std::getenv("GGML_OPENCL_MLDRIFT_FORCE_FINISH") != nullptr) {
                        CL_CHECK(clFinish(backend_ctx->queue));
                    }
                    return;
                } else if (mldrift_debug_all) {
                    GGML_LOG_WARN("ggml_opencl: mldrift(h30) buffer guard fail q=%zu out=%zu k_work=%zu v_work=%zu limits(q=%zu out=%zu k=%zu v=%zu)\n",
                                  q_bytes, out_bytes, k_work_bytes, v_work_bytes,
                                  mh::kBufferSizes[mh::kInputBufIds[0]],
                                  mh::kBufferSizes[mh::kOutputBufId],
                                  mh::kBufferSizes[mh::kInputBufIds[1]],
                                  mh::kBufferSizes[mh::kInputBufIds[2]]);
                }
            } else if (mldrift_debug_all) {
                GGML_LOG_WARN("ggml_opencl: mldrift(h30) init failed, fallback to native flash\n");
            }
        }
    }
mldrift_fallback:

    // Default to non-causal when no explicit mask is provided.
    // Legacy auto-causal heuristic can be re-enabled with GGML_OPENCL_FLASH_LEGACY_AUTO_CAUSAL=1.
    const bool disable_flash_causal = std::getenv("GGML_OPENCL_FLASH_DISABLE_CAUSAL") != nullptr;
    const bool force_flash_causal = std::getenv("GGML_OPENCL_FLASH_FORCE_CAUSAL") != nullptr;
    const bool legacy_auto_causal = std::getenv("GGML_OPENCL_FLASH_LEGACY_AUTO_CAUSAL") != nullptr;
    const bool auto_causal_guess = (mask == NULL && n_q > 1 && n_q == n_kv);
    const int is_causal = force_flash_causal ? 1 : ((!disable_flash_causal && legacy_auto_causal && auto_causal_guess) ? 1 : 0);

    if (std::getenv("GGML_OPENCL_FLASH_DEBUG") != nullptr) {
        static std::atomic<int> flash_debug_calls {0};
        const int call_id = flash_debug_calls.fetch_add(1, std::memory_order_relaxed);
        if (call_id < 8) {
            GGML_LOG_INFO(
                "ggml_opencl: flash dbg call=%d n_q=%d n_kv=%d d_q=%d d_v=%d n_head=%d n_head_kv=%d batch=%d q_type=%s k_type=%s v_type=%s scale=%g max_bias=%g logit_softcap=%g is_causal=%d mask=%d sinks=%d q_contig=%d k_contig=%d v_contig=%d",
                call_id,
                n_q, n_kv, d_head_q, d_head_v, n_head, n_head_kv, n_batch,
                ggml_type_name(q->type), ggml_type_name(k->type), ggml_type_name(v->type),
                (double) scale, (double) max_bias, (double) logit_softcap, is_causal,
                mask != nullptr, sinks != nullptr,
                ggml_is_contiguous(q), ggml_is_contiguous(k), ggml_is_contiguous(v));
        }
    }

    const int n_head_log2_val = n_head > 0 ? 1u << (int)floorf(log2f((float)n_head)) : 0;
    const float n_head_log2_f = n_head_log2_val > 0 ? (float)n_head_log2_val : 1.0f;
    const float m0 = powf(2.0f, -(max_bias) / n_head_log2_f);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2_f);

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra_q->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset_q));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra_k->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offset_k));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &extra_v->data_device));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong), &offset_v));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem),   &extra_o->data_device));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_ulong), &offset_o));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(float),    &scale));
    CL_CHECK(clSetKernelArg(kernel, 9, sizeof(int),      &n_q));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),     &n_kv));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),     &is_causal));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),     &n_head));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &q_nb1)); CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &q_nb2)); CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &q_nb3));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &k_nb1)); CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &k_nb2)); CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &k_nb3));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &v_nb1)); CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong), &v_nb2)); CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_ulong), &v_nb3));
    CL_CHECK(clSetKernelArg(kernel, 22, sizeof(cl_ulong), &o_nb1)); CL_CHECK(clSetKernelArg(kernel, 23, sizeof(cl_ulong), &o_nb2)); CL_CHECK(clSetKernelArg(kernel, 24, sizeof(cl_ulong), &o_nb3));
    CL_CHECK(clSetKernelArg(kernel, 25, sizeof(float),    &max_bias));
    CL_CHECK(clSetKernelArg(kernel, 26, sizeof(float),    &m0));
    CL_CHECK(clSetKernelArg(kernel, 27, sizeof(float),    &m1));
    CL_CHECK(clSetKernelArg(kernel, 28, sizeof(int),      &n_head_log2_val));
    CL_CHECK(clSetKernelArg(kernel, 29, sizeof(float),    &logit_softcap));
    CL_CHECK(clSetKernelArg(kernel, 30, sizeof(int),      &n_head_kv));
    CL_CHECK(clSetKernelArg(kernel, 31, sizeof(cl_mem),   &mask_buffer));
    CL_CHECK(clSetKernelArg(kernel, 32, sizeof(cl_ulong), &offset_mask));
    CL_CHECK(clSetKernelArg(kernel, 33, sizeof(cl_ulong), &mask_nb1));
    CL_CHECK(clSetKernelArg(kernel, 34, sizeof(cl_ulong), &mask_nb2));
    CL_CHECK(clSetKernelArg(kernel, 35, sizeof(cl_ulong), &mask_nb3));
    CL_CHECK(clSetKernelArg(kernel, 36, sizeof(int),      &mask_ne2));
    CL_CHECK(clSetKernelArg(kernel, 37, sizeof(int),      &mask_ne3));
    CL_CHECK(clSetKernelArg(kernel, 38, sizeof(cl_mem),   &sinks_buffer));
    CL_CHECK(clSetKernelArg(kernel, 39, sizeof(cl_ulong), &offset_sinks));

    if (n_q == 1) {
        const size_t wg_size = 64;
        size_t local_work_size[] = { wg_size, 1 };
        size_t global_work_size[] = { wg_size, (size_t)(n_head * n_batch) };
        backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_work_size, local_work_size, dst);
    } else {
        const int block_m = backend_ctx->kernels_flash_attn_bm.at(dk_dv);
        const size_t wg_size = block_m;
        size_t local_work_size[] = { wg_size, 1 };
        size_t global_work_size[] = { (size_t)((n_q + block_m - 1) / block_m) * wg_size, (size_t)(n_head * n_batch) };
        backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_work_size, local_work_size, dst);
    }
    if (flash_dump_this_call) {
        CL_CHECK(clFinish(backend_ctx->queue));
        ggml_cl_dump_buffer_region(backend_ctx, extra_o->data_device, offset_o, ggml_nbytes(dst),
                                   std::string(flash_dump_dir) + "/flash_call" + std::to_string(flash_dump_call) + "_out_native_f32.bin");
    }
}

static void ggml_cl_mul_mat_f16_f32_tiled(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const int M = src0->ne[1];
    const int N = src1->ne[1];
    const int K = src0->ne[0];

    cl_kernel kernel = backend_ctx->kernel_mul_mat_f16_f32_tiled;

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(int),      &M));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(int),      &N));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int),      &K));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_ulong), &offsetd));

    // Tiling parameters. These need to be tuned for optimal performance.
    // They must match the #defines in the kernel mul_mat_f16_f32.cl.
    //
    // OPWM / OPWN: Output tile size per Work-Group. A work-group computes a tile of size OPWM x OPWN.
    // TPWM / TPWN: Threads per Work-group. This is the work-group size.
    // OPTM / OPTN: Output elements per Thread. Each thread computes OPTM x OPTN elements.
    //
    // The following relationships must hold:
    //   OPWM = TPWM * OPTM
    //   OPWN = TPWN * OPTN
    //
    const int OPWM = 64;
    const int OPWN = 64;
    const int TPWM = 16;
    const int TPWN = 8;

    size_t local_work_size[2] = { TPWM, TPWN };
    size_t global_work_size[2] = {
        (size_t) ((M + OPWM - 1) / OPWM) * TPWM,
        (size_t) ((N + OPWN - 1) / OPWN) * TPWN,
    };

    backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_work_size, local_work_size, dst);
}

static void ggml_cl_conv_2d(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const cl_uint Cout = ne03; const cl_uint Cin = ne02; const cl_uint N = ne13;
    const cl_uint KW = ne00; const cl_uint KH = ne01; const cl_uint W = ne10; const cl_uint H = ne11; const cl_uint OW = ne0; const cl_uint OH = ne1;

    const cl_uint s0 = dst->op_params[0]; const cl_uint s1 = dst->op_params[1];
    const cl_uint p0 = dst->op_params[2]; const cl_uint p1 = dst->op_params[3];
    const cl_uint d0 = dst->op_params[4]; const cl_uint d1 = dst->op_params[5];

    const cl_uint cl_nb01 = nb01/ggml_type_size(src0->type); const cl_uint cl_nb02 = nb02/ggml_type_size(src0->type); const cl_uint cl_nb03 = nb03/ggml_type_size(src0->type);
    const cl_uint cl_nb11 = nb11/ggml_type_size(src1->type); const cl_uint cl_nb12 = nb12/ggml_type_size(src1->type); const cl_uint cl_nb13 = nb13/ggml_type_size(src1->type);
    const cl_uint cl_nb1 = nb1/ggml_type_size(dst->type); const cl_uint cl_nb2 = nb2/ggml_type_size(dst->type); const cl_uint cl_nb3 = nb3/ggml_type_size(dst->type);

    const int64_t NPQ = (int64_t)N * OW * OH;

    uint32_t BS_K = 64;
    uint32_t BS_NPQ = 64;
    uint32_t BS_CRS = 16;
    const uint32_t VEC_SIZE = 4;

    uint32_t TS_K = 4;
    uint32_t TS_NPQ = 8;

    auto splitWork = [](uint32_t work_size, uint32_t block_size) { return (block_size + work_size - 1) / block_size; };

    cl_kernel kernel;
    size_t shmem_size;

    const bool is_vae3x3_hotshape =
        KW == 3 && KH == 3 &&
        s0 == 1 && s1 == 1 &&
        p0 == 1 && p1 == 1 &&
        d0 == 1 && d1 == 1 &&
        N == 1 &&
        Cin == Cout &&
        (Cin == 128 || Cin == 256 || Cin == 512);
    const bool enable_conv2d_tuned = std::getenv("GGML_OPENCL_CONV2D_TUNED") != nullptr;

    const bool use_vae3x3_tuned_f16 =
        enable_conv2d_tuned &&
        src0->type == GGML_TYPE_F16 &&
        src1->type == GGML_TYPE_F16 &&
        backend_ctx->kernel_conv_2d_f16_vae3x3 != nullptr &&
        is_vae3x3_hotshape;

    const bool use_vae3x3_tuned_f32 =
        enable_conv2d_tuned &&
        src0->type == GGML_TYPE_F32 &&
        src1->type == GGML_TYPE_F32 &&
        backend_ctx->kernel_conv_2d_f32_vae3x3 != nullptr &&
        is_vae3x3_hotshape;

    const bool use_vae3x3_tuned_f16_f32 =
        enable_conv2d_tuned &&
        src0->type == GGML_TYPE_F16 &&
        src1->type == GGML_TYPE_F32 &&
        backend_ctx->kernel_conv_2d_f16_f32_vae3x3 != nullptr &&
        is_vae3x3_hotshape;

    if (std::getenv("GGML_OPENCL_CONV2D_TUNED_LOG") != nullptr) {
        static bool logged_tuned_f16 = false;
        static bool logged_tuned_f32 = false;
        static bool logged_tuned_f16_f32 = false;
        if (use_vae3x3_tuned_f16 && !logged_tuned_f16) {
            GGML_LOG_INFO("ggml_opencl: conv2d tuned f16 kernel active (Cin=%u, OW=%u, OH=%u)\n", Cin, OW, OH);
            logged_tuned_f16 = true;
        }
        if (use_vae3x3_tuned_f32 && !logged_tuned_f32) {
            GGML_LOG_INFO("ggml_opencl: conv2d tuned f32 kernel active (Cin=%u, OW=%u, OH=%u)\n", Cin, OW, OH);
            logged_tuned_f32 = true;
        }
        if (use_vae3x3_tuned_f16_f32 && !logged_tuned_f16_f32) {
            GGML_LOG_INFO("ggml_opencl: conv2d tuned f16_f32 kernel active (Cin=%u, OW=%u, OH=%u)\n", Cin, OW, OH);
            logged_tuned_f16_f32 = true;
        }
    }

    if (use_vae3x3_tuned_f16 || use_vae3x3_tuned_f32 || use_vae3x3_tuned_f16_f32) {
        BS_CRS = 32;
        BS_NPQ = 128;
        TS_K = 4;
        TS_NPQ = 8;
    }

    const uint32_t WG_K = BS_K / TS_K;
    const uint32_t WG_NPQ = BS_NPQ / TS_NPQ;
    const uint32_t NB_K = splitWork(Cout, BS_K);
    const uint32_t NB_NPQ = splitWork((uint32_t) NPQ, BS_NPQ);

    if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        kernel = use_vae3x3_tuned_f16 ? backend_ctx->kernel_conv_2d_f16_vae3x3 : backend_ctx->kernel_conv_2d_f16;
        shmem_size = (size_t)(BS_K * BS_CRS * sizeof(cl_half) + BS_CRS * (BS_NPQ / VEC_SIZE) * sizeof(cl_half4));
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        kernel = use_vae3x3_tuned_f32 ? backend_ctx->kernel_conv_2d_f32_vae3x3 : backend_ctx->kernel_conv_2d_f32;
        shmem_size = (size_t)(BS_K * BS_CRS * sizeof(cl_float) + BS_CRS * (BS_NPQ / VEC_SIZE) * sizeof(cl_float4));
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
        kernel = use_vae3x3_tuned_f16_f32 ? backend_ctx->kernel_conv_2d_f16_f32_vae3x3 : backend_ctx->kernel_conv_2d_f16_f32;
        shmem_size = (size_t)(BS_K * BS_CRS * sizeof(cl_half) + BS_CRS * (BS_NPQ / VEC_SIZE) * sizeof(cl_float4));
    } else {
        GGML_ASSERT(false && "Unsupported data type combination for conv2d");
    }

    cl_uint idx = 0;
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_mem), &extra0->data_device)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_mem), &extra1->data_device)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_mem), &extrad->data_device)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel, idx++, shmem_size, NULL));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &Cout)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &Cin)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &N));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &KW)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &KH)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &W)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &H));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &OW)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &OH));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &s0)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &s1)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &p0)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &p1));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &d0)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &d1));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &cl_nb01)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &cl_nb02)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &cl_nb03));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &cl_nb11)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &cl_nb12)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &cl_nb13));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &cl_nb1)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &cl_nb2)); CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &cl_nb3));

    size_t global_work_size[] = { (size_t)NB_K * WG_K, (size_t)NB_NPQ * WG_NPQ, 1 };
    size_t local_work_size[] = { (size_t)WG_K, (size_t)WG_NPQ, 1 };

    backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_work_size, local_work_size, dst);
}

static void ggml_cl_mul_mat_kq_kqv_adreno(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    const int  ne00 = src0->ne[0];
    const int  ne01 = src0->ne[1];
    const int  ne02 = src0->ne[2];

    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];

    const int  ne10 = src1->ne[0];
    const int  ne11 = src1->ne[1];
    const int  ne12 = src1->ne[2];

    const cl_ulong nb10 = src1->nb[0];

    const int  ne0 = dst->ne[0];
    const int  ne1 = dst->ne[1];

    GGML_ASSERT(ne00 == ne10);

    cl_kernel kernel;
    cl_context context = backend_ctx->context;

    cl_int              status;
    cl_image_format     img_fmt_1d;
    cl_image_desc       img_desc_1d;
    cl_buffer_region    region;
    cl_mem              A_image1d;
    cl_mem              A_sub_buffer;
    cl_mem              B_sub_buffer;
    cl_mem              D_image1d;
    cl_mem              D_sub_buffer;

    int M = ne01;
    int N = ne1;
    int K = ne00;

    if (nb01 > nb02) {
        // KQ
        kernel = backend_ctx->kernel_mul_mm_f16_f32_kq;
    } else {
        // KQV
        kernel = backend_ctx->kernel_mul_mm_f16_f32_kqv;
    }
    // create sub-buffer for A
    // <--------------------------------------------> //
    extra0 = src0->view_src ? (ggml_tensor_extra_cl *)src0->view_src->extra : (ggml_tensor_extra_cl *)src0->extra;

    region.origin = (extra0->offset);
    if (nb01 > nb02) {
        // KQ
        region.size = nb01 * ne01;
    } else {
        // KQV
        region.size = nb02 * ne02;
    }

    A_sub_buffer = clCreateSubBuffer((extra0->data_device), 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
    CL_CHECK(status);

    // <--------------------------------------------> //

    // create sub-buffer for B
    // <--------------------------------------------> //
    region.origin = (extra1->offset);
    region.size = nb10 * ne10 * ne11 * ne12;
    B_sub_buffer = clCreateSubBuffer((extra1->data_device), 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
    CL_CHECK(status);
    // <--------------------------------------------> //

    img_fmt_1d = {CL_RGBA, CL_FLOAT};
    memset(&img_desc_1d, 0, sizeof(img_desc_1d));
    img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    if (nb01 > nb02) {
        img_desc_1d.image_width = (nb01 * ne01 / 4)/4;
    }
    else {
        img_desc_1d.image_width = (nb02 * ne02 / 4)/4;
    }
    img_desc_1d.buffer = A_sub_buffer;
    A_image1d = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt_1d, &img_desc_1d, NULL, &status);
    CL_CHECK(status);

    // create sub-buffer for output C
    // <--------------------------------------------> //
    region.origin = (extrad->offset);
    region.size = ne0 * ne1 * dst->ne[2] * dst->nb[0]; // size of C in bytes
    D_sub_buffer = clCreateSubBuffer((extrad->data_device), 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
    CL_CHECK(status);
    // <--------------------------------------------> //

    // create image for C output
    // <--------------------------------------------> //
    img_fmt_1d = {CL_R, CL_FLOAT};
    memset(&img_desc_1d, 0, sizeof(img_desc_1d));
    img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    img_desc_1d.image_width = ne0 * ne1 * dst->ne[2] * dst->nb[0] / 4;
    img_desc_1d.buffer = D_sub_buffer;
    D_image1d = clCreateImage(context, CL_MEM_WRITE_ONLY, &img_fmt_1d, &img_desc_1d, NULL, &status);
    CL_CHECK(status);
    // <--------------------------------------------> //

    int offset_src0 = 0;
    int offset_src1 = 0;

    // set kernel args
    // <--------------------------------------------> //
    cl_uint k_arg = 0;
    CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(cl_mem), &A_image1d));
    CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),    &offset_src0));
    CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(cl_mem), &B_sub_buffer));
    CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),    &offset_src1));
    CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(cl_mem), &D_image1d));
    CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),    &extrad->offset));
    CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),    &M));
    CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),    &K));
    CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),    &N));
    CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),    &ne02));
    CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),    &ne12));
    CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),    &nb01));

    size_t global_work_size[3] = {64, static_cast<size_t>(((M+63)/64)), static_cast<size_t>(((N+31)/32)*ne12)};
    size_t local_work_size[3] = {64, 1, 2};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

    // deallocate sub buffers and images
    // <--------------------------------------------> //
    CL_CHECK(clReleaseMemObject(A_image1d));
    CL_CHECK(clReleaseMemObject(D_image1d));
    CL_CHECK(clReleaseMemObject(A_sub_buffer));
    CL_CHECK(clReleaseMemObject(B_sub_buffer));
    CL_CHECK(clReleaseMemObject(D_sub_buffer));
}

static void ggml_cl_mul_mat(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_opencl_log_tensor_layout("mul_mat", dst, src0, src1);
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    const enum ggml_type src0t = src0 ? src0->type : GGML_TYPE_COUNT;
    const enum ggml_type src1t = src1 ? src1->type : GGML_TYPE_COUNT;

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

#ifdef GGML_OPENCL_SOA_Q
    ggml_tensor_extra_cl_q4_0 * extra0_q4_0 = (ggml_tensor_extra_cl_q4_0 *)src0->extra;
    ggml_tensor_extra_cl_mxfp4 * extra0_mxfp4 = (ggml_tensor_extra_cl_mxfp4 *)src0->extra;
    ggml_tensor_extra_cl_q8_0 * extra0_q8_0 = (ggml_tensor_extra_cl_q8_0 *)src0->extra;
    ggml_tensor_extra_cl_q6_K * extra0_q6_K = (ggml_tensor_extra_cl_q6_K *)src0->extra;
#endif

    const int  ne00 = src0 ? src0->ne[0] : 0;
    const int  ne01 = src0 ? src0->ne[1] : 0;
    const int  ne02 = src0 ? src0->ne[2] : 0;
    const int  ne03 = src0 ? src0->ne[3] : 0;

    const cl_ulong nb00 = src0 ? src0->nb[0] : 0;
    const cl_ulong nb01 = src0 ? src0->nb[1] : 0;
    const cl_ulong nb02 = src0 ? src0->nb[2] : 0;
    const cl_ulong nb03 = src0 ? src0->nb[3] : 0;

    const int  ne10 = src1 ? src1->ne[0] : 0;
    const int  ne11 = src1 ? src1->ne[1] : 0;
    const int  ne12 = src1 ? src1->ne[2] : 0;
    const int  ne13 = src1 ? src1->ne[3] : 0;

    const cl_ulong nb10 = src1 ? src1->nb[0] : 0;
    const cl_ulong nb11 = src1 ? src1->nb[1] : 0;
    const cl_ulong nb12 = src1 ? src1->nb[2] : 0;
    const cl_ulong nb13 = src1 ? src1->nb[3] : 0;

    const int  ne0 = dst ? dst->ne[0] : 0;
    const int  ne1 = dst ? dst->ne[1] : 0;

    int r2 = ne12/ne02;
    int r3 = ne13/ne03;

    GGML_ASSERT(ne00 == ne10);

#ifdef GGML_OPENCL_SOA_Q
    if (src0t == GGML_TYPE_Q4_0 && std::getenv("GGML_OPENCL_Q4_0_REF") != nullptr) {
        GGML_ASSERT(extra0_q4_0 && extra0_q4_0->q && extra0_q4_0->d);
        cl_kernel kernel_ref = backend_ctx->kernel_mul_mat_q4_0_f32_ref;

        CL_CHECK(clSetKernelArg(kernel_ref,  0, sizeof(cl_mem),   &extra0_q4_0->q));
        CL_CHECK(clSetKernelArg(kernel_ref,  1, sizeof(cl_mem),   &extra0_q4_0->d));
        CL_CHECK(clSetKernelArg(kernel_ref,  2, sizeof(cl_mem),   &extra1->data_device));
        CL_CHECK(clSetKernelArg(kernel_ref,  3, sizeof(cl_ulong), &offset1));
        CL_CHECK(clSetKernelArg(kernel_ref,  4, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel_ref,  5, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel_ref,  6, sizeof(int),      &ne00));
        CL_CHECK(clSetKernelArg(kernel_ref,  7, sizeof(int),      &ne01));
        CL_CHECK(clSetKernelArg(kernel_ref,  8, sizeof(int),      &ne02));
        CL_CHECK(clSetKernelArg(kernel_ref,  9, sizeof(int),      &ne10));
        CL_CHECK(clSetKernelArg(kernel_ref, 10, sizeof(int),      &ne12));
        CL_CHECK(clSetKernelArg(kernel_ref, 11, sizeof(int),      &ne0));
        CL_CHECK(clSetKernelArg(kernel_ref, 12, sizeof(int),      &ne1));
        CL_CHECK(clSetKernelArg(kernel_ref, 13, sizeof(int),      &r2));
        CL_CHECK(clSetKernelArg(kernel_ref, 14, sizeof(int),      &r3));

        size_t global_work_size[] = {(size_t)ne01, (size_t)ne11, (size_t)ne12 * ne13};
        size_t local_work_size[]  = {1, 1, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel_ref, 3, global_work_size, local_work_size, dst);
        return;
    }
#endif

    int nth0 = 32;
    int nth1 = 1;
    int nrows = 1;
    // The number of values produced by each subgroup
    int ndst = 4;

    cl_kernel kernel;

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    cl_context context = backend_ctx->context;

    if(src0t == GGML_TYPE_F16 && src1t == GGML_TYPE_F32){
        if (ne01 >= 64 && ne1 >= 32 && ne00 >= 16 && (ne12 % ne02) == 0  &&
            // dst is wrapped with image1d_buffer, the size limit applies, also src0
            (ne0 * ne1 * dst->ne[2] * dst->nb[0] / 4 <= backend_ctx->image_max_buffer_size)) {
            // For KQ
            if (ggml_is_permuted(src0) && ggml_is_permuted(src1) &&
                ((nb01 * ne01 / 4)/4 <= backend_ctx->image_max_buffer_size) &&
                nb00 <= nb02 &&
                nb02 <= nb01 &&
                nb01 <= nb03 &&
                nb10 <= nb12 &&
                nb12 <= nb11 &&
                nb11 <= nb13) {
                ggml_cl_mul_mat_kq_kqv_adreno(backend, src0, src1, dst);
                return;
            }
            // For KQV
            if (!ggml_is_contiguous(src0) && ggml_is_contiguous(src1) &&
                ((nb02 * ne02 / 4)/4 <= backend_ctx->image_max_buffer_size)) {
                ggml_cl_mul_mat_kq_kqv_adreno(backend, src0, src1, dst);
                return;
            }
        }
    }

    if (ne01 && ne1 && use_adreno_kernels(backend_ctx, src0) && !ggml_opencl_force_no_reorder(src0)) {

    // init CL objects
    // <--------------------------------------------> //
    cl_int              status;
    cl_image_format     img_fmt_1d;
    cl_image_desc       img_desc_1d;
    cl_buffer_region    region;
    cl_mem              A_image1d = nullptr;
    cl_mem              B_image1d = nullptr;
    cl_mem              B_sub_buffer = nullptr;
    cl_mem              C_d = nullptr;
    // for B transpose
    cl_mem B_d = nullptr;
    cl_mem B_d_input_image = nullptr;
    // <--------------------------------------------> //

    // define matrix dimensions
    // <--------------------------------------------> //
    int M = ne01;
    int N = ne1;
    int K = ne00;
    int padding;
    // <--------------------------------------------> //

    // q4_0 x fp32
    if(src0t == GGML_TYPE_Q4_0 && src1t == GGML_TYPE_F32) {
        // TODO: remove duplicate definitions of image description + format -- move to top

        // create an image for A
        // <--------------------------------------------> //
        if (N == 1) {
            img_fmt_1d = { CL_R, CL_UNSIGNED_INT32};
        } else {
            img_fmt_1d = { CL_R, CL_FLOAT};
        }
        memset(&img_desc_1d, 0, sizeof(img_desc_1d));
        img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc_1d.image_width = M * K / 2 / 4;    // Divide by 4 for char -> float
        img_desc_1d.buffer = extra0_q4_0->q;
        A_image1d = clCreateImage(
            context,
            CL_MEM_READ_ONLY,
            &img_fmt_1d,
            &img_desc_1d,
            NULL,
            &status);
        CL_CHECK(status);
        // <--------------------------------------------> //


        // create a sub_buffer for B
        // <--------------------------------------------> //
        cl_mem   src1_mem    = extra1->data_device;
        cl_ulong src1_offset = offset1;
        if (!ggml_is_contiguous(src1) || src1->view_offs != 0) {
            cl_ulong nb10_cont = nb10;
            cl_ulong nb11_cont = nb11;
            cl_ulong nb12_cont = nb12;
            cl_ulong nb13_cont = nb13;
            backend_ctx->prealloc_src1.allocate(context, ggml_nbytes(src1));
            ggml_cl_copy_to_contiguous(backend, src1, backend_ctx->prealloc_src1.buffer,
                                       nb10_cont, nb11_cont, nb12_cont, nb13_cont);
            src1_mem    = backend_ctx->prealloc_src1.buffer;
            src1_offset = 0;
        }

        region.origin = src1_offset;
        region.size = K * N * sizeof(float);
        B_sub_buffer = clCreateSubBuffer(
            src1_mem,
            0,
            CL_BUFFER_CREATE_TYPE_REGION,
            &region,
            &status);
        CL_CHECK(status);
        // <--------------------------------------------> //

        // transpose activation for Skyler's gemm
        if (N != 1) {
            //how many extra elements beyond multiple of 8
            int extra_elements = N % 8;

            //how much padding to add
            padding = 0;
            if (extra_elements > 0){
                padding = 8 - extra_elements;
            }

            // Optional path: keep activations in FP32 during transpose for selected Q4 GEMM ops.
            const bool force_f32_act = (std::getenv("SD_OCL_Q4_GEMM_F32_ACT") != nullptr) ||
                                       ggml_opencl_force_f32_act_gemm(src0);
            const bool use_f32_act_transpose = force_f32_act;

            // Specify the starting offset (in bytes)
            region.origin = 0;
            if (use_f32_act_transpose) {
                region.size = K * (N + padding) * sizeof(float);
            } else {
                // Specify the size of the sub-buffer (divide by 2 for FP16).
                region.size = K * (N + padding) * sizeof(float) / 2;
            }
            backend_ctx->prealloc_act_trans.allocate(context, region.size);

            B_d = clCreateSubBuffer(
                backend_ctx->prealloc_act_trans.buffer,
                0,
                CL_BUFFER_CREATE_TYPE_REGION,
                &region,
                &status);
            CL_CHECK(status);

            cl_image_format image_format_B_d_input = { CL_RGBA, CL_FLOAT };
            cl_image_desc image_desc_B_d_input = {
                CL_MEM_OBJECT_IMAGE1D_BUFFER,
                static_cast<size_t>(K * N / 4),
                0, 0, 0, 0, 0, 0, 0, { B_sub_buffer }
            };
            B_d_input_image = clCreateImage(
                context,
                0,
                &image_format_B_d_input,
                &image_desc_B_d_input,
                NULL,
                &status);
            CL_CHECK(status);

            cl_image_format image_format_B_d_output = {
                CL_RGBA, static_cast<cl_channel_type>(use_f32_act_transpose ? CL_FLOAT : CL_HALF_FLOAT)
            };
            cl_image_desc image_desc_B_d_output = {
                CL_MEM_OBJECT_IMAGE1D_BUFFER,
                static_cast<size_t>(K * (N + padding)/4),
                0, 0, 0, 0, 0, 0, 0, { B_d }
            };
            B_image1d = clCreateImage(
                context,
                0,
                &image_format_B_d_output,
                &image_desc_B_d_output,
                NULL,
                &status);
            CL_CHECK(status);

            // Number of RGBA rows in src1 image (ceil-div by 4 channels).
            int height_B = (N + 3) / 4;
            int width_B = K/4;
            int padded_height_B = (N + padding)/4;

            if (use_f32_act_transpose) {
                kernel = (padding == 0) ? backend_ctx->kernel_transpose_32
                                        : backend_ctx->kernel_transpose_32_32;
                CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &B_d_input_image));
                CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &B_image1d));
                const int transpose_src_rows = (padding == 0) ? height_B : N;
                CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int),    &transpose_src_rows));
                CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int),    &width_B));
                if (padding != 0) {
                    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), &padded_height_B));
                }
            } else {
                kernel = backend_ctx->kernel_transpose_32_16;
                CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &B_d_input_image));
                CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &B_image1d));
                CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int),    &N));
                CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int),    &width_B));
                CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),    &padded_height_B));
            }

            size_t local_size_t[2] = { 1, 16 };
            //WGS tuning
            if (ne0 == 4096 && ne1 == 128 && ne10 == 4096) {
                local_size_t[0]=4;
                local_size_t[1]=8;
            } else if (ne0 == 11008 && ne1 == 128 && ne10 == 4096) {
                local_size_t[0]=2;
                local_size_t[1]=8;
            } else if(ne0 == 4096 && ne1 == 128 && ne10 == 11008) {
                local_size_t[0]=1;
                local_size_t[1]=8;
            } else if(ne0 == 32000 && ne1 == 128 && ne10 == 4096) {
                local_size_t[0]=2;
                local_size_t[1]=8;
            }

            const int transpose_rows = (use_f32_act_transpose && padding == 0) ? height_B : padded_height_B;
            size_t global_size_t[2] = {
                static_cast<size_t>(width_B),
                static_cast<size_t>(transpose_rows)
            };

            backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_size_t, local_size_t, dst);
        } else {
            // no need to transpose B in other cases
            // create an image for B from sub_buffer
            // <--------------------------------------------> //
            img_fmt_1d = {CL_RGBA, CL_FLOAT};

            memset(&img_desc_1d, 0, sizeof(img_desc_1d));
            img_desc_1d.image_width = K * N / 4;
            img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            img_desc_1d.buffer = B_sub_buffer;
            B_image1d = clCreateImage(
                context,
                CL_MEM_READ_ONLY,
                &img_fmt_1d,
                &img_desc_1d,
                NULL,
                &status);
            CL_CHECK(status);
            // <--------------------------------------------> //
        }

        // choose gemm or gemv kernel
        // <--------------------------------------------> //
        if (N == 1) {
            kernel = backend_ctx->CL_mul_mat_vec_q4_0_f32_1d_4x_flat_general;
            if (M == 4096 && K == 4096) {
                kernel = backend_ctx->CL_mul_mat_vec_q4_0_f32_1d_4x_flat_4096_1_4096;
            } else if (M == 4096 && K == 11008) {
                kernel = backend_ctx->CL_mul_mat_vec_q4_0_f32_1d_4x_flat_4096_1_11008;
            } else if (M == 11008 && K == 4096) {
                kernel = backend_ctx->CL_mul_mat_vec_q4_0_f32_1d_4x_flat_11008_1_4096;
            } else if (M == 32000 && K == 4096) {
                kernel = backend_ctx->CL_mul_mat_vec_q4_0_f32_1d_4x_flat_32000_1_4096;
            }
        } else {
            kernel = backend_ctx->CL_mul_mat_Ab_Bi_8x4;
            if (backend_ctx->CL_mul_mat_Ab_Bi_8x4_fp16chunk != nullptr && ggml_opencl_force_fp16_chunk_acc_gemm(src0)) {
                kernel = backend_ctx->CL_mul_mat_Ab_Bi_8x4_fp16chunk;
            }
            if (backend_ctx->CL_mul_mat_Ab_Bi_8x4_halfscale != nullptr && ggml_opencl_force_half_scale_gemm(src0)) {
                kernel = backend_ctx->CL_mul_mat_Ab_Bi_8x4_halfscale;
            }
            if (backend_ctx->CL_mul_mat_Ab_Bi_8x4_halfclamp != nullptr && ggml_opencl_force_half_acc_clamp_gemm(src0)) {
                kernel = backend_ctx->CL_mul_mat_Ab_Bi_8x4_halfclamp;
            }
            if (backend_ctx->CL_mul_mat_Ab_Bi_8x4_f32acc != nullptr && ggml_opencl_force_f32_acc_gemm(src0)) {
                kernel = backend_ctx->CL_mul_mat_Ab_Bi_8x4_f32acc;
            }
            if (backend_ctx->CL_mul_mat_Ab_Bi_8x4_f32read != nullptr && ggml_opencl_force_f32_read_gemm(src0)) {
                kernel = backend_ctx->CL_mul_mat_Ab_Bi_8x4_f32read;
            }
            if (backend_ctx->CL_mul_mat_Ab_Bi_8x4_f32act != nullptr && ggml_opencl_force_f32_act_gemm(src0)) {
                kernel = backend_ctx->CL_mul_mat_Ab_Bi_8x4_f32act;
            }
            if (backend_ctx->CL_mul_mat_Ab_Bi_8x4_kahan != nullptr && ggml_opencl_force_kahan_gemm(src0)) {
                kernel = backend_ctx->CL_mul_mat_Ab_Bi_8x4_kahan;
            }
        }
        // <--------------------------------------------> //

        // set kernel args
        // <--------------------------------------------> //
        cl_uint k_arg = 0;

        if (N == 1) {
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(cl_mem),   &A_image1d));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(cl_mem),   &extra0_q4_0->d));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(cl_mem),   &B_image1d));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(cl_ulong), &src1_offset));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(cl_ulong), &extrad->offset));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),      &ne02));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),      &ne10));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),      &r3));
        } else {
            region.origin = extrad->offset; // Specify the starting offset (in bytes)
            region.size = M * N * sizeof(float); // Specify the size of the sub-buffer
            C_d = clCreateSubBuffer(extrad->data_device, CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
            CL_CHECK(status);

            int padded_N = ne1 + padding;

            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra0_q4_0->q)); //A_q_dextra0_q4_0->q
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra0_q4_0->d)); //A_s_d
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &B_image1d)); //B_d
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &C_d)); //C_d
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),    &ne01)); //M
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),    &padded_N)); //N with padding
            CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),    &ne00)); //K
            CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int),    &ne1)); //N without padding
        }
        // <--------------------------------------------> //

        // choose workgroup size
        // <--------------------------------------------> //
        size_t global_work_size[3] = {
            64, static_cast<size_t>((M+63)/64), static_cast<size_t>((N+31)/32)};
        size_t local_work_size[3] = {64, 2, 4};

        global_work_size[0] = (size_t)(ceil((float)ne1/8));
        global_work_size[1] = (size_t)(ne01/4);
        global_work_size[2] = (size_t)(1);

        local_work_size[0]  = (size_t)(1); //4x32 for FP32
        local_work_size[1]  = (size_t)(128);
        local_work_size[2]  = (size_t)(1);

        //WGS tuning
        if (ne0 == 4096 && ne1 == 128 && ne10 == 4096) {
            local_work_size[0] = 1;
            local_work_size[1] = 128;
        } else if (ne0 == 11008 && ne1 == 128 && ne10 == 4096) {
            local_work_size[0] = 2;
            local_work_size[1] = 64;
        } else if (ne0 == 4096 && ne1 == 128 && ne10 == 11008) {
            local_work_size[0] = 2;
            local_work_size[1] = 64;
        } else if (ne0 == 32000 && ne1 == 128 && ne10 == 4096) {
            local_work_size[0] = 2;
            local_work_size[1] = 64;
        }

        if (N == 1) {
            size_t wavesize = backend_ctx->adreno_wave_size;
            local_work_size[0] = wavesize; // localsize
            local_work_size[1] = 4; // reduce factor
            local_work_size[2] = 1;

            global_work_size[0] = (((M / 2) + wavesize - 1) / wavesize) * wavesize;
            global_work_size[1] = 4; // reduce factor
            global_work_size[2] = 1;
        }
        // <--------------------------------------------> //

        // enqueue kernel with profiling
        // <--------------------------------------------> //
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
        // <--------------------------------------------> //

        // deallocate sub buffers and images
        // <--------------------------------------------> //
        CL_CHECK(clReleaseMemObject(A_image1d));
        CL_CHECK(clReleaseMemObject(B_sub_buffer));
        CL_CHECK(clReleaseMemObject(B_image1d));

        if (N != 1) {
            CL_CHECK(clReleaseMemObject(B_d));
            CL_CHECK(clReleaseMemObject(B_d_input_image));
            CL_CHECK(clReleaseMemObject(C_d));
        }
        // <--------------------------------------------> //

        return;
    }
    } // if (ne01 && ne1)
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

    // GEMM using local memory
    // Current BK = 16, so ne00 % 16 == 0
    if (src1t == GGML_TYPE_F32 &&
        ne00 % 16 == 0 &&
        ne11 > 1) {
        switch(src0t) {
            case GGML_TYPE_F32: {
                kernel = backend_ctx->kernel_mul_mm_f32_f32_l4_lm;
                nth0 = 128; // calculated as (BM*BN)/(TM*TN)

                int batch_stride_a = ne00*ne01;
                int batch_stride_b = ne10*ne11;
                int batch_stride_d = ne0*ne1;

                cl_mem mem_src0 = extra0->data_device;
                cl_mem mem_src1 = extra1->data_device;

                cl_ulong nb00_cont = nb00;
                cl_ulong nb01_cont = nb01;
                cl_ulong nb02_cont = nb02;
                cl_ulong nb03_cont = nb03;

                cl_ulong nb10_cont = nb10;
                cl_ulong nb11_cont = nb11;
                cl_ulong nb12_cont = nb12;
                cl_ulong nb13_cont = nb13;

                cl_ulong offset0_cont = offset0;
                cl_ulong offset1_cont = offset1;

                if (!ggml_is_contiguous(src0)) {
                    backend_ctx->prealloc_src0.allocate(backend_ctx->context, ggml_nbytes(src0));
                    ggml_cl_copy_to_contiguous(backend, src0, backend_ctx->prealloc_src0.buffer,
                        nb00_cont, nb01_cont, nb02_cont, nb03_cont);
                    mem_src0 = backend_ctx->prealloc_src0.buffer;
                    offset0_cont = 0;
                }

                if (!ggml_is_contiguous(src1)) {
                    backend_ctx->prealloc_src1.allocate(backend_ctx->context, ggml_nbytes(src1));
                    ggml_cl_copy_to_contiguous(backend, src1, backend_ctx->prealloc_src1.buffer,
                        nb10_cont, nb11_cont, nb12_cont, nb13_cont);
                    mem_src1 = backend_ctx->prealloc_src1.buffer;
                    offset1_cont = 0;
                }

                CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &mem_src0));
                CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0_cont));
                CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &mem_src1));
                CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1_cont));
                CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
                CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
                CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
                CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
                CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
                CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne11));
                CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
                CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne10)); // stride_a
                CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne10)); // stride_b
                CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne01)); // stride_d
                CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &batch_stride_a));
                CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &batch_stride_b));
                CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &batch_stride_d));
                CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &r2));
                CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &r3));

                // 64 is block tile size BM and BN - change here when BM and BN in the kernel are changed.
                size_t global_work_size[] = {(size_t)(CEIL_DIV(ne01, 64)*nth0), (size_t)(CEIL_DIV(ne11, 64)), (size_t)ne12*ne13};
                size_t local_work_size[] = {(size_t)nth0, 1, 1};

                backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
                return;
            }
            case GGML_TYPE_F16: {
                kernel = backend_ctx->kernel_mul_mm_f16_f32_l4_lm;
                nth0 = 128; // calculated as (BM*BN)/(TM*TN)

                int batch_stride_a = ne00*ne01;
                int batch_stride_b = ne10*ne11;
                int batch_stride_d = ne0*ne1;

                cl_mem mem_src0 = extra0->data_device;
                cl_mem mem_src1 = extra1->data_device;

                cl_ulong nb00_cont = nb00;
                cl_ulong nb01_cont = nb01;
                cl_ulong nb02_cont = nb02;
                cl_ulong nb03_cont = nb03;

                cl_ulong nb10_cont = nb10;
                cl_ulong nb11_cont = nb11;
                cl_ulong nb12_cont = nb12;
                cl_ulong nb13_cont = nb13;

                cl_ulong offset0_cont = offset0;
                cl_ulong offset1_cont = offset1;

                if (!ggml_is_contiguous(src0)) {
                    backend_ctx->prealloc_src0.allocate(backend_ctx->context, ggml_nbytes(src0));
                    ggml_cl_copy_to_contiguous(backend, src0, backend_ctx->prealloc_src0.buffer,
                        nb00_cont, nb01_cont, nb02_cont, nb03_cont);
                    mem_src0 = backend_ctx->prealloc_src0.buffer;
                    offset0_cont = 0;
                }

                if (!ggml_is_contiguous(src1)) {
                    backend_ctx->prealloc_src1.allocate(backend_ctx->context, ggml_nbytes(src1));
                    ggml_cl_copy_to_contiguous(backend, src1, backend_ctx->prealloc_src1.buffer,
                            nb10_cont, nb11_cont, nb12_cont, nb13_cont);
                    mem_src1 = backend_ctx->prealloc_src1.buffer;
                    offset1_cont = 0;
                }

                CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &mem_src0));
                CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0_cont));
                CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &mem_src1));
                CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1_cont));
                CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
                CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
                CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
                CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
                CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
                CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne11));
                CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
                CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne10)); // stride_a
                CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne10)); // stride_b
                CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne01)); // stride_d
                CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &batch_stride_a));
                CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &batch_stride_b));
                CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &batch_stride_d));
                CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &r2));
                CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &r3));

                // 64 is block tile size BM and BN - change here when BM and BN in the kernel are changed.
                size_t global_work_size[] = {(size_t)(CEIL_DIV(ne01, 64)*nth0), (size_t)(CEIL_DIV(ne11, 64)), (size_t)ne12*ne13};
                size_t local_work_size[] = {(size_t)nth0, 1, 1};

                backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
                return;
            }
            case GGML_TYPE_Q8_0: {
                if (ne11 < 32) {
                    break;
                }
                if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
                    break;
                }

                kernel = backend_ctx->kernel_mul_mm_q8_0_f32_l4_lm;
                nth0 = 128; // calculated as (BM*BN)/(TM*TN)

                int batch_stride_a = ne00*ne01;
                int batch_stride_b = ne10*ne11;
                int batch_stride_d = ne0*ne1;

                CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q8_0->q));
                CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q8_0->d));
                CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
                CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
                CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
                CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
                CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
                CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
                CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
                CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne11));
                CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
                CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne10)); // stride_a
                CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne10)); // stride_b
                CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne01)); // stride_d
                CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &batch_stride_a));
                CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &batch_stride_b));
                CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &batch_stride_d));
                CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &r2));
                CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &r3));

                // 64 is block tile size BM and BN - change here when BM and BN in the kernel are changed.
                size_t global_work_size[] = {(size_t)(CEIL_DIV(ne01, 64)*nth0), (size_t)(CEIL_DIV(ne11, 64)), (size_t)ne12*ne13};
                size_t local_work_size[] = {(size_t)nth0, 1, 1};

                backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
                return;
            }
            default:
                break;
        }
    }

    if (src0t == GGML_TYPE_F16 && src1t == GGML_TYPE_F32 &&
        src0->ne[1] > 32 &&   // M > 32
        src1->ne[1] > 32 &&   // N > 32
        src0->ne[0] > 32 &&   // K > 32
        src0->ne[2] == 1 && src0->ne[3] == 1 &&
        src1->ne[2] == 1 && src1->ne[3] == 1 &&
        ggml_is_contiguous(src0) && ggml_is_contiguous(src1) &&
        backend_ctx->kernel_mul_mat_f16_f32_tiled != NULL) {
        ggml_cl_mul_mat_f16_f32_tiled(backend, src0, src1, dst);
        return;
    }

    if (!ggml_is_transposed(src0) &&
        !ggml_is_transposed(src1) &&
        src1t == GGML_TYPE_F32 &&
        ne00%32 == 0 &&
        ne11 > 2) {
#ifdef GGML_OPENCL_SOA_Q
        // Set up kernel.
        switch(src0t) {
            case GGML_TYPE_Q4_0:
                // This should have been satisfied.
                GGML_ASSERT(ne11 == ne1);
                GGML_ASSERT(ne01 == ne0);

                if (backend_ctx->gpu_family == INTEL) {
                    nth0 = 16;
                    nth1 = 1;

                    kernel = backend_ctx->kernel_mul_mat_q4_0_f32_1d_16x_flat;
                } else if (backend_ctx->gpu_family == ADRENO) {
                    nth0 = 64;
                    nth1 = 1;

                    kernel = backend_ctx->kernel_mul_mat_q4_0_f32_1d_8x_flat;
                } else {
                    GGML_ASSERT(false && "TODO: Unknown GPU");
                }

                CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q4_0->q));
                CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q4_0->d));
                CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
                CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
                CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
                CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
                CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
                CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
                CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
                CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne10));
                CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
                CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne0));
                CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne1));
                CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &r2));
                CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &r3));
                break;
            default:
                break;
        }

        // Launch kernel.
        if (src0t == GGML_TYPE_Q4_0) {
            size_t global_work_size[] = {(size_t)(ne01 + 7)/8*nth0, (size_t)ne11*nth1, (size_t)ne12*ne13};
            size_t local_work_size[] = {(size_t)nth0, (size_t)nth1, 1};

            if (backend_ctx->gpu_family == INTEL) {
                // Set global size for Intel. It uses 16x output values.
                global_work_size[0] = (size_t)(ne01 + 15)/16*nth0;
                global_work_size[1] = (size_t)ne11*nth1;
                global_work_size[2] = (size_t)ne12*ne13;
            }

            backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
            return;
        }
#else // GGML_OPENCL_SOA_Q
        // TODO: add block_q4_0 variant.
#endif // GGML_OPENCL_SOA_Q
    }

    // use custom matrix x vector kernel
    switch (src0t) {
        case GGML_TYPE_F32:
            //GGML_ASSERT(ne02 == ne12);
            GGML_ASSERT(src1t == GGML_TYPE_F32);
            kernel = backend_ctx->kernel_mul_mat_f32_f32;
            nrows = 4;

            if (backend_ctx->gpu_family == INTEL) {
                nth0 = 32;
                nth1 = 1;
            } else if (backend_ctx->gpu_family == ADRENO) {
                nth0 = 64;
                nth1 = 1;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb00));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb03));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne10));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne11));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb10));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb13));
            CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 21, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),      &r3));
            break;
        case GGML_TYPE_F16:
            //GGML_ASSERT(ne02 == ne12);
            if (backend_ctx->gpu_family == INTEL) {
                nth0 = 32;
                nth1 = 1;
            } else if (backend_ctx->gpu_family == ADRENO) {
                nth0 = 64;
                nth1 = 1;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            if (src1t == GGML_TYPE_F32) {
                if (ne11 * ne12 < 4) {
                    kernel = backend_ctx->kernel_mul_mat_f16_f32_1row;
                } else if (ne00 >= 128 && ne01 >= 8 && ne00%4 == 0) {
                    kernel = backend_ctx->kernel_mul_mat_f16_f32_l4;
                    nrows = ne11;
                } else {
                    kernel = backend_ctx->kernel_mul_mat_f16_f32;
                    nrows = 4;
                }
            } else {
                kernel = backend_ctx->kernel_mul_mat_f16_f16;
                nrows = 4;
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb00));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb03));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne10));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne11));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb10));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb13));
            CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 21, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),      &r3));
            break;
        case GGML_TYPE_Q4_0:
            // This should have been satisfied.
            GGML_ASSERT(ne11 == ne1);
            GGML_ASSERT(ne01 == ne0);

#ifdef GGML_OPENCL_SOA_Q
            if (backend_ctx->gpu_family == INTEL) {
                nth0 = 16;
                nth1 = 1;

                kernel = backend_ctx->kernel_mul_mat_q4_0_f32_8x_flat;
                ndst = 8;
            } else if (backend_ctx->gpu_family == ADRENO) {
                nth0 = 64;
                nth1 = 1;

                kernel = backend_ctx->kernel_mul_mat_q4_0_f32_8x_flat;
                ndst =8;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q4_0->q));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q4_0->d));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne10));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &r3));
#else // GGML_OPENCL_SOA_Q
            if (backend_ctx->gpu_family == INTEL) {
                // Use 1D local size. Each workgroup is a SIMD group. Each SIMD
                // group produces N_DST (4 for Q4_0 kernel) values in the result.
                // The number of workgroups on dim 0 (the leading dimension) is
                // the nearest multiple of 4 that covers ne0 (equals ne01).
                nth0 = 16;
                nth1 = 1;

                kernel = backend_ctx->kernel_mul_mat_q4_0_f32;
                ndst = 4;
            } else if (backend_ctx->gpu_family == ADRENO) {
                nth0 = 64;
                nth1 = 1;

                kernel = backend_ctx->kernel_mul_mat_q4_0_f32_v;
                ndst = 4;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne10));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &r3));
#endif // GGML_OPENCL_SOA_Q
            break;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q8_0: {
#ifdef GGML_OPENCL_SOA_Q
            kernel = backend_ctx->kernel_mul_mv_q8_0_f32_flat;

            // nth0 - subgroup size
            // nth1 - number of subgroups per workgroup
            // ndst - number of output values per workgroup = output per subgroup * number of subgroups
            if (backend_ctx->gpu_family == INTEL) {
                nth0 = 16;
                nth1 = 2;
                ndst = nth1*4;
            } else if (backend_ctx->gpu_family == ADRENO) {
                nth0 = 64;
                nth1 = 2;
                ndst = nth1*4;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q8_0->q));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q8_0->d));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb03));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb13));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &r3));
#else
            kernel = backend_ctx->kernel_mul_mv_q8_0_f32;

            // nth0 - subgroup size
            // nth1 - number of subgroups per workgroup
            // ndst - number of output values per workgroup = output per subgroup * number of subgroups
            if (backend_ctx->gpu_family == INTEL) {
                nth0 = 16;
                nth1 = 2;
                ndst = nth1*4;
            } else if (backend_ctx->gpu_family == ADRENO) {
                nth0 = 64;
                nth1 = 2;
                ndst = nth1*4;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb03));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb13));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &r3));
#endif // GGML_OPENCL_SOA_Q
            break;
        }
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
#ifdef GGML_OPENCL_SOA_Q
            kernel = backend_ctx->kernel_mul_mv_q6_K_f32_flat;

            if (backend_ctx->gpu_family == INTEL) {
                nth0 = 16;
                nth1 = 2;
                ndst = 4;
            } else if (backend_ctx->gpu_family == ADRENO) {
                nth0 = 64;
                nth1 = 2;
                ndst = 4;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q6_K->ql));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q6_K->qh));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra0_q6_K->s));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_mem),   &extra0_q6_K->d));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne02));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne10));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &r3));
#else
            kernel = backend_ctx->kernel_mul_mv_q6_K_f32;

            if (backend_ctx->gpu_family == INTEL) {
                nth0 = 16;
                nth1 = 2;
                ndst = 1;
            } else if (backend_ctx->gpu_family == ADRENO) {
                nth0 = 64;
                nth1 = 2;
                ndst = 1;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne10));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &r3));
#endif // GGML_OPENCL_SOA_Q
            break;
        case GGML_TYPE_MXFP4: {
#ifdef GGML_OPENCL_SOA_Q
            kernel = backend_ctx->kernel_mul_mv_mxfp4_f32_flat;

            cl_mem q;
            if (backend_ctx->gpu_family == INTEL) {
                nth0 = 16;
                nth1 = 2;
                ndst = nth1*2;

                q = extra0_mxfp4->q;
            } else if (backend_ctx->gpu_family == ADRENO) {
                nth0 = 64;
                nth1 = 2;
                ndst = nth1*2;

                q = extra0_mxfp4->q_img;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &q));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_mxfp4->e));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb03));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb13));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &r3));
#else
            kernel = backend_ctx->kernel_mul_mv_mxfp4_f32;

            if (backend_ctx->gpu_family == INTEL) {
                nth0 = 16;
                nth1 = 2;
                ndst = nth1*2;
            } else if (backend_ctx->gpu_family == ADRENO) {
                nth0 = 64;
                nth1 = 2;
                ndst = nth1*2;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb03));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb13));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &r3));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(float)*nth0,nullptr));
#endif
            break;
        }
        default:
            GGML_ASSERT(false && "not implemented");
    }

    if (src0t == GGML_TYPE_Q4_0 || src0t == GGML_TYPE_MXFP4 ||
        src0t == GGML_TYPE_Q4_1 ||
        src0t == GGML_TYPE_Q8_0 ||
        src0t == GGML_TYPE_Q2_K) {
        // Each SIMD group produces N_DST values in the result. Assuming each
        // workgroup has N_SIMDGROUP SIMD groups, then each workgroup will
        // produce N_DST*N_SIMDGROUP values in the result. Hence, the grid size
        // (number of workgroups) will be a nearest multiple of
        // N_DST*N_SIMDGROUP to cover the size of the dimension. Below, 4 is
        // N_DST*N_SIMDGROUP (see the kernel for Q4_0 matmul).
        size_t global_work_size[] = {(size_t)(ne01 + ndst-1)/ndst*nth0, (size_t)ne11*nth1, (size_t)ne12*ne13};
        size_t local_work_size[] = {(size_t)nth0, (size_t)nth1, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    } else if (src0t == GGML_TYPE_Q4_K) {
        GGML_ASSERT(false && "not implemented");
    } else if (src0t == GGML_TYPE_Q3_K) {
        GGML_ASSERT(false && "not implemented");
    } else if (src0t == GGML_TYPE_Q5_K) {
        GGML_ASSERT(false && "not implemented");
    } else if (src0t == GGML_TYPE_Q6_K) {
        size_t global_work_size[] = {(size_t)(ne01+ndst*nth1-1)/(ndst*nth1)*nth0, (size_t)ne11*nth1, (size_t)ne12*ne13};
        size_t local_work_size[] = {(size_t)nth0, (size_t)nth1, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    } else {
        int64_t ny = (ne11 + nrows - 1)/nrows;

        size_t global_work_size[] = {(size_t)ne01*nth0, (size_t)ny*nth1, (size_t)ne12*ne13};
        size_t local_work_size[] = {(size_t)nth0, (size_t)nth1, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    }
}

static void ggml_cl_mul_mat_id(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    const ggml_tensor * src2 = dst->src[2];
    GGML_ASSERT(src2);
    GGML_ASSERT(src2->extra);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extra2 = (ggml_tensor_extra_cl *)src2->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offset2 = extra2->offset + src2->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    GGML_UNUSED(offset0);

#ifdef GGML_OPENCL_SOA_Q
    ggml_tensor_extra_cl_q4_0 * extra0_q4_0 = (ggml_tensor_extra_cl_q4_0 *)src0->extra;
    ggml_tensor_extra_cl_mxfp4 * extra0_mxfp4 = (ggml_tensor_extra_cl_mxfp4 *)src0->extra;
    ggml_tensor_extra_cl_q8_0 * extra0_q8_0 = (ggml_tensor_extra_cl_q8_0 *)src0->extra;
#endif

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const cl_ulong nb00 = src0->nb[0];
    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3];

    const cl_ulong nb11 = src1->nb[1];
    const cl_ulong nb12 = src1->nb[2];
    const cl_ulong nb13 = src1->nb[3];

    const int ne20 = src2->ne[0];
    const int ne21 = src2->ne[1];

    const cl_ulong nb21 = src2->nb[1];
    const cl_ulong nb20 = src2->nb[0];

    UNUSED(nb20);

    const int ne0 = dst->ne[0];
    const int ne1 = dst->ne[1];

    const int r2 = ne12/ne02;
    const int r3 = ne13/ne03;
    const int dst_rows = ne20*ne21; // ne20 = n_used_experts, ne21 = n_rows

    GGML_ASSERT(ne00 == ne10);

    int sgs   = 32; // subgroup size
    int nsg   = 1;  // number of subgroups
    int nrows = 1;  // number of row in src1
    int ndst  = 4;  // number of values produced by each subgroup

    cl_kernel kernel;

    // subgroup mat vec
    switch (src0->type) {
        case GGML_TYPE_Q4_0: {
            kernel = backend_ctx->kernel_mul_mv_id_q4_0_f32_8x_flat;

            if (backend_ctx->gpu_family == INTEL) {
                sgs  = 16;
                nsg  = 1;
                ndst = 8;
            } else if (backend_ctx->gpu_family == ADRENO) {
                sgs  = 64;
                nsg  = 1;
                ndst = 8;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q4_0->q));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q4_0->d));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra2->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset2));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne02));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb00));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne10));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne11));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &ne20));
            CL_CHECK(clSetKernelArg(kernel, 19, sizeof(int),      &ne21));
            CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong), &nb21));
            CL_CHECK(clSetKernelArg(kernel, 21, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 24, sizeof(int),      &r3));

            break;
        }
        case GGML_TYPE_Q8_0: {
#ifdef GGML_OPENCL_SOA_Q
            kernel = backend_ctx->kernel_mul_mv_id_q8_0_f32_flat;

            if (backend_ctx->gpu_family == INTEL) {
                sgs  = 16;
                nsg  = 2;
                ndst = 4;
            } else if (backend_ctx->gpu_family == ADRENO) {
                sgs  = 64;
                nsg  = 2;
                ndst = 4;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q8_0->q));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q8_0->d));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra2->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset2));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne11));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &ne20));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &ne21));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb21));
            CL_CHECK(clSetKernelArg(kernel, 19, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &ne1));
#else
            kernel = backend_ctx->kernel_mul_mv_id_q8_0_f32;

            if (backend_ctx->gpu_family == INTEL) {
                sgs  = 16;
                nsg  = 2;
                ndst = 4;
            } else if (backend_ctx->gpu_family == ADRENO) {
                sgs  = 64;
                nsg  = 2;
                ndst = 4;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra2->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset2));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne11));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &ne20));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &ne21));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb21));
            CL_CHECK(clSetKernelArg(kernel, 19, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &ne1));
#endif // GGML_OPENCL_SOA_Q
            break;
        }
        case GGML_TYPE_MXFP4: {
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
            if (use_adreno_moe_kernels(backend_ctx, src0)) {
                cl_int status;

                size_t local_size[3] = {64, 2, 1};
                size_t global_size[3] = {64, 2, 1};

                cl_mem src1_sub_buffer, buf_src1_image, buf_src2;

                int tile_size = 320;
                if (ne12 == 1) { // for gemv
                    kernel = backend_ctx->kernel_gemv_moe_mxfp4_f32;

                    // create a sub_buffer for src2
                    cl_buffer_region region;
                    region.origin = offset2;
                    region.size = ne20 * ne21 * sizeof(int);
                    buf_src2 = clCreateSubBuffer(extra2->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // set thread grid
                    global_size[0] = static_cast<size_t>(ne01);
                    global_size[1] = 4;
                    global_size[2] = static_cast<size_t>(ne20);
                    local_size[1] = 4;
                } else { // for gemm
                    kernel = backend_ctx->kernel_gemm_moe_mxfp4_f32;

                    // preprocess router table
                    int num_tiles_per_expert = (ne01 + tile_size - 1) / tile_size;
                    void * host_src2_reorder = malloc(ne20 * ne21 * 4 * num_tiles_per_expert * sizeof(short));
                    void * host_src2 = malloc(ne21 * nb21);
                    CL_CHECK(clEnqueueReadBuffer(backend_ctx->queue, extra2->data_device, CL_TRUE, offset2, ne21 * nb21, host_src2, 0, NULL, NULL));
                    int total_experts = nb21 / nb20;
                    int out_idx = 0;
                    for (int i_expert = 0; i_expert < ne02; i_expert++) {
                        for (int i_tile = 0; i_tile < num_tiles_per_expert; i_tile++) {
                            for (int j = 0; j < ne21; j++) {
                                for (int i = 0; i < ne20; i++) {
                                    int expert = ((int *)host_src2)[j * total_experts + i];
                                    if (i_expert == expert) {
                                        ((short *)host_src2_reorder)[out_idx] = static_cast<short>(expert);
                                        ((short *)host_src2_reorder)[out_idx + 1] = static_cast<short>(j * ne11 + (i % ne11));
                                        ((short *)host_src2_reorder)[out_idx + 2] = static_cast<short>(j * ne20 + i);
                                        ((short *)host_src2_reorder)[out_idx + 3] = static_cast<short>(i_tile);
                                        out_idx += 4;
                                    }
                                }
                            }
                        }
                    }
                    buf_src2 = clCreateBuffer(backend_ctx->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ne20 * ne21 * 4 * num_tiles_per_expert * sizeof(short), host_src2_reorder, &status);
                    CL_CHECK(status);

                    // set thread grid
                    global_size[0] = static_cast<size_t>(tile_size);
                    global_size[2] = static_cast<size_t>(ne20 * ne21 * num_tiles_per_expert);
                }

                // create a sub_buffer for src1
                cl_buffer_region region;
                region.origin = offset1;
                region.size = ne10 * ne11 * ne12 * sizeof(float);
                src1_sub_buffer = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                CL_CHECK(status);

                // create image for src1
                cl_image_format image_format_buf_src1 = {CL_RGBA, CL_FLOAT};
                cl_image_desc image_desc_buf_src1 = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne10 * ne11 * ne12 / 4), 0,0,0,0,0,0,0, {src1_sub_buffer}};
                buf_src1_image = clCreateImage(backend_ctx->context, CL_MEM_READ_ONLY, &image_format_buf_src1, &image_desc_buf_src1, NULL, &status);
                CL_CHECK(status);

                // Set kernel args
                int arg_idx = 0;
                CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_mxfp4->q));
                CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_mxfp4->e));
                CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src1_image));
                CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src2));
                CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extrad->data_device));
                CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_ulong),  &offsetd));
                CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne00));
                CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne01));
                if (ne12 == 1) {
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne11));
                } else {
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &tile_size));
                }

                // launch kernel
                backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_size, local_size, dst);

                // deallocate sub buffers and images
                CL_CHECK(clReleaseMemObject(src1_sub_buffer));
                CL_CHECK(clReleaseMemObject(buf_src1_image));
                CL_CHECK(clReleaseMemObject(buf_src2));
                return;
            } // else fallback to generic kernel
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

#ifdef GGML_OPENCL_SOA_Q
            kernel = backend_ctx->kernel_mul_mv_id_mxfp4_f32_flat;

            cl_mem q;
            if (backend_ctx->gpu_family == INTEL) {
                sgs  = 16;
                nsg  = 2;
                ndst = 2;

                q = extra0_mxfp4->q;
            } else if (backend_ctx->gpu_family == ADRENO) {
                sgs  = 64;
                nsg  = 1;
                ndst = 4;

                q = extra0_mxfp4->q_img;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &q));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_mxfp4->e));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra2->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset2));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb03));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne11));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb13));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &ne20));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &ne21));
            CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb21));
            CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 21, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),      &r3));
#else // GGML_OPENCL_SOA_Q
            kernel = backend_ctx->kernel_mul_mv_id_mxfp4_f32;

            if (backend_ctx->gpu_family == INTEL) {
                sgs  = 16;
                nsg  = 2;
                ndst = 2;
            } else if (backend_ctx->gpu_family == ADRENO) {
                sgs  = 64;
                nsg  = 2;
                ndst = 2;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra2->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset2));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb03));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne11));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb13));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &ne20));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &ne21));
            CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb21));
            CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 21, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),      &r3));
            CL_CHECK(clSetKernelArg(kernel, 24, sizeof(float)*sgs,nullptr));
#endif // GGML_OPENCL_SOA_Q
            break;
        }
        default:
            GGML_ASSERT(false && "not implemented");;
    }

    int _ne1 = 1;
    int ne123 = dst_rows;

    size_t global_work_size[] = {(size_t)(ne01+ndst*nsg-1)/(ndst*nsg)*sgs, (size_t)(_ne1+nrows-1)/nrows*nsg, (size_t)ne123};
    size_t local_work_size[] = {(size_t)sgs, (size_t)nsg, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_scale(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    GGML_UNUSED(src1);

    GGML_ASSERT(ggml_is_contiguous(src0));

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    float scale;
    float bias;
    memcpy(&scale, ((int32_t *) dst->op_params) + 0, sizeof(float));
    memcpy(&bias,  ((int32_t *) dst->op_params) + 1, sizeof(float));

    const int64_t ne = ggml_nelements(dst);
    if (ne < 4 || (ne % 4) != 0) {
        if (std::getenv("GGML_OPENCL_DEBUG_SCALE_FALLBACK") != nullptr) {
            static int debug_scale_fallback_count = 0;
            ++debug_scale_fallback_count;
            GGML_LOG_INFO("ggml_opencl: scale fallback #%d ne=%lld src='%s' dst='%s'\n",
                          debug_scale_fallback_count,
                          (long long) ne,
                          src0->name[0] ? src0->name : "(unnamed)",
                          dst->name[0] ? dst->name : "(unnamed)");
        }
        std::vector<float> tmp(static_cast<size_t>(ne));
        ggml_backend_tensor_get(src0, tmp.data(), 0, tmp.size() * sizeof(float));
        for (int64_t i = 0; i < ne; ++i) {
            tmp[static_cast<size_t>(i)] = tmp[static_cast<size_t>(i)] * scale + bias;
        }
        ggml_backend_tensor_set(dst, tmp.data(), 0, tmp.size() * sizeof(float));
        return;
    }

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel = backend_ctx->kernel_scale;

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(float),    &scale));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(float),    &bias));

    int n = ggml_nelements(dst)/4;

    size_t global_work_size[] = {(size_t)n, 1, 1};
    size_t local_work_size[] = {64, 1, 1};

    size_t * local_work_size_ptr = local_work_size;
    if (n % 64 != 0 && !backend_ctx->non_uniform_workgroups) {
        local_work_size_ptr = nullptr;  // Let driver choose the work-group sizes.
    }

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
}

static void ggml_cl_cpy(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);

    // GGML_OP_CPY happens between src0 and src1.
    // GGML_OP_DUP and GGML_OP_CONT happen between src0 and dst.
    UNUSED(dst);

    const int ne00 = src0 ? src0->ne[0] : 0;
    const int ne01 = src0 ? src0->ne[1] : 0;
    const int ne02 = src0 ? src0->ne[2] : 0;
    const int ne03 = src0 ? src0->ne[3] : 0;

    const cl_ulong nb00 = src0 ? src0->nb[0] : 0;
    const cl_ulong nb01 = src0 ? src0->nb[1] : 0;
    const cl_ulong nb02 = src0 ? src0->nb[2] : 0;
    const cl_ulong nb03 = src0 ? src0->nb[3] : 0;

    const int ne10 = src1 ? src1->ne[0] : 0;
    const int ne11 = src1 ? src1->ne[1] : 0;
    const int ne12 = src1 ? src1->ne[2] : 0;
    const int ne13 = src1 ? src1->ne[3] : 0;

    const cl_ulong nb10 = src1 ? src1->nb[0] : 0;
    const cl_ulong nb11 = src1 ? src1->nb[1] : 0;
    const cl_ulong nb12 = src1 ? src1->nb[2] : 0;
    const cl_ulong nb13 = src1 ? src1->nb[3] : 0;

    const enum ggml_type src0t = src0 ? src0->type : GGML_TYPE_COUNT;
    const enum ggml_type src1t = src1 ? src1->type : GGML_TYPE_COUNT;

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;

    cl_kernel kernel;

    switch (src0t) {
        case GGML_TYPE_F32:
            switch (src1t) {
                case GGML_TYPE_F16:
                    kernel = backend_ctx->kernel_cpy_f32_f16;
                    break;
                case GGML_TYPE_F32:
                    kernel = backend_ctx->kernel_cpy_f32_f32;
                    break;
                default:
                    GGML_ASSERT(false && "not implemented");
            }
            break;
        case GGML_TYPE_F16:
            switch (src1t) {
                case GGML_TYPE_F16:
                    kernel = backend_ctx->kernel_cpy_f16_f16;
                    break;
                case GGML_TYPE_F32:
                    kernel = backend_ctx->kernel_cpy_f16_f32;
                    break;
                default:
                    GGML_ASSERT(false && "not implemented");
            }
            break;
        default:
            GGML_ASSERT(false && "not implemented");
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne03));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb00));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne10));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne11));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne13));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb10));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb12));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb13));

    const int nth = MIN(64, ne00);

    size_t global_work_size[] = {(size_t)ne01*nth, (size_t)ne02, (size_t)ne03};
    size_t local_work_size[] = {(size_t)nth, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, src1);
}

static bool ggml_cl_use_cpy_1d(const ggml_tensor * src0, const ggml_tensor * dst) {
    if (src0 == nullptr || dst == nullptr) {
        return false;
    }
    if (src0->type != dst->type) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        return false;
    }
    const char * force_cpy_1d = std::getenv("GGML_OPENCL_CPY_1D_ALL");
    if (force_cpy_1d != nullptr && force_cpy_1d[0] != '0') {
        return ggml_is_contiguous(dst);
    }
    if (src0->op != GGML_OP_PERMUTE) {
        return false;
    }
    if (!ggml_is_contiguous(dst)) {
        return false;
    }
    return true;
}

static bool ggml_cl_use_cpy_repack(const ggml_backend_opencl_context * backend_ctx, const ggml_tensor * src0, const ggml_tensor * dst) {
    if (backend_ctx == nullptr || src0 == nullptr || dst == nullptr) {
        return false;
    }
    if (backend_ctx->gpu_family != ADRENO) {
        return false;
    }
    if (src0->op != GGML_OP_PERMUTE) {
        return false;
    }
    if (src0->type != dst->type) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
        return false;
    }
    if (src0->view_src == nullptr) {
        return false;
    }
    if (src0->view_offs != 0) {
        return false;
    }
    if (ggml_nbytes(src0) != ggml_nbytes(src0->view_src)) {
        return false;
    }
    const char * env = std::getenv("GGML_OPENCL_PERMUTE_REPACK");
    if (env == nullptr || env[0] == '0') {
        return false;
    }
    size_t max_bytes = SIZE_MAX;
    if (env[0] != '1') {
        char * end = nullptr;
        unsigned long long v = std::strtoull(env, &end, 10);
        if (end != env) {
            max_bytes = (size_t) v;
        }
    }
    return ggml_nbytes(src0) <= max_bytes;
}

static void ggml_cl_cpy_repack_host(ggml_backend_t backend, const ggml_tensor * src0, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(dst);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst->extra);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const size_t src_bytes = ggml_nbytes(src0);
    const size_t dst_bytes = ggml_nbytes(dst);
    const size_t elem_size = ggml_type_size(src0->type);

    std::vector<uint8_t> src_buf(src_bytes);
    std::vector<uint8_t> dst_buf(dst_bytes);

    CL_CHECK(clEnqueueReadBuffer(backend_ctx->queue, extra0->data_device, CL_TRUE, offset0, src_bytes, src_buf.data(), 0, NULL, NULL));

    const int64_t ne0 = src0->ne[0];
    const int64_t ne1 = src0->ne[1];
    const int64_t ne2 = src0->ne[2];
    const int64_t ne3 = src0->ne[3];

    const size_t nb0 = src0->nb[0];
    const size_t nb1 = src0->nb[1];
    const size_t nb2 = src0->nb[2];
    const size_t nb3 = src0->nb[3];

    size_t dst_idx = 0;
    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            for (int64_t i1 = 0; i1 < ne1; ++i1) {
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    const size_t src_off = (size_t)i0 * nb0 + (size_t)i1 * nb1 + (size_t)i2 * nb2 + (size_t)i3 * nb3;
                    const size_t dst_off = dst_idx * elem_size;
                    memcpy(dst_buf.data() + dst_off, src_buf.data() + src_off, elem_size);
                    ++dst_idx;
                }
            }
        }
    }

    CL_CHECK(clEnqueueWriteBuffer(backend_ctx->queue, extrad->data_device, CL_TRUE, offsetd, dst_bytes, dst_buf.data(), 0, NULL, NULL));
}

static void ggml_cl_cpy_1d_f32(ggml_backend_t backend, const ggml_tensor * src0, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const cl_ulong nb00 = src0->nb[0];
    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];

    const int ne0 = dst->ne[0];
    const int ne1 = dst->ne[1];
    const int ne2 = dst->ne[2];
    const int ne3 = dst->ne[3];

    const cl_ulong nb0 = dst->nb[0];
    const cl_ulong nb1 = dst->nb[1];
    const cl_ulong nb2 = dst->nb[2];
    const cl_ulong nb3 = dst->nb[3];

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel = backend_ctx->kernel_cpy_f32_f32_1d;

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne03));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb00));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne1));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne2));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne3));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb0));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb1));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb2));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb3));

    const size_t total = (size_t)ne0 * (size_t)ne1 * (size_t)ne2 * (size_t)ne3;
    size_t global_work_size[] = { total };
    size_t local_work_size[] = { 256 };

    size_t * local_work_size_ptr = local_work_size;
    if (total % local_work_size[0] != 0 && !backend_ctx->non_uniform_workgroups) {
        local_work_size_ptr = nullptr;
    }

    backend_ctx->enqueue_ndrange_kernel(kernel, 1, global_work_size, local_work_size_ptr, dst);
}

static void ggml_cl_cpy_1d_f16(ggml_backend_t backend, const ggml_tensor * src0, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const cl_ulong nb00 = src0->nb[0];
    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];

    const int ne0 = dst->ne[0];
    const int ne1 = dst->ne[1];
    const int ne2 = dst->ne[2];
    const int ne3 = dst->ne[3];

    const cl_ulong nb0 = dst->nb[0];
    const cl_ulong nb1 = dst->nb[1];
    const cl_ulong nb2 = dst->nb[2];
    const cl_ulong nb3 = dst->nb[3];

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel = backend_ctx->kernel_cpy_f16_f16_1d;

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne03));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb00));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne1));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne2));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne3));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb0));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb1));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb2));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb3));

    const size_t total = (size_t)ne0 * (size_t)ne1 * (size_t)ne2 * (size_t)ne3;
    size_t global_work_size[] = { total };
    size_t local_work_size[] = { 256 };

    size_t * local_work_size_ptr = local_work_size;
    if (total % local_work_size[0] != 0 && !backend_ctx->non_uniform_workgroups) {
        local_work_size_ptr = nullptr;
    }

    backend_ctx->enqueue_ndrange_kernel(kernel, 1, global_work_size, local_work_size_ptr, dst);
}

static void ggml_cl_dup(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    if (ggml_cl_use_cpy_repack(backend_ctx, src0, dst)) {
        if (std::getenv("GGML_OPENCL_DEBUG_REPACK")) {
            GGML_LOG_INFO("opencl: using host repack for permute cont (ne=%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ", bytes=%zu)\n",
                          dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], ggml_nbytes(dst));
        }
        ggml_cl_cpy_repack_host(backend, src0, dst);
        UNUSED(src1);
        return;
    }
    if (ggml_cl_use_cpy_1d(src0, dst)) {
        if (std::getenv("GGML_OPENCL_DEBUG_CPY1D")) {
            GGML_LOG_INFO("opencl: using 1d cpy for permute cont (type=%s, ne=%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ")\n",
                          ggml_type_name(src0->type), dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
        }
        if (src0->type == GGML_TYPE_F16 && backend_ctx->kernel_cpy_f16_f16_1d) {
            ggml_cl_cpy_1d_f16(backend, src0, dst);
        } else if (src0->type == GGML_TYPE_F32 && backend_ctx->kernel_cpy_f32_f32_1d) {
            ggml_cl_cpy_1d_f32(backend, src0, dst);
        } else {
            ggml_cl_cpy(backend, src0, dst, nullptr);
        }
    } else {
        ggml_cl_cpy(backend, src0, dst, nullptr);
    }
    UNUSED(src1);
}

static void ggml_cl_diag_mask_inf(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    int n_past = ((int32_t *)(dst->op_params))[0];

    const int  ne00 = src0 ? src0->ne[0] : 0;
    const int  ne01 = src0 ? src0->ne[1] : 0;
    const int  ne02 = src0 ? src0->ne[2] : 0;

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel;

    if (ne00%8 == 0) {
        kernel = backend_ctx->kernel_diag_mask_inf_8;

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),      &ne00));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),      &ne01));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &n_past));

        size_t global_work_size[] = {(size_t)ne00*ne01*ne02/8, 1, 1};
        size_t local_work_size[] = {64, 1, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    } else {
        kernel = backend_ctx->kernel_diag_mask_inf;

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),      &ne00));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),      &ne01));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &n_past));

        size_t global_work_size[] = {(size_t)ne00, (size_t)ne01, (size_t)ne02};
        size_t local_work_size[] = {64, 1, 1};

        size_t * local_work_size_ptr = local_work_size;
        if (ne00 % 64 != 0 && !backend_ctx->non_uniform_workgroups) {
            local_work_size_ptr = nullptr;  // Let driver choose the work-group sizes.
        }

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
    }
}

static void ggml_cl_soft_max(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    // Softmax can now fuse KQ mask and KQ scale, which used to be two additional
    // ops before softmax. It now also fuses alibi if `max_bias > 0`. For llama,
    // alibi is not used; however, for some other models, it is used.
    // KQ_mask
    if (src1) {
        GGML_ASSERT(src1);
        GGML_ASSERT(src1->extra);
    }

    const ggml_tensor * src2 = dst->src[2];
    if (src2) {
        GGML_ASSERT(src2->extra);
    }

    if (getenv("GGML_OPENCL_SOFTMAX_CPU") != nullptr) {
        GGML_ASSERT(src0->type == GGML_TYPE_F32);

        const size_t nbytes_src0 = ggml_nbytes(src0);
        const size_t nbytes_dst  = ggml_nbytes(dst);

        std::vector<char> h_src0(nbytes_src0);
        std::vector<char> h_dst(nbytes_dst);
        ggml_backend_tensor_get(src0, h_src0.data(), 0, nbytes_src0);

        std::vector<char> h_src1;
        std::vector<char> h_src2;

        if (src1) {
            h_src1.resize(ggml_nbytes(src1));
            ggml_backend_tensor_get(src1, h_src1.data(), 0, h_src1.size());
        }
        if (src2) {
            h_src2.resize(ggml_nbytes(src2));
            ggml_backend_tensor_get(src2, h_src2.data(), 0, h_src2.size());
        }

        const int64_t ne00 = src0->ne[0];
        const int64_t ne01 = src0->ne[1];
        const int64_t ne02 = src0->ne[2];
        const int64_t ne03 = src0->ne[3];

        const int64_t nb00 = src0->nb[0];
        const int64_t nb01 = src0->nb[1];
        const int64_t nb02 = src0->nb[2];
        const int64_t nb03 = src0->nb[3];

        const int64_t nb10 = src1 ? src1->nb[0] : 0;
        const int64_t nb11 = src1 ? src1->nb[1] : 0;
        const int64_t nb12 = src1 ? src1->nb[2] : 0;
        const int64_t nb13 = src1 ? src1->nb[3] : 0;

        const int64_t ne12 = src1 ? src1->ne[2] : 1;
        const int64_t ne13 = src1 ? src1->ne[3] : 1;

        const int64_t nbd0 = dst->nb[0];
        const int64_t nbd1 = dst->nb[1];
        const int64_t nbd2 = dst->nb[2];
        const int64_t nbd3 = dst->nb[3];

        float scale, max_bias;
        memcpy(&scale,    dst->op_params + 0, sizeof(float));
        memcpy(&max_bias, dst->op_params + 1, sizeof(float));

        const uint32_t n_head      = (uint32_t) ne02;
        const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

        const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
        const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

        const bool use_f16 = src1 && src1->type == GGML_TYPE_F16;
        const float * sk = src2 ? (const float *) h_src2.data() : nullptr;

        std::vector<float> wrow((size_t) ne00);

        for (int64_t i03 = 0; i03 < ne03; ++i03) {
            for (int64_t i02 = 0; i02 < ne02; ++i02) {
                const uint32_t h = (uint32_t) i02;
                const float slope = (max_bias > 0.0f) ? (h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1)) : 1.0f;

                for (int64_t i01 = 0; i01 < ne01; ++i01) {
                    const int64_t i11 = i01;
                    const int64_t i12 = i02 % ne12;
                    const int64_t i13 = i03 % ne13;

                    const char * sp = h_src0.data() + i01*nb01 + i02*nb02 + i03*nb03;
                    char * dp = h_dst.data() + i01*nbd1 + i02*nbd2 + i03*nbd3;

                    const char * mp = src1 ? (h_src1.data() + i11*nb11 + i12*nb12 + i13*nb13) : nullptr;

                    float maxv = -INFINITY;
                    for (int64_t i0 = 0; i0 < ne00; ++i0) {
                        const float v = *(const float *)(sp + i0*nb00);
                        float x = v * scale;
                        if (mp) {
                            if (use_f16) {
                                x += slope * ggml_fp16_to_fp32(*(const ggml_fp16_t *)(mp + i0*nb10));
                            } else {
                                x += slope * *(const float *)(mp + i0*nb10);
                            }
                        }
                        wrow[(size_t) i0] = x;
                        if (x > maxv) {
                            maxv = x;
                        }
                    }

                    if (sk) {
                        maxv = MAX(maxv, sk[i02]);
                    }

                    double sum = 0.0;
                    for (int64_t i0 = 0; i0 < ne00; ++i0) {
                        sum += expf(wrow[(size_t) i0] - maxv);
                    }
                    if (sk) {
                        sum += expf(sk[i02] - maxv);
                    }
                    const float inv_sum = (float)(1.0 / sum);
                    for (int64_t i0 = 0; i0 < ne00; ++i0) {
                        const float y = expf(wrow[(size_t) i0] - maxv) * inv_sum;
                        *(float *)(dp + i0*nbd0) = y;
                    }
                }
            }
        }

        ggml_backend_tensor_set(dst, h_dst.data(), 0, nbytes_dst);
        return;
    }

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    ggml_tensor_extra_cl * extra1 = src1 ? (ggml_tensor_extra_cl *)src1->extra : nullptr;
    ggml_tensor_extra_cl * extra2 = src2 ? (ggml_tensor_extra_cl *)src2->extra : nullptr;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_ulong offset1 = extra1 ? extra1->offset + src1->view_offs : offset0;
    cl_ulong offset2 = extra2 ? extra2->offset + src2->view_offs : offset0;

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const cl_long nb01 = src0->nb[1];
    const cl_long nb02 = src0->nb[2];
    const cl_long nb03 = src0->nb[3];

    const int ne12 = src1 ? src1->ne[2] : 0;
    const int ne13 = src1 ? src1->ne[3] : 0;

    const cl_long nb11 = src1 ? src1->nb[1] : 0;
    const cl_long nb12 = src1 ? src1->nb[2] : 0;
    const cl_long nb13 = src1 ? src1->nb[3] : 0;

    const cl_long nb1 = dst->nb[1];
    const cl_long nb2 = dst->nb[2];
    const cl_long nb3 = dst->nb[3];

    float scale, max_bias;
    memcpy(&scale,    dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, dst->op_params + 1, sizeof(float));

    const int n_head      = src0->ne[2];
    const int n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    // Local size must be wave size. Each workgroup is a wave, working on a row,
    // where a row corresponds to leading dimension.
    int nth = MIN(32, ne00);

    if (backend_ctx->gpu_family == INTEL) {
        // This is the same as the initial value.
        nth = MIN(32, ne00);
    }
    else if (backend_ctx->gpu_family == ADRENO) {
        nth = 64;
    } else {
        GGML_ASSERT(false && "TODO: Unknown GPU");
    }

    cl_kernel kernel;

    if (ne00%4 == 0) {
        if (use_f16) {
            kernel = backend_ctx->kernel_soft_max_4_f16;
        } else {
            kernel = backend_ctx->kernel_soft_max_4;
        }
    } else {
        if (use_f16) {
            kernel = backend_ctx->kernel_soft_max_f16;
        } else {
            kernel = backend_ctx->kernel_soft_max;
        }
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   extra1 ? &extra1->data_device : &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   extra2 ? &extra2->data_device : &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset2));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne13));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb12));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb13));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb1));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb2));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb3));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(float),    &scale));
    CL_CHECK(clSetKernelArg(kernel, 21, sizeof(float),    &max_bias));
    CL_CHECK(clSetKernelArg(kernel, 22, sizeof(float),    &m0));
    CL_CHECK(clSetKernelArg(kernel, 23, sizeof(float),    &m1));
    CL_CHECK(clSetKernelArg(kernel, 24, sizeof(int),      &n_head_log2));

    size_t global_work_size[] = {(size_t)ne01*nth, (size_t)ne02, (size_t)ne03};
    size_t local_work_size[] = {(size_t)nth, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_rope(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    if (getenv("GGML_OPENCL_ROPE_CPU") != nullptr) {
        const size_t nbytes_src0 = ggml_nbytes(src0);
        const size_t nbytes_src1 = ggml_nbytes(src1);
        const size_t nbytes_dst  = ggml_nbytes(dst);

        std::vector<char> h_src0(nbytes_src0);
        std::vector<char> h_src1(nbytes_src1);
        std::vector<char> h_dst(nbytes_dst);

        ggml_backend_tensor_get(src0, h_src0.data(), 0, nbytes_src0);
        ggml_backend_tensor_get(src1, h_src1.data(), 0, nbytes_src1);

        if (getenv("GGML_OPENCL_ROPE_CPU_DEBUG") != nullptr) {
            const int64_t npos = ggml_nelements(src1);
            const int32_t * pos = (const int32_t *) h_src1.data();
            int32_t pmin = pos[0];
            int32_t pmax = pos[0];
            for (int64_t i = 1; i < npos; ++i) {
                if (pos[i] < pmin) pmin = pos[i];
                if (pos[i] > pmax) pmax = pos[i];
            }
            GGML_LOG_INFO("opencl: rope cpu debug pos n=%lld min=%d max=%d first=%d,%d,%d,%d\n",
                          (long long) npos, pmin, pmax,
                          npos > 0 ? pos[0] : 0,
                          npos > 1 ? pos[1] : 0,
                          npos > 2 ? pos[2] : 0,
                          npos > 3 ? pos[3] : 0);
            if (src0->type == GGML_TYPE_F32) {
                const float * v = (const float *) h_src0.data();
                const int64_t n = (int64_t) (nbytes_src0 / sizeof(float));
                int64_t n_nan = 0;
                for (int64_t i = 0; i < n; ++i) {
                    if (isnan(v[i])) {
                        ++n_nan;
                        if (n_nan > 16) break;
                    }
                }
                GGML_LOG_INFO("opencl: rope cpu debug src0 f32 n=%lld nan_count=%lld\n",
                              (long long) n, (long long) n_nan);
            }
        }

        std::vector<char> h_src2;
        ggml_tensor host_src0 = *src0;
        ggml_tensor host_src1 = *src1;
        ggml_tensor host_src2;
        ggml_tensor host_dst  = *dst;

        host_src0.data = h_src0.data();
        host_src0.buffer = nullptr;
        host_src0.extra = nullptr;
        host_src0.view_src = nullptr;
        host_src0.view_offs = 0;

        host_src1.data = h_src1.data();
        host_src1.buffer = nullptr;
        host_src1.extra = nullptr;
        host_src1.view_src = nullptr;
        host_src1.view_offs = 0;

        ggml_tensor * src2 = dst->src[2];
        if (src2) {
            const size_t nbytes_src2 = ggml_nbytes(src2);
            h_src2.resize(nbytes_src2);
            ggml_backend_tensor_get(src2, h_src2.data(), 0, nbytes_src2);
            host_src2 = *src2;
            host_src2.data = h_src2.data();
            host_src2.buffer = nullptr;
            host_src2.extra = nullptr;
            host_src2.view_src = nullptr;
            host_src2.view_offs = 0;
        }

        host_dst.data = h_dst.data();
        host_dst.buffer = nullptr;
        host_dst.extra = nullptr;
        host_dst.view_src = nullptr;
        host_dst.view_offs = 0;
        host_dst.src[0] = &host_src0;
        host_dst.src[1] = &host_src1;
        host_dst.src[2] = src2 ? &host_src2 : nullptr;

        ggml_compute_params params;
        params.ith = 0;
        params.nth = 1;
        params.threadpool = nullptr;

        std::vector<float> wdata((size_t)host_src0.ne[0] + CACHE_LINE_SIZE_F32);
        params.wdata = wdata.data();
        params.wsize = wdata.size() * sizeof(float);

        ggml_compute_forward_rope(&params, &host_dst);
        ggml_backend_tensor_set(dst, h_dst.data(), 0, nbytes_dst);
        return;
    }

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    ggml_tensor * src2 = dst->src[2];
    ggml_tensor_extra_cl * extra2 = src2 ? (ggml_tensor_extra_cl *)src2->extra : nullptr;

    cl_ulong offset2 = extra2 ? extra2->offset + src2->view_offs : offset0;

    const int  ne00 = src0 ? src0->ne[0] : 0;
    const int  ne01 = src0 ? src0->ne[1] : 0;
    const int  ne02 = src0 ? src0->ne[2] : 0;
    const int  ne03 = src0 ? src0->ne[3] : 0;

    const cl_ulong  nb00 = src0 ? src0->nb[0] : 0;
    const cl_ulong  nb01 = src0 ? src0->nb[1] : 0;
    const cl_ulong  nb02 = src0 ? src0->nb[2] : 0;
    const cl_ulong  nb03 = src0 ? src0->nb[3] : 0;

    const int ne10 = src1 ? src1->ne[0] : 0;
    const int ne11 = src1 ? src1->ne[1] : 0; UNUSED(ne11);
    const int ne12 = src1 ? src1->ne[2] : 0; UNUSED(ne12);
    const int ne13 = src1 ? src1->ne[3] : 0; UNUSED(ne13);

    const int  ne0 = dst ? dst->ne[0] : 0;
    const int  ne1 = dst ? dst->ne[1] : 0;
    const int  ne2 = dst ? dst->ne[2] : 0;
    const int  ne3 = dst ? dst->ne[3] : 0;

    const cl_ulong  nb0 = dst ? dst->nb[0] : 0;
    const cl_ulong  nb1 = dst ? dst->nb[1] : 0;
    const cl_ulong  nb2 = dst ? dst->nb[2] : 0;
    const cl_ulong  nb3 = dst ? dst->nb[3] : 0;

    GGML_ASSERT(ne10 % ne02 == 0);
    GGML_ASSERT(ne10 >= ne02);

    int nth = MIN(64, ne00);

    const int n_past     = ((int *) dst->op_params)[0];
    const int n_dims     = ((int *) dst->op_params)[1];
    const int mode       = ((int *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    int32_t sections[4];

    memcpy(&freq_base,   (int32_t *) dst->op_params + 5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params + 6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params + 7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params + 8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params + 9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    memcpy(&sections,    (int32_t *) dst->op_params + 11, sizeof(int32_t)*4);

    const bool is_neox = mode & 2;
    const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;
    const int  is_imrope = mode == GGML_ROPE_TYPE_IMROPE;

    if (is_mrope) {
        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }

    if (is_vision) {
        GGML_ASSERT(n_dims == ne00/2);
    }

    cl_kernel kernel;

    if (is_neox) {
        switch (src0->type) {
            case GGML_TYPE_F32:
                kernel = backend_ctx->kernel_rope_neox_f32;
                break;
            case GGML_TYPE_F16:
                kernel = backend_ctx->kernel_rope_neox_f16;
                break;
            default:
                GGML_ASSERT(false);
        };
    } else if (is_mrope && !is_vision) {
        switch (src0->type) {
            case GGML_TYPE_F32:
                kernel = backend_ctx->kernel_rope_multi_f32;
                break;
            case GGML_TYPE_F16:
                kernel = backend_ctx->kernel_rope_multi_f16;
                break;
            default:
                GGML_ASSERT(false);
        };
    } else if (is_vision) {
        switch (src0->type) {
            case GGML_TYPE_F32:
                kernel = backend_ctx->kernel_rope_vision_f32;
                break;
            case GGML_TYPE_F16:
                kernel = backend_ctx->kernel_rope_vision_f16;
                break;
            default:
                GGML_ASSERT(false);
        }
    } else {
        switch (src0->type) {
            case GGML_TYPE_F32:
                kernel = backend_ctx->kernel_rope_norm_f32;
                break;
            case GGML_TYPE_F16:
                kernel = backend_ctx->kernel_rope_norm_f16;
                break;
            default:
                GGML_ASSERT(false);
        };
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   extra2 ? &extra2->data_device : &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset2));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne03));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb00));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &ne1));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &ne2));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(int),      &ne3));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong), &nb0));
    CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_ulong), &nb1));
    CL_CHECK(clSetKernelArg(kernel, 22, sizeof(cl_ulong), &nb2));
    CL_CHECK(clSetKernelArg(kernel, 23, sizeof(cl_ulong), &nb3));
    CL_CHECK(clSetKernelArg(kernel, 24, sizeof(int),      &n_past));
    CL_CHECK(clSetKernelArg(kernel, 25, sizeof(int),      &n_dims));
    CL_CHECK(clSetKernelArg(kernel, 26, sizeof(int),      &n_ctx_orig));
    CL_CHECK(clSetKernelArg(kernel, 27, sizeof(float),    &freq_base));
    CL_CHECK(clSetKernelArg(kernel, 28, sizeof(float),    &freq_scale));
    CL_CHECK(clSetKernelArg(kernel, 29, sizeof(float),    &ext_factor));
    CL_CHECK(clSetKernelArg(kernel, 30, sizeof(float),    &attn_factor));
    CL_CHECK(clSetKernelArg(kernel, 31, sizeof(float),    &beta_fast));
    CL_CHECK(clSetKernelArg(kernel, 32, sizeof(float),    &beta_slow));
    // both mrope and vision kernels have sections
    if (is_mrope || is_vision) {
        CL_CHECK(clSetKernelArg(kernel, 33, sizeof(int32_t)*4, &sections));
    }
    // only mrope has is_imrope
    if (is_mrope && !is_vision) {
        CL_CHECK(clSetKernelArg(kernel, 34, sizeof(int), &is_imrope));
    }

    size_t global_work_size[] = {(size_t)ne01*nth, (size_t)ne02, (size_t)ne03};
    size_t local_work_size[] = {(size_t)nth, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_solve_tri(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel = backend_ctx->kernel_solve_tri_f32;
    GGML_ASSERT(kernel != nullptr);

    const int n = src0->ne[0];
    const int k = src1->ne[0];

    const cl_ulong nb00 = src0->nb[0];
    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];

    const cl_ulong nb10 = src1->nb[0];
    const cl_ulong nb11 = src1->nb[1];
    const cl_ulong nb12 = src1->nb[2];
    const cl_ulong nb13 = src1->nb[3];

    const cl_ulong nb0 = dst->nb[0];
    const cl_ulong nb1 = dst->nb[1];
    const cl_ulong nb2 = dst->nb[2];
    const cl_ulong nb3 = dst->nb[3];

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &n));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int),      &k));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_ulong), &nb00));
    CL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong),&nb02));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong),&nb03));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong),&nb10));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong),&nb11));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong),&nb12));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong),&nb13));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong),&nb0));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong),&nb1));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong),&nb2));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong),&nb3));

    size_t global_work_size[3]= { (size_t)k, (size_t)dst->ne[2], (size_t)dst->ne[3]};
    size_t local_work_size[] = {16, 4, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_im2col(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    // src0 - filter, src1 - input
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const int32_t s0 = ((const int32_t*)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t*)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t*)(dst->op_params))[2];
    const int32_t p1 = ((const int32_t*)(dst->op_params))[3];
    const int32_t d0 = ((const int32_t*)(dst->op_params))[4];
    const int32_t d1 = ((const int32_t*)(dst->op_params))[5];

    const bool is_2D = ((const int32_t*)(dst->op_params))[6] == 1;

    const cl_long IC = src1->ne[is_2D ? 2 : 1];
    const cl_long IH = is_2D ? src1->ne[1] : 1;
    const cl_long IW =         src1->ne[0];

    const cl_long KH = is_2D ? src0->ne[1] : 1;
    const cl_long KW =         src0->ne[0];

    const cl_long OH = is_2D ? dst->ne[2] : 1;
    const cl_long OW =         dst->ne[1];

    // nb is byte offset, src is type float32
    const cl_ulong delta_offset = src1->nb[is_2D ? 2 : 1]/4;
    const cl_long  batch        = src1->ne[is_2D ? 3 : 2];
    const cl_ulong batch_offset = src1->nb[is_2D ? 3 : 2]/4;

    const cl_long pelements = OW*KW*KH;
    const cl_long CHW       = IC*KH*KW;

    cl_kernel kernel;

    if(dst->type == GGML_TYPE_F16) {
        kernel = backend_ctx->kernel_im2col_f16;
    } else {
        kernel = backend_ctx->kernel_im2col_f32;
    }

    CL_CHECK(clSetKernelArg(kernel,   0, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,   1, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,   2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,   3, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,   4, sizeof(cl_ulong), &batch_offset));
    CL_CHECK(clSetKernelArg(kernel,   5, sizeof(cl_ulong), &delta_offset));
    CL_CHECK(clSetKernelArg(kernel,   6, sizeof(cl_long),  &IW));
    CL_CHECK(clSetKernelArg(kernel,   7, sizeof(cl_long),  &IH));
    CL_CHECK(clSetKernelArg(kernel,   8, sizeof(cl_long),  &IC));
    CL_CHECK(clSetKernelArg(kernel,   9, sizeof(cl_long),  &OW));
    CL_CHECK(clSetKernelArg(kernel,  10, sizeof(cl_long),  &OH));
    CL_CHECK(clSetKernelArg(kernel,  11, sizeof(cl_long),  &KW));
    CL_CHECK(clSetKernelArg(kernel,  12, sizeof(cl_long),  &KH));
    CL_CHECK(clSetKernelArg(kernel,  13, sizeof(cl_long),  &pelements));
    CL_CHECK(clSetKernelArg(kernel,  14, sizeof(cl_long),  &CHW));
    CL_CHECK(clSetKernelArg(kernel,  15, sizeof(int),      &s0));
    CL_CHECK(clSetKernelArg(kernel,  16, sizeof(int),      &s1));
    CL_CHECK(clSetKernelArg(kernel,  17, sizeof(int),      &p0));
    CL_CHECK(clSetKernelArg(kernel,  18, sizeof(int),      &p1));
    CL_CHECK(clSetKernelArg(kernel,  19, sizeof(int),      &d0));
    CL_CHECK(clSetKernelArg(kernel,  20, sizeof(int),      &d1));

    const int num_blocks = (pelements + 256 - 1) / 256;
    size_t global_work_size[] = {(size_t)num_blocks*256, (size_t)OH, (size_t)batch*IC};
    size_t local_work_size[] = {256, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_argsort(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    GGML_UNUSED(src1);

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const int ne00  = src0->ne[0];
    const int nrows = ggml_nrows(src0);

    int ne00_padded = 1;
    while (ne00_padded < ne00) {
        ne00_padded *= 2;
    }

    int order = (enum ggml_sort_order) dst->op_params[0];

    cl_kernel kernel = backend_ctx->kernel_argsort_f32_i32;

    CL_CHECK(clSetKernelArg(kernel,   0, sizeof(cl_mem),            &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,   1, sizeof(cl_ulong),          &offset0));
    CL_CHECK(clSetKernelArg(kernel,   2, sizeof(cl_mem),            &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,   3, sizeof(cl_ulong),          &offsetd));
    CL_CHECK(clSetKernelArg(kernel,   4, sizeof(int),               &ne00));
    CL_CHECK(clSetKernelArg(kernel,   5, sizeof(int),               &ne00_padded));
    CL_CHECK(clSetKernelArg(kernel,   6, sizeof(int),               &order));
    CL_CHECK(clSetKernelArg(kernel,   7, ne00_padded*sizeof(int),   NULL));

    size_t global_work_size[] = {(size_t)ne00_padded, (size_t)nrows, (size_t)1};
    size_t local_work_size[] = {(size_t)ne00_padded, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_sum_rows(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    GGML_UNUSED(src1);

    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(ggml_is_contiguous(src0));

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];

    const cl_ulong nb1  = dst->nb[1];
    const cl_ulong nb2  = dst->nb[2];
    const cl_ulong nb3  = dst->nb[3];

    cl_kernel kernel = backend_ctx->kernel_sum_rows_f32;

    CL_CHECK(clSetKernelArg(kernel,   0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,   1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,   2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,   3, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,   4, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,   5, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,   6, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel,   7, sizeof(int),      &ne03));
    CL_CHECK(clSetKernelArg(kernel,   8, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel,   9, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel,  10, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel,  11, sizeof(cl_ulong), &nb1));
    CL_CHECK(clSetKernelArg(kernel,  12, sizeof(cl_ulong), &nb2));
    CL_CHECK(clSetKernelArg(kernel,  13, sizeof(cl_ulong), &nb3));

    size_t global_work_size[] = {(size_t)ne01, (size_t)ne02, (size_t)ne03};
    size_t local_work_size[] = {(size_t)64, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_glu(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    GGML_ASSERT(ggml_is_contiguous_1(src0));

    if (src1) {
        GGML_ASSERT(src1);
        GGML_ASSERT(src1->extra);
        GGML_ASSERT(ggml_are_same_shape(src0, src1));
    }

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    cl_kernel kernel;
    switch (ggml_get_glu_op(dst)) {
        case GGML_GLU_OP_GEGLU:
            if (dst->type == GGML_TYPE_F32) {
                kernel = backend_ctx->kernel_geglu;
            } else {
                kernel = backend_ctx->kernel_geglu_f16;
            }
            break;
        case GGML_GLU_OP_REGLU:
            if (dst->type == GGML_TYPE_F32) {
                kernel = backend_ctx->kernel_reglu;
            } else {
                kernel = backend_ctx->kernel_reglu_f16;
            }
            break;
        case GGML_GLU_OP_SWIGLU:
            if (dst->type == GGML_TYPE_F32) {
                kernel = backend_ctx->kernel_swiglu;
            } else {
                kernel = backend_ctx->kernel_swiglu_f16;
            }
            break;
        case GGML_GLU_OP_SWIGLU_OAI:
            kernel = backend_ctx->kernel_swiglu_oai;
            break;
        case GGML_GLU_OP_GEGLU_ERF:
            if (dst->type == GGML_TYPE_F32) {
                kernel = backend_ctx->kernel_geglu_erf;
            } else {
                kernel = backend_ctx->kernel_geglu_erf_f16;
            }
            break;
        case GGML_GLU_OP_GEGLU_QUICK:
            if (dst->type == GGML_TYPE_F32) {
                kernel = backend_ctx->kernel_geglu_quick;
            } else {
                kernel = backend_ctx->kernel_geglu_quick_f16;
            }
            break;
        default:
            GGML_ABORT("Unsupported glu op");
    }

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    ggml_tensor_extra_cl * extra1 = src1 ? (ggml_tensor_extra_cl *)src1->extra : nullptr;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_ulong offset1 = extra1 ? extra1->offset + src1->view_offs : offset0;

    const int ne0       = dst->ne[0];

    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb11 = src1 ? src1->nb[1] : nb01;

    const cl_ulong nb1  = dst->nb[1];

    const int   swp   = ggml_get_op_params_i32(dst, 1);
    const float alpha = ggml_get_op_params_f32(dst, 2);
    const float limit = ggml_get_op_params_f32(dst, 3);

    const int ne00_off = src1 ? 0 : (swp ? ne0 : 0);
    const int ne10_off = src1 ? 0 : (swp ? 0 : ne0);

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   src1 ? &extra1->data_device : &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb1));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne00_off));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne10_off));

    if (ggml_get_glu_op(dst) == GGML_GLU_OP_SWIGLU_OAI) {
        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(float), &limit));
        CL_CHECK(clSetKernelArg(kernel, 13, sizeof(float), &alpha));
    }

    const size_t nrows = ggml_nrows(src0);
    size_t nth = 512;
    size_t global_work_size[] = {nrows*nth, 1, 1};
    size_t local_work_size[] = {nth, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

//------------------------------------------------------------------------------
// Op offloading
//------------------------------------------------------------------------------

typedef void (*ggml_cl_func_t)(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

bool ggml_cl_compute_forward(ggml_backend_t backend, struct ggml_tensor * tensor) {
    ggml_cl_func_t func = nullptr;

    ggml_tensor * src0 = tensor->src[0];
    ggml_tensor * src1 = tensor->src[1];

    const bool any_on_device = tensor->extra
        || (src0 != nullptr && src0->extra)
        || (src1 != nullptr && src1->extra);

    switch (tensor->op) {
        case GGML_OP_GET_ROWS:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_get_rows;
            break;
        case GGML_OP_SET_ROWS:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_set_rows;
            break;
        case GGML_OP_CPY:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_cpy;
            break;
        case GGML_OP_DUP:
        case GGML_OP_CONT:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_dup;
            break;
        case GGML_OP_ADD:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_add;
            break;
        case GGML_OP_ADD_ID:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_add_id;
            break;
        case GGML_OP_MUL:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_mul;
            break;
        case GGML_OP_DIV:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_div;
            break;
        case GGML_OP_SUB:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_sub;
            break;
        case GGML_OP_SQR:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_sqr;
            break;
        case GGML_OP_SQRT:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_sqrt;
            break;
        case GGML_OP_MEAN:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_mean;
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(tensor)) {
                case GGML_UNARY_OP_GELU:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cl_gelu;
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cl_gelu_erf;
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cl_gelu_quick;
                    break;
                case GGML_UNARY_OP_SILU:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cl_silu;
                    break;
                case GGML_UNARY_OP_RELU:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cl_relu;
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cl_sigmoid;
                    break;
                case GGML_UNARY_OP_TANH:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cl_tanh;
                    break;
                case GGML_UNARY_OP_EXPM1:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cl_expm1;
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cl_softplus;
                    break;
                default:
                    return false;
            } break;
        case GGML_OP_GLU:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_glu;
            break;
        case GGML_OP_TRI:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_tri;
            break;
        case GGML_OP_FILL:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_fill;
            break;
        case GGML_OP_CLAMP:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_clamp;
            break;
        case GGML_OP_NORM:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_norm;
            break;
        case GGML_OP_RMS_NORM:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_rms_norm;
            break;
        case GGML_OP_GROUP_NORM:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_group_norm;
            break;
                case GGML_OP_REPEAT:
             if (!any_on_device) {
                return false;
            }
            func = ggml_cl_repeat;
            break;
        case GGML_OP_PAD:
            if (!any_on_device) {
                return false;
            }
            ggml_cl_pad(backend, tensor->src[0], tensor);
            return true;
        case GGML_OP_UPSCALE:
            if (!any_on_device) {
                return false;
            }
            ggml_cl_upscale(backend, tensor->src[0], tensor);
            return true;
        case GGML_OP_CONV_2D:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_conv_2d;
            break;
        case GGML_OP_SSM_CONV:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_ssm_conv;
            break;
        case GGML_OP_CONCAT:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_concat;
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            if (!any_on_device) {
                return false;
            }
            ggml_cl_timestep_embedding(backend, tensor->src[0], tensor);
            return true;
        case GGML_OP_MUL_MAT:
            if (!any_on_device && !ggml_cl_can_mul_mat(tensor->src[0], tensor->src[1], tensor)) {
                return false;
            }
            func = ggml_cl_mul_mat;
            break;
        case GGML_OP_MUL_MAT_ID:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_mul_mat_id;
            break;
        case GGML_OP_SCALE:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_scale;
            break;
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_nop;
            break;
        case GGML_OP_DIAG_MASK_INF:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_diag_mask_inf;
            break;
        case GGML_OP_SOFT_MAX:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_soft_max;
            break;
        case GGML_OP_ROPE:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_rope;
            break;
        case GGML_OP_SOLVE_TRI:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_solve_tri;
            break;
        case GGML_OP_IM2COL:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_im2col;
            break;
        case GGML_OP_ARGSORT:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_argsort;
            break;
        case GGML_OP_SUM_ROWS:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_sum_rows;
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            if (!any_on_device) {
                return false;
            }
            ggml_cl_flash_attn(backend, tensor->src[0], tensor->src[1], tensor);
            return true;
        default:
            return false;
    }

    func(backend, tensor->src[0], tensor->src[1], tensor);
    return true;
}
