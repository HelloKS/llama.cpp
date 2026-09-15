#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"
#include "dequantize.cuh"

#include <cstdint>
#include <cstdlib>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
static constexpr int Q2_K_BF16_M = 64;
static constexpr int Q2_K_BF16_N = 32;
static constexpr int Q2_K_BF16_STRIDE = QK_K + 8;
static constexpr int Q2_K_BF16_THREADS = 256;
static constexpr int Q2_K_BF16_SHARED = (Q2_K_BF16_M + Q2_K_BF16_N)*Q2_K_BF16_STRIDE*sizeof(nv_bfloat16) +
                                       Q2_K_BF16_N*(sizeof(int64_t) + sizeof(int32_t));

__launch_bounds__(Q2_K_BF16_THREADS)
static __global__ void mul_mat_q2_k_bf16_expert_tiles(
        const block_q2_K * __restrict__ x, const float * __restrict__ y, float * __restrict__ dst,
        const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ expert_bounds,
        const uint2 * __restrict__ tiles, const int * __restrict__ ntiles, const uint3 nty,
        const int nrows_x, const int nblocks_x, const int nused, const int nchannels_y,
        const int64_t stride_row_x, const int64_t stride_expert_x,
        const int64_t stride_channel_y, const int64_t stride_token_y,
        const int64_t stride_channel_dst, const int64_t stride_token_dst) {
#ifdef AMPERE_MMA_AVAILABLE
    using namespace ggml_cuda_mma;
    using tile_A = tile<16, 8, nv_bfloat162>;
    using tile_B = tile<8, 8, nv_bfloat162>;
    using tile_C = tile<16, 8, float>;

    extern __shared__ __align__(16) char data_q2_k_bf16[];
    nv_bfloat16 * sx = (nv_bfloat16 *) data_q2_k_bf16;
    nv_bfloat16 * sy = sx + Q2_K_BF16_M*Q2_K_BF16_STRIDE;
    int64_t * y_offsets = (int64_t *) (sy + Q2_K_BF16_N*Q2_K_BF16_STRIDE);
    int32_t * ids = (int32_t *) (y_offsets + Q2_K_BF16_N);
    const int tid = threadIdx.y*32 + threadIdx.x;
    const int warp_row = (threadIdx.y % 4)*16;
    const int warp_col = (threadIdx.y / 4)*16;
    const int64_t nwork = int64_t(*ntiles)*nty.z;

    for (int64_t work = blockIdx.x; work < nwork; work += gridDim.x) {
        const uint2 index = fast_div_modulo(work, nty);
        const uint2 work_tile = tiles[index.x];
        const int row = index.y*Q2_K_BF16_M;
        const int remaining = expert_bounds[work_tile.x + 1] - work_tile.y;

        if (tid < Q2_K_BF16_N) {
            const int id = tid < remaining ? ids_dst[work_tile.y + tid] : -1;
            ids[tid] = id;
            y_offsets[tid] = id < 0 ? 0 : (id/nused)*stride_token_y + ((id % nused) % nchannels_y)*stride_channel_y;
        }
        __syncthreads();

        tile_C sum[2];
        for (int kb = 0; kb < nblocks_x; ++kb) {
            for (int i = tid/64; i < Q2_K_BF16_M; i += Q2_K_BF16_THREADS/64) {
                if (row + i < nrows_x) {
                    dequantize_q2_K<nv_bfloat16>(x, work_tile.x*stride_expert_x + (row + i)*stride_row_x + kb,
                                               sx + i*Q2_K_BF16_STRIDE, tid % 64);
                } else {
                    for (int k = tid % 64; k < QK_K; k += 64) {
                        sx[i*Q2_K_BF16_STRIDE + k] = ggml_cuda_cast<nv_bfloat16>(0.0f);
                    }
                }
            }

            for (int i = tid; i < Q2_K_BF16_N*(QK_K/2); i += Q2_K_BF16_THREADS) {
                const int j = i/(QK_K/2);
                const int k = 2*(i % (QK_K/2));
                const float * src = y + y_offsets[j] + int64_t(kb)*QK_K + k;
                const float2 value = ids[j] < 0 ? make_float2(0.0f, 0.0f) : make_float2(src[0], src[1]);
                ((nv_bfloat162 *) sy)[j*(Q2_K_BF16_STRIDE/2) + k/2] = ggml_cuda_cast<nv_bfloat162>(value);
            }
            __syncthreads();

#pragma unroll
            for (int k = 0; k < QK_K; k += 16) {
                tile_A a;
                load_ldmatrix(a, (const nv_bfloat162 *) sx + warp_row*(Q2_K_BF16_STRIDE/2) + k/2, Q2_K_BF16_STRIDE/2);
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    tile_B b;
                    load_ldmatrix(b, (const nv_bfloat162 *) sy + (warp_col + j*8)*(Q2_K_BF16_STRIDE/2) + k/2, Q2_K_BF16_STRIDE/2);
                    mma(sum[j], a, b);
                }
            }
            __syncthreads();
        }

#pragma unroll
        for (int j = 0; j < 2; ++j) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = row + warp_row + tile_C::get_i(l);
                const int id = ids[warp_col + j*8 + tile_C::get_j(l)];
                if (i < nrows_x && id >= 0) {
                    dst[(id/nused)*stride_token_dst + (id % nused)*stride_channel_dst + i] = sum[j].x[l];
                }
            }
        }
        __syncthreads();
    }
#else
    GGML_UNUSED_VARS(x, y, dst, ids_dst, expert_bounds, tiles, ntiles, nty, nrows_x, nblocks_x, nused, nchannels_y,
                    stride_row_x, stride_expert_x, stride_channel_y, stride_token_y, stride_channel_dst, stride_token_dst);
    NO_DEVICE_CODE;
#endif
}

static bool ggml_cuda_mul_mat_q2_k_bf16(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const int32_t * ids_dst, const int32_t * expert_bounds, const int64_t nused, cudaStream_t stream) {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CUDA_Q2_K_MOE_BF16");
        return value && std::atoi(value) == 1;
    }();
    const int id = ggml_cuda_get_device();
    const auto & device = ggml_cuda_info().devices[id];
    const int64_t assignments = src1->ne[2]*nused;
    if (!enabled || src0->type != GGML_TYPE_Q2_K || src1->ne[2] < 32 || assignments/src0->ne[2] < 16 ||
        device.cc < GGML_CUDA_CC_BLACKWELL || device.cc >= GGML_CUDA_CC_RUBIN ||
        ggml_cuda_highest_compiled_arch(device.cc) < GGML_CUDA_CC_AMPERE || device.smpbo < Q2_K_BF16_SHARED ||
        src0->ne[3] != 1 || src0->ne[1] > INT_MAX || src0->ne[0]/QK_K > INT_MAX) {
        return false;
    }

    const int64_t nty = (src0->ne[1] + Q2_K_BF16_M - 1)/Q2_K_BF16_M;
    const int64_t max_tiles = assignments/Q2_K_BF16_N + src0->ne[2];
    if (max_tiles > INT_MAX/nty) {
        return false;
    }

    CUDA_SET_SHARED_MEMORY_LIMIT(mul_mat_q2_k_bf16_expert_tiles, Q2_K_BF16_SHARED);
    static int blocks_per_sm[GGML_CUDA_MAX_DEVICES] = {};
    if (blocks_per_sm[id] == 0) {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm[id], mul_mat_q2_k_bf16_expert_tiles, Q2_K_BF16_THREADS, Q2_K_BF16_SHARED));
        GGML_ASSERT(blocks_per_sm[id] > 0);
        GGML_LOG_INFO("%s: device %d using Q2_K BF16 prefill (64x32x256 tiles, %d blocks/SM)\n", __func__, id, blocks_per_sm[id]);
    }
    ggml_cuda_pool_alloc<uint2> tiles(ctx.pool(id), max_tiles);
    ggml_cuda_pool_alloc<int> ntiles(ctx.pool(id), 1);
    mmq_make_expert_tiles<Q2_K_BF16_N><<<1, 128, 0, stream>>>
        (expert_bounds, tiles.get(), ntiles.get(), src0->ne[2]);
    CUDA_CHECK(cudaGetLastError());
    mul_mat_q2_k_bf16_expert_tiles<<<device.nsm*blocks_per_sm[id], dim3(32, 8, 1), Q2_K_BF16_SHARED, stream>>>
        ((const block_q2_K *) src0->data, (const float *) src1->data, (float *) dst->data,
         ids_dst, expert_bounds, tiles.get(), ntiles.get(), init_fastdiv_values(nty),
         src0->ne[1], src0->ne[0]/QK_K, nused, src1->ne[1],
         src0->nb[1]/sizeof(block_q2_K), src0->nb[2]/sizeof(block_q2_K),
         src1->nb[1]/sizeof(float), src1->nb[2]/sizeof(float), dst->nb[1]/sizeof(float), dst->nb[2]/sizeof(float));
    CUDA_CHECK(cudaGetLastError());
    return true;
}
#endif

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_q_case<GGML_TYPE_Q1_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_q_case<GGML_TYPE_Q2_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_MXFP4:
            mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool fallback = ne01 % 128 != 0;

    const bool use_native_fp4 = blackwell_mma_available(cc) && (src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4);
    const size_t y_block_size       = use_native_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4 ? QK_FP4_MMQ            : QK8_1_MMQ;

    if (!ids) {
        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * y_block_size/y_values_per_block +
            ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
        ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
        if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
            src1_scale.alloc(ne13*ne12*ne11);
        }

        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (use_native_fp4) {
                static constexpr size_t align_float8 = 32;
                const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_fp4_cuda(src1_d, nullptr, src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, stream);

            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        // Stride depends on quantization format
        const int64_t s12 = use_native_fp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
            src0->type == GGML_TYPE_NVFP4 && use_native_fp4 ? src1_scale.ptr : nullptr,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            ne1, ne1};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);

    // gate/up activations are broadcast across experts (ne11 == 1): quantize each token once and
    // scatter to its slots. ids_src1 then holds the inverse map (token slot -> compact row).
    const bool dedup_bcast = ne11 == 1 && n_expert_used > 1;

    {
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, /*write_inverse =*/ dedup_bcast, stream);
        CUDA_CHECK(cudaGetLastError());
    }

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (ggml_cuda_mul_mat_q2_k_bf16(ctx, src0, src1, dst, ids_dst.get(), expert_bounds.get(), n_expert_used, stream)) {
        return;
    }
#endif

    const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * y_block_size/y_values_per_block +
        ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
    ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
    if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
        src1_scale.alloc(ne12*n_expert_used);
    }

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        if (use_native_fp4) {
            static constexpr size_t align_float8 = 32;
            const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
            if (dedup_bcast) {
                quantize_scatter_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10,
                                        /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
            } else {
                quantize_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13,
                                        ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
            }
        } else if (dedup_bcast) {
            quantize_scatter_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10,
                                    /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    static_assert(QK_FP4_MMQ == 8 * QK_MXFP4, "QK_FP4_MMQ needs to be 8 * QK_MXFP4");
    const int64_t s12 = use_native_fp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                         ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    // Each expert only sees ne12*n_expert_used/ne02 tokens on average.
    // On RDNA3 and RDNA4 it is faster to pick the tile size against this value instead of ne12.
    int64_t ncols_opt = ne12;
    if (GGML_CUDA_CC_IS_RDNA3_0(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
        ncols_opt = (ne12*n_expert_used + ne02 - 1) / ne02;
    }

    if (src0->type == GGML_TYPE_Q2_K && ne12 >= 32 &&
        cc >= GGML_CUDA_CC_BLACKWELL && cc < GGML_CUDA_CC_RUBIN) {
        static const int moe_ncols = [] {
            const char * value = std::getenv("GGML_CUDA_Q2_K_MOE_NCOLS");
            if (!value) {
                return -1;
            }
            char * end = nullptr;
            const long parsed = std::strtol(value, &end, 10);
            return end != value && *end == '\0' && (parsed == -1 || parsed == 32 || parsed == 64 || parsed == 128) ? int(parsed) : 0;
        }();
        // Keep the launch bound unchanged so skewed expert routing remains valid.
        if (moe_ncols != 0) {
            ncols_opt = moe_ncols == -1 ? (ne12*n_expert_used + ne02 - 1) / ne02 : moe_ncols;
        }
    }

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d,
        src1_scale.ptr,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        ne12, ncols_opt};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
// -------------------------------------------------
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
// -------------------------------------------------
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
// -------------------------------------------------
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    // MMQ tiles require at least 48 KiB per-block shared memory; fall back to BLAS otherwise.
    {
        const int    id    = ggml_cuda_get_device();
        const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
        if (smpbo < 48 * 1024) {
            return false;
        }
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        // for MoE, mmq is faster even without native dp4a
        // TODO: check if cards older than pascal might benefit from this as well
        return cc >= GGML_CUDA_CC_PASCAL && n_experts > 0;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    // gfx900 (Vega 10), gfx909, and gfx90c lack native dp4a, losing to dequant + hipBLAS
    // for dense matrices; keep MMQ only for MoE, where the
    // hipBLAS path is much slower.
    if (cc == GGML_CUDA_CC_VEGA || GGML_CUDA_CC_IS_GCN_APU(cc)) {
        return n_experts > 0;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}
