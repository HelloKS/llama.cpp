#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11070
#define USE_CUB
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11070

#ifdef USE_CUB
#include <cub/cub.cuh>
using namespace cub;
#endif // USE_CUB

#include "ssm-scan.cuh"

#include <cstdlib>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <mma.h>
#endif

// Minimum number of tokens to use SSD (State Space Duality) matmul path instead of scan path.
// For n_tok <= this threshold, the scan kernel is used (lower overhead for short sequences).
#define SSM_SSD_MIN_TOKENS 128

// prepare_dt kernel dimensions: one block per (head, seq), each block handles DT_MAX_ITEMS items.
#define SSM_SSD_DT_BLOCK     256
#define SSM_SSD_DT_MAX_ITEMS  32

// Maximum tokens the SSD path supports, derived from the prepare_dt kernel block capacity.
#define SSM_SSD_MAX_TOKENS (SSM_SSD_DT_BLOCK * SSM_SSD_DT_MAX_ITEMS)

// Chunk size for chunked SSD. Caps matmul cost at O(chunk^2) per chunk.
#define SSM_SSD_CHUNK_SIZE 256

// We would like to keep pragma unroll for cases where L_template is not 0,
// so we suppress the clang transformation warning.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template <size_t splitD, size_t N, size_t L_template>
__global__ void __launch_bounds__(splitD, 1)
    ssm_scan_f32(const float * src0_ptr, const float * src1_ptr, const float * src2_ptr,
                 const float * src3_ptr, const float * src4_ptr, const float * src5_ptr,
                 const int32_t * src6_ptr, float * dst_ptr,
                 const int src0_nb2, const int src0_nb3, const int src1_nb2, const int src1_nb3,
                 const int src2_nb1, const int src2_nb2, const int src3_nb1,
                 const int src4_nb2, const int src4_nb3, const int src5_nb2, const int src5_nb3,
                 const int64_t s_off, const int64_t d_inner, const int64_t L_param)
{
    const float   * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float   * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float   * GGML_CUDA_RESTRICT src2 = src2_ptr;
    const float   * GGML_CUDA_RESTRICT src3 = src3_ptr;
    const float   * GGML_CUDA_RESTRICT src4 = src4_ptr;
    const float   * GGML_CUDA_RESTRICT src5 = src5_ptr;
    const int32_t * GGML_CUDA_RESTRICT src6 = src6_ptr;
    float         * GGML_CUDA_RESTRICT dst  = dst_ptr;
    const size_t L = L_template == 0 ? L_param : L_template;
    ggml_cuda_pdl_sync();
    const float *s0_block = (const float *)((const char *)src0 + src6[blockIdx.x] * src0_nb3 + blockIdx.y * splitD * src0_nb2);
    const float *x_block = (const float *)((const char *)src1 + (blockIdx.x * src1_nb3) + blockIdx.y * splitD * sizeof(float));
    const float *dt_block = (const float *)((const char *)src2 + (blockIdx.x * src2_nb2) + blockIdx.y * splitD * sizeof(float));
    const float *A_block = (const float *)((const char *)src3 + blockIdx.y * splitD * src3_nb1);
    const float *B_block = (const float *)((const char *)src4 + (blockIdx.x * src4_nb3));
    const float *C_block = (const float *)((const char *)src5 + (blockIdx.x * src5_nb3));
    float *y_block = (float *)((char *)dst + (blockIdx.x * d_inner * L * sizeof(float)) + blockIdx.y * splitD * sizeof(float));
    float *s_block = (float *)((char *)dst + s_off + blockIdx.x * src0_nb3 + blockIdx.y * splitD * src0_nb2);

    const int stride_x = src1_nb2 / sizeof(float);
    const int stride_dt = src2_nb1 / sizeof(float);
    const int stride_B = src4_nb2 / sizeof(float);
    const int stride_C = src5_nb2 / sizeof(float);
    const int stride_y = d_inner;

    float regA[N];
    float regs0[N];

    __shared__ float smemB[N];
    __shared__ float smemC[N];

#ifdef USE_CUB
    using BlockLoad = cub::BlockLoad<float, splitD, N, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStore = cub::BlockStore<float, splitD, N, cub::BLOCK_STORE_WARP_TRANSPOSE>;

    union CubTempStorage {
        typename BlockLoad::TempStorage load_temp;
        typename BlockStore::TempStorage store_temp;
    };
    __shared__ CubTempStorage cub_temp_storage;

    BlockLoad(cub_temp_storage.load_temp).Load(A_block, regA);
    __syncthreads();
    BlockLoad(cub_temp_storage.load_temp).Load(s0_block, regs0);
#else
    const int stride_s0 = src0_nb2 / sizeof(float);
    const int stride_A = src3_nb1 / sizeof(float);
#pragma unroll
    for (size_t n = 0; n < N; ++n)
    {
        regA[n] = A_block[threadIdx.x * stride_A + n];
        regs0[n] = s0_block[threadIdx.x * stride_s0 + n];
    }
#endif

#pragma unroll
    for (size_t i = 0; i < L; i++)
    {
        if (threadIdx.x < N)
        {
            smemB[threadIdx.x] = B_block[i * stride_B + threadIdx.x];
            smemC[threadIdx.x] = C_block[i * stride_C + threadIdx.x];
        }
        __syncthreads();

        float dt_soft_plus = dt_block[i * stride_dt + threadIdx.x];
        if (dt_soft_plus <= 20.0f)
        {
            dt_soft_plus = log1pf(expf(dt_soft_plus));
        }
        float x_dt = x_block[i * stride_x + threadIdx.x] * dt_soft_plus;

        float sumf = 0.0f;
#pragma unroll
        for (size_t n = 0; n < N; n++)
        {
            float state = regs0[n] * expf(dt_soft_plus * regA[n]) + smemB[n] * x_dt;
            sumf += state * smemC[n];
            regs0[n] = state;
        }
        y_block[i * stride_y + threadIdx.x] = sumf;
        __syncthreads();
    }

#ifdef USE_CUB
    BlockStore(cub_temp_storage.store_temp).Store(s_block, regs0);
#else
    const int stride_s = stride_s0;
#pragma unroll
    for (size_t n = 0; n < N; ++n)
    {
        s_block[threadIdx.x * stride_s + n] = regs0[n];
    }
#endif
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

// assumes as many threads as d_state
template <int c_factor, int d_state>
__global__ void __launch_bounds__(d_state, 1)
    ssm_scan_f32_group(
        const float * src0_ptr, const float * src1_ptr, const float * src2_ptr,
        const float * src3_ptr, const float * src4_ptr, const float * src5_ptr,
        const int32_t * src6_ptr, float * dst_ptr,
        const int src0_nb2, const int src0_nb3, const int src1_nb2, const int src1_nb3,
        const int src2_nb1, const int src2_nb2, const int src3_nb1,
        const int src4_nb2, const int src4_nb3, const int src5_nb2, const int src5_nb3,
        const int64_t s_off, const int64_t n_head, const int64_t d_head, const int64_t n_group, const int64_t n_tok, const int64_t K) {
    const float   * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float   * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float   * GGML_CUDA_RESTRICT src2 = src2_ptr;
    const float   * GGML_CUDA_RESTRICT src3 = src3_ptr;
    const float   * GGML_CUDA_RESTRICT src4 = src4_ptr;
    const float   * GGML_CUDA_RESTRICT src5 = src5_ptr;
    const int32_t * GGML_CUDA_RESTRICT src6 = src6_ptr;
    float         * GGML_CUDA_RESTRICT dst  = dst_ptr;

    const int warp     = threadIdx.x / WARP_SIZE;
    const int lane     = threadIdx.x % WARP_SIZE;
    const int warp_idx = blockIdx.x  * c_factor + warp;

    const int head_idx =  warp_idx / d_head;
    const int head_off = (warp_idx % d_head) * sizeof(float);
    const int seq_idx  = blockIdx.y;

    const int group_off = (head_idx / (n_head / n_group)) * d_state * sizeof(float);

    ggml_cuda_pdl_sync();
    // TODO: refactor strides to be in elements/floats instead of bytes to be cleaner and consistent with the rest of the codebase
    const float * s0_warp = (const float *) ((const char *) src0 + src6[seq_idx] * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);
    const float * x_warp  = (const float *) ((const char *) src1 + (seq_idx * src1_nb3) + (warp_idx * sizeof(float)));
    const float * dt_warp = (const float *) ((const char *) src2 + (seq_idx * src2_nb2) + head_idx * sizeof(float));
    const float * A_warp  = (const float *) ((const char *) src3 + head_idx * src3_nb1);
    const float * B_warp  = (const float *) ((const char *) src4 + (seq_idx * src4_nb3) + (group_off));
    const float * C_warp  = (const float *) ((const char *) src5 + (seq_idx * src5_nb3) + (group_off));
    float *       y_warp  = dst + (seq_idx * n_tok * n_head * d_head) + warp_idx;
    float *       s_warp  = (float *) ((char *) dst + s_off + seq_idx * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);

    // strides across n_seq_tokens
    const int stride_x  = src1_nb2 / sizeof(float);
    const int stride_dt = src2_nb1 / sizeof(float);
    const int stride_B  = src4_nb2 / sizeof(float);
    const int stride_C  = src5_nb2 / sizeof(float);
    const int stride_y  = n_head * d_head;

    float state[c_factor];
    float state_sum = 0.0f;

#pragma unroll
    for (int j = 0; j < c_factor; j++) {
        state[j] = s0_warp[WARP_SIZE * j + lane];
    }

    for (int64_t i = 0; i < n_tok; i++) {
        // NOTE: dt_soft_plus, dA and x_dt have the same value for a warp here.
        // Recalculation is intentional; sharing via shuffles/smem proved slower due to sync overhead.
        const float dt_soft_plus = (dt_warp[i * stride_dt] <= 20.0f ? log1pf(expf(dt_warp[i * stride_dt])) : dt_warp[i * stride_dt]);

        state_sum = 0.0f;
        const float dA   = expf(dt_soft_plus * A_warp[0]);
        const float x_dt = x_warp[i * stride_x] * dt_soft_plus;
#pragma unroll
        for (int j = 0; j < c_factor; j++) {
            const float B_val = B_warp[i * stride_B + WARP_SIZE * j + lane];
            const float C_val = C_warp[i * stride_C + WARP_SIZE * j + lane];
            state[j] = (state[j] * dA) + (B_val * x_dt);
            state_sum += state[j] * C_val;
        }

        // parallel accumulation for output
        state_sum = warp_reduce_sum(state_sum);

        if (lane == 0) {
            y_warp[i * stride_y] = state_sum;
        }

        // Slot 0 is the final state written below; slots 1..K-1 are rollback snapshots.
        const int64_t slot = n_tok - 1 - i;
        if (K > 1 && slot > 0 && slot < K) {
            float * s_snapshot_warp = (float *) ((char *) dst + s_off + (slot * gridDim.y + seq_idx) * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);
#pragma unroll
            for (int j = 0; j < c_factor; j++) {
                s_snapshot_warp[WARP_SIZE * j + lane] = state[j];
            }
        }
    }

    // write back the state
#pragma unroll
    for (int j = 0; j < c_factor; j++) {
        s_warp[WARP_SIZE * j + lane] = state[j];
    }
}

static void ssm_scan_f32_cuda(const float * src0, const float * src1, const float * src2, const float * src3,
                              const float * src4, const float * src5, const int32_t * src6, float * dst,
                              const int src0_nb2, const int src0_nb3, const int src1_nb2, const int src1_nb3, const int src2_nb1,
                              const int src2_nb2, const int src3_nb1, const int src4_nb2, const int src4_nb3, const int src5_nb2,
                              const int src5_nb3, const int64_t s_off, const int64_t d_state, const int64_t head_dim,
                              const int64_t n_head, const int64_t n_group, const int64_t n_tok, const int64_t n_seq,
                              const int64_t K, cudaStream_t stream) {
    // NOTE: if you change conditions here, be sure to update the corresponding supports_op condition!
    if (src3_nb1 == sizeof(float)) {
        // Mamba-2
        if (d_state == 128) {
            constexpr int threads   = 128;
            constexpr int num_warps = threads/WARP_SIZE;

            const dim3 blocks((n_head * head_dim + (num_warps - 1)) / num_warps, n_seq, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_scan_f32_group<128/WARP_SIZE, 128>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, head_dim, n_group, n_tok, K);
        } else if (d_state == 256) { // Falcon-H1
            constexpr int threads   = 256;
            constexpr int num_warps = threads/WARP_SIZE;

            const dim3 blocks((n_head * head_dim + (num_warps - 1)) / num_warps, n_seq, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_scan_f32_group<256/WARP_SIZE, 256>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, head_dim, n_group, n_tok, K);
        } else {
            GGML_ABORT("doesn't support d_state!=(128 or 256).");
        }
    } else {
        // Mamba-1
        GGML_ASSERT(K == 1);
        constexpr int threads = 128;
        GGML_ASSERT(n_head % threads == 0);
        GGML_ASSERT(head_dim == 1);
        GGML_ASSERT(n_group == 1);
        const dim3 blocks(n_seq, (n_head + threads - 1) / threads, 1);
        if (d_state == 16) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            switch (n_tok)
            {
            case 1:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 1>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 2:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 2>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 3:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 3>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 4:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 4>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 5:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 5>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 6:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 6>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 7:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 7>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 8:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 8>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            default:
                ggml_cuda_kernel_launch(ssm_scan_f32<threads, 16, 0>, launch_params,
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            }
        } else {
            GGML_ABORT("doesn't support d_state!=16.");
        }
    }
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// ============================================================================
// SSD (State Space Duality) kernels for Mamba-2 prefill (n_tok > SSM_SSD_MIN_TOKENS)
//
// Instead of a sequential scan, SSD reformulates the output as:
//   Y = (L (.) (C @ B^T)) @ (X * dt)  +  decay * C @ s_init
// where L is a causal decay mask derived from A and dt.
//
// This converts the O(T*N) sequential scan into parallel matmuls.
// ============================================================================
// Softplus(dt) and inclusive prefix sum per head using CUB BlockScan.
// Grid: (n_head, n_seqs)
template <int BLOCK_SIZE, int MAX_ITEMS>
__global__ void ssm_ssd_prepare_dt_kernel(
        const float * __restrict__ dt_raw,
        float * __restrict__ dt_sp_out,
        float * __restrict__ cs_out,
        const int n_head, const int n_tok,
        const int dt_stride_tok,   // elements between tokens in dt
        const int dt_stride_seq) { // elements between sequences in dt

    const int h = blockIdx.x;
    const int s = blockIdx.y;

    const float * dt_seq = dt_raw + s * dt_stride_seq;

    float * dt_sp_seq = dt_sp_out + s * n_tok * n_head;
    float * cs_seq    = cs_out    + s * n_tok * n_head;

    const int items_per_thread = (n_tok + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Phase 1: softplus with interleaved distribution (t = i*BLOCK_SIZE + threadIdx.x).
    // Each warp reads BLOCK_SIZE consecutive tokens, giving coalesced dt_raw loads
    // (stride n_head between threads vs. items_per_thread*n_head in blocked layout).
    float local_vals[MAX_ITEMS];
    for (int i = 0; i < items_per_thread; i++) {
        const int t = i * BLOCK_SIZE + threadIdx.x;
        if (t < n_tok) {
            float val = dt_seq[h + t * dt_stride_tok];
            float sp = (val <= 20.0f) ? log1pf(expf(val)) : val;
            local_vals[i] = sp;
            dt_sp_seq[t * n_head + h] = sp;
        } else {
            local_vals[i] = 0.0f;
        }
    }

    // Phase 2+3: per-step inclusive scan to build cs[] in token order.
    // With interleaved distribution the per-thread total scan would not give token-order
    // prefix sums, so we scan one BLOCK_SIZE slab at a time and carry a running total.
#ifdef USE_CUB
    using BlockScan = cub::BlockScan<float, BLOCK_SIZE>;
    __shared__ typename BlockScan::TempStorage scan_temp;
    __shared__ float step_total;

    float running = 0.0f;
    for (int i = 0; i < items_per_thread; i++) {
        float inclusive;
        BlockScan(scan_temp).InclusiveSum(local_vals[i], inclusive);
        const int t = i * BLOCK_SIZE + threadIdx.x;
        if (t < n_tok) {
            cs_seq[t * n_head + h] = running + inclusive;
        }
        if (threadIdx.x == BLOCK_SIZE - 1) {
            step_total = inclusive;
        }
        __syncthreads();
        running += step_total;
    }
#else
    // Fallback: sequential prefix scan in shared memory, one slab at a time.
    __shared__ float sdata[BLOCK_SIZE];
    float running = 0.0f;
    for (int i = 0; i < items_per_thread; i++) {
        const int t = i * BLOCK_SIZE + threadIdx.x;
        sdata[threadIdx.x] = local_vals[i];
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int j = 1; j < BLOCK_SIZE; j++) {
                sdata[j] += sdata[j - 1];
            }
        }
        __syncthreads();
        if (t < n_tok) {
            cs_seq[t * n_head + h] = running + sdata[threadIdx.x];
        }
        running += sdata[BLOCK_SIZE - 1];
        __syncthreads();
    }
#endif
}

// Prepare SSD matmul inputs for one chunk: X_dt, B_weighted, C_scaled.
// T_matmul controls precision for X_dt, B_weighted (float or half).
// C_scaled is always float (pairs with float s_cur in step 3c).
// Computation is always FP32; only the final store converts to T_matmul.
// The fallback also materializes the causal M matrix = exp(A*(cs_out - cs_in)) * CB.
// Grid covers X_dt, B_weighted, C_scaled, and fallback M.
template <int BLOCK_SIZE, typename T_matmul, bool MATERIALIZE_M>
__global__ void ssm_ssd_pre_matmul_kernel(
        const float * __restrict__ cs,         // {n_tok, n_head} cumulative dt sums
        const float * __restrict__ dt_sp,      // {n_tok, n_head} softplus(dt)
        const float * __restrict__ A,          // {1, n_head}
        const float * __restrict__ x,          // {head_dim, n_head, n_tok, n_seqs}
        const float * __restrict__ B,          // {d_state, n_group, n_tok, n_seqs}
        const float * __restrict__ C_src,      // {d_state, n_group, n_tok, n_seqs}
        T_matmul * __restrict__ X_dt,          // {head_dim, C, n_head} x * dt, d-fastest
        T_matmul * __restrict__ B_weighted,    // {d_state, C, n_head} B * decay_from_end
        float * __restrict__ C_scaled,         // {d_state, C, n_head} C * decay_to_pos (always float)
        const float * __restrict__ CB,         // {chunk_len, chunk_len, n_group, n_seqs}
        half * __restrict__ M_out,             // {chunk_len, chunk_len, n_head, n_seqs}
        const int chunk_len, const int head_dim, const int n_head, const int n_group,
        const int d_state, const int A_stride,
        const int x_stride_tok, const int x_stride_seq,
        const int B_stride_tok, const int B_stride_seq,
        const int C_stride_tok, const int C_stride_seq,
        const int chunk_offset,
        const int n_tok_total) {

    const int h = blockIdx.y;
    const int s = blockIdx.z;
    const int g = h / (n_head / n_group);

    const float A_h = A[h * A_stride];
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    const int cs_seq_off = s * n_tok_total * n_head;
    const float cs_base = (chunk_offset > 0) ? cs[cs_seq_off + (chunk_offset - 1) * n_head + h] : 0.0f;
    const float cs_last = cs[cs_seq_off + (chunk_offset + chunk_len - 1) * n_head + h] - cs_base;

    // Prepare X_dt = x * dt, stored d-fastest for coalesced reads and writes.
    const int n_xdt = chunk_len * head_dim;
    if (idx < n_xdt) {
        const int d = idx % head_dim;
        const int t = idx / head_dim;

        const float x_val = x[s * x_stride_seq + (chunk_offset + t) * x_stride_tok + d + h * head_dim];
        const float dt_val = dt_sp[cs_seq_off + (chunk_offset + t) * n_head + h];

        X_dt[d + t * head_dim + h * n_xdt + s * n_xdt * n_head] = (T_matmul)(x_val * dt_val);
    }

    // Prepare B_weighted and C_scaled together: both share the same index space (d_state * chunk_len)
    // and the same cs_t load, so merging halves the cs[] global memory traffic.
    const int n_bw = d_state * chunk_len;
    if (idx < n_bw) {
        const int n = idx % d_state;
        const int t = idx / d_state;

        const float cs_t = cs[cs_seq_off + (chunk_offset + t) * n_head + h] - cs_base;

        const float B_val = B[s * B_stride_seq + (chunk_offset + t) * B_stride_tok + g * d_state + n];
        B_weighted[n + t * d_state + h * n_bw + s * n_bw * n_head] = (T_matmul)(B_val * __expf(A_h * (cs_last - cs_t)));

        const float C_val = C_src[s * C_stride_seq + (chunk_offset + t) * C_stride_tok + g * d_state + n];
        C_scaled[n + t * d_state + h * n_bw + s * n_bw * n_head] = C_val * __expf(A_h * cs_t);
    }

    if constexpr (MATERIALIZE_M) {
        // Materialize M = exp(A*(cs_out - cs_in)) * CB with causal mask.
        const int n_M = chunk_len * chunk_len;
        if (idx < n_M) {
            const int t_out = idx % chunk_len;
            const int t_in  = idx / chunk_len;

            half val;
            if (t_in <= t_out) {
                const float cs_out = cs[cs_seq_off + (chunk_offset + t_out) * n_head + h] - cs_base;
                const float cs_in  = cs[cs_seq_off + (chunk_offset + t_in)  * n_head + h] - cs_base;
                const float decay  = __expf(A_h * (cs_out - cs_in));
                const float * CB_g = CB + (int64_t)s * chunk_len * chunk_len * n_group
                                       + (int64_t)g * chunk_len * chunk_len;
                const float cb_val = CB_g[t_out + t_in * chunk_len];
                val = __float2half(decay * cb_val);
            } else {
                val = __float2half(0.0f);
            }

            M_out[(int64_t)s * n_M * n_head + (int64_t)h * n_M + t_in * chunk_len + t_out] = val;
        }
    }
}

template <int TILE>
__global__ void ssm_ssd_fused_output_kernel(
        const half * __restrict__ X_dt,
        const float * __restrict__ CB,
        const float * __restrict__ cs_chunk,
        const float * __restrict__ A,
        float * __restrict__ dst_chunk,
        const int chunk_len, const int head_dim, const int n_head, const int n_group,
        const int A_stride,
        const int64_t xdt_stride_tok, const int64_t xdt_stride_head, const int64_t xdt_stride_seq,
        const int64_t cb_stride_col, const int64_t cb_stride_group, const int64_t cb_stride_seq,
        const int64_t cs_stride_tok, const int64_t cs_stride_seq,
        const int64_t dst_stride_tok, const int64_t dst_stride_head, const int64_t dst_stride_seq) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    static_assert(TILE == 16, "WMMA tile must be 16x16");
    const int lane = threadIdx.x;
    const int h = blockIdx.y;
    const int s = blockIdx.z;
    const int n_d_tiles = (head_dim + TILE - 1) / TILE;
    const int tile_t = blockIdx.x / n_d_tiles;
    const int tile_d = blockIdx.x - tile_t * n_d_tiles;
    const int t0 = tile_t * TILE;
    const int d0 = tile_d * TILE;
    const int g = h / (n_head / n_group);

    const half * X_h = X_dt + (int64_t)s * xdt_stride_seq + (int64_t)h * xdt_stride_head;
    const float * CB_g = CB + (int64_t)s * cb_stride_seq + (int64_t)g * cb_stride_group;
    const float * cs_s = cs_chunk + (int64_t)s * cs_stride_seq;
    float * dst_s = dst_chunk + (int64_t)s * dst_stride_seq + (int64_t)h * dst_stride_head;

    __shared__ __align__(32) half tile_M[TILE * TILE];
    __shared__ __align__(32) half tile_X[TILE * TILE];
    __shared__ __align__(32) float tile_Y[TILE * TILE];

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TILE, TILE, TILE, float> acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);

    const float A_h = A[(int64_t)h * A_stride];
    const int k_limit = t0 + TILE < chunk_len ? t0 + TILE : chunk_len;
    for (int k0 = 0; k0 < k_limit; k0 += TILE) {
        for (int i = lane; i < TILE * TILE; i += WARP_SIZE) {
            const int row = i / TILE;
            const int col = i - row * TILE;
            const int t_out = t0 + row;
            const int t_in = k0 + col;
            float m = 0.0f;
            if (t_out < chunk_len && t_in < chunk_len && t_in <= t_out) {
                const float decay = __expf(A_h * (cs_s[(int64_t)t_out * cs_stride_tok + h] - cs_s[(int64_t)t_in * cs_stride_tok + h]));
                m = decay * CB_g[t_out + (int64_t)t_in * cb_stride_col];
            }
            tile_M[i] = __float2half_rn(m);

            const int x_t = k0 + row;
            const int x_d = d0 + col;
            tile_X[i] = x_t < chunk_len && x_d < head_dim ? X_h[(int64_t)x_t * xdt_stride_tok + x_d] : __float2half(0.0f);
        }
        __syncwarp();

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TILE, TILE, TILE, half, nvcuda::wmma::row_major> frag_M;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TILE, TILE, TILE, half, nvcuda::wmma::row_major> frag_X;
        nvcuda::wmma::load_matrix_sync(frag_M, tile_M, TILE);
        nvcuda::wmma::load_matrix_sync(frag_X, tile_X, TILE);
        nvcuda::wmma::mma_sync(acc, frag_M, frag_X, acc);
        __syncwarp();
    }

    nvcuda::wmma::store_matrix_sync(tile_Y, acc, TILE, nvcuda::wmma::mem_row_major);
    __syncwarp();
    for (int i = lane; i < TILE * TILE; i += WARP_SIZE) {
        const int row = i / TILE;
        const int col = i - row * TILE;
        const int t = t0 + row;
        const int d = d0 + col;
        if (t < chunk_len && d < head_dim) {
            dst_s[(int64_t)t * dst_stride_tok + d] += tile_Y[i];
        }
    }
#else
    GGML_UNUSED_VARS(X_dt, CB, cs_chunk, A, dst_chunk, chunk_len, head_dim, n_head, n_group, A_stride,
        xdt_stride_tok, xdt_stride_head, xdt_stride_seq, cb_stride_col, cb_stride_group, cb_stride_seq,
        cs_stride_tok, cs_stride_seq, dst_stride_tok, dst_stride_head, dst_stride_seq);
#endif
}

static void ssm_ssd_launch_fused_output_cuda(
        const half * X_dt, const float * CB, const float * cs_chunk, const float * A, float * dst_chunk,
        const int chunk_len, const int head_dim, const int n_head, const int n_group, const int n_seq,
        const int A_stride,
        const int64_t xdt_stride_tok, const int64_t xdt_stride_head, const int64_t xdt_stride_seq,
        const int64_t cb_stride_col, const int64_t cb_stride_group, const int64_t cb_stride_seq,
        const int64_t cs_stride_tok, const int64_t cs_stride_seq,
        const int64_t dst_stride_tok, const int64_t dst_stride_head, const int64_t dst_stride_seq,
        cudaStream_t stream) {
    constexpr int TILE = 16;
    const int n_tiles = ((chunk_len + TILE - 1) / TILE) * ((head_dim + TILE - 1) / TILE);
    const dim3 grid(n_tiles, n_head, n_seq);
    ssm_ssd_fused_output_kernel<TILE><<<grid, WARP_SIZE, 0, stream>>>(
        X_dt, CB, cs_chunk, A, dst_chunk, chunk_len, head_dim, n_head, n_group, A_stride,
        xdt_stride_tok, xdt_stride_head, xdt_stride_seq, cb_stride_col, cb_stride_group, cb_stride_seq,
        cs_stride_tok, cs_stride_seq, dst_stride_tok, dst_stride_head, dst_stride_seq);
    CUDA_CHECK(cudaGetLastError());
}

static bool ssm_ssd_fused_output_requested() {
    static const bool requested = []() {
        const char * value = getenv("GGML_CUDA_SSM_SSD_FUSION");
        const bool enabled = value == nullptr || strcmp(value, "0") != 0;
        if (enabled) {
            GGML_LOG_INFO("CUDA Mamba-2 SSD fused output requested\n");
        }
        return enabled;
    }();
    return requested;
}

// Scale running state in-place: s_cur *= decay_total(chunk).
// Called BEFORE cuBLAS state update (beta=1) to fuse inter-chunk decay.
// Eliminates the s_old buffer and D2D memcpy vs the old approach of:
//   memcpy(s_old, s_cur) -> cuBLAS(beta=0) -> s_cur += decay * s_old
// Grid: (ceil(d_state * head_dim / BLOCK), n_head, n_seqs)
template <int BLOCK_SIZE>
__global__ void ssm_ssd_scale_state_kernel(
        float * __restrict__ s_cur,            // {d_state, head_dim, n_head, n_seqs}
        const float * __restrict__ cs,         // {n_tok, n_head} cumulative dt sums
        const float * __restrict__ A,          // {1, n_head}
        const int d_state, const int head_dim, const int n_head,
        const int chunk_offset, const int chunk_len,
        const int n_tok_total, const int A_stride) {

    const int h = blockIdx.y;
    const int s = blockIdx.z;
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int state_per_head = d_state * head_dim;
    if (idx >= state_per_head) return;

    const float A_h = A[h * A_stride];
    const int cs_seq_off = s * n_tok_total * n_head;
    const float cs_base = (chunk_offset > 0) ? cs[cs_seq_off + (chunk_offset - 1) * n_head + h] : 0.0f;
    const float cs_last = cs[cs_seq_off + (chunk_offset + chunk_len - 1) * n_head + h] - cs_base;
    const float decay_total = __expf(A_h * cs_last);

    const int off = s * state_per_head * n_head + h * state_per_head + idx;
    s_cur[off] *= decay_total;
}

// Copy initial state from src0[ids[s]] into s_cur for each sequence.
// Grid: (ceil(d_state * head_dim * n_head / BLOCK), n_seqs)
template <int BLOCK_SIZE>
__global__ void ssm_ssd_init_state_kernel(
        const float * __restrict__ src0,       // {d_state, head_dim, n_head, n_rs}
        const int32_t * __restrict__ ids,      // {n_seqs}
        float * __restrict__ s_cur,            // {d_state, head_dim, n_head, n_seqs}
        const int state_size,                  // d_state * head_dim * n_head
        const int64_t s0_stride_seq) {         // elements between state rows
    const int s = blockIdx.y;
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= state_size) return;

    const float * s_src = src0 + (int64_t)ids[s] * s0_stride_seq;
    s_cur[s * state_size + idx] = s_src[idx];
}

#define SSM_SSD_PARALLEL_SCRATCH_MAX (256ull * 1024ull * 1024ull)

template <int BLOCK_SIZE>
__global__ void ssm_ssd_prepare_dt_chunks_kernel(
        const float * __restrict__ dt_raw,
        float * __restrict__ dt_sp_out,
        float * __restrict__ cs_out,
        const int n_head, const int n_tok,
        const int dt_stride_tok, const int dt_stride_seq) {
    const int h = blockIdx.x;
    const int s = blockIdx.y;
    const int chunk = blockIdx.z;
    const int t = chunk * BLOCK_SIZE + threadIdx.x;

    float dt = 0.0f;
    if (t < n_tok) {
        dt = dt_raw[s * dt_stride_seq + t * dt_stride_tok + h];
        dt = dt <= 20.0f ? log1pf(expf(dt)) : dt;
        dt_sp_out[(s * n_tok + t) * n_head + h] = dt;
    }

#ifdef USE_CUB
    using BlockScan = cub::BlockScan<float, BLOCK_SIZE>;
    __shared__ typename BlockScan::TempStorage scan_temp;
    float inclusive;
    BlockScan(scan_temp).InclusiveSum(dt, inclusive);
#else
    __shared__ float scan[BLOCK_SIZE];
    scan[threadIdx.x] = dt;
    __syncthreads();
    for (int offset = 1; offset < BLOCK_SIZE; offset *= 2) {
        const float add = threadIdx.x >= offset ? scan[threadIdx.x - offset] : 0.0f;
        __syncthreads();
        scan[threadIdx.x] += add;
        __syncthreads();
    }
    const float inclusive = scan[threadIdx.x];
#endif
    if (t < n_tok) {
        cs_out[(s * n_tok + t) * n_head + h] = inclusive;
    }
}

__global__ void ssm_ssd_parallel_cb_ptrs_kernel(
        const float * __restrict__ B,
        const float * __restrict__ C,
        float * __restrict__ CB,
        const float ** __restrict__ C_ptrs,
        const float ** __restrict__ B_ptrs,
        float ** __restrict__ CB_ptrs,
        const int base_chunk, const int wave_chunks,
        const int d_state, const int n_group, const int n_seq,
        const int B_stride_tok, const int B_stride_seq,
        const int C_stride_tok, const int C_stride_seq) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = n_seq * wave_chunks * n_group;
    if (idx >= count) {
        return;
    }

    const int g = idx % n_group;
    const int batch = idx / n_group;
    const int wk = batch % wave_chunks;
    const int s = batch / wave_chunks;
    const int chunk_offset = (base_chunk + wk) * SSM_SSD_CHUNK_SIZE;
    C_ptrs[idx] = C + s * C_stride_seq + chunk_offset * C_stride_tok + g * d_state;
    B_ptrs[idx] = B + s * B_stride_seq + chunk_offset * B_stride_tok + g * d_state;
    CB_ptrs[idx] = CB + (int64_t)idx * SSM_SSD_CHUNK_SIZE * SSM_SSD_CHUNK_SIZE;
}

__global__ void ssm_ssd_parallel_output_ptrs_kernel(
        const void * __restrict__ A,
        const void * __restrict__ B,
        float * __restrict__ dst,
        const void ** __restrict__ A_ptrs,
        const void ** __restrict__ B_ptrs,
        void ** __restrict__ dst_ptrs,
        const int base_chunk, const int wave_chunks,
        const int n_head, const int n_seq,
        const int n_tok, const int head_dim,
        const int64_t A_stride_seq, const int64_t A_stride_chunk, const int64_t A_stride_head,
        const int64_t B_stride_seq, const int64_t B_stride_chunk, const int64_t B_stride_head,
        const int A_element_size, const int B_element_size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = n_seq * wave_chunks * n_head;
    if (idx >= count) {
        return;
    }

    const int h = idx % n_head;
    const int batch = idx / n_head;
    const int wk = batch % wave_chunks;
    const int s = batch / wave_chunks;
    const int64_t a_off = s * A_stride_seq + wk * A_stride_chunk + h * A_stride_head;
    const int64_t b_off = s * B_stride_seq + wk * B_stride_chunk + h * B_stride_head;
    A_ptrs[idx] = (const char *)A + a_off * A_element_size;
    B_ptrs[idx] = (const char *)B + b_off * B_element_size;
    dst_ptrs[idx] = dst + ((int64_t)s * n_tok + (base_chunk + wk) * SSM_SSD_CHUNK_SIZE) * n_head * head_dim + h * head_dim;
}

template <int BLOCK_SIZE, typename T_matmul>
__global__ void ssm_ssd_parallel_pre_matmul_kernel(
        const float * __restrict__ cs,
        const float * __restrict__ dt_sp,
        const float * __restrict__ A,
        const float * __restrict__ x,
        const float * __restrict__ B,
        const float * __restrict__ C_src,
        T_matmul * __restrict__ X_dt,
        T_matmul * __restrict__ B_weighted,
        float * __restrict__ C_scaled,
        const float * __restrict__ CB,
        half * __restrict__ M_out,
        const int base_chunk, const int wave_chunks,
        const int head_dim, const int n_head, const int n_group,
        const int d_state, const int n_tok, const int A_stride,
        const int x_stride_tok, const int x_stride_seq,
        const int B_stride_tok, const int B_stride_seq,
        const int C_stride_tok, const int C_stride_seq,
        const bool write_all, const bool write_M) {
    const int h = blockIdx.y;
    const int batch = blockIdx.z;
    const int wk = batch % wave_chunks;
    const int s = batch / wave_chunks;
    const int chunk_offset = (base_chunk + wk) * SSM_SSD_CHUNK_SIZE;
    const int g = h / (n_head / n_group);
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int cs_seq_off = s * n_tok * n_head;
    const float A_h = A[h * A_stride];
    const float cs_last = cs[cs_seq_off + (chunk_offset + SSM_SSD_CHUNK_SIZE - 1) * n_head + h];
    const int64_t xdt_stride = (int64_t)SSM_SSD_CHUNK_SIZE * head_dim;
    const int64_t state_stride = (int64_t)d_state * SSM_SSD_CHUNK_SIZE;
    const int64_t M_stride = (int64_t)SSM_SSD_CHUNK_SIZE * SSM_SSD_CHUNK_SIZE;

    if (write_all && idx < SSM_SSD_CHUNK_SIZE * head_dim) {
        const int d = idx % head_dim;
        const int t = idx / head_dim;
        const float x_val = x[s * x_stride_seq + (chunk_offset + t) * x_stride_tok + h * head_dim + d];
        const float dt_val = dt_sp[cs_seq_off + (chunk_offset + t) * n_head + h];
        X_dt[(batch * n_head + h) * xdt_stride + idx] = (T_matmul)(x_val * dt_val);
    }

    if (idx < d_state * SSM_SSD_CHUNK_SIZE) {
        const int n = idx % d_state;
        const int t = idx / d_state;
        const float cs_t = cs[cs_seq_off + (chunk_offset + t) * n_head + h];
        if (write_all) {
            const float B_val = B[s * B_stride_seq + (chunk_offset + t) * B_stride_tok + g * d_state + n];
            B_weighted[(batch * n_head + h) * state_stride + idx] = (T_matmul)(B_val * __expf(A_h * (cs_last - cs_t)));
        } else {
            const float C_val = C_src[s * C_stride_seq + (chunk_offset + t) * C_stride_tok + g * d_state + n];
            C_scaled[(batch * n_head + h) * state_stride + idx] = C_val * __expf(A_h * cs_t);
        }
    }

    if (write_all && write_M && idx < SSM_SSD_CHUNK_SIZE * SSM_SSD_CHUNK_SIZE) {
        const int t_out = idx % SSM_SSD_CHUNK_SIZE;
        const int t_in = idx / SSM_SSD_CHUNK_SIZE;
        half val = __float2half(0.0f);
        if (t_in <= t_out) {
            const float cs_out = cs[cs_seq_off + (chunk_offset + t_out) * n_head + h];
            const float cs_in = cs[cs_seq_off + (chunk_offset + t_in) * n_head + h];
            const float cb = CB[((int64_t)batch * n_group + g) * M_stride + t_out + t_in * SSM_SSD_CHUNK_SIZE];
            val = __float2half(__expf(A_h * (cs_out - cs_in)) * cb);
        }
        M_out[((int64_t)batch * n_head + h) * M_stride + idx] = val;
    }
}

template <int BLOCK_SIZE>
__global__ void ssm_ssd_parallel_state_passing_kernel(
        const float * __restrict__ src0,
        const int32_t * __restrict__ ids,
        const float * __restrict__ cs,
        const float * __restrict__ A,
        float * __restrict__ chunk_states,
        float * __restrict__ final_states,
        const int64_t s0_stride_seq,
        const int d_state, const int head_dim, const int n_head,
        const int n_chunks, const int n_tok, const int A_stride) {
    const int s = blockIdx.y;
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int state_per_head = d_state * head_dim;
    const int state_size = state_per_head * n_head;
    if (idx >= state_size) {
        return;
    }

    const int h = idx / state_per_head;
    float state = src0[(int64_t)ids[s] * s0_stride_seq + idx];
    for (int k = 0; k < n_chunks; ++k) {
        const int64_t off = ((int64_t)s * n_chunks + k) * state_size + idx;
        const float delta = chunk_states[off];
        chunk_states[off] = state;
        const int t_last = (k + 1) * SSM_SSD_CHUNK_SIZE - 1;
        const float chunk_sum = cs[(s * n_tok + t_last) * n_head + h];
        state = __expf(A[h * A_stride] * chunk_sum) * state + delta;
    }
    final_states[(int64_t)s * state_size + idx] = state;
}

static bool ssm_scan_ssd_parallel_f32_cuda(
        ggml_backend_cuda_context & ctx,
        const float * src0_d, const float * src1_d, const float * src2_d, const float * src3_d,
        const float * src4_d, const float * src5_d, const int32_t * src6_d, float * dst_d,
        const int64_t s0_stride_seq,
        const int x_stride_tok,  const int x_stride_seq,
        const int dt_stride_tok, const int dt_stride_seq,
        const int A_stride,
        const int B_stride_tok,  const int B_stride_seq,
        const int C_stride_tok,  const int C_stride_seq,
        const int64_t s_off, const int64_t d_state, const int64_t head_dim,
        const int64_t n_head, const int64_t n_group, const int64_t n_tok, const int64_t n_seq) {
    if (n_tok % SSM_SSD_CHUNK_SIZE != 0 || B_stride_tok < d_state * n_group || C_stride_tok < d_state * n_group) {
        return false;
    }

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool use_fused_output = ssm_ssd_fused_output_requested()
        && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING;

    const int64_t n_chunks = n_tok / SSM_SSD_CHUNK_SIZE;
    const int64_t state_per_head = d_state * head_dim;
    const uint64_t dt_cs_bytes = 2ull * n_tok * n_head * n_seq * sizeof(float);
    const uint64_t chunk_states_bytes = (uint64_t)n_chunks * n_seq * n_head * state_per_head * sizeof(float);
    const uint64_t fixed_bytes = dt_cs_bytes + chunk_states_bytes;
    const uint64_t pointer_bytes_per_chunk = (uint64_t)3 * n_seq * (n_group + n_head) * sizeof(void *);
    const uint64_t wave_bytes_per_chunk = (uint64_t)n_seq * (
        SSM_SSD_CHUNK_SIZE * SSM_SSD_CHUNK_SIZE * n_group * sizeof(float) +
        SSM_SSD_CHUNK_SIZE * head_dim * n_head * sizeof(half) +
        d_state * SSM_SSD_CHUNK_SIZE * n_head * sizeof(half) +
        d_state * SSM_SSD_CHUNK_SIZE * n_head * sizeof(float) +
        (use_fused_output ? 0 : SSM_SSD_CHUNK_SIZE * SSM_SSD_CHUNK_SIZE * n_head * sizeof(half))) + pointer_bytes_per_chunk;
    if (fixed_bytes >= SSM_SSD_PARALLEL_SCRATCH_MAX || wave_bytes_per_chunk == 0) {
        return false;
    }
    const int64_t wave_capacity = (SSM_SSD_PARALLEL_SCRATCH_MAX - fixed_bytes) / wave_bytes_per_chunk;
    if (wave_capacity < 1) {
        return false;
    }
    const int64_t wave_chunks_max = wave_capacity < n_chunks ? wave_capacity : n_chunks;

    cudaStream_t stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();
    const int64_t d_inner = head_dim * n_head;
    const int64_t state_size = state_per_head * n_head;
    const float alpha_one = 1.0f;
    const float beta_zero = 0.0f;
    const float beta_one = 1.0f;
    const int64_t chunk_matrix = SSM_SSD_CHUNK_SIZE * SSM_SSD_CHUNK_SIZE;
    using matmul_t = half;

    ggml_cuda_pool_alloc<float> dt_sp_buf(ctx.pool(), n_tok * n_head * n_seq);
    ggml_cuda_pool_alloc<float> cs_buf(ctx.pool(), n_tok * n_head * n_seq);
    ggml_cuda_pool_alloc<float> chunk_states_buf(ctx.pool(), n_chunks * n_seq * state_size);
    ggml_cuda_pool_alloc<float> CB_buf(ctx.pool(), wave_chunks_max * n_seq * n_group * chunk_matrix);
    ggml_cuda_pool_alloc<matmul_t> X_dt_buf(ctx.pool(), wave_chunks_max * n_seq * n_head * SSM_SSD_CHUNK_SIZE * head_dim);
    ggml_cuda_pool_alloc<matmul_t> B_w_buf(ctx.pool(), wave_chunks_max * n_seq * n_head * d_state * SSM_SSD_CHUNK_SIZE);
    ggml_cuda_pool_alloc<float> C_s_buf(ctx.pool(), wave_chunks_max * n_seq * n_head * d_state * SSM_SSD_CHUNK_SIZE);
    ggml_cuda_pool_alloc<half> M_buf(ctx.pool());
    if (!use_fused_output) {
        M_buf.alloc(wave_chunks_max * n_seq * n_head * chunk_matrix);
    }
    ggml_cuda_pool_alloc<const float *> cb_src_ptrs(ctx.pool(), 2 * wave_chunks_max * n_seq * n_group);
    ggml_cuda_pool_alloc<float *> cb_dst_ptrs(ctx.pool(), wave_chunks_max * n_seq * n_group);
    ggml_cuda_pool_alloc<const void *> output_src_ptrs(ctx.pool(), 2 * wave_chunks_max * n_seq * n_head);
    ggml_cuda_pool_alloc<void *> output_dst_ptrs(ctx.pool(), wave_chunks_max * n_seq * n_head);

    float * dt_sp = dt_sp_buf.get();
    float * cs = cs_buf.get();
    float * chunk_states = chunk_states_buf.get();
    float * CB = CB_buf.get();
    matmul_t * X_dt = X_dt_buf.get();
    matmul_t * B_weighted = B_w_buf.get();
    float * C_scaled = C_s_buf.get();
    half * M_mat = M_buf.get();
    float * final_states = (float *)((char *)dst_d + s_off);

    if (use_fused_output) {
        CUDA_CHECK(cudaMemsetAsync(dst_d, 0, n_tok * d_inner * n_seq * sizeof(float), stream));
    }

    {
        dim3 grid(n_head, n_seq, n_chunks);
        ssm_ssd_prepare_dt_chunks_kernel<SSM_SSD_CHUNK_SIZE><<<grid, SSM_SSD_CHUNK_SIZE, 0, stream>>>(
            src2_d, dt_sp, cs, n_head, n_tok, dt_stride_tok, dt_stride_seq);
        CUDA_CHECK(cudaGetLastError());
    }

    for (int64_t base_chunk = 0; base_chunk < n_chunks; base_chunk += wave_chunks_max) {
        const int wave_chunks = (int)((base_chunk + wave_chunks_max <= n_chunks) ? wave_chunks_max : n_chunks - base_chunk);
        const int cb_count = n_seq * wave_chunks * n_group;
        {
            constexpr int BLOCK = 256;
            ssm_ssd_parallel_cb_ptrs_kernel<<<(cb_count + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
                src4_d, src5_d, CB,
                cb_src_ptrs.get(), cb_src_ptrs.get() + wave_chunks_max * n_seq * n_group, cb_dst_ptrs.get(),
                base_chunk, wave_chunks, d_state, n_group, n_seq,
                B_stride_tok, B_stride_seq, C_stride_tok, C_stride_seq);
            CUDA_CHECK(cudaGetLastError());
        }
        CUBLAS_CHECK(cublasSgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            SSM_SSD_CHUNK_SIZE, SSM_SSD_CHUNK_SIZE, d_state,
            &alpha_one,
            cb_src_ptrs.get(), C_stride_tok,
            cb_src_ptrs.get() + wave_chunks_max * n_seq * n_group, B_stride_tok,
            &beta_zero,
            cb_dst_ptrs.get(), SSM_SSD_CHUNK_SIZE,
            cb_count));

        {
            constexpr int BLOCK = 256;
            int64_t max_work = SSM_SSD_CHUNK_SIZE * head_dim;
            if (d_state * SSM_SSD_CHUNK_SIZE > max_work) max_work = d_state * SSM_SSD_CHUNK_SIZE;
            if (!use_fused_output && chunk_matrix > max_work) max_work = chunk_matrix;
            dim3 grid((max_work + BLOCK - 1) / BLOCK, n_head, n_seq * wave_chunks);
            ssm_ssd_parallel_pre_matmul_kernel<BLOCK, matmul_t><<<grid, BLOCK, 0, stream>>>(
                cs, dt_sp, src3_d, src1_d, src4_d, src5_d,
                X_dt, B_weighted, C_scaled, CB, M_mat,
                base_chunk, wave_chunks, head_dim, n_head, n_group, d_state, n_tok, A_stride,
                x_stride_tok, x_stride_seq, B_stride_tok, B_stride_seq, C_stride_tok, C_stride_seq, true, !use_fused_output);
            CUDA_CHECK(cudaGetLastError());
        }

        const int64_t xdt_stride = SSM_SSD_CHUNK_SIZE * head_dim;
        const int64_t weighted_stride = d_state * SSM_SSD_CHUNK_SIZE;
        const int output_count = n_seq * wave_chunks * n_head;
        if (!use_fused_output) {
            constexpr int BLOCK = 256;
            ssm_ssd_parallel_output_ptrs_kernel<<<(output_count + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
                X_dt, M_mat, dst_d,
                output_src_ptrs.get(), output_src_ptrs.get() + wave_chunks_max * n_seq * n_head, output_dst_ptrs.get(),
                base_chunk, wave_chunks, n_head, n_seq, n_tok, head_dim,
                wave_chunks * n_head * xdt_stride, n_head * xdt_stride, xdt_stride,
                wave_chunks * n_head * chunk_matrix, n_head * chunk_matrix, chunk_matrix,
                sizeof(half), sizeof(half));
            CUDA_CHECK(cudaGetLastError());
            CUBLAS_CHECK(cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                head_dim, SSM_SSD_CHUNK_SIZE, SSM_SSD_CHUNK_SIZE,
                &alpha_one,
                output_src_ptrs.get(), CUDA_R_16F, head_dim,
                output_src_ptrs.get() + wave_chunks_max * n_seq * n_head, CUDA_R_16F, SSM_SSD_CHUNK_SIZE,
                &beta_zero,
                output_dst_ptrs.get(), CUDA_R_32F, d_inner,
                output_count, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }
        for (int64_t s = 0; s < n_seq; ++s) {
            const int wave_head_count = wave_chunks * n_head;
            CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                d_state, head_dim, SSM_SSD_CHUNK_SIZE,
                &alpha_one,
                B_weighted + s * wave_chunks * n_head * weighted_stride, CUDA_R_16F, d_state, weighted_stride,
                X_dt + s * wave_chunks * n_head * xdt_stride, CUDA_R_16F, head_dim, xdt_stride,
                &beta_zero,
                chunk_states + (s * n_chunks + base_chunk) * state_size, CUDA_R_32F, d_state, state_per_head,
                wave_head_count, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

            if (use_fused_output) {
                for (int wk = 0; wk < wave_chunks; ++wk) {
                    const int64_t batch = s * wave_chunks + wk;
                    const int64_t chunk_offset = (base_chunk + wk) * SSM_SSD_CHUNK_SIZE;
                    ssm_ssd_launch_fused_output_cuda(
                        X_dt + batch * n_head * xdt_stride,
                        CB + batch * n_group * chunk_matrix,
                        cs + (s * n_tok + chunk_offset) * n_head,
                        src3_d,
                        dst_d + (s * n_tok + chunk_offset) * d_inner,
                        SSM_SSD_CHUNK_SIZE, head_dim, n_head, n_group, 1, A_stride,
                        head_dim, xdt_stride, xdt_stride * n_head,
                        SSM_SSD_CHUNK_SIZE, chunk_matrix, chunk_matrix * n_group,
                        n_head, n_tok * n_head,
                        d_inner, head_dim, n_tok * d_inner,
                        stream);
                }
            }
        }
    }

    {
        constexpr int BLOCK = 256;
        dim3 grid((state_size + BLOCK - 1) / BLOCK, n_seq);
        ssm_ssd_parallel_state_passing_kernel<BLOCK><<<grid, BLOCK, 0, stream>>>(
            src0_d, src6_d, cs, src3_d, chunk_states, final_states,
            s0_stride_seq, d_state, head_dim, n_head, n_chunks, n_tok, A_stride);
        CUDA_CHECK(cudaGetLastError());
    }

    for (int64_t base_chunk = 0; base_chunk < n_chunks; base_chunk += wave_chunks_max) {
        const int wave_chunks = (int)((base_chunk + wave_chunks_max <= n_chunks) ? wave_chunks_max : n_chunks - base_chunk);
        {
            constexpr int BLOCK = 256;
            const int64_t work = d_state * SSM_SSD_CHUNK_SIZE;
            dim3 grid((work + BLOCK - 1) / BLOCK, n_head, n_seq * wave_chunks);
            ssm_ssd_parallel_pre_matmul_kernel<BLOCK, matmul_t><<<grid, BLOCK, 0, stream>>>(
                cs, dt_sp, src3_d, src1_d, src4_d, src5_d,
                X_dt, B_weighted, C_scaled, CB, M_mat,
                base_chunk, wave_chunks, head_dim, n_head, n_group, d_state, n_tok, A_stride,
                x_stride_tok, x_stride_seq, B_stride_tok, B_stride_seq, C_stride_tok, C_stride_seq, false, false);
            CUDA_CHECK(cudaGetLastError());
        }
        const int64_t scaled_stride = d_state * SSM_SSD_CHUNK_SIZE;
        const int output_count = n_seq * wave_chunks * n_head;
        {
            constexpr int BLOCK = 256;
            ssm_ssd_parallel_output_ptrs_kernel<<<(output_count + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
                chunk_states + base_chunk * state_size, C_scaled, dst_d,
                output_src_ptrs.get(), output_src_ptrs.get() + wave_chunks_max * n_seq * n_head, output_dst_ptrs.get(),
                base_chunk, wave_chunks, n_head, n_seq, n_tok, head_dim,
                n_chunks * state_size, state_size, state_per_head,
                wave_chunks * n_head * scaled_stride, n_head * scaled_stride, scaled_stride,
                sizeof(float), sizeof(float));
            CUDA_CHECK(cudaGetLastError());
        }
        CUBLAS_CHECK(cublasGemmBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            head_dim, SSM_SSD_CHUNK_SIZE, d_state,
            &alpha_one,
            output_src_ptrs.get(), CUDA_R_32F, d_state,
            output_src_ptrs.get() + wave_chunks_max * n_seq * n_head, CUDA_R_32F, d_state,
            &beta_one,
            output_dst_ptrs.get(), CUDA_R_32F, d_inner,
            output_count, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
    return true;
}

// SSD (State Space Duality) dispatch for Mamba-2 prefill.
// Chunked matmuls: CB, intra-chunk Y, S@C, B@X_dt.
// All strides are in elements (floats), not bytes.
static void ssm_scan_ssd_f32_cuda(
        ggml_backend_cuda_context & ctx,
        const float * src0_d, const float * src1_d, const float * src2_d, const float * src3_d,
        const float * src4_d, const float * src5_d, const int32_t * src6_d, float * dst_d,
        const int64_t s0_stride_seq,                                   // state (src0) stride between seqs
        const int x_stride_tok,  const int x_stride_seq,               // x (src1) strides
        const int dt_stride_tok, const int dt_stride_seq,              // dt (src2) strides
        const int A_stride,                                            // A (src3) stride between heads
        const int B_stride_tok,  const int B_stride_seq,               // B (src4) strides
        const int C_stride_tok,  const int C_stride_seq,               // C (src5) strides
        const int64_t s_off, const int64_t d_state, const int64_t head_dim,
        const int64_t n_head, const int64_t n_group, const int64_t n_tok, const int64_t n_seq) {

    static const bool use_parallel = []() {
        const char * value = getenv("GGML_CUDA_SSM_SSD_PARALLEL");
        return value == nullptr || strcmp(value, "0") != 0;
    }();
    if (use_parallel && ssm_scan_ssd_parallel_f32_cuda(ctx,
            src0_d, src1_d, src2_d, src3_d, src4_d, src5_d, src6_d, dst_d,
            s0_stride_seq, x_stride_tok, x_stride_seq, dt_stride_tok, dt_stride_seq, A_stride,
            B_stride_tok, B_stride_seq, C_stride_tok, C_stride_seq,
            s_off, d_state, head_dim, n_head, n_group, n_tok, n_seq)) {
        return;
    }

    cudaStream_t stream = ctx.stream();
    const int64_t d_inner = head_dim * n_head;

    const int64_t chunk_size = SSM_SSD_CHUNK_SIZE;
    const int64_t n_chunks = (n_tok + chunk_size - 1) / chunk_size;

    const int64_t state_per_head = d_state * head_dim;

    using matmul_t = half;
    static constexpr cudaDataType_t matmul_dtype = CUDA_R_16F;

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool use_fused_output = ssm_ssd_fused_output_requested()
        && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING;

    ggml_cuda_pool_alloc<float>    dt_sp_buf(ctx.pool(), n_tok * n_head * n_seq);
    ggml_cuda_pool_alloc<float>    cs_buf(ctx.pool(), n_tok * n_head * n_seq);
    ggml_cuda_pool_alloc<float>    CB_buf(ctx.pool(), chunk_size * chunk_size * n_group * n_seq);
    ggml_cuda_pool_alloc<matmul_t> X_dt_buf(ctx.pool(), chunk_size * head_dim * n_head * n_seq);
    ggml_cuda_pool_alloc<matmul_t> B_w_buf(ctx.pool(), d_state * chunk_size * n_head * n_seq);
    ggml_cuda_pool_alloc<float>    C_s_buf(ctx.pool(), d_state * chunk_size * n_head * n_seq);
    float    * dt_sp      = dt_sp_buf.get();
    float    * cs         = cs_buf.get();
    float    * CB         = CB_buf.get();
    matmul_t * X_dt       = X_dt_buf.get();
    matmul_t * B_weighted = B_w_buf.get();
    float    * C_scaled   = C_s_buf.get();
    float    * s_cur      = (float *)((char *)dst_d + s_off); // write state directly to dst

    // Step 1: softplus(dt) and parallel prefix sum over full sequence
    {
        dim3 grid(n_head, n_seq);
        ssm_ssd_prepare_dt_kernel<SSM_SSD_DT_BLOCK, SSM_SSD_DT_MAX_ITEMS><<<grid, SSM_SSD_DT_BLOCK, 0, stream>>>(
            src2_d, dt_sp, cs, n_head, n_tok, dt_stride_tok, dt_stride_seq);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 2: initialize running state from src0[ids[s]]
    {
        constexpr int BLOCK = 256;
        const int64_t state_size = d_state * head_dim * n_head;
        dim3 grid((state_size + BLOCK - 1) / BLOCK, n_seq);
        ssm_ssd_init_state_kernel<BLOCK><<<grid, BLOCK, 0, stream>>>(
            src0_d, src6_d, s_cur, state_size, s0_stride_seq);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 3: chunked SSD loop
    // Per chunk: pre_matmul + CB, intra-chunk Y, S@C, state update + scale_state
    cublasHandle_t handle = ctx.cublas_handle();
    const float alpha_one  = 1.0f;
    const float beta_zero  = 0.0f;
    const float beta_one   = 1.0f;
    const int lda_C_src = C_stride_tok;  // leading dim for C in CB = C^T @ B
    const int ldb_B_src = B_stride_tok;  // leading dim for B in CB = C^T @ B

    // The fallback reuses this causal M scratch buffer across chunks.
    const int64_t n_M_max = chunk_size * chunk_size;
    ggml_cuda_pool_alloc<half> M_buf(ctx.pool());
    half * M_mat = use_fused_output ? nullptr : M_buf.alloc(n_M_max * n_head * n_seq);

    for (int64_t k = 0; k < n_chunks; k++) {
        const int64_t chunk_offset = k * chunk_size;
        const int64_t chunk_len = (chunk_offset + chunk_size <= n_tok) ? chunk_size : (n_tok - chunk_offset);

        // 3a: CB = C^T @ B per group
        for (int64_t s = 0; s < n_seq; s++) {
            const float * C_s = src5_d + s * C_stride_seq + chunk_offset * C_stride_tok;
            const float * B_s = src4_d + s * B_stride_seq + chunk_offset * B_stride_tok;
            float *      CB_s = CB + s * chunk_len * chunk_len * n_group;

            if (n_group == 1) {
                CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    chunk_len, chunk_len, d_state,
                    &alpha_one, C_s, lda_C_src, B_s, ldb_B_src,
                    &beta_zero, CB_s, (int)chunk_len));
            } else {
                CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    chunk_len, chunk_len, d_state,
                    &alpha_one,
                    C_s, CUDA_R_32F, lda_C_src, d_state,
                    B_s, CUDA_R_32F, ldb_B_src, d_state,
                    &beta_zero,
                    CB_s, CUDA_R_32F, (int)chunk_len, (long long)(chunk_len * chunk_len),
                    n_group,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            }
        }

        // 3b: prepare X_dt, B_weighted, C_scaled, and fallback M.
        const int64_t n_M = chunk_len * chunk_len;
        {
            constexpr int BLOCK = 256;
            const int64_t n_xdt   = chunk_len * head_dim;
            const int64_t n_bw    = d_state * chunk_len;
            int64_t max_work = n_xdt;
            if (n_bw  > max_work) max_work = n_bw;
            if (!use_fused_output && n_M > max_work) max_work = n_M;
            dim3 grid((max_work + BLOCK - 1) / BLOCK, n_head, n_seq);
            if (use_fused_output) {
                ssm_ssd_pre_matmul_kernel<BLOCK, matmul_t, false><<<grid, BLOCK, 0, stream>>>(
                    cs, dt_sp, src3_d, src1_d, src4_d, src5_d,
                    X_dt, B_weighted, C_scaled,
                    CB, M_mat,
                    chunk_len, head_dim, n_head, n_group, d_state, A_stride,
                    x_stride_tok, x_stride_seq, B_stride_tok, B_stride_seq, C_stride_tok, C_stride_seq,
                    chunk_offset, n_tok);
            } else {
                ssm_ssd_pre_matmul_kernel<BLOCK, matmul_t, true><<<grid, BLOCK, 0, stream>>>(
                    cs, dt_sp, src3_d, src1_d, src4_d, src5_d,
                    X_dt, B_weighted, C_scaled,
                    CB, M_mat,
                    chunk_len, head_dim, n_head, n_group, d_state, A_stride,
                    x_stride_tok, x_stride_seq, B_stride_tok, B_stride_seq, C_stride_tok, C_stride_seq,
                    chunk_offset, n_tok);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        // 3c: dst = S_cur^T @ C_scaled (state contribution)
        {
            const int64_t stride_S  = state_per_head;
            const int64_t stride_Cs = d_state * chunk_len;

            for (int64_t s = 0; s < n_seq; s++) {
                float * dst_chunk = dst_d + s * d_inner * n_tok + chunk_offset * d_inner;

                CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    head_dim, chunk_len, d_state,
                    &alpha_one,
                    s_cur    + s * stride_S  * n_head, CUDA_R_32F, d_state, stride_S,
                    C_scaled + s * stride_Cs * n_head, CUDA_R_32F, d_state, stride_Cs,
                    &beta_zero,
                    dst_chunk, CUDA_R_32F, d_inner, head_dim,
                    n_head,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            }
        }

        // 3d: add the intra-chunk contribution to the result from 3c.
        if (use_fused_output) {
            const int64_t stride_X_h = (int64_t)chunk_len * head_dim;
            const int64_t stride_CB_g = (int64_t)chunk_len * chunk_len;
            ssm_ssd_launch_fused_output_cuda(
                X_dt, CB, cs + chunk_offset * n_head, src3_d, dst_d + chunk_offset * d_inner,
                chunk_len, head_dim, n_head, n_group, n_seq, A_stride,
                head_dim, stride_X_h, stride_X_h * n_head,
                chunk_len, stride_CB_g, stride_CB_g * n_group,
                n_head, n_tok * n_head,
                d_inner, head_dim, n_tok * d_inner,
                stream);
        } else {
            // M is stored as M[t_out, t_in] (lower-triangular), transpose needed for Y = X @ M^T.
            const int64_t stride_M = n_M;
            const int64_t stride_X_h = (int64_t)chunk_len * head_dim;

            for (int64_t s = 0; s < n_seq; s++) {
                float * dst_chunk = dst_d + s * d_inner * n_tok + chunk_offset * d_inner;
                CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    head_dim, chunk_len, chunk_len,
                    &alpha_one,
                    X_dt       + s * stride_X_h * n_head, matmul_dtype, head_dim, stride_X_h,
                    M_mat      + s * stride_M   * n_head, matmul_dtype, chunk_len, stride_M,
                    &beta_one,
                    dst_chunk, CUDA_R_32F, d_inner, head_dim,
                    n_head,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            }
        }

        // 3e: s_cur = B_weighted @ X_dt^T + decay_total * s_cur_old (state update)
        {
            // Scale s_cur in-place by per-head decay_total BEFORE cuBLAS overwrites it
            constexpr int BLOCK = 256;
            dim3 grid((state_per_head + BLOCK - 1) / BLOCK, n_head, n_seq);
            ssm_ssd_scale_state_kernel<BLOCK><<<grid, BLOCK, 0, stream>>>(
                s_cur, cs, src3_d,
                d_state, head_dim, n_head,
                chunk_offset, chunk_len, n_tok, A_stride);
            CUDA_CHECK(cudaGetLastError());

            // cuBLAS with beta=1: s_cur = B_weighted @ X_dt^T + 1.0 * s_cur (already scaled)
            const int64_t stride_Bw = d_state * chunk_len;
            const int64_t stride_X  = chunk_len * head_dim;
            const int64_t stride_S  = state_per_head;

            for (int64_t s = 0; s < n_seq; s++) {
                // X_dt is d-fastest {hd, C}, read as OP_T to get {C, hd}
                CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    d_state, head_dim, chunk_len,
                    &alpha_one,
                    B_weighted + s * stride_Bw * n_head, matmul_dtype, d_state, stride_Bw,
                    X_dt       + s * stride_X  * n_head, matmul_dtype, head_dim, stride_X,
                    &beta_one,
                    s_cur      + s * stride_S  * n_head, CUDA_R_32F, d_state, stride_S,
                    n_head,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            }
        }
    }
}

template <int BLOCK_SIZE>
__global__ void ssm_scan_identity_ids_kernel(int32_t * ids, const int n_seq) {
    const int s = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (s < n_seq) {
        ids[s] = s;
    }
}

template <int BLOCK_SIZE>
__global__ void ssm_scan_rollback_scatter_kernel(
        const float * prefix, const float * tail, float * dst,
        const int64_t prefix_y_elems, const int64_t tail_y_elems,
        const int64_t prefix_seq_elems, const int64_t tail_seq_elems, const int64_t dst_seq_elems,
        const int64_t prefix_tokens_elems, const int64_t s_off_elems, const int64_t total_elems) {
    const int64_t i = (int64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= total_elems) {
        return;
    }

    if (i < prefix_y_elems) {
        const int64_t s = i / prefix_seq_elems;
        const int64_t j = i - s * prefix_seq_elems;
        dst[s * dst_seq_elems + j] = prefix[i];
    } else if (i < prefix_y_elems + tail_y_elems) {
        const int64_t j = i - prefix_y_elems;
        const int64_t s = j / tail_seq_elems;
        const int64_t k = j - s * tail_seq_elems;
        dst[s * dst_seq_elems + prefix_tokens_elems + k] = tail[j];
    } else {
        const int64_t j = i - prefix_y_elems - tail_y_elems;
        dst[s_off_elems + j] = tail[tail_y_elems + j];
    }
}

static bool ssm_scan_rollback_scratch_supported(
        const int64_t d_state, const int64_t head_dim, const int64_t n_head,
        const int64_t n_tok, const int64_t n_seq, const int64_t K, const int64_t prefix_tokens) {
    if (d_state <= 0 || head_dim <= 0 || n_head <= 0 || n_tok <= 0 || n_seq <= 0 || n_seq > INT_MAX || K <= 0 || prefix_tokens <= 0) {
        return false;
    }
    if (prefix_tokens >= n_tok) {
        return false;
    }

    const size_t max_elems = (size_t)INT64_MAX / sizeof(float);
    auto checked_mul = [max_elems](size_t a, size_t b, size_t & result) {
        if (a != 0 && b > max_elems / a) {
            return false;
        }
        result = a * b;
        return true;
    };
    auto checked_add = [max_elems](size_t a, size_t b, size_t & result) {
        if (b > max_elems - a) {
            return false;
        }
        result = a + b;
        return true;
    };

    size_t d_inner;
    size_t state_per_head;
    size_t state_size;
    size_t prefix_y_elems;
    size_t tail_y_elems;
    size_t state_elems;
    size_t prefix_alloc;
    size_t tail_alloc;
    size_t scatter_elems;
    const size_t tail_tokens = n_tok - prefix_tokens;

    return checked_mul((size_t)head_dim, (size_t)n_head, d_inner)
        && checked_mul((size_t)d_state, (size_t)head_dim, state_per_head)
        && checked_mul((size_t)d_state, d_inner, state_size)
        && state_per_head <= INT_MAX / sizeof(float)
        && state_size <= INT_MAX / sizeof(float)
        && checked_mul((size_t)prefix_tokens, d_inner, prefix_y_elems)
        && checked_mul(prefix_y_elems, (size_t)n_seq, prefix_y_elems)
        && checked_mul(tail_tokens, d_inner, tail_y_elems)
        && checked_mul(tail_y_elems, (size_t)n_seq, tail_y_elems)
        && checked_mul((size_t)K, state_size, state_elems)
        && checked_mul(state_elems, (size_t)n_seq, state_elems)
        && checked_mul(state_size, (size_t)n_seq, prefix_alloc)
        && checked_add(prefix_y_elems, prefix_alloc, prefix_alloc)
        && checked_add(tail_y_elems, state_elems, tail_alloc)
        && checked_add(prefix_y_elems, tail_y_elems, scatter_elems)
        && checked_add(scatter_elems, state_elems, scatter_elems)
        && scatter_elems <= (size_t)INT_MAX * 256;
}

static void ssm_scan_ssd_rollback_f32_cuda(
        ggml_backend_cuda_context & ctx,
        const float * src0_d, const float * src1_d, const float * src2_d, const float * src3_d,
        const float * src4_d, const float * src5_d, const int32_t * src6_d, float * dst_d,
        const int64_t s0_stride_seq,
        const int x_stride_tok,  const int x_stride_seq,
        const int dt_stride_tok, const int dt_stride_seq,
        const int A_stride,
        const int B_stride_tok,  const int B_stride_seq,
        const int C_stride_tok,  const int C_stride_seq,
        const int64_t d_state, const int64_t head_dim, const int64_t n_head, const int64_t n_group,
        const int64_t n_tok, const int64_t n_seq, const int64_t K, const int64_t prefix_tokens) {
    cudaStream_t stream = ctx.stream();

    const int64_t tail_tokens = n_tok - prefix_tokens;
    const int64_t d_inner = head_dim * n_head;
    const int64_t state_size = d_state * d_inner;
    const int64_t prefix_y_elems = prefix_tokens * d_inner * n_seq;
    const int64_t tail_y_elems = tail_tokens * d_inner * n_seq;
    const int64_t state_elems = K * state_size * n_seq;

    ggml_cuda_pool_alloc<float> prefix_buf(ctx.pool(), prefix_y_elems + state_size * n_seq);
    ggml_cuda_pool_alloc<float> tail_buf(ctx.pool(), tail_y_elems + state_elems);
    ggml_cuda_pool_alloc<int32_t> ids_buf(ctx.pool(), n_seq);

    float * prefix = prefix_buf.get();
    float * tail = tail_buf.get();
    int32_t * ids = ids_buf.get();

    ssm_scan_ssd_f32_cuda(ctx,
        src0_d, src1_d, src2_d, src3_d, src4_d, src5_d, src6_d, prefix,
        s0_stride_seq,
        x_stride_tok, x_stride_seq, dt_stride_tok, dt_stride_seq, A_stride,
        B_stride_tok, B_stride_seq, C_stride_tok, C_stride_seq,
        prefix_y_elems * sizeof(float), d_state, head_dim, n_head, n_group, prefix_tokens, n_seq);

    {
        constexpr int BLOCK = 256;
        ssm_scan_identity_ids_kernel<BLOCK><<<((int)n_seq + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(ids, (int)n_seq);
        CUDA_CHECK(cudaGetLastError());
    }

    const float * tail_src1 = src1_d + prefix_tokens * x_stride_tok;
    const float * tail_src2 = src2_d + prefix_tokens * dt_stride_tok;
    const float * tail_src4 = src4_d + prefix_tokens * B_stride_tok;
    const float * tail_src5 = src5_d + prefix_tokens * C_stride_tok;
    const float * prefix_state = prefix + prefix_y_elems;

    ssm_scan_f32_cuda(prefix_state, tail_src1, tail_src2, src3_d, tail_src4, tail_src5, ids, tail,
                      state_size / n_head * sizeof(float), state_size * sizeof(float),
                      x_stride_tok * sizeof(float), x_stride_seq * sizeof(float),
                      dt_stride_tok * sizeof(float), dt_stride_seq * sizeof(float), A_stride * sizeof(float),
                      B_stride_tok * sizeof(float), B_stride_seq * sizeof(float),
                      C_stride_tok * sizeof(float), C_stride_seq * sizeof(float),
                      tail_y_elems * sizeof(float), d_state, head_dim, n_head, n_group, tail_tokens, n_seq, K, stream);

    {
        constexpr int BLOCK = 256;
        const int64_t total_elems = prefix_y_elems + tail_y_elems + state_elems;
        ssm_scan_rollback_scatter_kernel<BLOCK><<<(total_elems + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
            prefix, tail, dst_d,
            prefix_y_elems, tail_y_elems,
            prefix_tokens * d_inner, tail_tokens * d_inner, n_tok * d_inner,
            prefix_tokens * d_inner, n_tok * d_inner * n_seq, total_elems);
        CUDA_CHECK(cudaGetLastError());
    }
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

void ggml_cuda_op_ssm_scan(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // s
    const struct ggml_tensor * src1 = dst->src[1];  // x
    const struct ggml_tensor * src2 = dst->src[2];  // dt
    const struct ggml_tensor * src3 = dst->src[3];  // A
    const struct ggml_tensor * src4 = dst->src[4];  // B
    const struct ggml_tensor * src5 = dst->src[5];  // C
    const struct ggml_tensor * src6 = dst->src[6];  // ids

    const int64_t nc  = src0->ne[0];  // d_state
    const int64_t nr  = src0->ne[1];  // head_dim or 1
    const int64_t nh  = src1->ne[1];  // n_head
    const int64_t ng  = src4->ne[1];  // n_group
    const int64_t n_t = src1->ne[2];  // number of tokens per sequence
    const int64_t n_s = src1->ne[3];  // number of sequences in the batch
    const int32_t K_param = ggml_get_op_params_i32(dst, 0);
    const int64_t K = K_param > 0 ? K_param : 1;

    const int64_t s_off = ggml_nelements(src1) * sizeof(float);

    GGML_ASSERT(ggml_nelements(src1) + K*nc*nr*nh*n_s == ggml_nelements(dst));
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src2->nb[0] == sizeof(float));
    GGML_ASSERT(src3->nb[0] == sizeof(float));
    GGML_ASSERT(src4->nb[0] == sizeof(float));
    GGML_ASSERT(src5->nb[0] == sizeof(float));
    GGML_ASSERT(src6->nb[0] == sizeof(int32_t));
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * src2_d = (const float *) src2->data;
    const float * src3_d = (const float *) src3->data;
    const float * src4_d = (const float *) src4->data;
    const float * src5_d = (const float *) src5->data;
    const int32_t * src6_d = (const int32_t *) src6->data;
    float *       dst_d  = (float *) dst->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src6->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // Byte strides are narrowed to int for both scan and SSD paths.
    GGML_ASSERT(src0->nb[2] <= (size_t)INT_MAX);
    GGML_ASSERT(src0->nb[3] <= (size_t)INT_MAX);
    GGML_ASSERT(src1->nb[2] <= (size_t)INT_MAX);
    GGML_ASSERT(src1->nb[3] <= (size_t)INT_MAX);
    GGML_ASSERT(src2->nb[1] <= (size_t)INT_MAX);
    GGML_ASSERT(src2->nb[2] <= (size_t)INT_MAX);
    GGML_ASSERT(src3->nb[1] <= (size_t)INT_MAX);
    GGML_ASSERT(src4->nb[2] <= (size_t)INT_MAX);
    GGML_ASSERT(src4->nb[3] <= (size_t)INT_MAX);
    GGML_ASSERT(src5->nb[2] <= (size_t)INT_MAX);
    GGML_ASSERT(src5->nb[3] <= (size_t)INT_MAX);

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // Mamba-2 with scalar A per head: use SSD matmul path for long sequences.
    // Requires NVIDIA Turing+ otherwise fallback to scan.
    const bool is_mamba2 = (src3->nb[1] == sizeof(float));
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool ssd_device_supported = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
    const bool ssd_layout_supported = src0->nb[1] == nc * sizeof(float)
                                   && src0->nb[2] == nc * nr * sizeof(float);
    const bool use_ssd = is_mamba2 && n_t > SSM_SSD_MIN_TOKENS
                      && K == 1
                      && n_t <= SSM_SSD_MAX_TOKENS
                      && ssd_device_supported
                      && nr % 8 == 0;  // cuBLAS requires 8-element (16-byte) alignment

    if (use_ssd) {
        // ssm_ssd_init_state_kernel uses flat linear indexing within each sequence,
        // so src0 must be fully contiguous across all inner dimensions.
        // The scan path handles non-contiguous nb[2] via src0_nb2 but does not handle nb[1].
        GGML_ASSERT(src0->nb[1] == nc         * sizeof(float));
        GGML_ASSERT(src0->nb[2] == nc * nr    * sizeof(float));

        ssm_scan_ssd_f32_cuda(ctx,
            src0_d, src1_d, src2_d, src3_d, src4_d, src5_d, src6_d, dst_d,
            (int64_t)(src0->nb[3] / sizeof(float)),
            (int)(src1->nb[2] / sizeof(float)), (int)(src1->nb[3] / sizeof(float)),
            (int)(src2->nb[1] / sizeof(float)), (int)(src2->nb[2] / sizeof(float)),
            (int)(src3->nb[1] / sizeof(float)),
            (int)(src4->nb[2] / sizeof(float)), (int)(src4->nb[3] / sizeof(float)),
            (int)(src5->nb[2] / sizeof(float)), (int)(src5->nb[3] / sizeof(float)),
            s_off, nc, nr, nh, ng, n_t, n_s);
        return;
    }

    static const char * ssd_rollback_env = getenv("GGML_CUDA_SSM_SSD_ROLLBACK");
    static const bool ssd_rollback_enabled = ssd_rollback_env == nullptr || strcmp(ssd_rollback_env, "0") != 0;
    const int64_t rollback_prefix = K > 1 && n_t >= K ? ((n_t - K) / SSM_SSD_CHUNK_SIZE) * SSM_SSD_CHUNK_SIZE : 0;
    const int64_t rollback_tail = n_t - rollback_prefix;
    const bool rollback_geometry_supported = rollback_prefix > SSM_SSD_MIN_TOKENS
                                          && rollback_prefix <= SSM_SSD_MAX_TOKENS
                                          && K <= SSM_SSD_CHUNK_SIZE
                                          && rollback_tail >= K
                                          && rollback_tail <= SSM_SSD_CHUNK_SIZE + K - 1;
    const bool use_ssd_rollback = ssd_rollback_enabled && is_mamba2 && K > 1
                               && ssd_device_supported
                               && nr % 8 == 0
                               && ssd_layout_supported
                               && rollback_geometry_supported
                               && ssm_scan_rollback_scratch_supported(nc, nr, nh, n_t, n_s, K, rollback_prefix);

    if (use_ssd_rollback) {
        ssm_scan_ssd_rollback_f32_cuda(ctx,
            src0_d, src1_d, src2_d, src3_d, src4_d, src5_d, src6_d, dst_d,
            (int64_t)(src0->nb[3] / sizeof(float)),
            (int)(src1->nb[2] / sizeof(float)), (int)(src1->nb[3] / sizeof(float)),
            (int)(src2->nb[1] / sizeof(float)), (int)(src2->nb[2] / sizeof(float)),
            (int)(src3->nb[1] / sizeof(float)),
            (int)(src4->nb[2] / sizeof(float)), (int)(src4->nb[3] / sizeof(float)),
            (int)(src5->nb[2] / sizeof(float)), (int)(src5->nb[3] / sizeof(float)),
            nc, nr, nh, ng, n_t, n_s, K, rollback_prefix);
        return;
    }
#endif
    ssm_scan_f32_cuda(src0_d, src1_d, src2_d, src3_d, src4_d, src5_d, src6_d, dst_d,
                      src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3], src2->nb[1], src2->nb[2],
                      src3->nb[1], src4->nb[2], src4->nb[3], src5->nb[2], src5->nb[3],
                      s_off, nc, nr, nh, ng, n_t, n_s, K, stream);
}
