/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#ifndef USE_ROCM
    #include <cub/block/block_load.cuh>
    #include <cub/block/block_store.cuh>
    #include <cub/block/block_scan.cuh>
    #include <cub/device/device_scan.cuh>
#else
    #include <hipcub/hipcub.hpp>
    namespace cub = hipcub;
#endif

#include "selective_scan2.h"
#include "selective_scan2_common.h"
#include "static_switch.h"

template<int kNThreads_, int kNItems_, int kNRows_, bool kIsEvenLen_,
         bool kIsVariableB_, bool kIsVariableC_,
         bool kHasZ_, typename input_t_, typename weight_t_>
struct Selective_Scan2_fwd_kernel_traits {
    static_assert(kNItems_ % 4 == 0);
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    // Setting MinBlocksPerMP to be 3 (instead of 2) for 128 threads improves occupancy.
    static constexpr int kMinBlocks = kNThreads < 128 ? 5 : 3;
    static constexpr int kNItems = kNItems_;
    static constexpr int kNRows = kNRows_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : constexpr_min(8, kNItems);
    static_assert(kNItems % kNElts == 0);
    static constexpr int kNLoads = kNItems / kNElts;
    static constexpr bool kIsComplex = std::is_same_v<weight_t, complex_t>;
    static constexpr bool kIsEvenLen = kIsEvenLen_;
    static constexpr bool kIsVariableB = kIsVariableB_;
    static constexpr bool kIsVariableC = kIsVariableC_;
    static constexpr bool kHasZ = kHasZ_;

    static constexpr bool kDirectIO = kIsEvenLen && kNLoads == 1;

    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using scan_t = std::conditional_t<!kIsComplex, float2, float4>;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE : cub::BLOCK_LOAD_DIRECT>;
    using BlockLoadWeightT = cub::BlockLoad<input_t, kNThreads, !kIsComplex ? kNItems : kNItems * 2, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightVecT = cub::BlockLoad<vec_t, kNThreads, !kIsComplex ? kNLoads : kNLoads * 2,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE  : cub::BLOCK_LOAD_DIRECT>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_STORE_WARP_TRANSPOSE : cub::BLOCK_STORE_DIRECT>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING>;
    using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    static constexpr int kSmemIOSize = custom_max({sizeof(typename BlockLoadT::TempStorage),
                                                 sizeof(typename BlockLoadVecT::TempStorage),
                                                 (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockLoadWeightT::TempStorage),
                                                 (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockLoadWeightVecT::TempStorage),
                                                 sizeof(typename BlockStoreT::TempStorage),
                                                 sizeof(typename BlockStoreVecT::TempStorage)});
    static constexpr int kSmemSize = kSmemIOSize + sizeof(typename BlockScanT::TempStorage);
};

// Part 1: Pre-kernel that computes everything up to the scan
template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void selective_scan2_fwd_pre_kernel(SSMParamsBase params,
                                    typename Ktraits::scan_t* thread_data_out,
                                    typename Ktraits::scan_t* running_prefix_in) {
    constexpr bool kIsComplex = Ktraits::kIsComplex;
    constexpr bool kIsVariableB = Ktraits::kIsVariableB;
    constexpr bool kIsVariableC = Ktraits::kIsVariableC;
    constexpr bool kHasZ = Ktraits::kHasZ;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNItems = Ktraits::kNItems;
    constexpr int kNRows = Ktraits::kNRows;
    constexpr bool kDirectIO = Ktraits::kDirectIO;
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;
    using scan_t = typename Ktraits::scan_t;

    // Shared memory.
    extern __shared__ char smem_[];
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_weight = reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage&>(smem_);
    auto& smem_load_weight1 = *reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage*>(smem_ + sizeof(typename Ktraits::BlockLoadWeightT::TempStorage));
    scan_t *smem_running_prefix = reinterpret_cast<scan_t *>(smem_ + Ktraits::kSmemSize);

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    const int group_id = dim_id / (params.dim_ngroups_ratio);
    input_t *u = reinterpret_cast<input_t *>(params.u_ptr) + batch_id * params.u_batch_stride
        + dim_id * kNRows * params.u_d_stride;
    input_t *delta = reinterpret_cast<input_t *>(params.delta_ptr) + batch_id * params.delta_batch_stride
        + dim_id * kNRows * params.delta_d_stride;
    weight_t *A = reinterpret_cast<weight_t *>(params.A_ptr) + dim_id * kNRows * params.A_d_stride;
    weight_t *B = reinterpret_cast<weight_t *>(params.B_ptr) + dim_id * kNRows * params.B_d_stride;
    input_t *Bvar = reinterpret_cast<input_t *>(params.B_ptr) + batch_id * params.B_batch_stride + group_id * params.B_group_stride;
    weight_t *C = reinterpret_cast<weight_t *>(params.C_ptr) + dim_id * kNRows * params.C_d_stride;
    input_t *Cvar = reinterpret_cast<input_t *>(params.C_ptr) + batch_id * params.C_batch_stride + group_id * params.C_group_stride;
    scan_t *x = reinterpret_cast<scan_t *>(params.x_ptr) + (batch_id * params.dim + dim_id * kNRows) * params.n_chunks * params.dstate;

    float D_val[kNRows] = {0};
    if (params.D_ptr != nullptr) {
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            D_val[r] = reinterpret_cast<float *>(params.D_ptr)[dim_id * kNRows + r];
        }
    }
    float delta_bias[kNRows] = {0};
    if (params.delta_bias_ptr != nullptr) {
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            delta_bias[r] = reinterpret_cast<float *>(params.delta_bias_ptr)[dim_id * kNRows + r];
        }
    }

    constexpr int kChunkSize = kNThreads * kNItems;
    for (int chunk = 0; chunk < params.n_chunks; ++chunk) {
        input_t u_vals[kNRows][kNItems], delta_vals_load[kNRows][kNItems];
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            if constexpr (!kDirectIO) {
                if (r > 0) { __syncthreads(); }
            }
            load_input<Ktraits>(u + r * params.u_d_stride, u_vals[r], smem_load, params.seqlen - chunk * kChunkSize);
            if constexpr (!kDirectIO) { __syncthreads(); }
            load_input<Ktraits>(delta + r * params.delta_d_stride, delta_vals_load[r], smem_load, params.seqlen - chunk * kChunkSize);
        }
        u += kChunkSize;
        delta += kChunkSize;

        float delta_vals[kNRows][kNItems], delta_u_vals[kNRows][kNItems], out_vals[kNRows][kNItems];
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float u_val = float(u_vals[r][i]);
                delta_vals[r][i] = float(delta_vals_load[r][i]) + delta_bias[r];
                if (params.delta_softplus) {
                    delta_vals[r][i] = delta_vals[r][i] <= 20.f ? log1pf(expf(delta_vals[r][i])) : delta_vals[r][i];
                }
                delta_u_vals[r][i] = delta_vals[r][i] * u_val;
                out_vals[r][i] = D_val[r] * u_val;
            }
        }

        __syncthreads();
        for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {
            weight_t A_val[kNRows];
            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                A_val[r] = A[state_idx * params.A_dstate_stride + r * params.A_d_stride];
                // Multiply the real part of A with LOG2E so we can use exp2f instead of expf.
                constexpr float kLog2e = M_LOG2E;
                if constexpr (!kIsComplex) {
                    A_val[r] *= kLog2e;
                } else {
                    A_val[r].real_ *= kLog2e;
                }
            }
            // This variable holds B * C if both B and C are constant across seqlen. If only B varies
            // across seqlen, this holds C. If only C varies across seqlen, this holds B.
            // If both B and C vary, this is unused.
            weight_t BC_val[kNRows];
            weight_t B_vals[kNItems], C_vals[kNItems];
            if constexpr (kIsVariableB) {
                load_weight<Ktraits>(Bvar + state_idx * params.B_dstate_stride, B_vals,
                    smem_load_weight, (params.seqlen - chunk * kChunkSize) * (!kIsComplex ? 1 : 2));
                if constexpr (!kIsVariableC) {
                    #pragma unroll
                    for (int r = 0; r < kNRows; ++r) {
                        BC_val[r] = C[state_idx * params.C_dstate_stride + r * params.C_d_stride];
                    }
                }
            }
            if constexpr (kIsVariableC) {
                auto &smem_load_weight_C = !kIsVariableB ? smem_load_weight : smem_load_weight1;
                load_weight<Ktraits>(Cvar + state_idx * params.C_dstate_stride, C_vals,
                    smem_load_weight_C, (params.seqlen - chunk * kChunkSize) * (!kIsComplex ? 1 : 2));
                if constexpr (!kIsVariableB) {
                    #pragma unroll
                    for (int r = 0; r < kNRows; ++r) {
                        BC_val[r] = B[state_idx * params.B_dstate_stride + r * params.B_d_stride];
                    }
                }
            }
            if constexpr (!kIsVariableB && !kIsVariableC) {
                #pragma unroll
                for (int r = 0; r < kNRows; ++r) {
                    BC_val[r] = B[state_idx * params.B_dstate_stride + r * params.B_d_stride] * C[state_idx * params.C_dstate_stride + r * params.C_d_stride];
                }
            }

            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                if (r > 0) { __syncthreads(); }  // Scan could be using the same smem
                scan_t thread_data[kNItems];
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    if constexpr (!kIsComplex) {
                        thread_data[i] = make_float2(exp2f(delta_vals[r][i] * A_val[r]),
                                                     !kIsVariableB ? delta_u_vals[r][i] : B_vals[i] * delta_u_vals[r][i]);
                        if constexpr (!Ktraits::kIsEvenLen) {  // So that the last state is correct
                            if (threadIdx.x * kNItems + i >= params.seqlen - chunk * kChunkSize) {
                                thread_data[i] = make_float2(1.f, 0.f);
                            }
                        }
                    } else {
                        // Pytorch's implementation of complex exp (which calls thrust) is very slow
                        complex_t delta_a_exp = cexp2f(delta_vals[r][i] * A_val[r]);
                        weight_t B_delta_u_val = !kIsVariableB ? delta_u_vals[r][i] : B_vals[i] * delta_u_vals[r][i];
                        thread_data[i] = make_float4(delta_a_exp.real_, delta_a_exp.imag_, B_delta_u_val.real_, B_delta_u_val.imag_);
                        if constexpr (!Ktraits::kIsEvenLen) {  // So that the last state is correct
                            if (threadIdx.x * kNItems + i >= params.seqlen - chunk * kChunkSize) {
                                thread_data[i] = make_float4(1.f, 0.f, 0.f, 0.f);
                            }
                        }
                    }
                }

                // Store thread_data for the scan kernel
                int data_offset = (batch_id * params.dim + dim_id * kNRows + r) * params.n_chunks * params.dstate * kNThreads * kNItems +
                                 chunk * params.dstate * kNThreads * kNItems +
                                 state_idx * kNThreads * kNItems +
                                 threadIdx.x * kNItems;
                for (int i = 0; i < kNItems; ++i) {
                    thread_data_out[data_offset + i] = thread_data[i];
                }

                // Store initial running prefix for the scan
                if (threadIdx.x == 0 && chunk > 0) {
                    int prefix_offset = (batch_id * params.dim + dim_id * kNRows + r) * params.n_chunks * params.dstate +
                                        (chunk - 1) * params.dstate +
                                        state_idx;
                    running_prefix_in[prefix_offset] = smem_running_prefix[state_idx + r * MAX_DSTATE];
                }
            }
        }

        Bvar += kChunkSize * (!kIsComplex ? 1 : 2);
        Cvar += kChunkSize * (!kIsComplex ? 1 : 2);
    }
}

// Part 2: Scan kernel using cub::DeviceScan::InclusiveScan
template<typename weight_t, typename scan_t>
struct SSMScanOpDevice {
    __host__ __device__ __forceinline__ scan_t operator()(const scan_t &a, const scan_t &b) const {
        if constexpr (std::is_same_v<scan_t, float2>) {
            return make_float2(a.x * b.x, a.x * b.y + a.y);
        } else {
            // Complex version
            float real_a = a.x;
            float imag_a = a.y;
            complex_t a_complex(real_a, imag_a);
            complex_t b_complex_first(b.x, b.y);
            complex_t ab_complex = a_complex * b_complex_first;
            complex_t ab_complex_second(b.z, b.w);
            return make_float4(ab_complex.real_, ab_complex.imag_,
                               b.z * a.x - b.w * a.y + a.z,
                               b.w * a.x + b.z * a.y + a.w);
        }
    }
};

template<typename Ktraits>
void selective_scan2_fwd_scan_kernel(
    typename Ktraits::scan_t* d_in,
    typename Ktraits::scan_t* d_out,
    typename Ktraits::scan_t* running_prefix_in,
    typename Ktraits::scan_t* running_prefix_out,
    int batch, int dim, int n_chunks, int dstate, int seq_len,
    cudaStream_t stream) {

    using scan_t = typename Ktraits::scan_t;
    using weight_t = typename Ktraits::weight_t;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNItems = Ktraits::kNItems;
    constexpr int kNRows = Ktraits::kNRows;
    constexpr bool kIsComplex = Ktraits::kIsComplex;

    // Calculate some constants
    const int elements_per_chunk = kNThreads * kNItems;
    const int elements_per_batch_dim_row_state = n_chunks * elements_per_chunk;

    // Allocate temporary storage and initialize variables
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Create temporary memory for segmented scan
    const int max_elements = elements_per_chunk;
    scan_t *d_temp_in = nullptr;
    cudaMalloc(&d_temp_in, max_elements * sizeof(scan_t));

    // Determine size needed for CUB
    cub::DeviceScan::InclusiveScan(
        d_temp_storage,
        temp_storage_bytes,
        d_temp_in,
        d_out,
        SSMScanOpDevice<weight_t, scan_t>(),
        max_elements,
        stream
    );

    // Allocate temporary storage for CUB
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Process each segment of data one at a time
    for (int b = 0; b < batch; ++b) {
        for (int d = 0; d < dim; ++d) {
            for (int r = 0; r < kNRows; ++r) {
                for (int s = 0; s < dstate; ++s) {
                    scan_t prefix = {0};

                    for (int c = 0; c < n_chunks; ++c) {
                        // Calculate offset into the data arrays
                        const int data_offset =
                            (b * dim + d * kNRows + r) * n_chunks * dstate * elements_per_chunk +
                            c * dstate * elements_per_chunk +
                            s * elements_per_chunk;

                        // Initialize prefix for first chunk, otherwise use running prefix
                        if (c == 0) {
                            // Identity value
                            if constexpr (!kIsComplex) {
                                prefix = make_float2(1.0f, 0.0f);
                            } else {
                                prefix = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
                            }
                        } else {
                            // Get previous chunk's running prefix
                            const int prefix_idx = (b * dim + d * kNRows + r) * n_chunks * dstate + (c-1) * dstate + s;
                            cudaMemcpyAsync(&prefix,
                                           running_prefix_out + prefix_idx,
                                           sizeof(scan_t),
                                           cudaMemcpyDeviceToHost,
                                           stream);
                            cudaStreamSynchronize(stream);
                        }

                        // Copy chunk to temporary buffer
                        cudaMemcpyAsync(d_temp_in,
                                       d_in + data_offset,
                                       elements_per_chunk * sizeof(scan_t),
                                       cudaMemcpyDeviceToDevice,
                                       stream);

                        // Process the first element to incorporate the prefix
                        if (c > 0) {
                            scan_t first_element;
                            cudaMemcpyAsync(&first_element,
                                          d_temp_in,
                                          sizeof(scan_t),
                                          cudaMemcpyDeviceToHost,
                                          stream);
                            cudaStreamSynchronize(stream);

                            // Apply the scan operation to merge the prefix with the first element
                            SSMScanOpDevice<weight_t, scan_t> scan_op;
                            scan_t result = scan_op(prefix, first_element);

                            // Write the result back
                            cudaMemcpyAsync(d_temp_in,
                                          &result,
                                          sizeof(scan_t),
                                          cudaMemcpyHostToDevice,
                                          stream);
                        }

                        // Perform scan on the chunk
                        cub::DeviceScan::InclusiveScan(
                            d_temp_storage,
                            temp_storage_bytes,
                            d_temp_in,
                            d_out + data_offset,
                            SSMScanOpDevice<weight_t, scan_t>(),
                            elements_per_chunk,
                            stream
                        );

                        // Store the running prefix for this chunk
                        const int prefix_out_idx = (b * dim + d * kNRows + r) * n_chunks * dstate + c * dstate + s;
                        cudaMemcpyAsync(
                            running_prefix_out + prefix_out_idx,
                            d_out + data_offset + elements_per_chunk - 1,
                            sizeof(scan_t),
                            cudaMemcpyDeviceToDevice,
                            stream
                        );
                    }
                }
            }
        }
    }

    // Clean up temporary memory
    cudaFree(d_temp_storage);
    cudaFree(d_temp_in);
}

// Part 3: Post-kernel that processes the scan results and computes the final output
template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void selective_scan2_fwd_post_kernel(SSMParamsBase params,
                                     typename Ktraits::scan_t* thread_data_scanned,
                                     typename Ktraits::scan_t* running_prefix_out) {
    constexpr bool kIsComplex = Ktraits::kIsComplex;
    constexpr bool kIsVariableB = Ktraits::kIsVariableB;
    constexpr bool kIsVariableC = Ktraits::kIsVariableC;
    constexpr bool kHasZ = Ktraits::kHasZ;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNItems = Ktraits::kNItems;
    constexpr int kNRows = Ktraits::kNRows;
    constexpr bool kDirectIO = Ktraits::kDirectIO;
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;
    using scan_t = typename Ktraits::scan_t;

    // Shared memory
    extern __shared__ char smem_[];
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_load_weight = reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage&>(smem_);
    auto& smem_load_weight1 = *reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage*>(smem_ + sizeof(typename Ktraits::BlockLoadWeightT::TempStorage));

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    const int group_id = dim_id / (params.dim_ngroups_ratio);

    // Get pointers to data
    weight_t *B = reinterpret_cast<weight_t *>(params.B_ptr) + dim_id * kNRows * params.B_d_stride;
    input_t *Bvar = reinterpret_cast<input_t *>(params.B_ptr) + batch_id * params.B_batch_stride + group_id * params.B_group_stride;
    weight_t *C = reinterpret_cast<weight_t *>(params.C_ptr) + dim_id * kNRows * params.C_d_stride;
    input_t *Cvar = reinterpret_cast<input_t *>(params.C_ptr) + batch_id * params.C_batch_stride + group_id * params.C_group_stride;
    scan_t *x = reinterpret_cast<scan_t *>(params.x_ptr) + (batch_id * params.dim + dim_id * kNRows) * params.n_chunks * params.dstate;

    // Get D values for the output calculation
    float D_val[kNRows] = {0};
    if (params.D_ptr != nullptr) {
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            D_val[r] = reinterpret_cast<float *>(params.D_ptr)[dim_id * kNRows + r];
        }
    }

    constexpr int kChunkSize = kNThreads * kNItems;

    for (int chunk = 0; chunk < params.n_chunks; ++chunk) {
        float out_vals[kNRows][kNItems];
        // Initialize out_vals with D_val * u_val (which was already computed in pre-kernel)
        // We'll need to re-read u_vals here or pass them through

        // For simplicity, start with zero and we'll add D_val * u_val in the inner loop below
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                out_vals[r][i] = 0.0f;
            }
        }

        for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {
            // This variable holds B * C if both B and C are constant across seqlen. If only B varies
            // across seqlen, this holds C. If only C varies across seqlen, this holds B.
            // If both B and C vary, this is unused.
            weight_t BC_val[kNRows];
            weight_t B_vals[kNItems], C_vals[kNItems];

            if constexpr (kIsVariableB) {
                load_weight<Ktraits>(Bvar + state_idx * params.B_dstate_stride, B_vals,
                    smem_load_weight, (params.seqlen - chunk * kChunkSize) * (!kIsComplex ? 1 : 2));
                if constexpr (!kIsVariableC) {
                    #pragma unroll
                    for (int r = 0; r < kNRows; ++r) {
                        BC_val[r] = C[state_idx * params.C_dstate_stride + r * params.C_d_stride];
                    }
                }
            }

            if constexpr (kIsVariableC) {
                auto &smem_load_weight_C = !kIsVariableB ? smem_load_weight : smem_load_weight1;
                load_weight<Ktraits>(Cvar + state_idx * params.C_dstate_stride, C_vals,
                    smem_load_weight_C, (params.seqlen - chunk * kChunkSize) * (!kIsComplex ? 1 : 2));
                if constexpr (!kIsVariableB) {
                    #pragma unroll
                    for (int r = 0; r < kNRows; ++r) {
                        BC_val[r] = B[state_idx * params.B_dstate_stride + r * params.B_d_stride];
                    }
                }
            }

            if constexpr (!kIsVariableB && !kIsVariableC) {
                #pragma unroll
                for (int r = 0; r < kNRows; ++r) {
                    BC_val[r] = B[state_idx * params.B_dstate_stride + r * params.B_d_stride] * C[state_idx * params.C_dstate_stride + r * params.C_d_stride];
                }
            }

            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                // Get the scanned data for this state
                int data_offset = (batch_id * params.dim + dim_id * kNRows + r) * params.n_chunks * params.dstate * kNThreads * kNItems +
                                 chunk * params.dstate * kNThreads * kNItems +
                                 state_idx * kNThreads * kNItems +
                                 threadIdx.x * kNItems;

                // Copy running prefix to x for backward pass
                if (threadIdx.x == 0) {
                    int prefix_offset = (batch_id * params.dim + dim_id * kNRows + r) * params.n_chunks * params.dstate +
                                      chunk * params.dstate +
                                      state_idx;
                    x[(r * params.n_chunks + chunk) * params.dstate + state_idx] = running_prefix_out[prefix_offset];
                }

                // Process scanned data to calculate output
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    scan_t scanned_val = thread_data_scanned[data_offset + i];
                    const weight_t C_val = !kIsVariableC
                        ? BC_val[r]
                        : (!kIsVariableB ? BC_val[r] * C_vals[i] : C_vals[i]);

                    if constexpr (!kIsComplex) {
                        out_vals[r][i] += scanned_val.y * C_val;
                    } else {
                        out_vals[r][i] += (complex_t(scanned_val.z, scanned_val.w) * C_val).real_ * 2;
                    }
                }
            }
        }

        // Store the output
        input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
            + dim_id * kNRows * params.out_d_stride + chunk * kChunkSize;
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            if constexpr (!kDirectIO) {
                if (r > 0) { __syncthreads(); }
            }
            store_output<Ktraits>(out + r * params.out_d_stride, out_vals[r], smem_store, params.seqlen - chunk * kChunkSize);
        }

        if constexpr (kHasZ) {
            input_t *z = reinterpret_cast<input_t *>(params.z_ptr) + batch_id * params.z_batch_stride
                + dim_id * kNRows * params.z_d_stride + chunk * kChunkSize;
            input_t *out_z = reinterpret_cast<input_t *>(params.out_z_ptr) + batch_id * params.out_z_batch_stride
                + dim_id * kNRows * params.out_z_d_stride + chunk * kChunkSize;
            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                input_t z_vals[kNItems];
                __syncthreads();
                load_input<Ktraits>(z + r * params.z_d_stride, z_vals, smem_load, params.seqlen - chunk * kChunkSize);
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    float z_val = z_vals[i];
                    out_vals[r][i] *= z_val / (1 + expf(-z_val));
                }
                __syncthreads();
                store_output<Ktraits>(out_z + r * params.out_z_d_stride, out_vals[r], smem_store, params.seqlen - chunk * kChunkSize);
            }
        }

        Bvar += kChunkSize * (!kIsComplex ? 1 : 2);
        Cvar += kChunkSize * (!kIsComplex ? 1 : 2);
    }
}

template<int kNThreads, int kNItems, typename input_t, typename weight_t>
void selective_scan2_fwd_launch(SSMParamsBase &params, cudaStream_t stream) {
    // Only kNRows == 1 is tested for now, which ofc doesn't differ from previously when we had each block
    // processing 1 row.
    constexpr int kNRows = 1;
    BOOL_SWITCH(params.seqlen % (kNThreads * kNItems) == 0, kIsEvenLen, [&] {
        BOOL_SWITCH(params.is_variable_B, kIsVariableB, [&] {
            BOOL_SWITCH(params.is_variable_C, kIsVariableC, [&] {
                BOOL_SWITCH(params.z_ptr != nullptr , kHasZ, [&] {
                    using Ktraits = Selective_Scan2_fwd_kernel_traits<kNThreads, kNItems, kNRows, kIsEvenLen, kIsVariableB, kIsVariableC, kHasZ, input_t, weight_t>;
                    using scan_t = typename Ktraits::scan_t;

                    constexpr int kSmemSize = Ktraits::kSmemSize + kNRows * MAX_DSTATE * sizeof(typename Ktraits::scan_t);
                    dim3 grid(params.batch, params.dim / kNRows);

                    // Calculate sizes
                    constexpr int kChunkSize = kNThreads * kNItems;
                    const int elements_per_scan = kNThreads * kNItems;

                    // Allocate device memory for intermediate data
                    scan_t* d_thread_data;
                    scan_t* d_thread_data_scanned;
                    scan_t* d_running_prefix_in;
                    scan_t* d_running_prefix_out;

                    size_t thread_data_size = params.batch * params.dim * kNRows * params.n_chunks * params.dstate * elements_per_scan * sizeof(scan_t);
                    size_t running_prefix_size = params.batch * params.dim * kNRows * params.n_chunks * params.dstate * sizeof(scan_t);

                    cudaMalloc(&d_thread_data, thread_data_size);
                    cudaMalloc(&d_thread_data_scanned, thread_data_size);
                    cudaMalloc(&d_running_prefix_in, running_prefix_size);
                    cudaMalloc(&d_running_prefix_out, running_prefix_size);

                    // Setup kernel attributes
                    if (kSmemSize >= 48 * 1024) {
                        #ifndef USE_ROCM
                        C10_CUDA_CHECK(cudaFuncSetAttribute(
                            &selective_scan2_fwd_pre_kernel<Ktraits>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
                        C10_CUDA_CHECK(cudaFuncSetAttribute(
                            &selective_scan2_fwd_post_kernel<Ktraits>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
                        #else
                        C10_CUDA_CHECK(cudaFuncSetAttribute(
                            (void *)&selective_scan2_fwd_pre_kernel<Ktraits>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
                        C10_CUDA_CHECK(cudaFuncSetAttribute(
                            (void *)&selective_scan2_fwd_post_kernel<Ktraits>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
                        std::cerr << "Warning (selective_scan2_fwd_kernel): attempting to set maxDynamicSharedMemorySize on an AMD GPU which is currently a non-op (in ROCm versions <= 6.1). This might lead to undefined behavior. \n" << std::endl;
                        #endif
                    }

                    // 1. Pre-compute data for scanning
                    selective_scan2_fwd_pre_kernel<Ktraits><<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(
                        params, d_thread_data, d_running_prefix_in);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();

                    // 2. Perform scan operation
                    selective_scan2_fwd_scan_kernel<Ktraits>(
                        d_thread_data, d_thread_data_scanned, d_running_prefix_in, d_running_prefix_out,
                        params.batch, params.dim, params.n_chunks, params.dstate, params.seqlen,
                        stream);

                    // 3. Process scan results
                    selective_scan2_fwd_post_kernel<Ktraits><<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(
                        params, d_thread_data_scanned, d_running_prefix_out);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();

                    // Free allocated memory
                    cudaFree(d_thread_data);
                    cudaFree(d_thread_data_scanned);
                    cudaFree(d_running_prefix_in);
                    cudaFree(d_running_prefix_out);
                });
            });
        });
    });
}

template<typename input_t, typename weight_t>
void selective_scan2_fwd_cuda(SSMParamsBase &params, cudaStream_t stream) {

    #ifndef USE_ROCM
        if (params.seqlen <= 128) {
            selective_scan2_fwd_launch<32, 4, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 256) {
            selective_scan2_fwd_launch<32, 8, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 512) {
            selective_scan2_fwd_launch<32, 16, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 1024) {
            selective_scan2_fwd_launch<64, 16, input_t, weight_t>(params, stream);
        } else {
            selective_scan2_fwd_launch<128, 16, input_t, weight_t>(params, stream);
        }
    #else
        if (params.seqlen <= 256) {
            selective_scan2_fwd_launch<64, 4, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 512) {
            selective_scan2_fwd_launch<64, 8, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 1024) {
            selective_scan2_fwd_launch<64, 16, input_t, weight_t>(params, stream);
        } else {
            selective_scan2_fwd_launch<128, 16, input_t, weight_t>(params, stream);
        }
    #endif
}
