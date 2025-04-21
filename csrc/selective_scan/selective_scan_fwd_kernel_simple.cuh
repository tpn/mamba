// Simplified version of selective_scan_fwd_kernel.cuh based on:
//      void selective_scan_fwd_kernel<
//          Selective_Scan_fwd_kernel_traits<
//              (int)32,    // kNThreads
//              (int)4,     // kNItems
//              (int)1,     // kNRows
//              (bool)0,    // kIsEvenLen
//              (bool)1,    // kIsVariableB
//              (bool)1,    // kIsVariableC
//              (bool)1,    // kHasZ
//              c10::Half,  // input type
//              float,      // weight type
//          >
//      >(SSMParamsBase)

//------------------------------------------------------------------------------
//  Fixed traits  (matches   void selective_scan_fwd_kernel<…>(…))
//------------------------------------------------------------------------------
using KernelTraits =
    Selective_Scan_fwd_kernel_traits<
        /*kNThreads   =*/ 32,
        /*kNItems     =*/ 4,
        /*kNRows      =*/ 1,
        /*kIsEvenLen  =*/ false,
        /*kIsVariableB=*/ true,
        /*kIsVariableC=*/ true,
        /*kHasZ       =*/ true,
        /*input_t     =*/ c10::Half,
        /*weight_t    =*/ float>;

//------------------------------------------------------------------------------
//  Simplified kernel ‒ *no complex numbers*, *variable B & C*, *uneven length*
//------------------------------------------------------------------------------
__global__ __launch_bounds__(KernelTraits::kNThreads, KernelTraits::kMinBlocks)
void selective_scan_fwd_kernel(SSMParamsBase params)
{
    // --- Compile‑time aliases -------------------------------------------------
    constexpr int  kNThreads  = KernelTraits::kNThreads;   // 32
    constexpr int  kNItems    = KernelTraits::kNItems;     // 4
    constexpr int  kNRows     = KernelTraits::kNRows;      // 1
    constexpr int  kChunkSize = kNThreads * kNItems;       // 128
    constexpr bool kIsEvenLen = KernelTraits::kIsEvenLen;  // false

    using input_t      = typename KernelTraits::input_t;      // c10::Half
    using weight_t     = typename KernelTraits::weight_t;     // float
    using scan_t       = typename KernelTraits::scan_t;       // float2

    using BlockLoadT          = typename KernelTraits::BlockLoadT;
    using BlockLoadVecT       = typename KernelTraits::BlockLoadVecT;
    using BlockLoadWeightT    = typename KernelTraits::BlockLoadWeightT;
    using BlockLoadWeightVecT = typename KernelTraits::BlockLoadWeightVecT;
    using BlockStoreT         = typename KernelTraits::BlockStoreT;
    using BlockStoreVecT      = typename KernelTraits::BlockStoreVecT;
    using BlockScanT          = typename KernelTraits::BlockScanT;

    // --- Shared memory --------------------------------------------------------
    extern __shared__ char smem_[];

    auto& smem_load        = reinterpret_cast<BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_B      = reinterpret_cast<BlockLoadWeightT::TempStorage&>(smem_);
    auto& smem_load_C      =
        *reinterpret_cast<BlockLoadWeightT::TempStorage*>(smem_ + sizeof(BlockLoadWeightT::TempStorage));
    auto& smem_store       = reinterpret_cast<BlockStoreT::TempStorage&>(smem_);
    auto& smem_scan        =
        *reinterpret_cast<BlockScanT::TempStorage*>(smem_ + KernelTraits::kSmemIOSize);

    scan_t* smem_running_prefix =
        reinterpret_cast<scan_t*>(smem_ + KernelTraits::kSmemSize);

    // --- CTA‑level offsets ----------------------------------------------------
    const int batch_id = blockIdx.x;
    const int dim_id   = blockIdx.y;
    const int group_id = dim_id / params.dim_ngroups_ratio;

    input_t*  u      = reinterpret_cast<input_t*>(params.u_ptr)
                     + batch_id * params.u_batch_stride
                     + dim_id   * kNRows * params.u_d_stride;

    input_t*  delta  = reinterpret_cast<input_t*>(params.delta_ptr)
                     + batch_id * params.delta_batch_stride
                     + dim_id   * kNRows * params.delta_d_stride;

    weight_t* A      = reinterpret_cast<weight_t*>(params.A_ptr)
                     + dim_id * kNRows * params.A_d_stride;

    input_t*  Bvar   = reinterpret_cast<input_t*>(params.B_ptr)
                     + batch_id * params.B_batch_stride
                     + group_id * params.B_group_stride;

    input_t*  Cvar   = reinterpret_cast<input_t*>(params.C_ptr)
                     + batch_id * params.C_batch_stride
                     + group_id * params.C_group_stride;

    scan_t*   x      = reinterpret_cast<scan_t*>(params.x_ptr)
                     + (batch_id * params.dim + dim_id * kNRows)
                     * params.n_chunks * params.dstate;

    // --- Optional per‑row constants ------------------------------------------
    float D_val[kNRows]{};
    if (params.D_ptr) {
        #pragma unroll
        for (int r = 0; r < kNRows; ++r)
            D_val[r] = reinterpret_cast<float*>(params.D_ptr)[dim_id * kNRows + r];
    }

    float delta_bias[kNRows]{};
    if (params.delta_bias_ptr) {
        #pragma unroll
        for (int r = 0; r < kNRows; ++r)
            delta_bias[r] = reinterpret_cast<float*>(params.delta_bias_ptr)[dim_id * kNRows + r];
    }

    //==========================================================================*
    //  MAIN LOOP ‒ one “chunk” (128 elts) per iteration
    //==========================================================================*
    for (int chunk = 0; chunk < params.n_chunks; ++chunk)
    {
        //----------------------------------------------------------------------
        //  1.  Load U and Δ for this chunk
        //----------------------------------------------------------------------
        input_t u_vals[kNRows][kNItems];
        input_t delta_vals_load[kNRows][kNItems];

        __syncthreads();
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            load_input<KernelTraits>(u + r * params.u_d_stride,
                                     u_vals[r], smem_load,
                                     params.seqlen - chunk * kChunkSize);

            load_input<KernelTraits>(delta + r * params.delta_d_stride,
                                     delta_vals_load[r], smem_load,
                                     params.seqlen - chunk * kChunkSize);
        }
        u     += kChunkSize;
        delta += kChunkSize;

        //----------------------------------------------------------------------
        //  2.  Per‑element preprocessing
        //----------------------------------------------------------------------
        float delta_vals[kNRows][kNItems];
        float delta_u_vals[kNRows][kNItems];
        float out_vals[kNRows][kNItems];

        #pragma unroll
        for (int r = 0; r < kNRows; ++r)
        {
            for (int i = 0; i < kNItems; ++i)
            {
                const float u_val = float(u_vals[r][i]);
                float d          = float(delta_vals_load[r][i]) + delta_bias[r];

                if (params.delta_softplus)
                    d = (d <= 20.f) ? log1pf(expf(d)) : d;

                delta_vals[r][i]  = d;
                delta_u_vals[r][i] = d * u_val;
                out_vals[r][i]     = D_val[r] * u_val;
            }
        }

        //----------------------------------------------------------------------
        //  3.  Iterate over state dimension
        //----------------------------------------------------------------------
        __syncthreads();
        for (int state_idx = 0; state_idx < params.dstate; ++state_idx)
        {
            //-------------------- 3a.  Load A (constant) -----------------------
            weight_t A_val = A[state_idx * params.A_dstate_stride] * M_LOG2E;

            //-------------------- 3b.  Load B & C (both variable) -------------
            weight_t B_vals[kNItems];
            weight_t C_vals[kNItems];

            load_weight<KernelTraits>(Bvar + state_idx * params.B_dstate_stride,
                                      B_vals, smem_load_B,
                                      params.seqlen - chunk * kChunkSize);

            load_weight<KernelTraits>(Cvar + state_idx * params.C_dstate_stride,
                                      C_vals, smem_load_C,
                                      params.seqlen - chunk * kChunkSize);

            //-------------------- 3c.  Row loop (kNRows == 1) ------------------
            scan_t thread_data[kNItems];

            #pragma unroll
            for (int i = 0; i < kNItems; ++i)
            {
                thread_data[i] = make_float2(
                    exp2f(delta_vals[0][i] * A_val),   // x
                    B_vals[i] * delta_u_vals[0][i]);   // y

                if constexpr (!kIsEvenLen)
                {
                    if (threadIdx.x * kNItems + i >=
                        params.seqlen - chunk * kChunkSize)
                    {
                        thread_data[i] = make_float2(1.f, 0.f);
                    }
                }
            }

            // Inclusive scan ---------------------------------------------------
            scan_t running_prefix =
                (chunk > 0 && (threadIdx.x & 31) == 0)
                    ? smem_running_prefix[state_idx]
                    : make_float2(1.f, 0.f);

            SSMScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);

            BlockScanT(smem_scan)
                .InclusiveScan(thread_data, thread_data,
                               SSMScanOp<weight_t>(), prefix_op);

            if (threadIdx.x == 0) {
                smem_running_prefix[state_idx] = prefix_op.running_prefix;
                x[chunk * params.dstate + state_idx] = prefix_op.running_prefix;
            }

            //-------------------- 3d.  Finish output ---------------------------
            #pragma unroll
            for (int i = 0; i < kNItems; ++i)
            {
                out_vals[0][i] += thread_data[i].y * C_vals[i];
            }
        }  // ‑‑ end dstate loop

        //----------------------------------------------------------------------
        //  4.  Store result
        //----------------------------------------------------------------------
        input_t* out = reinterpret_cast<input_t*>(params.out_ptr)
                     + batch_id * params.out_batch_stride
                     + dim_id   * params.out_d_stride
                     + chunk    * kChunkSize;

        __syncthreads();
        store_output<KernelTraits>(out, out_vals[0], smem_store,
                                   params.seqlen - chunk * kChunkSize);

        //----------------------------------------------------------------------
        //  5.  Optional Z‑gate
        //----------------------------------------------------------------------
        if constexpr (KernelTraits::kHasZ)
        {
            input_t* z = reinterpret_cast<input_t*>(params.z_ptr)
                       + batch_id * params.z_batch_stride
                       + dim_id   * params.z_d_stride
                       + chunk    * kChunkSize;

            input_t* out_z = reinterpret_cast<input_t*>(params.out_z_ptr)
                           + batch_id * params.out_z_batch_stride
                           + dim_id   * params.out_z_d_stride
                           + chunk    * kChunkSize;

            input_t z_vals[kNItems];
            load_input<KernelTraits>(z, z_vals, smem_load,
                                     params.seqlen - chunk * kChunkSize);

            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                const float zv = z_vals[i];
                out_vals[0][i] *= zv / (1.f + expf(-zv));
            }

            store_output<KernelTraits>(out_z, out_vals[0], smem_store,
                                       params.seqlen - chunk * kChunkSize);
        }

        //----------------------------------------------------------------------
        //  6.  Advance variable pointers
        //----------------------------------------------------------------------
        Bvar += kChunkSize;
        Cvar += kChunkSize;
    } // ‑‑ end chunk loop
}
