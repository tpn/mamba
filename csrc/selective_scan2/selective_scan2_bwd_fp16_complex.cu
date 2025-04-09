/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

// Split into multiple files to compile in paralell

#include "selective_scan2_bwd_kernel.cuh"

template void selective_scan2_bwd_cuda<at::Half, complex_t>(SSMParamsBwd &params, cudaStream_t stream);