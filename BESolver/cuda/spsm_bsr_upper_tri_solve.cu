
extern "C" __global__ void spsm_bsr_upper_tri_solve(const double* __restrict__ const Asp_data_ptr,
                                                    const int* __restrict__ const Asp_indptr_ptr,
                                                    const int* __restrict__ const Asp_indices_ptr,
                                                    const int nbr, const int nrhs, const int bsz,
                                                    const double* const b, double* out){

    // copy b to out
    const int gid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid    = threadIdx.x;
    const int n      = nbr * bsz;
    const int rhs_id = blockIdx.x;

    extern __shared__ double xs[];
    
    
    for(int i=nbr-1; i>=0; i--){

        // out[rhs_id * n + i * bsz + tid] = b[rhs_id * n + i * bsz + tid];
        // __syncthreads();

        double bi           = b[rhs_id * n + i * bsz + tid];

        const int row_start = Asp_indptr_ptr[i];
        const int row_end   = Asp_indptr_ptr[i+1];

        for(int idx=row_start; idx<row_end; idx++){
            
            const int j = Asp_indices_ptr[idx];
            if(j <= i)
                continue; // skip diagonal and lower blocks

            const double* __restrict__ A_block = Asp_data_ptr + idx * bsz * bsz;

            xs[tid] = out[rhs_id * n + j * bsz + tid];
            __syncthreads();
            
            double s = 0.0;
            
            //#pragma unroll 4
            for (int c=0; c<bsz; c++)
                s += A_block[tid * bsz + c] * xs[c];
            
            //printf("Thread %d processing block (%d, %d), s = %f\n", tid, i, j, s);
            bi -= s;
            __syncthreads();


        }

        out[rhs_id * n + i * bsz + tid] = bi;
        __syncthreads();


    }

    return ;
}
    


