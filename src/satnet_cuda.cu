//#include <stdio.h>
#include <math.h>
//#include <assert.h>
#include <stdint.h>
#include <float.h>

#include <cuda_runtime.h>
#include "satnet.h"

const double MEPS = 1e-24;
const int WARP_SIZE = 32;
const int WARP_NUM = 32;
const int MBUF_SIZE = 320;

// Warp level dot product
__device__
float warpdot(const float * x, const float * z, int k)
{
    if (k==0) return 0;
    int lane = threadIdx.x % WARP_SIZE;

    float val = 0;
    #pragma unroll 2
    for (int i=lane; i<k; i+=WARP_SIZE) val += x[i]*z[i];
    __syncwarp();

    unsigned int active = __activemask();
    #pragma unroll
    for (int off=WARP_SIZE/2; off; off/=2) 
        val += __shfl_xor_sync(active, val, off);

    return val;
}

__global__ void mix_init(int32_t *perm, int n, int k, const int32_t *is_input, int32_t *index, const float *z, float *V)
{
    z +=         n   * blockIdx.x;
    is_input += n   * blockIdx.x;
    V +=         n*k * blockIdx.x;
    index +=     n   * blockIdx.x;

    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    for (int i=warp; i<n; i+=WARP_NUM) {
        if (is_input[i]) {
            for (int kk=lane; kk<k; kk+=WARP_SIZE) {
                if (kk==0) V[i*k] = -cos(z[i]*M_PI);
                else if (kk==1) V[i*k+1] = copysign(sin(z[i]*M_PI), V[i*k+1]);
                else V[i*k+kk] = 0;
            }
            __syncwarp();
        } else {
            float s = warpdot(V+i*k, V+i*k, k);
            s = rsqrtf(s);
            __syncwarp();
            for (int kk=lane; kk<k; kk+=WARP_SIZE) V[i*k+kk] *= s;
        }
    }
    if (threadIdx.x==0) {
        int i_=0, j=0;
        for (; i_<n-1; i_++) {
            int i = perm[i_]+1;
            //int i = i_+1;
            if (!is_input[i]) index[j++] = i;
        }
        for (; j<n; j++) index[j] = 0;
    }
    __syncthreads();
    //__threadfence_system();
}

/*  The mix kernel perform a cycle of block coordinate descent for all Vi.
 */
__forceinline__
__device__ float mix_kernel(const int is_forward, float prox_lam,
        int m, int k, const int32_t *__restrict__ index, 
        const float *__restrict__ S, const float *__restrict__ dz, float *__restrict__ V, const float *__restrict__ Vproj, float *__restrict__ W, 
        float *__restrict__ gnrm, const float *__restrict__ Snrms, float *smem)
{

    const int kk = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    float * __restrict__ g =    smem;
    float * __restrict__ Si =   smem+k;
    float * __restrict__ Wbuf = smem+k+m; // smem buf for the first MBUF_SIZE W

    int mbuf = m>MBUF_SIZE ?   MBUF_SIZE : m; // mbuf = # of m inside buffer (in smem)
    int mrem = m>MBUF_SIZE ? m-MBUF_SIZE : 0; // mrem = # of m outside buffer (in global mem)
    for (int j=lane; j<mbuf; j+=WARP_SIZE) Wbuf[kk*mbuf+j] = W[kk*m+j];

    __shared__ float delta;
    if (threadIdx.x==0) delta = 0;

    for (int i, i_=0; (i=index[i_]); i_++) {
        for (int j=threadIdx.x; j<m; j += blockDim.x) Si[j] = S[i*m+j];
        __syncthreads();

        const float Sii = Snrms[i], Vik = V[i*k+kk];

        // val = Wk'Si - Sii Vik
        const float val = warpdot(Wbuf+kk*mbuf, Si, mbuf) 
                        + warpdot(W+kk*m+mbuf, Si+mbuf, mrem) 
                        - Sii * Vik;
        if (lane == 0) g[kk] = val;
        __syncthreads();

        float gnrmi, t;
        if (is_forward) { // gnrm is calculated in the forward pass
            gnrmi = sqrtf(warpdot(g,g,k));
            t = -val;
        } else { // In the backward pass, t = -(I-vi vi')(g + v0 dzi) 
            gnrmi = gnrm[i]+prox_lam;
            float c = warpdot(Vproj+i*k, g, k) + dz[i] * Vproj[i*k];
            t = -val + c * Vproj[i*k+kk] - dz[i] * Vproj[kk];
        }
        t = t/gnrmi-Vik;
        __syncthreads();

        if (lane==0) g[kk] = t, V[i*k+kk] += t;

        // W += (vi^new-vi^old) Si'
        #pragma unroll 2
        for (int j=lane; j<mbuf; j+=WARP_SIZE) Wbuf[kk*mbuf+j] += t* Si[j];
        for (int j=lane; j<mrem; j+=WARP_SIZE) W[kk*m+mbuf+j] += t* Si[j+mbuf];
        __syncthreads();
        if (is_forward) {
            // Calc function decrease
            float gg = warpdot(g, g, k);
            if (threadIdx.x == 0) delta += gnrmi * gg, gnrm[i] = gnrmi;
        }
        __threadfence_block();
    }
    __syncthreads();

    for (int j=lane; j<mbuf; j+=WARP_SIZE) W[kk*m+j] = Wbuf[kk*mbuf+j];
    __threadfence_block();

    return delta;
}

// consider the \min unsat problem,
__global__ void mix_forward(int max_iter, float eps, int n, int m, int k, const int32_t *index, int32_t *niter, const float *S, float *z, float *V, float *W, float *gnrm, float *Snrms, float *cache)
{
    z +=        n * blockIdx.x;
    index +=    n * blockIdx.x;
    V +=        n*k*blockIdx.x;
    W +=        m*k*blockIdx.x;
    gnrm +=     n * blockIdx.x;

    extern __shared__ float smem[];

    float delta;
    int iter = 0;
    for (; iter < max_iter; iter++) {
        delta = mix_kernel(1, 0, m, k, index, S, NULL, V, NULL, W, gnrm, Snrms, smem);
        if (iter && delta < eps) break;
        if (iter == 0) eps = delta*eps;
    }
    niter[blockIdx.x] = iter;

    for (int i,i_=0; (i=index[i_]); i_++) {
        float zi = V[i*k];
        zi = saturate((zi+1)/2)*2-1;
        zi = saturate(1-acosf(zi)/M_PI);
        if (threadIdx.x == 0) z[i] = zi;
    }

}

__global__ void mix_backward(float prox_lam, int n, int m, int k, int32_t *is_input, int32_t *index, int32_t *niter, const float *S, float *dS, float *z, float *dz, const float *V, float *U, float *W, float *Phi, float *gnrm, float *Snrms, float *cache)
{
    gnrm += n * blockIdx.x;
    z +=    n * blockIdx.x;
    index += n* blockIdx.x;
    V +=    n*k*blockIdx.x;
    W +=    m*k*blockIdx.x;
    Phi +=   m*k*blockIdx.x;
    U +=   n*k*blockIdx.x;
    dz +=   n * blockIdx.x;
    dS +=   n*m*blockIdx.x;

    extern __shared__ float smem[];

    __shared__ int invalid_flag;
    if (threadIdx.x == 0) invalid_flag = 0;
    __syncthreads();


    for (int i,i_=0; (i=index[i_]); i_++) 
        if (threadIdx.x==0) { 
            float zi = z[i];
            float dzi = dz[i]/M_PI/sinpif(zi);
            if (isnan(dzi) || isinf(dzi) || gnrm[i] < MEPS) invalid_flag = 1;
            dz[i] = dzi;
        }
    __syncthreads();
    __threadfence_block();

    if (invalid_flag) {
        for (int i=threadIdx.x; i<n; i+=blockDim.x) dz[i] = 0;
        return;
    }

    // solve P (S'S+D_z-D_sii)xI_k P U = -dz P v0
    int iter = 0;
    for (; iter<niter[blockIdx.x]; iter++) {
        mix_kernel(0, prox_lam, m, k, index, S, dz, U, V, Phi, gnrm, Snrms, smem);
    }

    // sanity check
    for (int ik=threadIdx.x; ik<n*k; ik+=blockDim.x) 
        if (isnan(U[ik]) || isinf(U[ik])) invalid_flag = 1;
    __syncthreads();
    if (invalid_flag) {
        for (int i=threadIdx.x; i<n; i+=blockDim.x) dz[i] = 0;
        return;
    }

    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    // dS = U W + V Phi
    for (int ij=threadIdx.x; ij<n*m; ij+=blockDim.x) {
        float val = 0;
        for (int kk=0; kk<k; kk++)
            val += U[ij/m*k+kk]*W[kk*m+ij%m] + V[ij/m*k+kk]*Phi[kk*m+ij%m];
        dS[ij] = val;
    }

    // dzi = v0'Phi si
    __syncthreads();
    for (int i=1; i<n; i++) {
        if (!is_input[i]) {
            if (threadIdx.x == 0) dz[i] = 0;
            continue;
        }
        __shared__ float val1, val2;
        __syncthreads();
        for (int kk=warp; kk<k; kk+=WARP_NUM) {
            float val = warpdot(S+i*m, Phi+kk*m, m);
            __syncwarp();
            if (kk == 0) val1 = val;
            if (kk == 1) val2 = val;
            __syncwarp();
        }
        __syncthreads();
        if (threadIdx.x == 0){
            dz[i] = (dz[i] + val1) * sinpif(z[i])*M_PI + val2 * copysign(cospif(z[i])*M_PI, V[i*k+1])*M_PI;
        }
        __syncthreads();
    }
}

void mix_init_launcher_cuda(mix_t mix, int32_t *perm, cudaStream_t stream)
{
        mix_init<<<mix.b,WARP_SIZE*WARP_NUM,0,stream>>>(perm,
                mix.n, mix.k, mix.is_input, mix.index, mix.z,
                mix.V);
}

void mix_forward_launcher_cuda(mix_t mix, int max_iter, float eps, cudaStream_t stream)
{
    int smem_size = (mix.m+mix.k*(1+MBUF_SIZE))*sizeof(float);
    mix_forward<<<mix.b,WARP_SIZE*WARP_NUM,smem_size,stream>>>(max_iter, eps,
            mix.n, mix.m, mix.k, mix.index, mix.niter, 
            mix.S, mix.z, mix.V, mix.W, mix.gnrm, mix.Snrms, mix.cache);
}

void mix_backward_launcher_cuda(mix_t mix, float prox_lam, cudaStream_t stream)
{
    int smem_size = (mix.m+mix.k*(1+MBUF_SIZE))*sizeof(float);
    mix_backward<<<mix.b,WARP_SIZE*WARP_NUM,smem_size,stream>>>(prox_lam,
           mix.n, mix.m, mix.k, mix.is_input, mix.index, mix.niter, 
           mix.S, mix.dS, mix.z, mix.dz, mix.V, mix.U, mix.W, mix.Phi, mix.gnrm, mix.Snrms, mix.cache);
}
