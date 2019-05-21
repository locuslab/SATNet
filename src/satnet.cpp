#ifdef MIX_USE_GPU
    #include <ATen/cuda/CUDAContext.h>
#endif
#include <torch/extension.h>

#ifdef MIX_USE_GPU
	#define DEVICE_NAME cuda
	#define _MIX_DEV_STR "cuda"
	#define _MIX_CUDA_DECL , cudaStream_t stream
	#define _MIX_CUDA_ARG , stream
	#define _MIX_CUDA_HEAD cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	#define _MIX_CUDA_TAIL  AT_CUDA_CHECK(cudaGetLastError()); 
                            //AT_CUDA_CHECK(cudaStreamSynchronize(stream));
#else
	#define DEVICE_NAME cpu
	#define _MIX_DEV_STR "cpu"
	#define _MIX_CUDA_DECL
	#define _MIX_CUDA_ARG
	#define _MIX_CUDA_HEAD
	#define _MIX_CUDA_TAIL
#endif

// name mangling for CPU and CUDA
#define _MIX_CAT(x,y) x ## _ ## y
#define _MIX_EVAL(x,y) _MIX_CAT(x,y)
#define _MIX_FUNC(name) _MIX_EVAL(name, DEVICE_NAME)

#include "satnet.h"

void _MIX_FUNC(mix_init_launcher)    (mix_t mix, int32_t *perm             _MIX_CUDA_DECL);
void _MIX_FUNC(mix_forward_launcher) (mix_t mix, int max_iter, float eps   _MIX_CUDA_DECL);
void _MIX_FUNC(mix_backward_launcher)(mix_t mix, float prox_lam            _MIX_CUDA_DECL);

void mix_init(at::Tensor perm,
        at::Tensor is_input, at::Tensor index, at::Tensor z, at::Tensor V)
{
	_MIX_CUDA_HEAD;

    mix_t mix;
    mix.b = V.size(0); mix.n = V.size(1); mix.k = V.size(2);
    mix.is_input = is_input.data<int>();
    mix.index = index.data<int>();
    mix.z = z.data<float>();
    mix.V = V.data<float>();
    
    _MIX_FUNC(mix_init_launcher)(mix, perm.data<int>() _MIX_CUDA_ARG);

	_MIX_CUDA_TAIL;
}

void mix_forward(int max_iter, float eps,
        at::Tensor index, at::Tensor niter, at::Tensor S, at::Tensor z, at::Tensor V, at::Tensor W, at::Tensor gnrm, at::Tensor Snrms, at::Tensor cache)
{
	_MIX_CUDA_HEAD;

    mix_t mix;
    mix.b = V.size(0); mix.n = V.size(1); mix.m = S.size(1); mix.k = V.size(2);
    mix.index = index.data<int>();
    mix.niter = niter.data<int>();
    mix.S = S.data<float>();
    mix.z = z.data<float>();
    mix.V = V.data<float>();
    mix.W = W.data<float>();
    mix.gnrm = gnrm.data<float>(); mix.Snrms = Snrms.data<float>();
    mix.cache = cache.data<float>();

    _MIX_FUNC(mix_forward_launcher)(mix, max_iter, eps _MIX_CUDA_ARG);

	_MIX_CUDA_TAIL;
}

void mix_backward(float prox_lam,
        at::Tensor is_input, at::Tensor index, at::Tensor niter, at::Tensor S, at::Tensor dS, at::Tensor z, at::Tensor dz,
        at::Tensor V, at::Tensor U, at::Tensor W, at::Tensor Phi, at::Tensor gnrm, at::Tensor Snrms, at::Tensor cache)
{
	_MIX_CUDA_HEAD;

    mix_t mix;
    mix.b = V.size(0); mix.n = V.size(1); mix.m = S.size(1); mix.k = V.size(2);
    mix.is_input = is_input.data<int>();
    mix.index = index.data<int>();
    mix.niter = niter.data<int>();
    mix.S = S.data<float>(); mix.dS = dS.data<float>();
    mix.z = z.data<float>(); mix.dz = dz.data<float>();
    mix.V = V.data<float>(); mix.U = U.data<float>();
    mix.W = W.data<float>(); mix.Phi = Phi.data<float>();
    mix.gnrm = gnrm.data<float>(); mix.Snrms = Snrms.data<float>();
    mix.cache = cache.data<float>();

    _MIX_FUNC(mix_backward_launcher)(mix, prox_lam _MIX_CUDA_ARG);

	_MIX_CUDA_TAIL;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init" , &mix_init, "SATNet init (" _MIX_DEV_STR ")");
    m.def("forward" , &mix_forward, "SATNet forward (" _MIX_DEV_STR ")");
    m.def("backward" , &mix_backward, "SATNet backward (" _MIX_DEV_STR ")");
}
