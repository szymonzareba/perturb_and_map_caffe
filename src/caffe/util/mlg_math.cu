#include "caffe/common.hpp"
#include "caffe/util/mlg_math.hpp"

namespace caffe {

template <typename Dtype>
__global__ void replicate_kernel(const int xcount, const int repxcount, const Dtype* x, Dtype* repx) {
  CUDA_KERNEL_LOOP(index, repxcount) {
   repx[index] = x[index % xcount];
  }
}

template <typename Dtype>
__global__ void sample_ge0_kernel(const int n, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
	if(y[index] > (Dtype)0.){
		y[index] = (Dtype) 1.;
	}else{
		y[index] = (Dtype) 0.;
	}
  }
}

template <typename Dtype>
__global__ void sample_ge0_5_kernel(const int n, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
	if(y[index] > (Dtype)0.5){
		y[index] = (Dtype) 1.;
	}else{
		y[index] = (Dtype) 0.;
	}
  }
}

template <typename Dtype>
__global__ void add_scaled_kernel(const int n, const Dtype alpha, const Dtype* a, const Dtype beta, const Dtype* b, Dtype* x) {
  CUDA_KERNEL_LOOP(index, n) {
	x[index] = alpha * a[index] + beta * b[index];
  }
}

template <typename Dtype>
__global__ void relax_0_1_kernel(const int n, Dtype* x) {
  CUDA_KERNEL_LOOP(index, n) {
	if(x[index] > 1){
		x[index] = 1;
	}

	if(x[index] < 0){
		x[index] = 0;
	}
  }
}

template <typename Dtype>
__global__ void negate_kernel(const int n, Dtype* x){
	CUDA_KERNEL_LOOP(index, n){
		x[index] = 1 - x[index];
	}
}

template <typename Dtype>
__global__ void add_with_mask_kernel(const int n, const Dtype* a, const Dtype* bMask, const Dtype* b, Dtype* x){
	CUDA_KERNEL_LOOP(index, n){
		x[index] = a[index] + bMask[index] * b[index];
	}
}

}
