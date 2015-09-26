#include "caffe/common.hpp"
#include "caffe/util/mlg_math.hpp"

namespace caffe {

template <typename Dtype>
__global__ void binarization_kernel(const int count, const Dtype threshold, const Dtype* x, Dtype* y) {
  CUDA_KERNEL_LOOP(index, count) {
   if(x[index] > threshold){
	y[index] = (Dtype) 1.;
   }else{
	y[index] = (Dtype) 0.;
   }
  }
}

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
__global__ void sample_ge0_kernel(const int n, const Dtype* src, Dtype* dst) {
  CUDA_KERNEL_LOOP(index, n) {
	if(src[index] > (Dtype)0.){
		dst[index] = (Dtype) 1.;
	}else{
		dst[index] = (Dtype) 0.;
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
	if(x[index] > (Dtype)1.){
		x[index] = (Dtype)1.;
	}

	if(x[index] < (Dtype)0.){
		x[index] = (Dtype)0.;
	}
  }
}

template <typename Dtype>
__global__ void negate_0_1_kernel(const int n, Dtype* x){
	CUDA_KERNEL_LOOP(index, n){
		x[index] = (Dtype)1. - x[index];
	}
}

template <typename Dtype>
__global__ void add_with_mask_kernel(const int n, const Dtype* a, const Dtype* bMask, const Dtype* b, Dtype* x){
	CUDA_KERNEL_LOOP(index, n){
		x[index] = a[index] + bMask[index] * b[index];
	}
}

template <typename Dtype>
__global__ void negate_g_kernel(const int n, const Dtype threshold, const Dtype* mask, Dtype* x){
	CUDA_KERNEL_LOOP(index, n){
		if(mask[index] > threshold){
			x[index] = - x[index];
		}
	}
}

template <typename Dtype>
__global__ void negate_l_kernel(const int n, const Dtype threshold, const Dtype* mask, Dtype* x){
	CUDA_KERNEL_LOOP(index, n){
		if(mask[index] < threshold){
			x[index] = - x[index];
		}
	}
}

template <typename Dtype>
__global__ void negate_0_1_g_kernel(const int n, const Dtype threshold, const Dtype* mask, Dtype* x){
	CUDA_KERNEL_LOOP(index, n){
		if(mask[index] > threshold){
			x[index] = (Dtype)1. - x[index];
		}
	}
}

template <typename Dtype>
__global__ void negate_0_1_l_kernel(const int n, const Dtype threshold, const Dtype* mask, Dtype* x){
	CUDA_KERNEL_LOOP(index, n){
		if(mask[index] < threshold){
			x[index] = (Dtype)1. - x[index];
		}
	}
}

template
__global__ void binarization_kernel(const int count, const float threshold, const float* x, float* y);

template
__global__ void binarization_kernel(const int count, const double threshold, const double* x, double* y);

template
__global__ void replicate_kernel<float>(const int xcount, const int repxcount, const float* x, float* repx);

template
__global__ void replicate_kernel<double>(const int xcount, const int repxcount, const double* x, double* repx);

template
__global__ void sample_ge0_kernel<float>(const int n, float* y);

template
__global__ void sample_ge0_kernel<double>(const int n, double* y);

template
__global__ void sample_ge0_kernel<float>(const int n, const float* src, float* dst);

template
__global__ void sample_ge0_kernel<double>(const int n, const double* src, double* dst);

template
__global__ void sample_ge0_5_kernel<float>(const int n, float* y);

template
__global__ void sample_ge0_5_kernel<double>(const int n, double* y);

template
__global__ void add_scaled_kernel<float>(const int n, const float alpha, const float* a, const float beta, const float* b, float* x);

template
__global__ void add_scaled_kernel<double>(const int n, const double alpha, const double* a, const double beta, const double* b, double* x);

template
__global__ void relax_0_1_kernel<float>(const int n, float* x);

template
__global__ void relax_0_1_kernel<double>(const int n, double* x);

template
__global__ void negate_0_1_kernel<float>(const int n, float* x);

template
__global__ void negate_0_1_kernel<double>(const int n, double* x);

template
__global__ void negate_g_kernel<float>(const int n, const float threshold, const float* mask, float* x);

template
__global__ void negate_g_kernel<double>(const int n, const double threshold, const double* mask, double* x);

template
__global__ void negate_l_kernel<float>(const int n, const float threshold, const float* mask, float* x);

template
__global__ void negate_l_kernel<double>(const int n, const double threshold, const double* mask, double* x);

template
__global__ void negate_0_1_g_kernel<float>(const int n, const float threshold, const float* mask, float* x);

template
__global__ void negate_0_1_g_kernel<double>(const int n, const double threshold, const double* mask, double* x);

template
__global__ void negate_0_1_l_kernel<float>(const int n, const float threshold, const float* mask, float* x);

template
__global__ void negate_0_1_l_kernel<double>(const int n, const double threshold, const double* mask, double* x);

template
__global__ void add_with_mask_kernel<float>(const int n, const float* a, const float* bMask, const float* b, float* x);

template
__global__ void add_with_mask_kernel<double>(const int n, const double* a, const double* bMask, const double* b, double* x);
}
