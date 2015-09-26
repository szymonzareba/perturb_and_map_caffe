#include "caffe/util/mlg_assert.hpp"


namespace caffe {


template <typename Dtype>
__global__ void mlg_check_not_finite_kernel(const int n, const Dtype* data, Dtype* result) {
 CUDA_KERNEL_LOOP(index, n) {
	 if(isfinite(data[index])){
		 //is finite
		 result[index] = (Dtype)0.;
	 }else{
		 //not finite
		 result[index] = (Dtype)1.;
	 }
 }
}

template <typename Dtype>
__global__ void mlg_check_range_kernel(const int n, const Dtype* data, const Dtype min, const Dtype max, Dtype* result) {
 CUDA_KERNEL_LOOP(index, n) {
	 if(data[index] >= min && data[index] <= max){
		 //is ok
		 result[index] = (Dtype)0.;
	 }else{
		 //not not
		 result[index] = (Dtype)1.;
	 }
 }
}

template <typename Dtype>
__global__ void mlg_check_not_range_kernel(const int n, const Dtype* data, const Dtype min, const Dtype max, Dtype* result) {
 CUDA_KERNEL_LOOP(index, n) {
	 if(data[index] <= min || data[index] >= max){
		 //is ok
		 result[index] = (Dtype)0.;
	 }else{
		 //not not
		 result[index] = (Dtype)1.;
	 }
 }
}

template <typename Dtype>
bool MLGASSERT<Dtype>::mlg_gpu_finite(const int N, const Dtype* data){
	if(work){
		Dtype* result;
		cudaMalloc((void**) &result, N * sizeof(Dtype));

		mlg_check_not_finite_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, data, result);
		CUDA_POST_KERNEL_CHECK;

		Dtype asum;
		caffe_gpu_asum(N, result, &asum);

		cudaFree(result);

		return asum > 0;
	} else {
		return 0;
	}
}

template <typename Dtype>
bool MLGASSERT<Dtype>::mlg_gpu_range(const int N, const Dtype* data){
	if(work){
		Dtype min = 1e-44;
		Dtype max = 1e+37;

		Dtype* result1;
		cudaMalloc((void**) &result1, N * sizeof(Dtype));

		Dtype* result2;
		cudaMalloc((void**) &result2, N * sizeof(Dtype));

		Dtype* result3;
		cudaMalloc((void**) &result3, N * sizeof(Dtype));

		mlg_check_not_range_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, data, (Dtype)0., min, result1);
		CUDA_POST_KERNEL_CHECK;

		mlg_check_not_range_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, data, -min, (Dtype)0., result2);
		CUDA_POST_KERNEL_CHECK;

		mlg_check_range_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, data, -max, max, result3);
		CUDA_POST_KERNEL_CHECK;

		Dtype asum1;
		caffe_gpu_asum(N, result1, &asum1);

		Dtype asum2;
		caffe_gpu_asum(N, result2, &asum2);

		Dtype asum3;
		caffe_gpu_asum(N, result3, &asum3);

		cudaFree(result1);
		cudaFree(result2);
		cudaFree(result3);

		return asum1 > 0 || asum2 > 0 || asum3 > 0;
	} else{
		return 0;
	}
}

template <typename Dtype>
bool MLGASSERT<Dtype>::mlg_gpu_range(const int N, const Dtype* data, const Dtype min, const Dtype max){
	if(work){
		Dtype* result;
		cudaMalloc((void**) &result, N * sizeof(Dtype));

		mlg_check_range_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, data, min, max, result);
		CUDA_POST_KERNEL_CHECK;

		Dtype asum;
		caffe_gpu_asum(N, result, &asum);

		cudaFree(result);

		return asum > 0;
	} else{
		return 0;
	}
}

template
bool MLGASSERT<float>::mlg_gpu_finite(const int N, const float* data);

template
bool MLGASSERT<double>::mlg_gpu_finite(const int N, const double* data);

template
bool MLGASSERT<float>::mlg_gpu_range(const int N, const float* data);

template
bool MLGASSERT<double>::mlg_gpu_range(const int N, const double* data);

template
bool MLGASSERT<float>::mlg_gpu_range(const int N, const float* data, const float min, const float max);

template
bool MLGASSERT<double>::mlg_gpu_range(const int N, const double* data, const double min, const double max);

}
