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
__global__ void add_with_mask_kernel_2(const int n, const Dtype* mask, const Dtype* a, const Dtype* b, Dtype* x){
	CUDA_KERNEL_LOOP(index, n){
		x[index] = mask[index] * a[index] +  (1.0 - mask[index]) * b[index];
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

template <typename Dtype>
__global__ void cumulative_sum_kernel(const int n, const Dtype* g_idata, Dtype* g_odata){
	CUDA_KERNEL_LOOP(index, n){
		g_odata[index] = 0;
		for(int i = 0; i <= index; i++){
			g_odata[index] = g_odata[index] + g_idata[i];
		}
	}
}

///
#define BLOCK_SIZE 512 	// You can change this
__global__ void fixup(float *input, float *aux, int len) {
    unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;
    if (blockIdx.x) {
       if (start + t < len)
          input[start + t] += aux[blockIdx.x - 1];
       if (start + BLOCK_SIZE + t < len)
          input[start + BLOCK_SIZE + t] += aux[blockIdx.x - 1];
    }
}

__global__ void scan(float * input, float * output, float *aux, int len) {
    // Load a segment of the input vector into shared memory
    __shared__ float scan_array[BLOCK_SIZE << 1];
    unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;
    if (start + t < len)
       scan_array[t] = input[start + t];
    else
       scan_array[t] = 0;
    if (start + BLOCK_SIZE + t < len)
       scan_array[BLOCK_SIZE + t] = input[start + BLOCK_SIZE + t];
    else
       scan_array[BLOCK_SIZE + t] = 0;
    __syncthreads();

    // Reduction
    int stride;
    for (stride = 1; stride <= BLOCK_SIZE; stride <<= 1) {
       int index = (t + 1) * stride * 2 - 1;
       if (index < 2 * BLOCK_SIZE)
          scan_array[index] += scan_array[index - stride];
       __syncthreads();
    }

    // Post reduction
    for (stride = BLOCK_SIZE >> 1; stride; stride >>= 1) {
       int index = (t + 1) * stride * 2 - 1;
       if (index + stride < 2 * BLOCK_SIZE)
          scan_array[index + stride] += scan_array[index];
       __syncthreads();
    }

    if (start + t < len)
       output[start + t] = scan_array[t];
    if (start + BLOCK_SIZE + t < len)
       output[start + BLOCK_SIZE + t] = scan_array[BLOCK_SIZE + t];

    if (aux && t == 0)
       aux[blockIdx.x] = scan_array[2 * BLOCK_SIZE - 1];
}

template <>
void efficientScan(const int numElements, float* deviceInput, float* deviceOutput)
{
	float *deviceAuxArray;
	float *deviceAuxScannedArray;
    cudaMalloc(&deviceAuxArray, (BLOCK_SIZE << 1) * sizeof(float));
    cudaMalloc(&deviceAuxScannedArray, (BLOCK_SIZE << 1) * sizeof(float));

    int numBlocks = ceil((float)numElements/(BLOCK_SIZE<<1));
    dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, deviceAuxArray, numElements);
    cudaDeviceSynchronize();
    scan<<<dim3(1,1,1), dimBlock>>>(deviceAuxArray, deviceAuxScannedArray, NULL, BLOCK_SIZE << 1);
    cudaDeviceSynchronize();
    fixup<<<dimGrid, dimBlock>>>(deviceOutput, deviceAuxScannedArray, numElements);
    cudaDeviceSynchronize();

    cudaFree(deviceAuxArray);
    cudaFree(deviceAuxScannedArray);
}

template <>
void efficientScan(const int numElements, double* deviceInput, double* deviceOutput)
{
	NOT_IMPLEMENTED;
}
///



/*
template <>
__global__ void cumulative_sum_kernel(const int n, const float* g_idata, float* g_odata){
	extern __shared__ float temp_cumulative_sum_float_kernel[];// allocated on invocation
	int thid = threadIdx.x;
	int offset = 1;

	int ai = thid;
	int bi = thid + (n/2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(ai);
	temp_cumulative_sum_float_kernel[ai + bankOffsetA] = g_idata[ai];
	temp_cumulative_sum_float_kernel[bi + bankOffsetB] = g_idata[bi];

	for (int d = n>>1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			temp_cumulative_sum_float_kernel[bi] += temp_cumulative_sum_float_kernel[ai];
		}
		offset *= 2;
	}
	if (thid==0) {
		temp_cumulative_sum_float_kernel[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
	}
	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			float t = temp_cumulative_sum_float_kernel[ai];
			temp_cumulative_sum_float_kernel[ai] = temp_cumulative_sum_float_kernel[bi];
			temp_cumulative_sum_float_kernel[bi] += t;
		}
	}
	__syncthreads();
	g_odata[ai] = temp_cumulative_sum_float_kernel[ai + bankOffsetA];
	g_odata[bi] = temp_cumulative_sum_float_kernel[bi + bankOffsetB];
}

template <>
__global__ void cumulative_sum_kernel(const int n, const double* g_idata, double* g_odata){
	extern __shared__ double temp_cumulative_sum_double_kernel[];// allocated on invocation
	int thid = threadIdx.x;
	int offset = 1;

	int ai = thid;
	int bi = thid + (n/2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(ai);
	temp_cumulative_sum_double_kernel[ai + bankOffsetA] = g_idata[ai];
	temp_cumulative_sum_double_kernel[bi + bankOffsetB] = g_idata[bi];

	for (int d = n>>1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			temp_cumulative_sum_double_kernel[bi] += temp_cumulative_sum_double_kernel[ai];
		}
		offset *= 2;
	}
	if (thid==0) {
		temp_cumulative_sum_double_kernel[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
	}
	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			float t = temp_cumulative_sum_double_kernel[ai];
			temp_cumulative_sum_double_kernel[ai] = temp_cumulative_sum_double_kernel[bi];
			temp_cumulative_sum_double_kernel[bi] += t;
		}
	}
	__syncthreads();
	g_odata[ai] = temp_cumulative_sum_double_kernel[ai + bankOffsetA];
	g_odata[bi] = temp_cumulative_sum_double_kernel[bi + bankOffsetB];
}
*/

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

template
__global__ void add_with_mask_kernel_2<float>(const int n, const float* mask, const float* a, const float* b, float* x);

template
__global__ void add_with_mask_kernel_2<double>(const int n, const double* mask, const double* a, const double* b, double* x);

template
__global__ void cumulative_sum_kernel(const int n, const float* g_idata, float* g_odata);

template
__global__ void cumulative_sum_kernel(const int n, const double* g_idata, double* g_odata);

}
