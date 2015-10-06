#include "caffe/util/mlg_rng.hpp"


namespace caffe {

__global__ void mlg_init_rng_kernel(unsigned int seed, int n, curandState_t* states) {
	CUDA_KERNEL_LOOP(index, n) {
  /* we have to initialize the state */
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
		  	  index, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[index]);
	}
}

template <typename Dtype>
__global__ void mlg_uniform_kernel(curandState_t* states, int n, Dtype* numbers) {
 CUDA_KERNEL_LOOP(index, n) {
  numbers[index] = curand_uniform(&states[index]);
 }
}

template <typename Dtype>
__global__ void mlg_gumbel_kernel(int n, Dtype* numbers) {
 CUDA_KERNEL_LOOP(index, n) {
  numbers[index] = log( numbers[index] ) - log( (Dtype)1.0 - numbers[index] );
 }
}

template <typename Dtype>
void MLGRNG<Dtype>::mlg_gpu_uniform(const int N, Dtype* data){
	if(stateCount < N)
	{
		//LOG(INFO) << "Random states reshaped " << stateCount << " to " << N << " " << std::endl;
		if(stateCount != 0)
		{
			cudaFree(states);
		}
		cudaMalloc((void**) &states, N * sizeof(curandState_t));
		stateCount = N;

		mlg_init_rng_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(time(0), N, states);
		CUDA_POST_KERNEL_CHECK;

		//LOG(INFO) << "Done" << std::endl;
	}

	mlg_uniform_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(states, N, data);
	CUDA_POST_KERNEL_CHECK;

}

template <typename Dtype>
__global__ void mlg_fix_0_1_kernel(int n, Dtype* numbers) {
 CUDA_KERNEL_LOOP(index, n) {
  if(numbers[index] < MLG_MIN_UNI){
   numbers[index] = MLG_MIN_UNI;
  }
  if(numbers[index] > MLG_MAX_UNI){
   numbers[index] = MLG_MAX_UNI;
  }
 }
}

template <typename Dtype>
void MLGRNG<Dtype>::mlg_gpu_gumbel(const int N, Dtype* data){
	mlg_gpu_uniform(N, data);

	// add or substract small value to avoid nans etc
	mlg_fix_0_1_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, data);
	CUDA_POST_KERNEL_CHECK;

	mlg_gumbel_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, data);
	CUDA_POST_KERNEL_CHECK;
}




template <typename Dtype>
__global__ void mlg_set_index_kernel(int n, int* numbers) {
 CUDA_KERNEL_LOOP(index, n) {
  numbers[index] = index;
 }
}


template <typename Dtype>
void MLGRNG<Dtype>::mlg_gpu_permutation(const int N, int* data){

	int* tmp = new int[N];
	mlg_cpu_permutation(N, tmp);
	cudaMemcpy(data, tmp, N * sizeof(int), cudaMemcpyHostToDevice);
	delete tmp;

	/*
	mlg_set_index_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, data);
	CUDA_POST_KERNEL_CHECK;

	int* range;
	cudaMalloc((void**) &range, N * sizeof(int));

	mlg_gpu_range(N, 0, N-1, range);

	int* local2 = new int[N];
	cudaMemcpy(local2, range, N * sizeof(int), cudaMemcpyDeviceToHost);

	int* tmp;
	cudaMalloc((void**) &tmp,  sizeof(int));
	for(int i = 0; i < N; i++){
		cudaMemcpy(tmp, data + i, 1 * sizeof(int), cudaMemcpyDeviceToDevice);
		cudaMemcpy(data + i, data + (range[i]), 1 * sizeof(int), cudaMemcpyDeviceToDevice);
		cudaMemcpy(data + (range[i]), tmp, 1 * sizeof(int), cudaMemcpyDeviceToDevice);
	}

	int* local1 = new int[N];
	cudaMemcpy(local1, data, N * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(range);
	*/
}

template <typename Dtype>
__global__ void mlg_set_min_max_kernel(int n, int min, int max, Dtype* rand, int* numbers) {
 CUDA_KERNEL_LOOP(index, n) {
  numbers[index] = (((int)(rand[index] * 100000)) % (max - min + 1) ) + min;
 }
}

template <typename Dtype>
void MLGRNG<Dtype>::mlg_gpu_range(const int N, const int min, const int max, int* data){
	Dtype* tmp;
	cudaMalloc((void**) &tmp, N * sizeof(Dtype));
	mlg_gpu_uniform(N, tmp);

	mlg_set_min_max_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, min, max, tmp, data);
	CUDA_POST_KERNEL_CHECK;

	cudaFree(tmp);
}

template
void MLGRNG<float>::mlg_gpu_uniform( \
		const int N, \
		float* data);

template
void MLGRNG<double>::mlg_gpu_uniform( \
		const int N, \
		double* data);

template
void MLGRNG<float>::mlg_gpu_gumbel( \
		const int N, \
		float* data);

template
void MLGRNG<double>::mlg_gpu_gumbel( \
		const int N, \
		double* data);

template
void MLGRNG<float>::mlg_gpu_permutation(const int N, int* data);

template
void MLGRNG<double>::mlg_gpu_permutation(const int N, int* data);

template
void MLGRNG<float>::mlg_gpu_range(const int N, const int min, const int max, int* data);

template
void MLGRNG<double>::mlg_gpu_range(const int N, const int min, const int max, int* data);
}
