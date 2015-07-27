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
  numbers[index] = log( 1 - numbers[index] ) - log( numbers[index] );
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
void MLGRNG<Dtype>::mlg_gpu_gumbel(const int N, Dtype* data){
	mlg_gpu_uniform(N, data);
	mlg_gumbel_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, data);
	CUDA_POST_KERNEL_CHECK;
}

template void MLGRNG<float>::mlg_gpu_uniform( \
		const int N, \
		float* data);

template void MLGRNG<double>::mlg_gpu_uniform( \
		const int N, \
		double* data);

template void MLGRNG<float>::mlg_gpu_gumbel( \
		const int N, \
		float* data);

template void MLGRNG<double>::mlg_gpu_gumbel( \
		const int N, \
		double* data);

}
