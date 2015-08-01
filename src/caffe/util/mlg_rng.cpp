#include "caffe/util/mlg_rng.hpp"


namespace caffe {

template <typename Dtype>
MLGRNG<Dtype>::MLGRNG(){
	stateCount = 0;
	states = 0;
}

template <typename Dtype>
MLGRNG<Dtype>::~MLGRNG(){
	  if(Caffe::mode() == Caffe::GPU)
	  {
	  #ifndef CPU_ONLY
		  if(stateCount > 0)
		  {
			  cudaFree(states);
		  }
	  #else
		NO_GPU;
	  #endif
	  }
}

template <typename Dtype>
void MLGRNG<Dtype>::mlg_cpu_uniform(const int N, Dtype* data){
    caffe_rng_uniform<Dtype>(N, Dtype(0.), Dtype(1.), data);
}

template <typename Dtype>
void MLGRNG<Dtype>::mlg_cpu_gumbel(const int N, Dtype* data){
	mlg_cpu_uniform(N, data);
	for(int i = 0; i < N; i++){
		data[i] = -( log( data[i] ) - log( 1 - data[i] ) );
	}
}

template <typename Dtype>
void MLGRNG<Dtype>::mlg_cpu_permutation(const int N, int* data){
	for(int i = 0; i < N; i++){
		data[i] = i;
	}

	int* order = new int[N];
	mlg_cpu_range(N, 0, N-1, order);

	for(int i = 0; i < N; i++){
		int tmp = data[i];
		data[i] = data[order[i]];
		data[order[i]] = tmp;
	}

	delete order;
}

template <typename Dtype>
void MLGRNG<Dtype>::mlg_cpu_range(const int N, const int min, const int max, int* data){
	Dtype* tmp = new Dtype[N];

	caffe_rng_uniform<Dtype>(N, Dtype(0.), Dtype(1.), tmp);
/*
	Dtype* device;
	cudaMalloc((void**) &device, N * sizeof(Dtype));
	mlg_gpu_uniform(N, device);
	cudaMemcpy(tmp, device, N * sizeof(Dtype), cudaMemcpyDeviceToHost);
	cudaFree(device);
*/

	for(int i = 0; i < N; i++){
		data[i] = (((int)(tmp[i] * 100000)) % (max - min + 1) ) + min;
	}

	delete tmp;
}

template MLGRNG<float>::MLGRNG();
template MLGRNG<double>::MLGRNG();

template MLGRNG<float>::~MLGRNG();
template MLGRNG<double>::~MLGRNG();

template
void MLGRNG<float>::mlg_cpu_uniform(const int N, float* data);

template
void MLGRNG<double>::mlg_cpu_uniform(const int N, double* data);

template
void MLGRNG<float>::mlg_cpu_gumbel(const int N, float* data);

template
void MLGRNG<double>::mlg_cpu_gumbel(const int N, double* data);

template
void MLGRNG<float>::mlg_cpu_permutation(const int N, int* data);

template
void MLGRNG<double>::mlg_cpu_permutation(const int N, int* data);

template
void MLGRNG<float>::mlg_cpu_range(const int N, const int min, const int max, int* data);

template
void MLGRNG<double>::mlg_cpu_range(const int N, const int min, const int max, int* data);


}
