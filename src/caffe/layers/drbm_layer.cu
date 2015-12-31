#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/drbm_layers.hpp"
#include "caffe/util/mlg_math.hpp"

namespace caffe {

template <typename Dtype>
void DRBMLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	// randomize previous data
	//LOG(INFO) << "Randomizing" << std::endl;
	for(int i = 0; i < probs_0.size(); i++)
	{
		caffe_gpu_rng_uniform(probs_0[i]->count(), (Dtype)0., (Dtype)1., probs_0[i]->mutable_gpu_data());
		caffe_gpu_rng_uniform(states_0[i]->count(), (Dtype)0., (Dtype)1., states_0[i]->mutable_gpu_data());
	}

	// copy visible data
	//LOG(INFO) << "Copy visible" << std::endl;
	caffe_copy(probs_0[0]->count(), bottom[0]->gpu_data(), probs_0[0]->mutable_gpu_data());
	caffe_copy(states_0[0]->count(), bottom[0]->gpu_data(), states_0[0]->mutable_gpu_data());

	//LOG(INFO) << "MF" << std::endl;
	for(int mf_step = 0; mf_step < this->layer_param_.drbm_param().mf_steps(); mf_step++)
	{
		//LOG(INFO) << "MF " << mf_step << std::endl;

		for(int layer_num = 1; layer_num < probs_0.size(); layer_num += 2)
		{
			//LOG(INFO) << "Layer " << layer_num << std::endl;
			// clear previous and add bias
			//LOG(INFO) << "Bias" << std::endl;
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
					M_, layer_sizes[layer_num], 1,
					(Dtype)1., ones_m.gpu_data(), biases[layer_num]->gpu_data(),
					(Dtype)0., probs_0[layer_num]->mutable_gpu_data());

			// multiply and add previous layer
			//LOG(INFO) << "Previous" << std::endl;
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
					M_, layer_sizes[layer_num], layer_sizes[layer_num - 1],
					(Dtype)1., probs_0[layer_num - 1]->gpu_data(), weights[layer_num - 1]->gpu_data(),
					(Dtype)1., probs_0[layer_num]->mutable_gpu_data());


			// check if next layer exists
			if(layer_num + 1 < probs_0.size())
			{
				// multiply and add next layer
				//LOG(INFO) << "Next" << std::endl;
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
						M_, layer_sizes[layer_num], layer_sizes[layer_num + 1],
						(Dtype)1., probs_0[layer_num + 1]->gpu_data(), weights[layer_num]->gpu_data(),
						(Dtype)1., probs_0[layer_num]->mutable_gpu_data());
			}

			//LOG(INFO) << "Sigmoid" << std::endl;
			sigmoid_gpu(probs_0[layer_num]->count(), probs_0[layer_num]->mutable_gpu_data());
		}

		for(int layer_num = 2; layer_num < probs_0.size(); layer_num += 2)
		{
			//LOG(INFO) << "Layer " << layer_num << std::endl;
			// clear previous and add bias
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
					M_, layer_sizes[layer_num], 1,
					(Dtype)1., ones_m.gpu_data(), biases[layer_num]->gpu_data(),
					(Dtype)0., probs_0[layer_num]->mutable_gpu_data());

			// multiply and add previous layer
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
					M_, layer_sizes[layer_num], layer_sizes[layer_num - 1],
					(Dtype)1., probs_0[layer_num - 1]->gpu_data(), weights[layer_num - 1]->gpu_data(),
					(Dtype)1., probs_0[layer_num]->mutable_gpu_data());


			// check if next layer exists
			if(layer_num + 1 < probs_0.size())
			{
				// multiply and add next layer
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
						M_, layer_sizes[layer_num], layer_sizes[layer_num + 1],
						(Dtype)1., probs_0[layer_num + 1]->gpu_data(), weights[layer_num]->gpu_data(),
						(Dtype)1., probs_0[layer_num]->mutable_gpu_data());
			}

			sigmoid_gpu(probs_0[layer_num]->count(), probs_0[layer_num]->mutable_gpu_data());
		}
	}

	// sample probs to states
	//LOG(INFO) << "Sample" << std::endl;
	for(int i = 0; i < probs_0.size(); i++)
	{
		sample_gpu(probs_0[i]->count(), probs_0[i]->gpu_data(), states_0[i]->mutable_gpu_data());
	}

	top[2]->mutable_cpu_data()[0] = ll_gpu();
}

template <typename Dtype>
__global__ void fix_0_1(const int n, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
	if(y[index] == (Dtype) 0.0){
		y[index] = (Dtype) 0.00001;
	}
	if(y[index] == (Dtype) 1.0){
		y[index] = (Dtype) ( 1.0 - 0.00001 );
	}
  }
}

template <typename Dtype>
__global__ void add_log(const int n, const Dtype* x0,  const Dtype* x1,  Dtype* combined) {
  CUDA_KERNEL_LOOP(index, n) {
	  if( x0[index] < 0.5 ){
		  combined[index] = log( 1.0 - x1[index] );
	  }
	  else{
		  combined[index] = log( x1[index] );
	  }
  }
}

template <typename Dtype>
Dtype DRBMLayer<Dtype>::ll_gpu()
{

	caffe_copy(probs_1[1]->count(), probs_0[1]->gpu_data() , probs_1[1]->mutable_gpu_data());
	caffe_copy(states_1[1]->count(), states_0[1]->gpu_data() , states_1[1]->mutable_gpu_data());

	int layer_num = 0;

	// clear previous and add bias
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
			M_, layer_sizes[layer_num], 1,
			(Dtype)1., ones_m.gpu_data(), biases[layer_num]->gpu_data(),
			(Dtype)0., probs_1[layer_num]->mutable_gpu_data());

	// multiply and add next layer
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
			M_, layer_sizes[layer_num], layer_sizes[layer_num + 1],
			(Dtype)1., probs_1[layer_num + 1]->gpu_data(), weights[layer_num]->gpu_data(),
			(Dtype)1., probs_1[layer_num]->mutable_gpu_data());


	sigmoid_gpu(probs_1[layer_num]->count(), probs_1[layer_num]->mutable_gpu_data());
	sample_gpu(probs_1[layer_num]->count(), probs_1[layer_num]->gpu_data(), states_1[layer_num]->mutable_gpu_data());



	Blob<Dtype> combined;
	combined.ReshapeLike(*probs_1[0]);

	caffe_gpu_sub(combined.count(), probs_0[0]->gpu_data(), probs_1[0]->gpu_data(), combined.mutable_gpu_data());

	// combined = x = [M,K]
	Dtype r;

	// sum over k and M
	caffe_gpu_asum<Dtype>(combined.count(), combined.gpu_data(), &r);
	// divide by m
	r = r / this->M_;

	return r;
}

/*
template <typename Dtype>
Dtype DRBMLayer<Dtype>::ll_gpu()
{

	// randomize previous data
	for(int i = 0; i < probs_1.size(); i++)
	{
		caffe_gpu_rng_uniform(probs_1[i]->count(), (Dtype)0., (Dtype)1., probs_1[i]->mutable_gpu_data());
		caffe_gpu_rng_uniform(states_1[i]->count(), (Dtype)0., (Dtype)1., states_1[i]->mutable_gpu_data());
	}

	// copy last hidden layer data
	caffe_copy(probs_1[probs_1.size() - 1]->count(), probs_0[probs_1.size() - 1]->gpu_data(), probs_1[probs_1.size() - 1]->mutable_gpu_data());
	caffe_copy(states_1[states_1.size() - 1]->count(), states_0[probs_1.size() - 1]->gpu_data(), states_1[states_1.size() - 1]->mutable_gpu_data());

	for(int mf_step = 0; mf_step < this->layer_param_.drbm_param().mf_steps(); mf_step++)
	{
		for(int layer_num = probs_1.size()-2; layer_num >= 0; layer_num -= 2)
		{
			// clear previous and add bias
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
					M_, layer_sizes[layer_num], 1,
					(Dtype)1., ones_m.gpu_data(), biases[layer_num]->gpu_data(),
					(Dtype)0., probs_1[layer_num]->mutable_gpu_data());

			// multiply and add next layer
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
					M_, layer_sizes[layer_num], layer_sizes[layer_num + 1],
					(Dtype)1., probs_1[layer_num + 1]->gpu_data(), weights[layer_num]->gpu_data(),
					(Dtype)1., probs_1[layer_num]->mutable_gpu_data());

			// check if previous layer exists
			if(layer_num - 1 >= 0)
			{
				// multiply and add next layer
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
						M_, layer_sizes[layer_num], layer_sizes[layer_num - 1],
						(Dtype)1., probs_1[layer_num - 1]->gpu_data(), weights[layer_num - 1]->gpu_data(),
						(Dtype)1., probs_1[layer_num]->mutable_gpu_data());
			}

			sigmoid_gpu(probs_1[layer_num]->count(), probs_1[layer_num]->mutable_gpu_data());
		}

		for(int layer_num = probs_1.size()-3; layer_num >= 0; layer_num -= 2)
		{
			// clear previous and add bias
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
					M_, layer_sizes[layer_num], 1,
					(Dtype)1., ones_m.gpu_data(), biases[layer_num]->gpu_data(),
					(Dtype)0., probs_1[layer_num]->mutable_gpu_data());

			// multiply and add next layer
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
					M_, layer_sizes[layer_num], layer_sizes[layer_num + 1],
					(Dtype)1., probs_1[layer_num + 1]->gpu_data(), weights[layer_num]->gpu_data(),
					(Dtype)1., probs_1[layer_num]->mutable_gpu_data());

			// check if previous layer exists
			if(layer_num - 1 >= 0)
			{
				// multiply and add next layer
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
						M_, layer_sizes[layer_num], layer_sizes[layer_num - 1],
						(Dtype)1., probs_1[layer_num - 1]->gpu_data(), weights[layer_num - 1]->gpu_data(),
						(Dtype)1., probs_1[layer_num]->mutable_gpu_data());
			}

			sigmoid_gpu(probs_1[layer_num]->count(), probs_1[layer_num]->mutable_gpu_data());
		}
	}

	// sample probs to states
	for(int i = 0; i < probs_1.size(); i++)
	{
		sample_gpu(probs_1[i]->count(), probs_1[i]->gpu_data(), states_1[i]->mutable_gpu_data());
	}


	//x1(x1==0) = eps;
	//x1(x1==1) = 1-eps;
	fix_0_1<Dtype><<<CAFFE_GET_BLOCKS(probs_1[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(probs_1[0]->count(), probs_1[0]->mutable_gpu_data());
	CUDA_POST_KERNEL_CHECK;

	//error = -mean(sum( x0 .* log( x1 ) + (1-x0) .* log( (1-x1) ) ));
	Blob<Dtype> combined;
	combined.ReshapeLike(*probs_1[0]);

	//x0 .* log( x1 )+ (1-x0) .* log( (1-x1) )
	add_log<Dtype><<<CAFFE_GET_BLOCKS(probs_1[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(probs_1[0]->count(),
			probs_0[0]->gpu_data(), probs_1[0]->gpu_data(),
			combined.mutable_gpu_data());
	CUDA_POST_KERNEL_CHECK;

	// combined = x = [M,K]
	Dtype r;

	// sum over k and M
	caffe_gpu_asum<Dtype>(combined.count(), combined.gpu_data(), &r);
	// divide by m
	r = r / this->M_;

	return r;
}
*/

template <typename Dtype>
void DRBMLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}



template <typename Dtype>
void DRBMLayer<Dtype>::gradient_gpu() {
	//TODO calculate and set blob_diffs

	Dtype scalar = -1. / this->M_;

	for(int i = 0; i < biases.size(); i++)
	{
		Blob<Dtype> tmp(probs_0[i]->shape());
		caffe_gpu_set(tmp.count(), (Dtype)0, tmp.mutable_gpu_data());
		caffe_gpu_sub(tmp.count(), states_0[i]->gpu_data(), states_1[i]->gpu_data(), tmp.mutable_gpu_data());

		caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
				layer_sizes[i], 1, this->M_,
		    	(Dtype)1., tmp.gpu_data(), ones_m.gpu_data(),
		    	(Dtype)0., biases[i]->mutable_gpu_diff());

		caffe_gpu_scal<Dtype>(biases[i]->count(), scalar, biases[i]->mutable_gpu_diff());
	}

	for(int i = 0; i < weights.size(); i++)
	{
		Blob<Dtype> tmp1(weights[i]->shape());
		caffe_gpu_set(tmp1.count(), (Dtype)0, tmp1.mutable_gpu_data());

		Blob<Dtype> tmp2(weights[i]->shape());
		caffe_gpu_set(tmp2.count(), (Dtype)0, tmp2.mutable_gpu_data());

	    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
	    		layer_sizes[i + 1], layer_sizes[i], M_,
	    		(Dtype)1., states_0[i + 1]->gpu_data(), states_0[i]->gpu_data(),
	    		(Dtype)0., tmp1.mutable_gpu_data());

    	caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			layer_sizes[i + 1], layer_sizes[i], M_,
    			(Dtype)1., states_1[i + 1]->gpu_data(), states_1[i]->gpu_data(),
	    		(Dtype)0., tmp2.mutable_gpu_data());

	    caffe_gpu_sub(tmp1.count(), tmp1.gpu_data(), tmp2.gpu_data(), weights[i]->mutable_gpu_diff());
	    caffe_gpu_scal<Dtype>(weights[i]->count(), scalar, weights[i]->mutable_gpu_diff());
	}
}


template <typename Dtype>
__global__ void sample_gpu2(const int n, Dtype* data, const Dtype* randoms) {
  CUDA_KERNEL_LOOP(index, n) {
	if(data[index] > randoms[index])
	{
		data[index] = 1;
	}
	else
	{
		data[index] = 0;
	}
  }
}

template <typename Dtype>
__global__ void sample_gpu2(const int n, const Dtype* src, Dtype* dst, const Dtype* randoms) {
  CUDA_KERNEL_LOOP(index, n) {
	if(src[index] > randoms[index])
	{
		dst[index] = 1;
	}
	else
	{
		dst[index] = 0;
	}
  }
}

template <typename Dtype>
void DRBMLayer<Dtype>::sample_gpu(int N, Dtype* mat)
{
	vector<int> shape(2);
	shape[0] = N;
	shape[1] = 1;
	randomContainer.Reshape(shape);

	MLGRNG<Dtype>::getInstance().mlg_gpu_uniform(N, randomContainer.mutable_gpu_data());
	//caffe_gpu_set(N, (Dtype)0.5, randomContainer.mutable_gpu_data());

	sample_gpu2<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
			N, mat, randomContainer.gpu_data());
	CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void DRBMLayer<Dtype>::sample_gpu(int N, const Dtype* src, Dtype* dst)
{
	vector<int> shape(2);
	shape[0] = N;
	shape[1] = 1;
	randomContainer.Reshape(shape);

	MLGRNG<Dtype>::getInstance().mlg_gpu_uniform(N, randomContainer.mutable_gpu_data());
	//caffe_gpu_set(N, (Dtype)0.5, randomContainer.mutable_gpu_data());

	sample_gpu2<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
			N, src, dst, randomContainer.gpu_data());
	CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void sigmoid_gpu2(const int n, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = 1. / (1. + exp(-out[index]));
  }
}

template <typename Dtype>
void DRBMLayer<Dtype>::sigmoid_gpu(int count, Dtype* data)
{
	sigmoid_gpu2<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			count, data);
	CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(DRBMLayer);

template float DRBMLayer<float>::ll_gpu();

template double DRBMLayer<double>::ll_gpu();

template void DRBMLayer<float>::gradient_gpu();

template void DRBMLayer<double>::gradient_gpu();

template void DRBMLayer<float>::sigmoid_gpu( \
		int count, \
		float* data);

template void DRBMLayer<double>::sigmoid_gpu( \
		int count, \
		double* data);

template void DRBMLayer<float>::sample_gpu( \
		int N, \
		float* mat);

template void DRBMLayer<double>::sample_gpu( \
		int N, \
		double* mat);

template void DRBMLayer<float>::sample_gpu( \
		int N, \
		const float* src, float* dst);

template void DRBMLayer<double>::sample_gpu( \
		int N, \
		const double* src, double* dst);

} // namespace caffe
