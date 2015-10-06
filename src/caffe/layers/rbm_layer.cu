#include <vector>

#include <curand.h>
#include <curand_kernel.h>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/rbm_layers.hpp"
#include "caffe/util/mlg_math.hpp"

namespace caffe {

template <typename Dtype>
void RBMLayer<Dtype>::replicate_data_gpu(const int N, const int R, const Dtype* src, Dtype* dst){
    replicate_kernel<Dtype><<<CAFFE_GET_BLOCKS(N*R), CAFFE_CUDA_NUM_THREADS>>>(N, R, src, dst);
    CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void RBMLayer<Dtype>::replicate_data_gpu(const int N, Blob<Dtype>* X, Blob<Dtype>* repX){

	const int axis = X->CanonicalAxisIndex(this->layer_param_.rbm_param().axis());

	vector<int> X_shape = X->shape();

    vector<int> repX_shape(2);
    repX_shape[0] = X_shape[0] * N;
    repX_shape[1] = X->count(axis);

    repX->Reshape(repX_shape);

    replicate_kernel<Dtype><<<CAFFE_GET_BLOCKS(repX->count()), CAFFE_CUDA_NUM_THREADS>>>(X->count(), repX->count(), X->gpu_data(), repX->mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;

	if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(X->count(), X->gpu_data())) LOG(INFO) << "X not finite" << std::endl;
	if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(repX->count(), repX->gpu_data())) LOG(INFO) << "repX not finite" << std::endl;
}

template <typename Dtype>
void RBMLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	// H  = 1 * X * W(T) + 0 * H
	// top_data = 1 * bottom_data * weight(T) + 0 * top_data
	// [m,n] = 1 * [m,k] * [k,n] + 0 * [m,n]
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
			M_, N_, K_,
			(Dtype)1., bottom[0]->gpu_data(), this->blobs_[0]->gpu_data(),
			(Dtype)0., this->H0.mutable_gpu_data());

	// H = 1 * cM * C + 1 * H
	// top_data = 1 * bias_c_multiplier * c + 1 * top_data
	// [m,n] = 1 * [m,1] * [1,n] + 1 * [m,n]
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
			M_, N_, 1,
			(Dtype)1., ones_m_.gpu_data(), this->blobs_[2]->gpu_data(),
			(Dtype)1., this->H0.mutable_gpu_data());

	this->sigmoid_gpu(top[0]->count(), this->H0.mutable_gpu_data());

	this->sample_gpu(top[0]->count(), this->H0.gpu_data(), top[0]->mutable_gpu_data());

	top[2]->mutable_cpu_data()[0] = this->ll_gpu(top, bottom);

	if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(bottom[0]->count(), bottom[0]->gpu_data())) LOG(INFO) << "X0S not finite" << std::endl;
	if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(this->H0.count(), this->H0.gpu_data())) LOG(INFO) << "H0 not finite" << std::endl;
	if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(top[0]->count(), top[0]->gpu_data())) LOG(INFO) << "H1S not finite" << std::endl;
}

template <typename Dtype>
void RBMLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

	// calculate gradient
	gradient_gpu(top, propagate_down, bottom);

	if(bottom.size() == 2) // layer has second input for reconstruction
	{
		// calculate layer backward output
		const Dtype* H1SData = top[1]->gpu_data();
		Dtype* X1SData = bottom[1]->mutable_gpu_data();

		const Dtype* W = this->blobs_[0]->gpu_data();
		const Dtype* b = this->blobs_[1]->gpu_data();
		const Dtype* c = this->blobs_[2]->gpu_data();

		// X = 1 * H * W + 0 * X
		// bottom_data = 1 * top_data * weights + 0 * bottom_data
		// [m,k] = 1 * [m,n] * [n,k] + 0 * [m,k]
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
				M_, K_, N_,
				(Dtype)1., H1SData, W,
				(Dtype)0., X1SData);


		// X = 1 * bM * b + 1 * X
		// bottom_data = 1 * bias_b_multiplier * b + 1 * bottom_data
		// [m,k] = 1 * [m,1] * [1,k] + 1 * [m,k]
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
				M_, K_, 1,
				(Dtype)1., ones_m_.gpu_data(), b,
				(Dtype)1., X1SData);


		sigmoid_gpu(bottom[1]->count(), X1SData);

		sample_gpu(bottom[1]->count(),  X1SData);
	}
}

template <typename Dtype>
void RBMLayer<Dtype>::gradient_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

	  if (this->param_propagate_down_[0]) {
		// calculate gradient with respect to W
		  LOG(FATAL) << this->type()
				  << " Layer cannot calculate W gradient.";
	  }

	  if (this->param_propagate_down_[1]) {
		// calculate gradient with respect to b
		  LOG(FATAL) << this->type()
				  << " Layer cannot calculate b gradient.";
	  }

	  if (this->param_propagate_down_[2]) {
		// calculate gradient with respect to c
		  LOG(FATAL) << this->type()
				  << " Layer cannot calculate c gradient.";
	  }
}

template <typename Dtype>
Dtype RBMLayer<Dtype>::ll_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom)
{
	Dtype loss = 0;

		switch(this->layer_param_.rbm_param().llaprox())
		{
			case RBMLayer::AIS:
			{}
			break;
			case RBMLayer::RAIS:
			{}
			break;
			case RBMLayer::CSL:
			{}
			break;
			case RBMLayer::REC:
			{
				Blob<Dtype> xTmp;
				xTmp.ReshapeLike(this->X1S);

				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
						M_, K_, N_,
						(Dtype)1., top[0]->gpu_data(), this->blobs_[0]->gpu_data(),
						(Dtype)0., xTmp.mutable_gpu_data());

				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
						M_, K_, 1,
						(Dtype)1., ones_m_.gpu_data(), this->blobs_[1]->gpu_data(),
						(Dtype)1., xTmp.mutable_gpu_data());

				int count = xTmp.count();
				this->sigmoid_gpu(count, xTmp.mutable_gpu_data());

				this->sample_gpu(count, xTmp.mutable_gpu_data());

				caffe_gpu_sub<Dtype>(xTmp.count(), bottom[0]->gpu_data(), xTmp.mutable_gpu_data(), xTmp.mutable_gpu_data());

				Dtype r;
				caffe_gpu_asum<Dtype>(xTmp.count(), xTmp.gpu_data(), &r);
				r = r / (Dtype)count;

				loss = r;

			}
			break;
			default:
			{
				LOG(INFO) << "No such ll approx";
			}
			break;
		}



		return loss;
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
void RBMLayer<Dtype>::sample_gpu(int N, Dtype* mat)
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
void RBMLayer<Dtype>::sample_gpu(int N, const Dtype* src, Dtype* dst)
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
void RBMLayer<Dtype>::sigmoid_gpu(int count, Dtype* data)
{
	sigmoid_gpu2<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			count, data);
	CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(RBMLayer);

template void RBMLayer<float>::sigmoid_gpu( \
		int count, \
		float* data);

template void RBMLayer<double>::sigmoid_gpu( \
		int count, \
		double* data);

template void RBMLayer<float>::sample_gpu( \
		int N, \
		float* mat);

template void RBMLayer<double>::sample_gpu( \
		int N, \
		double* mat);

template void RBMLayer<float>::sample_gpu( \
		int N, \
		const float* src, float* dst);

template void RBMLayer<double>::sample_gpu( \
		int N, \
		const double* src, double* dst);

template void RBMLayer<float>::gradient_gpu( \
     const std::vector<Blob<float>*>& top, \
     const std::vector<bool>& propagate_down, \
     const std::vector<Blob<float>*>& bottom);

template void RBMLayer<double>::gradient_gpu( \
     const std::vector<Blob<double>*>& top, \
     const std::vector<bool>& propagate_down, \
     const std::vector<Blob<double>*>& bottom);

template float RBMLayer<float>::ll_gpu( \
     const std::vector<Blob<float>*>& top, \
     const std::vector<Blob<float>*>& bottom);

template double RBMLayer<double>::ll_gpu( \
     const std::vector<Blob<double>*>& top, \
     const std::vector<Blob<double>*>& bottom);

template
void RBMLayer<float>::replicate_data_gpu(const int N, const int R, const float* src, float* dst);

template
void RBMLayer<double>::replicate_data_gpu(const int N, const int R, const double* src, double* dst);

template
void RBMLayer<float>::replicate_data_gpu(const int N, Blob<float>* src, Blob<float>* dst);

template
void RBMLayer<double>::replicate_data_gpu(const int N, Blob<double>* src, Blob<double>* dst);


} // namespace caffe
