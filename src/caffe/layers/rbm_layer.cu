#include <vector>

#include <curand.h>
#include <curand_kernel.h>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/rbm_layers.hpp"

namespace caffe {

template <typename Dtype>
void RBMLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* X0S = bottom[0]->gpu_data();
	Dtype* H0S = top[0]->mutable_gpu_data();
	const Dtype* W = this->blobs_[0]->gpu_data();
	const Dtype* c = this->blobs_[2]->gpu_data();

	  // H  = 1 * X * W(T) + 0 * H
	  // top_data = 1 * bottom_data * weight(T) + 0 * top_data
	  // [m,n] = 1 * [m,k] * [k,n] + 0 * [m,n]
	  // OK
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
			M_, N_, K_,
			(Dtype)1., X0S, W,
			(Dtype)0., H0S);

	  // H = 1 * cM * C + 1 * H
	  // top_data = 1 * bias_c_multiplier * c + 1 * top_data
	  // [m,n] = 1 * [m,1] * [1,n] + 1 * [m,n]
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
			M_, N_, 1,
			(Dtype)1., ones_m_.gpu_data(), c,
			(Dtype)1., H0S);

	int count = top[0]->count();
	sigmoid_gpu(count, top[0]->mutable_gpu_data());

	sample_gpu(count, top[0]->mutable_gpu_data());

	top[2]->mutable_cpu_data()[0] = ll_gpu(top, bottom);
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
		const Dtype* H1S = top[1]->gpu_data();
		const Dtype* W = this->blobs_[0]->gpu_data();
		const Dtype* b = this->blobs_[1]->gpu_data();
		Dtype* X1S = bottom[1]->mutable_gpu_data();


		// X = 1 * H * W + 0 * X
		// bottom_data = 1 * top_data * weights + 0 * bottom_data
		// [m,k] = 1 * [m,n] * [n,k] + 0 * [m,k]
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
				M_, K_, N_,
				(Dtype)1., H1S, W,
				(Dtype)0., X1S);


		// X = 1 * bM * b + 1 * X
		// bottom_data = 1 * bias_b_multiplier * b + 1 * bottom_data
		// [m,k] = 1 * [m,1] * [1,k] + 1 * [m,k]
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
				M_, K_, 1,
				(Dtype)1., ones_m_.gpu_data(), b,
				(Dtype)1., X1S);


		sigmoid_gpu(bottom[0]->count(), bottom[1]->mutable_gpu_data());

		sample_gpu(bottom[0]->count(),  bottom[1]->mutable_gpu_data());
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
				xTmp.ReshapeLike(this->X1S_);

				const Dtype* xData = bottom[0]->gpu_data();
				const Dtype* hData = top[0]->gpu_data();
				Dtype* xTmpData = xTmp.mutable_gpu_data();

				const Dtype* W = this->blobs_[0]->gpu_data();
				const Dtype* b = this->blobs_[1]->gpu_data();

				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
						M_, K_, N_,
						(Dtype)1., hData, W,
						(Dtype)0., xTmpData);

				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
						M_, K_, 1,
						(Dtype)1., ones_m_.gpu_data(), b,
						(Dtype)1., xTmpData);

				int count = xTmp.count();
				sigmoid_gpu(count, xTmpData);

				sample_gpu(count, xTmpData);

				caffe_gpu_sub<Dtype>(xTmp.count(), xData, xTmpData, xTmp.mutable_gpu_data());

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
	if(randoms[index] < data[index])
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
void RBMLayer<Dtype>::sample_gpu(int N, Dtype* mat)
{
	vector<int> shape(2);
	shape[0] = N;
	shape[1] = 1;
	randomContainer.Reshape(shape);

	MLGRNG<Dtype>::getInstance().mlg_gpu_uniform(N, randomContainer.mutable_gpu_data());

	sample_gpu2<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
			N, mat, randomContainer.gpu_data());
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

} // namespace caffe
