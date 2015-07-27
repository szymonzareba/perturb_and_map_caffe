#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/rbm_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void sample_ge0(const int n, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
	if(y[index] > (Dtype)0.){
		y[index] = (Dtype) 1.;
	}else{
		y[index] = (Dtype) 0.;
	}
  }
}

template <typename Dtype>
void RBMPMLayer<Dtype>::find_map_gpu(Blob<Dtype>* X, Blob<Dtype>* H, Blob<Dtype>* b, Blob<Dtype>* c, Blob<Dtype>* W){
	switch(this->layer_param_.rbm_param().rbm_pm_param().map_method()){
		case RBMPMLayer::CoordinateDescent:
		{
			Dtype* XS = X->mutable_gpu_data();
			Dtype* HS = H->mutable_gpu_data();
			int descentSteps = this->layer_param_.rbm_param().rbm_pm_param().coordinate_descent_param().descent_steps();

			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
					  this->M_, this->N_, this->K_,
					  (Dtype)1., XS, W->gpu_data(),
					  (Dtype)0., HS);

			caffe_gpu_add(H->count(), HS, c->gpu_data(), HS);

			sample_ge0<Dtype><<<CAFFE_GET_BLOCKS(H->count()), CAFFE_CUDA_NUM_THREADS>>>(H->count(), H->mutable_gpu_data());
			CUDA_POST_KERNEL_CHECK;

			for(int descent = 0; descent < descentSteps; descent++){

				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
						this->M_, this->K_, this->N_,
						(Dtype)1., HS, W->gpu_data(),
						(Dtype)0., XS);

				caffe_gpu_add(X->count(), XS, b->gpu_data(), XS);

				sample_ge0<Dtype><<<CAFFE_GET_BLOCKS(X->count()), CAFFE_CUDA_NUM_THREADS>>>(X->count(), X->mutable_gpu_data());
				CUDA_POST_KERNEL_CHECK;



				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
						this->M_, this->N_, this->K_,
						(Dtype)1., XS, W->gpu_data(),
						(Dtype)0., HS);

				caffe_gpu_add(H->count(), HS, c->gpu_data(), HS);

				sample_ge0<Dtype><<<CAFFE_GET_BLOCKS(H->count()), CAFFE_CUDA_NUM_THREADS>>>(H->count(), H->mutable_gpu_data());
				CUDA_POST_KERNEL_CHECK;
			}
		}
		break;
		default:
		{}
	}
}

template
void RBMPMLayer<float>::find_map_gpu(Blob<float>* X, Blob<float>* H, Blob<float>* b, Blob<float>* c, Blob<float>* W);

template
void RBMPMLayer<double>::find_map_gpu(Blob<double>* X, Blob<double>* H, Blob<double>* b, Blob<double>* c, Blob<double>* W);

}
