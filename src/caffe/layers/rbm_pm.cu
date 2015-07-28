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
__global__ void sample_ge0_5(const int n, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
	if(y[index] > (Dtype)0.5){
		y[index] = (Dtype) 1.;
	}else{
		y[index] = (Dtype) 0.;
	}
  }
}

template <typename Dtype>
__global__ void add_scaled(const int n, const Dtype alpha, const Dtype* a, const Dtype beta, const Dtype* b, Dtype* x) {
  CUDA_KERNEL_LOOP(index, n) {
	x[index] = alpha * a[index] + beta * b[index];
  }
}

template <typename Dtype>
__global__ void relax_0_1(const int n, Dtype* x) {
  CUDA_KERNEL_LOOP(index, n) {
		if(x[index] > 1){
			x[index] = 1;
		}

		if(x[index] < 0){
			x[index] = 0;
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
		case RBMPMLayer::FreeEnergyGradientDescent:
		{
			int descentSteps = this->layer_param_.rbm_param().rbm_pm_param().fegd_param().descent_steps();
			Dtype eta0 = this->layer_param_.rbm_param().rbm_pm_param().fegd_param().eta_0();
			Dtype etaDecay = this->layer_param_.rbm_param().rbm_pm_param().fegd_param().eta_decay();

			Dtype* XS = X->mutable_gpu_data();
			Dtype* HS = H->mutable_gpu_data();



			for(int descent = 0; descent < descentSteps; descent++){
				Dtype eta = eta0 / ( 1 + etaDecay * descent );

				// h = sigmoid ( c + w * x )
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
						  this->M_, this->N_, this->K_,
						  (Dtype)1., XS, W->gpu_data(),
						  (Dtype)0., HS);

				caffe_gpu_add(H->count(), HS, c->gpu_data(), HS);

				sigmoid_gpu(H->count(), HS);

				// x = x + eta * ( b + W * h )
				// x = x + (eta *  W * h) + (eta * b)

				// x = (1 * x) + (eta *  W * h)
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
						this->M_, this->K_, this->N_,
						eta, HS, W->gpu_data(),
						(Dtype)1., XS);

				// x = (1 * x) + (eta * b)
				add_scaled<Dtype><<<CAFFE_GET_BLOCKS(X->count()), CAFFE_CUDA_NUM_THREADS>>>(X->count(), (Dtype) 1., XS, eta, b->gpu_data(), XS);
				CUDA_POST_KERNEL_CHECK;

				// bring back to [0, 1]
				relax_0_1<Dtype><<<CAFFE_GET_BLOCKS(X->count()), CAFFE_CUDA_NUM_THREADS>>>(X->count(), XS);
				CUDA_POST_KERNEL_CHECK;
			}
			sample_ge0_5<Dtype><<<CAFFE_GET_BLOCKS(X->count()), CAFFE_CUDA_NUM_THREADS>>>(X->count(), XS);
			CUDA_POST_KERNEL_CHECK;

			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
					this->M_, this->N_, this->K_,
					(Dtype)1., XS, W->gpu_data(),
					(Dtype)0., HS);

			caffe_gpu_add(H->count(), HS, c->gpu_data(), HS);
			sigmoid_gpu(H->count(), HS);
			sample_gpu(H->count(), HS);

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
