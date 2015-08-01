#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/rbm_layers.hpp"
#include "caffe/util/mlg_math.hpp"

namespace caffe {

template <typename Dtype>
void RBMPMLayer<Dtype>::find_map_gpu(Blob<Dtype>* X, Blob<Dtype>* H, Blob<Dtype>* b, Blob<Dtype>* c, Blob<Dtype>* W){
	switch(this->layer_param_.rbm_param().rbm_pm_param().map_method()){
		case RBMPMLayer::CoordinateDescent:
		{
			Dtype* XS = X->mutable_gpu_data();
			Dtype* HS = H->mutable_gpu_data();
			const int descentSteps = this->layer_param_.rbm_param().rbm_pm_param().coordinate_descent_param().descent_steps();
			const int repTimes = this->layer_param_.rbm_param().rbm_pm_param().batch_repeats();

			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
					  this->M_ * repTimes, this->N_, this->K_,
					  (Dtype)1., XS, W->gpu_data(),
					  (Dtype)0., HS);

			caffe_gpu_add(H->count(), HS, c->gpu_data(), HS);

			sample_ge0_kernel<Dtype><<<CAFFE_GET_BLOCKS(H->count()), CAFFE_CUDA_NUM_THREADS>>>(H->count(), H->mutable_gpu_data());
			CUDA_POST_KERNEL_CHECK;

			for(int descent = 0; descent < descentSteps; descent++){

				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
						this->M_ * repTimes, this->K_, this->N_,
						(Dtype)1., HS, W->gpu_data(),
						(Dtype)0., XS);

				caffe_gpu_add(X->count(), XS, b->gpu_data(), XS);

				sample_ge0_kernel<Dtype><<<CAFFE_GET_BLOCKS(X->count()), CAFFE_CUDA_NUM_THREADS>>>(X->count(), X->mutable_gpu_data());
				CUDA_POST_KERNEL_CHECK;



				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
						this->M_ * repTimes, this->N_, this->K_,
						(Dtype)1., XS, W->gpu_data(),
						(Dtype)0., HS);

				caffe_gpu_add(H->count(), HS, c->gpu_data(), HS);

				sample_ge0_kernel<Dtype><<<CAFFE_GET_BLOCKS(H->count()), CAFFE_CUDA_NUM_THREADS>>>(H->count(), H->mutable_gpu_data());
				CUDA_POST_KERNEL_CHECK;
			}
		}
		break;
		case RBMPMLayer::FreeEnergyGradientDescent:
		{
			const int descentSteps = this->layer_param_.rbm_param().rbm_pm_param().fegd_param().descent_steps();
			const int repTimes = this->layer_param_.rbm_param().rbm_pm_param().batch_repeats();

			const Dtype eta0 = this->layer_param_.rbm_param().rbm_pm_param().fegd_param().eta_0();
			const Dtype etaDecay = this->layer_param_.rbm_param().rbm_pm_param().fegd_param().eta_decay();

			Dtype* XS = X->mutable_gpu_data();
			Dtype* HS = H->mutable_gpu_data();



			for(int descent = 0; descent < descentSteps; descent++){
				Dtype eta = eta0 / ( 1 + etaDecay * descent );

				// h = sigmoid ( c + w * x )
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
						  this->M_ * repTimes, this->N_, this->K_,
						  (Dtype)1., XS, W->gpu_data(),
						  (Dtype)0., HS);

				caffe_gpu_add(H->count(), HS, c->gpu_data(), HS);

				sigmoid_gpu(H->count(), HS);

				// x = x + eta * ( b + W * h )
				// x = x + (eta *  W * h) + (eta * b)

				// x = (1 * x) + (eta *  W * h)
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
						this->M_ * repTimes, this->K_, this->N_,
						eta, HS, W->gpu_data(),
						(Dtype)1., XS);

				// x = (1 * x) + (eta * b)
				add_scaled_kernel<Dtype><<<CAFFE_GET_BLOCKS(X->count()), CAFFE_CUDA_NUM_THREADS>>>(X->count(), (Dtype) 1., XS, eta, b->gpu_data(), XS);
				CUDA_POST_KERNEL_CHECK;

				// bring back to [0, 1]
				relax_0_1_kernel<Dtype><<<CAFFE_GET_BLOCKS(X->count()), CAFFE_CUDA_NUM_THREADS>>>(X->count(), XS);
				CUDA_POST_KERNEL_CHECK;
			}
			sample_ge0_5_kernel<Dtype><<<CAFFE_GET_BLOCKS(X->count()), CAFFE_CUDA_NUM_THREADS>>>(X->count(), XS);
			CUDA_POST_KERNEL_CHECK;

			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
					this->M_ * repTimes, this->N_, this->K_,
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

template <typename Dtype>
void RBMPMLayer<Dtype>::replicate_data_gpu(const int N, Blob<Dtype>* X, Blob<Dtype>* repX){

	const int axis = X->CanonicalAxisIndex(this->layer_param_.rbm_param().axis());

	vector<int> X_shape = X->shape();

    vector<int> repX_shape(2);
    repX_shape[0] = X_shape[0] * N;
    repX_shape[1] = X->count(axis);

    repX->Reshape(repX_shape);

    replicate_kernel<Dtype><<<CAFFE_GET_BLOCKS(repX->count()), CAFFE_CUDA_NUM_THREADS>>>(X->count(), repX->count(), X->gpu_data(), repX->mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;
}


template
void RBMPMLayer<float>::replicate_data_gpu(const int N, Blob<float>* X, Blob<float>* repX);

template
void RBMPMLayer<double>::replicate_data_gpu(const int N, Blob<double>* X, Blob<double>* repX);

template
void RBMPMLayer<float>::find_map_gpu(Blob<float>* X, Blob<float>* H, Blob<float>* b, Blob<float>* c, Blob<float>* W);

template
void RBMPMLayer<double>::find_map_gpu(Blob<double>* X, Blob<double>* H, Blob<double>* b, Blob<double>* c, Blob<double>* W);

}
