#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/rbm_layers.hpp"
#include "caffe/util/mlg_math.hpp"

namespace caffe {

template <typename Dtype>
void RBMPMLayer<Dtype>::find_map_gpu(Blob<Dtype>* X, Blob<Dtype>* H, Blob<Dtype>* b, Blob<Dtype>* c, Blob<Dtype>* W){

		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(b->count(), b->gpu_data())) LOG(INFO) << "b not finite" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(c->count(), c->gpu_data())) LOG(INFO) << "c not finite" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(W->count(), W->gpu_data())) LOG(INFO) << "W not finite" << std::endl;

		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(X->count(), X->gpu_data())) LOG(INFO) << "X not finite" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(H->count(), H->gpu_data())) LOG(INFO) << "H not finite" << std::endl;

		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(b->count(), b->gpu_data())) LOG(INFO) << "b not in float range" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(c->count(), c->gpu_data())) LOG(INFO) << "c not in float range" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(W->count(), W->gpu_data())) LOG(INFO) << "W not in float range" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(X->count(), X->gpu_data())) LOG(INFO) << "X not in float range" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(H->count(), H->gpu_data())) LOG(INFO) << "H not in float range" << std::endl;

	switch(this->layer_param_.rbm_param().rbm_pm_param().map_method()){
		case RBMPMLayer::CoordinateDescent:
		{
			Dtype* XS = X->mutable_gpu_data();
			Dtype* HS = H->mutable_gpu_data();
			const int descentSteps = this->layer_param_.rbm_param().rbm_pm_param().coordinate_descent_param().descent_steps();
			const int repTimes = this->layer_param_.rbm_param().rbm_pm_param().batch_repeats();

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

					if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(X->count(), X->gpu_data())) LOG(INFO) << "X not finite" << std::endl;
					if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(H->count(), H->gpu_data())) LOG(INFO) << "H not finite" << std::endl;
					if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(H->count(), H->gpu_data())) LOG(INFO) << "H not in float range" << std::endl;
					if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(X->count(), X->gpu_data())) LOG(INFO) << "X not in float range" << std::endl;
			}

			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
					this->M_ * repTimes, this->N_, this->K_,
					(Dtype)1., XS, W->gpu_data(),
					(Dtype)0., HS);

			caffe_gpu_add(H->count(), HS, c->gpu_data(), HS);
			this->sigmoid_gpu(H->count(), HS);

		}
		break;
		case RBMPMLayer::FreeEnergyGradientDescent:
		{
			const int descentSteps = this->layer_param_.rbm_param().rbm_pm_param().fegd_param().descent_steps();
			const int repTimes = this->layer_param_.rbm_param().rbm_pm_param().batch_repeats();

			const int m = this->M_ * repTimes;

			const Dtype eta0 = this->layer_param_.rbm_param().rbm_pm_param().fegd_param().eta_0();
			const Dtype etaDecay = this->layer_param_.rbm_param().rbm_pm_param().fegd_param().eta_decay();

			Dtype* XS = X->mutable_gpu_data();
			Dtype* HS = H->mutable_gpu_data();

			for(int descent = 0; descent < descentSteps; descent++){
				Dtype eta = eta0;

				if(etaDecay != 0){
					eta = eta0 * sqrt( etaDecay  / (etaDecay + descent) );
				}

				// hs = sigmoid ( c + w * x )
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
						  m, this->N_, this->K_,
						  (Dtype)1., XS, W->gpu_data(),
						  (Dtype)0., HS);


				caffe_gpu_add(H->count(), HS, c->gpu_data(), HS);

				this->sigmoid_gpu(H->count(), HS);

				// x = x + eta * ( b + W * h )
				// x = x + (eta *  W * h) + (eta * b)

				// x = (1 * x) + (eta *  W * h)
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
						m, this->K_, this->N_,
						eta, HS, W->gpu_data(),
						1., XS);

				// x = (1 * x) + (eta * b)
				add_scaled_kernel<Dtype><<<CAFFE_GET_BLOCKS(X->count()), CAFFE_CUDA_NUM_THREADS>>>(X->count(), (Dtype) 1., XS, eta, b->gpu_data(), XS);
				CUDA_POST_KERNEL_CHECK;

				// bring back to [0, 1]
				relax_0_1_kernel<Dtype><<<CAFFE_GET_BLOCKS(X->count()), CAFFE_CUDA_NUM_THREADS>>>(X->count(), XS);
				CUDA_POST_KERNEL_CHECK;

					if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(X->count(), XS)) LOG(INFO) << "X not finite" << std::endl;
					if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(H->count(), HS)) LOG(INFO) << "H not finite" << std::endl;
					if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(H->count(), HS)) LOG(INFO) << "H not in float range" << std::endl;
					if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(X->count(), XS)) LOG(INFO) << "X not in float range" << std::endl;
			}

			// 0.5 sample
			sample_ge0_5_kernel<Dtype><<<CAFFE_GET_BLOCKS(X->count()), CAFFE_CUDA_NUM_THREADS>>>(X->count(), XS);
			CUDA_POST_KERNEL_CHECK;

			// no sample
			// nothing

			// probabilistic sample
			//this->sample_gpu(X->count(), X->mutable_gpu_data());

			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
					this->M_ * repTimes, this->N_, this->K_,
					(Dtype)1., XS, W->gpu_data(),
					(Dtype)0., HS);

			caffe_gpu_add(H->count(), HS, c->gpu_data(), HS);
			this->sigmoid_gpu(H->count(), HS);

		}
		break;
		case RBMPMLayer::FreeEnergyGradientDescentEta2:
		{
			const int descentSteps = this->layer_param_.rbm_param().rbm_pm_param().fegd_param().descent_steps();
			const int repTimes = this->layer_param_.rbm_param().rbm_pm_param().batch_repeats();

			const int m = this->M_ * repTimes;

			const Dtype eta0 = this->layer_param_.rbm_param().rbm_pm_param().fegd_param().eta_0();
			const Dtype etaDecay = this->layer_param_.rbm_param().rbm_pm_param().fegd_param().eta_decay();

			Dtype* XS = X->mutable_gpu_data();
			Dtype* HS = H->mutable_gpu_data();

			for(int descent = 0; descent < descentSteps; descent++){
				Dtype eta = eta0 / ( 1 + etaDecay * descent );

				// hs = sigmoid ( c + w * x )
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
						  m, this->N_, this->K_,
						  (Dtype)1., XS, W->gpu_data(),
						  (Dtype)0., HS);


				caffe_gpu_add(H->count(), HS, c->gpu_data(), HS);

				this->sigmoid_gpu(H->count(), HS);

				// x = x + eta * ( b + W * h )
				// x = x + (eta *  W * h) + (eta * b)

				// x = (1 * x) + (eta *  W * h)
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
						m, this->K_, this->N_,
						eta, HS, W->gpu_data(),
						1., XS);

				// x = (1 * x) + (eta * b)
				add_scaled_kernel<Dtype><<<CAFFE_GET_BLOCKS(X->count()), CAFFE_CUDA_NUM_THREADS>>>(X->count(), (Dtype) 1., XS, eta, b->gpu_data(), XS);
				CUDA_POST_KERNEL_CHECK;

				// bring back to [0, 1]
				relax_0_1_kernel<Dtype><<<CAFFE_GET_BLOCKS(X->count()), CAFFE_CUDA_NUM_THREADS>>>(X->count(), XS);
				CUDA_POST_KERNEL_CHECK;

					if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(X->count(), X->gpu_data())) LOG(INFO) << "X not finite" << std::endl;
					if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(H->count(), H->gpu_data())) LOG(INFO) << "H not finite" << std::endl;
					if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(H->count(), H->gpu_data())) LOG(INFO) << "H not in float range" << std::endl;
					if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(X->count(), X->gpu_data())) LOG(INFO) << "X not in float range" << std::endl;
			}

			// 0.5 sample
			sample_ge0_5_kernel<Dtype><<<CAFFE_GET_BLOCKS(X->count()), CAFFE_CUDA_NUM_THREADS>>>(X->count(), XS);
			CUDA_POST_KERNEL_CHECK;

			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
					this->M_ * repTimes, this->N_, this->K_,
					(Dtype)1., XS, W->gpu_data(),
					(Dtype)0., HS);

			caffe_gpu_add(H->count(), HS, c->gpu_data(), HS);
			this->sigmoid_gpu(H->count(), HS);

		}
		break;
		case RBMPMLayer::NegativeGreedyEnergyOptimization:
		{
			const int steps = this->layer_param_.rbm_param().rbm_pm_param().geo_param().steps();
			const int repTimes = this->layer_param_.rbm_param().rbm_pm_param().batch_repeats();

			Dtype* XS = X->mutable_gpu_data();
			Dtype* HS = H->mutable_gpu_data();

			Blob<Dtype> dX;
			dX.ReshapeLike(*X);

			Blob<Dtype> dH;
			dH.ReshapeLike(*H);

			Dtype* dXData = dX.mutable_gpu_data();
			Dtype* dHData = dH.mutable_gpu_data();

			for(int step = 0; step < steps; step++){

				// dX
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
						this->M_ * repTimes, this->K_, this->N_,
						(Dtype)1., HS, W->gpu_data(),
						(Dtype)0., dXData);

				caffe_gpu_add(X->count(), dXData, b->gpu_data(), dXData);

				// dH
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
						this->M_ * repTimes, this->N_, this->K_,
						(Dtype)1., XS, W->gpu_data(),
						(Dtype)0., dHData);

				caffe_gpu_add(H->count(), dHData, c->gpu_data(), dHData);


				sample_ge0_kernel<Dtype><<<CAFFE_GET_BLOCKS(dH.count()), CAFFE_CUDA_NUM_THREADS>>>(dH.count(), dHData);
				CUDA_POST_KERNEL_CHECK;

				sample_ge0_kernel<Dtype><<<CAFFE_GET_BLOCKS(dX.count()), CAFFE_CUDA_NUM_THREADS>>>(dX.count(), dXData);
				CUDA_POST_KERNEL_CHECK;

				caffe_copy(dH.count(), dHData, HS);
				caffe_copy(dX.count(), dXData, XS);


					if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(X->count(), X->gpu_data())) LOG(INFO) << "X not finite" << std::endl;
					if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(H->count(), H->gpu_data())) LOG(INFO) << "H not finite" << std::endl;
					if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(H->count(), H->gpu_data())) LOG(INFO) << "H not in float range" << std::endl;
					if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(X->count(), X->gpu_data())) LOG(INFO) << "X not in float range" << std::endl;
			}

			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
					this->M_ * repTimes, this->N_, this->K_,
					(Dtype)1., XS, W->gpu_data(),
					(Dtype)0., HS);

			caffe_gpu_add(H->count(), HS, c->gpu_data(), HS);
			this->sigmoid_gpu(H->count(), HS);
		}
		break;
		default:
		{
			NOT_IMPLEMENTED;
		}
	}

		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(X->count(), X->gpu_data())) LOG(INFO) << "X not finite" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(H->count(), H->gpu_data())) LOG(INFO) << "H not finite" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(H->count(), H->gpu_data())) LOG(INFO) << "H not in float range" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(X->count(), X->gpu_data())) LOG(INFO) << "X not in float range" << std::endl;
}

template
void RBMPMLayer<float>::find_map_gpu(Blob<float>* X, Blob<float>* H, Blob<float>* b, Blob<float>* c, Blob<float>* W);

template
void RBMPMLayer<double>::find_map_gpu(Blob<double>* X, Blob<double>* H, Blob<double>* b, Blob<double>* c, Blob<double>* W);

}
