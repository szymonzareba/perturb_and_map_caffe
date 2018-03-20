#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/drbm_layers.hpp"
#include "caffe/util/mlg_math.hpp"

namespace caffe {


template <typename Dtype>
void DRBMPMLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	//generate samples
	generatePerturbations_gpu();
	optimizeEnergy_gpu();
	DRBMLayer<Dtype>::gradient_gpu();
	DRBMLayer<Dtype>::Backward_gpu(top, propagate_down, bottom);
}

template <typename Dtype>
void DRBMPMLayer<Dtype>::optimizeEnergy_gpu(){
	// copy probs and states
	for(int i = 0; i < this->probs_0.size(); i++)
	{
		caffe_copy(this->probs_0[i]->count(), this->probs_0[i]->gpu_data(), this->probs_1[i]->mutable_gpu_data());
		caffe_copy(this->states_0[i]->count(), this->states_0[i]->gpu_data(), this->states_1[i]->mutable_gpu_data());
	}

	switch(this->layer_param_.drbm_param().drbm_pm_param().map_method()){
		case DRBMPMLayer::CoordinateDescent:
		{
			for(int descent_step = 0; descent_step < this->layer_param_.drbm_param().drbm_pm_param().cd_param().descent_steps(); descent_step++)
			{
				for(int layer_num = 1; layer_num < this->probs_1.size(); layer_num += 2)
				{
					// clear previous and add bias
					caffe_copy(this->pertBiases[layer_num]->count(), this->pertBiases[layer_num]->gpu_data(), this->probs_1[layer_num]->mutable_gpu_data());
/*
					caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
							this->M_, this->layer_sizes[layer_num], 1,
							(Dtype)1., this->ones_m.gpu_data(), this->pertBiases[layer_num]->gpu_data(),
							(Dtype)0., this->probs_1[layer_num]->mutable_gpu_data());
*/
					// multiply and add previous layer
					caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
							this->M_, this->layer_sizes[layer_num], this->layer_sizes[layer_num - 1],
							(Dtype)1., this->probs_1[layer_num - 1]->gpu_data(), this->pertWeights[layer_num - 1]->gpu_data(),
							(Dtype)1., this->probs_1[layer_num]->mutable_gpu_data());


					// check if next layer exists
					if(layer_num + 1 < this->probs_1.size())
					{
						// multiply and add next layer
						caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
								this->M_, this->layer_sizes[layer_num], this->layer_sizes[layer_num + 1],
								(Dtype)1., this->probs_1[layer_num + 1]->gpu_data(), this->pertWeights[layer_num]->gpu_data(),
								(Dtype)1., this->probs_1[layer_num]->mutable_gpu_data());
					}

					sample_ge0_kernel<Dtype><<<CAFFE_GET_BLOCKS(this->probs_1[layer_num]->count()), CAFFE_CUDA_NUM_THREADS>>>(
							this->probs_1[layer_num]->count(), this->probs_1[layer_num]->mutable_gpu_data());
				}

				for(int layer_num = 2; layer_num < this->probs_1.size(); layer_num += 2)
				{
					// clear previous and add bias
/*
					caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
							this->M_, this->layer_sizes[layer_num], 1,
							(Dtype)1., this->ones_m.gpu_data(), this->pertBiases[layer_num]->gpu_data(),
							(Dtype)0., this->probs_1[layer_num]->mutable_gpu_data());
*/
					caffe_copy(this->pertBiases[layer_num]->count(), this->pertBiases[layer_num]->gpu_data(), this->probs_1[layer_num]->mutable_gpu_data());

					// multiply and add previous layer
					caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
							this->M_, this->layer_sizes[layer_num], this->layer_sizes[layer_num - 1],
							(Dtype)1., this->probs_1[layer_num - 1]->gpu_data(), this->pertWeights[layer_num - 1]->gpu_data(),
							(Dtype)1., this->probs_1[layer_num]->mutable_gpu_data());

					// check if next layer exists
					if(layer_num + 1 < this->probs_1.size())
					{
						// multiply and add next layer
						caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
								this->M_, this->layer_sizes[layer_num], this->layer_sizes[layer_num + 1],
								(Dtype)1., this->probs_1[layer_num + 1]->gpu_data(), this->pertWeights[layer_num]->gpu_data(),
								(Dtype)1., this->probs_1[layer_num]->mutable_gpu_data());
					}

					sample_ge0_kernel<Dtype><<<CAFFE_GET_BLOCKS(this->probs_1[layer_num]->count()), CAFFE_CUDA_NUM_THREADS>>>(
							this->probs_1[layer_num]->count(), this->probs_1[layer_num]->mutable_gpu_data());
				}
			}

			// copy probs to states
			for(int i = 0; i < this->probs_1.size(); i++)
			{
				caffe_copy(this->probs_1[i]->count(), this->probs_1[i]->gpu_data(), this->states_1[i]->mutable_gpu_data());
			}

			break;
		}
		case DRBMPMLayer::GreedyEnergyOptimization:
		{
			for(int step = 0; step < this->layer_param_.drbm_param().drbm_pm_param().geo_param().steps(); step++)
			{
				for(int layer_num = 1; layer_num < this->probs_1.size(); layer_num += 2)
				{
					// clear previous and add bias
/*
					caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
							this->M_, this->layer_sizes[layer_num], 1,
							(Dtype)1., this->ones_m.gpu_data(), this->pertBiases[layer_num]->gpu_data(),
							(Dtype)0., this->probs_1[layer_num]->mutable_gpu_data());
*/
					caffe_copy(this->pertBiases[layer_num]->count(), this->pertBiases[layer_num]->gpu_data(), this->probs_1[layer_num]->mutable_gpu_data());

					// multiply and add previous layer
					caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
							this->M_, this->layer_sizes[layer_num], this->layer_sizes[layer_num - 1],
							(Dtype)1., this->probs_1[layer_num - 1]->gpu_data(), this->pertWeights[layer_num - 1]->gpu_data(),
							(Dtype)1., this->probs_1[layer_num]->mutable_gpu_data());


					// check if next layer exists
					if(layer_num + 1 < this->probs_1.size())
					{
						// multiply and add next layer
						caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
								this->M_, this->layer_sizes[layer_num], this->layer_sizes[layer_num + 1],
								(Dtype)1., this->probs_1[layer_num + 1]->gpu_data(), this->pertWeights[layer_num]->gpu_data(),
								(Dtype)1., this->probs_1[layer_num]->mutable_gpu_data());
					}

					sample_ge0_kernel<Dtype><<<CAFFE_GET_BLOCKS(this->probs_1[layer_num]->count()), CAFFE_CUDA_NUM_THREADS>>>(
							this->probs_1[layer_num]->count(),
							this->probs_1[layer_num]->gpu_data(),
							this->states_1[layer_num]->mutable_gpu_data());
				}

				for(int layer_num = 2; layer_num < this->probs_1.size(); layer_num += 2)
				{
					// clear previous and add bias
/*
					caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
							this->M_, this->layer_sizes[layer_num], 1,
							(Dtype)1., this->ones_m.gpu_data(), this->pertBiases[layer_num]->gpu_data(),
							(Dtype)0., this->probs_1[layer_num]->mutable_gpu_data());
*/
					caffe_copy(this->pertBiases[layer_num]->count(), this->pertBiases[layer_num]->gpu_data(), this->probs_1[layer_num]->mutable_gpu_data());

					// multiply and add previous layer
					caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
							this->M_, this->layer_sizes[layer_num], this->layer_sizes[layer_num - 1],
							(Dtype)1., this->probs_1[layer_num - 1]->gpu_data(), this->pertWeights[layer_num - 1]->gpu_data(),
							(Dtype)1., this->probs_1[layer_num]->mutable_gpu_data());

					// check if next layer exists
					if(layer_num + 1 < this->probs_1.size())
					{
						// multiply and add next layer
						caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
								this->M_, this->layer_sizes[layer_num], this->layer_sizes[layer_num + 1],
								(Dtype)1., this->probs_1[layer_num + 1]->gpu_data(), this->pertWeights[layer_num]->gpu_data(),
								(Dtype)1., this->probs_1[layer_num]->mutable_gpu_data());
					}

					sample_ge0_kernel<Dtype><<<CAFFE_GET_BLOCKS(this->probs_1[layer_num]->count()), CAFFE_CUDA_NUM_THREADS>>>(
							this->probs_1[layer_num]->count(),
							this->probs_1[layer_num]->gpu_data(),
							this->states_1[layer_num]->mutable_gpu_data());
				}

				for(int i = 0; i < this->probs_1.size(); i++)
				{
					caffe_copy(this->probs_1[i]->count(), this->states_1[i]->gpu_data(), this->probs_1[i]->mutable_gpu_data());
				}
			}

			break;
		}
		case DRBMPMLayer::RandomizedGreedyEnergyOptimization:
		{
			for(int step = 0; step < this->layer_param_.drbm_param().drbm_pm_param().rgeo_param().steps(); step++)
			{
				for(int layer_num = 1; layer_num < this->probs_1.size(); layer_num += 2)
				{
					// clear previous and add bias
					caffe_copy(this->pertBiases[layer_num]->count(), this->pertBiases[layer_num]->gpu_data(), this->probs_1[layer_num]->mutable_gpu_data());

					// multiply and add previous layer
					caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
							this->M_, this->layer_sizes[layer_num], this->layer_sizes[layer_num - 1],
							(Dtype)1., this->probs_1[layer_num - 1]->gpu_data(), this->pertWeights[layer_num - 1]->gpu_data(),
							(Dtype)1., this->probs_1[layer_num]->mutable_gpu_data());


					// check if next layer exists
					if(layer_num + 1 < this->probs_1.size())
					{
						// multiply and add next layer
						caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
								this->M_, this->layer_sizes[layer_num], this->layer_sizes[layer_num + 1],
								(Dtype)1., this->probs_1[layer_num + 1]->gpu_data(), this->pertWeights[layer_num]->gpu_data(),
								(Dtype)1., this->probs_1[layer_num]->mutable_gpu_data());
					}

					sample_ge0_kernel<Dtype><<<CAFFE_GET_BLOCKS(this->probs_1[layer_num]->count()), CAFFE_CUDA_NUM_THREADS>>>(
							this->probs_1[layer_num]->count(),
							this->probs_1[layer_num]->gpu_data(),
							this->states_1[layer_num]->mutable_gpu_data());
				}

				for(int layer_num = 2; layer_num < this->probs_1.size(); layer_num += 2)
				{
					// clear previous and add bias
					caffe_copy(this->pertBiases[layer_num]->count(), this->pertBiases[layer_num]->gpu_data(), this->probs_1[layer_num]->mutable_gpu_data());

					// multiply and add previous layer
					caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
							this->M_, this->layer_sizes[layer_num], this->layer_sizes[layer_num - 1],
							(Dtype)1., this->probs_1[layer_num - 1]->gpu_data(), this->pertWeights[layer_num - 1]->gpu_data(),
							(Dtype)1., this->probs_1[layer_num]->mutable_gpu_data());

					// check if next layer exists
					if(layer_num + 1 < this->probs_1.size())
					{
						// multiply and add next layer
						caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
								this->M_, this->layer_sizes[layer_num], this->layer_sizes[layer_num + 1],
								(Dtype)1., this->probs_1[layer_num + 1]->gpu_data(), this->pertWeights[layer_num]->gpu_data(),
								(Dtype)1., this->probs_1[layer_num]->mutable_gpu_data());
					}

					sample_ge0_kernel<Dtype><<<CAFFE_GET_BLOCKS(this->probs_1[layer_num]->count()), CAFFE_CUDA_NUM_THREADS>>>(
							this->probs_1[layer_num]->count(),
							this->probs_1[layer_num]->gpu_data(),
							this->states_1[layer_num]->mutable_gpu_data());
				}

				double maskout = this->layer_param_.drbm_param().drbm_pm_param().rgeo_param().maskout();

				for(int i = 0; i < this->probs_1.size(); i++)
				{
					Blob<Dtype> maskoutBlob;
					maskoutBlob.ReshapeLike(*(this->states_1[i]));

					// uniform
					MLGRNG<Dtype>::getInstance().mlg_gpu_uniform(maskoutBlob.count(), maskoutBlob.mutable_gpu_data());

					// binarization, uniform > 1-maskout (1-0.9) = 1
					// 10% - 0, 90% - 1
					binarization_kernel<Dtype><<<CAFFE_GET_BLOCKS(maskoutBlob.count()), CAFFE_CUDA_NUM_THREADS>>>
							(maskoutBlob.count(), 1.0 - maskout, maskoutBlob.gpu_data(), maskoutBlob.mutable_gpu_data());
					CUDA_POST_KERNEL_CHECK;

					// X = 10% dXData + 90% XS
					add_with_mask_kernel_2<Dtype><<<CAFFE_GET_BLOCKS(maskoutBlob.count()), CAFFE_CUDA_NUM_THREADS>>>
							(maskoutBlob.count(),
									maskoutBlob.gpu_data(),
									this->states_1[i]->gpu_data(),
									this->probs_1[i]->gpu_data(),
									this->probs_1[i]->mutable_gpu_data());
					CUDA_POST_KERNEL_CHECK;

				}
			}

			break;
		}
		default:
		{
			NOT_IMPLEMENTED;
		}
	}
}

template void DRBMPMLayer<float>::Backward_gpu( \
		const vector<Blob<float>*>& top, \
	    const vector<bool>& propagate_down, \
	    const vector<Blob<float>*>& bottom);

template void DRBMPMLayer<double>::Backward_gpu( \
		const vector<Blob<double>*>& top, \
	    const vector<bool>& propagate_down, \
	    const vector<Blob<double>*>& bottom);

template void DRBMPMLayer<float>::optimizeEnergy_gpu();
template void DRBMPMLayer<double>::optimizeEnergy_gpu();

//INSTANTIATE_LAYER_GPU_FUNCS(DRBMPMLayer);

} // namespace caffe
