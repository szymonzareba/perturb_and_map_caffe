#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/drbm_layers.hpp"
#include "caffe/util/mlg_math.hpp"

namespace caffe {

template <typename Dtype>
void DRBMMeanFieldLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

	// copy previous probs
	//LOG(INFO) << "Copy" << std::endl;
	for(int i = 0; i < this->probs_0.size(); i++)
	{
		caffe_copy(this->probs_0[i]->count(), this->probs_0[i]->gpu_data(), this->probs_1[i]->mutable_gpu_data());
	}

	//LOG(INFO) << "MF" << std::endl;
	for(int mf_step = 0; mf_step < this->layer_param_.drbm_param().drbm_mf_param().steps(); mf_step++)
	{
		//LOG(INFO) << "MF " << mf_step << std::endl;
		for(int layer_num = 0; layer_num < this->probs_1.size(); layer_num += 2)
		{
			//LOG(INFO) << "Layer " << layer_num << std::endl;
			// clear previous and add bias
			//LOG(INFO) << "Bias" << std::endl;
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
					this->M_, this->layer_sizes[layer_num], 1,
					(Dtype)1., this->ones_m.gpu_data(), this->biases[layer_num]->gpu_data(),
					(Dtype)0., this->probs_1[layer_num]->mutable_gpu_data());

			// check if previous layer exists
			if(layer_num - 1 >= 0)
			{
				// multiply and add previous layer
				//LOG(INFO) << "Previous" << std::endl;
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
						this->M_, this->layer_sizes[layer_num], this->layer_sizes[layer_num - 1],
						(Dtype)1., this->probs_1[layer_num - 1]->gpu_data(), this->weights[layer_num - 1]->gpu_data(),
						(Dtype)1., this->probs_1[layer_num]->mutable_gpu_data());
			}

			// check if next layer exists
			if(layer_num + 1 < this->probs_1.size())
			{
				// multiply and add next layer
				//LOG(INFO) << "Next" << std::endl;
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
						this->M_, this->layer_sizes[layer_num], this->layer_sizes[layer_num + 1],
						(Dtype)1., this->probs_1[layer_num + 1]->gpu_data(), this->weights[layer_num]->gpu_data(),
						(Dtype)1., this->probs_1[layer_num]->mutable_gpu_data());
			}

			//LOG(INFO) << "Sigmoid" << std::endl;
			this->sigmoid_gpu(this->probs_1[layer_num]->count(), this->probs_1[layer_num]->mutable_gpu_data());
		}

		for(int layer_num = 1; layer_num < this->probs_1.size(); layer_num += 2)
		{
			// clear previous and add bias
			//LOG(INFO) << "Bias" << std::endl;
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
					this->M_, this->layer_sizes[layer_num], 1,
					(Dtype)1., this->ones_m.gpu_data(), this->biases[layer_num]->gpu_data(),
					(Dtype)0., this->probs_1[layer_num]->mutable_gpu_data());

			// check if previous layer exists
			if(layer_num - 1 >= 0)
			{
				// multiply and add previous layer
				//LOG(INFO) << "Previous" << std::endl;
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
						this->M_, this->layer_sizes[layer_num], this->layer_sizes[layer_num - 1],
						(Dtype)1., this->probs_1[layer_num - 1]->gpu_data(), this->weights[layer_num - 1]->gpu_data(),
						(Dtype)1., this->probs_1[layer_num]->mutable_gpu_data());
			}

			// check if next layer exists
			if(layer_num + 1 < this->probs_1.size())
			{
				// multiply and add next layer
				//LOG(INFO) << "Next" << std::endl;
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
						this->M_, this->layer_sizes[layer_num], this->layer_sizes[layer_num + 1],
						(Dtype)1., this->probs_1[layer_num + 1]->gpu_data(), this->weights[layer_num]->gpu_data(),
						(Dtype)1., this->probs_1[layer_num]->mutable_gpu_data());
			}

			//LOG(INFO) << "Sigmoid" << std::endl;
			this->sigmoid_gpu(this->probs_1[layer_num]->count(), this->probs_1[layer_num]->mutable_gpu_data());
		}
	}

	// sample probs to states
	//LOG(INFO) << "Sample" << std::endl;
	for(int i = 0; i < this->probs_1.size(); i++)
	{
		this->sample_gpu(this->probs_1[i]->count(), this->probs_1[i]->gpu_data(), this->states_1[i]->mutable_gpu_data());
	}

	DRBMLayer<Dtype>::gradient_gpu();
	DRBMLayer<Dtype>::Backward_gpu(top, propagate_down, bottom);
}

template void DRBMMeanFieldLayer<float>::Backward_gpu( \
		const vector<Blob<float>*>& top, \
	    const vector<bool>& propagate_down, \
	    const vector<Blob<float>*>& bottom);

template void DRBMMeanFieldLayer<double>::Backward_gpu( \
		const vector<Blob<double>*>& top, \
	    const vector<bool>& propagate_down, \
	    const vector<Blob<double>*>& bottom);

//INSTANTIATE_LAYER_GPU_FUNCS(DRBMMeanFieldLayer);

} // namespace caffe
