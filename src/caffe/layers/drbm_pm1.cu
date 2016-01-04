#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/drbm_layers.hpp"
#include "caffe/util/mlg_math.hpp"

namespace caffe {

template <typename Dtype>
void DRBMPM1Layer<Dtype>::generatePerturbations_gpu() {
	Blob<Dtype> ra;
	for(int i = 0; i < this->biases.size(); i++){
		caffe_copy(this->pertBiases[i]->count(), this->biases[i]->gpu_data(), this->pertBiases[i]->mutable_gpu_data());
		ra.ReshapeLike(*this->pertBiases[i]);
		MLGRNG<Dtype>::getInstance().mlg_gpu_gumbel(ra.count(), ra.mutable_gpu_data());
		caffe_gpu_add(this->pertBiases[i]->count(), this->pertBiases[i]->gpu_data(), ra.gpu_data(), this->pertBiases[i]->mutable_gpu_data());
	}

	for(int i = 0; i < this->weights.size(); i++){
		caffe_copy(this->pertWeights[i]->count(), this->weights[i]->gpu_data(), this->pertWeights[i]->mutable_gpu_data());
	}
}

template void DRBMPM1Layer<float>::generatePerturbations_gpu();
template void DRBMPM1Layer<double>::generatePerturbations_gpu();

//INSTANTIATE_LAYER_GPU_FUNCS(DRBMPM1Layer);

} // namespace caffe
