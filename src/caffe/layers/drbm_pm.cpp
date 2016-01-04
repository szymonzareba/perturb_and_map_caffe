#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/drbm_layers.hpp"

namespace caffe {

template <typename Dtype>
void DRBMPMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top) {
	DRBMLayer<Dtype>::LayerSetUp(bottom, top);

	for(int i = 0; i < this->biases.size(); i++){
		Blob<Dtype>* newBias = new Blob<Dtype>();
		newBias->ReshapeLike(*this->biases[i]);

		pertBiases.push_back(newBias);
	}

	for(int i = 0; i < this->weights.size(); i++){
		Blob<Dtype>* newWeights = new Blob<Dtype>();
		newWeights->ReshapeLike(*this->weights[i]);

		pertWeights.push_back(newWeights);
	}

}

template <typename Dtype>
void DRBMPMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	//generate samples
	generatePerturbations_cpu();
	optimizeEnergy_cpu();
	DRBMLayer<Dtype>::gradient_cpu();
	DRBMLayer<Dtype>::Backward_cpu(top, propagate_down, bottom);
}

template <typename Dtype>
void DRBMPMLayer<Dtype>::optimizeEnergy_cpu(){
	NOT_IMPLEMENTED;
}

template <typename Dtype>
DRBMPMLayer<Dtype>::~DRBMPMLayer(){
	for(int i = 1; i < pertBiases.size(); i++)
	{
		delete pertBiases[i];
	}

	for(int i = 1; i < pertWeights.size(); i++)
	{
		delete pertWeights[i];
	}
}

template void DRBMPMLayer<float>::optimizeEnergy_cpu();
template void DRBMPMLayer<double>::optimizeEnergy_cpu();

#ifdef CPU_ONLY
STUB_GPU(DRBMPMLayer);
#endif

INSTANTIATE_CLASS(DRBMPMLayer);
//REGISTER_LAYER_CLASS(DRBMPM);

} // namespace caffe
