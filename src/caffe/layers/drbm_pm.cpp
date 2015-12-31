#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/drbm_layers.hpp"

namespace caffe {

template <typename Dtype>
void DRBMPMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	NOT_IMPLEMENTED;
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

template void DRBMPMLayer<float>::optimizeEnergy_cpu();
template void DRBMPMLayer<double>::optimizeEnergy_cpu();

#ifdef CPU_ONLY
STUB_GPU(DRBMPMLayer);
#endif

INSTANTIATE_CLASS(DRBMPMLayer);
//REGISTER_LAYER_CLASS(DRBMPM);

} // namespace caffe
