#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/drbm_layers.hpp"
#include "caffe/util/mlg_math.hpp"

namespace caffe {


template <typename Dtype>
void DRBMPMLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	NOT_IMPLEMENTED;
	//generate samples
	generatePerturbations_gpu();
	optimizeEnergy_gpu();
	DRBMLayer<Dtype>::gradient_gpu();
	DRBMLayer<Dtype>::Backward_gpu(top, propagate_down, bottom);
}

template <typename Dtype>
void DRBMPMLayer<Dtype>::optimizeEnergy_gpu(){
	NOT_IMPLEMENTED;
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
