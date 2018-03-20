#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/drbm_layers.hpp"

namespace caffe {


template <typename Dtype>
void DRBMMeanFieldLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	NOT_IMPLEMENTED;
	// calculate samples
	DRBMLayer<Dtype>::gradient_cpu();
	DRBMLayer<Dtype>::Backward_cpu(top, propagate_down, bottom);
}


#ifdef CPU_ONLY
STUB_GPU(DRBMMeanFieldLayer);
#endif

INSTANTIATE_CLASS(DRBMMeanFieldLayer);
REGISTER_LAYER_CLASS(DRBMMeanField);

} // namespace caffe
