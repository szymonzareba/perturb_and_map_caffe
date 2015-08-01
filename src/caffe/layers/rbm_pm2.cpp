#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/rbm_layers.hpp"
#include "caffe/util/mlg_rng.hpp"

namespace caffe {

template <typename Dtype>
void RBMPM2Layer<Dtype>::gradient_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom){
	NOT_IMPLEMENTED;
}

template <typename Dtype>
void RBMPM2Layer<Dtype>::find_w_mask_cpu(Blob<Dtype>* W){
	NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(RBMPM2Layer);
#endif

INSTANTIATE_CLASS(RBMPM2Layer);
REGISTER_LAYER_CLASS(RBMPM2);

} // namespace caffe
