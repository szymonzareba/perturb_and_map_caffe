#include <vector>

#include "caffe/layer.hpp"
#include "caffe/rbm_loss_layers.hpp"

namespace caffe {

template <typename Dtype>
void RBMReturnLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	int count = bottom[0]->count();
	caffe_copy(count, bottom[0]->gpu_data(), bottom[1]->mutable_gpu_data());
}

template <typename Dtype>
void RBMReturnLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) { NOT_IMPLEMENTED; }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(RBMReturnLayer);

}  // namespace caffe
