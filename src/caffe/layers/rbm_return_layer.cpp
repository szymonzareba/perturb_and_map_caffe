#include <vector>

#include "caffe/layer.hpp"
#include "caffe/rbm_loss_layers.hpp"

namespace caffe {

template <typename Dtype>
void RBMReturnLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void RBMReturnLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void RBMReturnLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	int count = bottom[0]->count();
	caffe_copy(count, bottom[0]->cpu_data(), bottom[1]->mutable_cpu_data());
}

template <typename Dtype>
void RBMReturnLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) { NOT_IMPLEMENTED; }
  }
}

INSTANTIATE_CLASS(RBMReturnLayer);
REGISTER_LAYER_CLASS(RBMReturn);

}  // namespace caffe
