#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/mlg_neuron_layers.hpp"

namespace caffe {

template <typename Dtype>
void BunchOfNeuronsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
void BunchOfNeuronsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	  const double bunch_A = this->layer_param_.bunch_param().bunch_a();
	  const double bunch_B = this->layer_param_.bunch_param().bunch_b();

	  const Dtype* bottom_data = bottom[0]->cpu_data();
	  Dtype* top_data = top[0]->mutable_cpu_data();
	  const int count = bottom[0]->count();
	  for (int i = 0; i < count; ++i) {
		  top_data[i] = 1. - pow( 1. - pow( sigmoid(bottom_data[i]), bunch_A ), bunch_B ); // f(x) = 1 - (1 - sigmoid(x)^A)^B
	  }


}

template <typename Dtype>
void BunchOfNeuronsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
	  const double bunch_A = this->layer_param_.bunch_param().bunch_a();
	  const double bunch_B = this->layer_param_.bunch_param().bunch_b();

	  const Dtype* top_data = top[0]->cpu_data();
	  const Dtype* top_diff = top[0]->cpu_diff();
	  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	  const int count = bottom[0]->count();
	  for (int i = 0; i < count; ++i) {
		  const Dtype sigmoid_x = top_data[i];
		  const Dtype derivativeBoN = bunch_A * bunch_B * (1. - sigmoid_x) * pow( sigmoid_x, bunch_B ) * pow( 1. - pow( sigmoid_x, bunch_A ), ( bunch_B -1. ) );// f'(x) = A*B*(1-sigmoid(x))*sigmoid(x)^A*(1-sigmoid(x)^A)^(B-1)
		  bottom_diff[i] = top_diff[i] * derivativeBoN;
	  }
  }
}

#ifdef CPU_ONLY
STUB_GPU(BunchOfNeuronsLayer);
#endif

INSTANTIATE_CLASS(BunchOfNeuronsLayer);
REGISTER_LAYER_CLASS(BunchOfNeurons);

}  // namespace caffe
