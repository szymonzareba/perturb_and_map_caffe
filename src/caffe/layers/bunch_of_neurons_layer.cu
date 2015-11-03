#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mlg_neuron_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void BoNForward(const int n, const double bunch_A, const double bunch_B, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
	const double sigm = 1. / (1. + exp(-in[index]));
    out[index] = 1. - pow( 1. - pow( sigm, bunch_A ), bunch_B ); // f(x) = 1 - (1 - sigmoid(x)^A)^B
  }
}

template <typename Dtype>
void BunchOfNeuronsLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
  	Dtype* top_data = top[0]->mutable_gpu_data();
  	const int count = bottom[0]->count();
  	const double bunch_A = this->layer_param_.bunch_param().bunch_a();
  	const double bunch_B = this->layer_param_.bunch_param().bunch_b();
  // NOLINT_NEXT_LINE(whitespace/operators)
	BoNForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bunch_A, bunch_B, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void BoNBackward(const int n, const double bunch_A, const double bunch_B, const Dtype* in_diff,
    const Dtype* out_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
		const double sigmoid_x = out_data[index];
		//const Dtype derivativeBoN = bunch_A * bunch_B * (1. - sigmoid_x) * pow( sigmoid_x, bunch_B ) * pow( 1. - pow( sigmoid_x, bunch_A ), ( bunch_B - 1. ) );// f'(x) = A*B*(1-sigmoid(x))*sigmoid(x)^A*(1-sigmoid(x)^A)^(B-1)
		//out_diff[index] = in_diff[index] * bunch_A * bunch_B * (1. - sigmoid_x) * pow( sigmoid_x, bunch_B ) * pow( 1. - pow( sigmoid_x, bunch_A ), ( bunch_B - 1. ) );// f'(x) = A*B*(1-sigmoid(x))*sigmoid(x)^A*(1-sigmoid(x)^A)^(B-1)
		out_diff[index] = in_diff[index] * bunch_A * bunch_B * (1. - sigmoid_x) * pow( sigmoid_x, bunch_A ) * pow( 1. - pow( sigmoid_x, bunch_A ) , bunch_B - 1.);
  }
}

template <typename Dtype>
void BunchOfNeuronsLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    const double bunch_A = this->layer_param_.bunch_param().bunch_a();
    const double bunch_B = this->layer_param_.bunch_param().bunch_b();
    // NOLINT_NEXT_LINE(whitespace/operators)
    BoNBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bunch_A, bunch_B, top_diff, top_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BunchOfNeuronsLayer);


}  // namespace caffe
