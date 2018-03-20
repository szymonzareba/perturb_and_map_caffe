#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util_layers.hpp"
#include "caffe/util/mlg_math.hpp"

namespace caffe {

template <typename Dtype>
void BinarizationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	Dtype threshold = this->layer_param_.bin_param().threshold();
	bool work = this->layer_param_.bin_param().work();

	for(int i = 0; i < bottom.size(); i++){
		const Dtype* botData = bottom[i]->gpu_data();
		Dtype* topData = top[i]->mutable_gpu_data();
		if(work){
			binarization_kernel<Dtype><<<CAFFE_GET_BLOCKS(bottom[i]->count()), CAFFE_CUDA_NUM_THREADS>>>(bottom[i]->count(), threshold, botData, topData);
			CUDA_POST_KERNEL_CHECK;
		} else {
			caffe_copy(bottom[i]->count(), botData, topData);
		}
	}
}

template <typename Dtype>
void BinarizationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

	Dtype threshold = this->layer_param_.bin_param().threshold();
	bool work = this->layer_param_.bin_param().work();


	for(int i = 0; i < bottom.size(); i++){
		Dtype* botData = bottom[i]->mutable_gpu_data();
		const Dtype* topData = top[i]->gpu_data();
		if(work){
			binarization_kernel<Dtype><<<CAFFE_GET_BLOCKS(bottom[i]->count()), CAFFE_CUDA_NUM_THREADS>>>(bottom[i]->count(), threshold, topData, botData);
			CUDA_POST_KERNEL_CHECK;
		} else {
			caffe_copy(bottom[i]->count(), topData, botData);
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(BinarizationLayer);

} // namespace caffe
