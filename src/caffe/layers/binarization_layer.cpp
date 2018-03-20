#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util_layers.hpp"

namespace caffe {

template <typename Dtype>
void BinarizationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	for(int i = 0; i < bottom.size(); i++){
		top[i]->ReshapeLike(*bottom[i]);
	}

	// Calculate gradient, yes!
	this->param_propagate_down_.resize(this->blobs_.size(), false);
}

template <typename Dtype>
void BinarizationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	for(int i = 0; i < bottom.size(); i++){
		top[i]->ReshapeLike(*bottom[i]);
	}
}

template <typename Dtype>
void BinarizationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	float threshold = this->layer_param_.bin_param().threshold();
	bool work = this->layer_param_.bin_param().work();


	for(int i = 0; i < bottom.size(); i++){
		const Dtype* botData = bottom[i]->cpu_data();
		Dtype* topData = top[i]->mutable_cpu_data();
		if(work){
			for(int l = 0; l < bottom[i]->count(); l++){
				if(botData[l] > threshold){
					topData[l] = (Dtype) 1.;
				}else{
					topData[l] = (Dtype) 0.;
				}
			}
		} else {
			caffe_copy(bottom[i]->count(), botData, topData);
		}
	}
}

template <typename Dtype>
void BinarizationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

	float threshold = this->layer_param_.bin_param().threshold();
	bool work = this->layer_param_.bin_param().work();


	for(int i = 0; i < bottom.size(); i++){

		Dtype* botData = bottom[i]->mutable_cpu_data();
		const Dtype* topData = top[i]->cpu_data();

		if(work){
			for(int l = 0; l < bottom[i]->count(); l++){
				if(topData[l] > threshold){
					botData[l] = (Dtype) 1.;
				}else{
					botData[l] = (Dtype) 0.;
				}
			}
		} else {
			caffe_copy(bottom[i]->count(), topData, botData);
		}
	}

}




#ifdef CPU_ONLY
STUB_GPU(BinarizationLayer);
#endif

INSTANTIATE_CLASS(BinarizationLayer);
REGISTER_LAYER_CLASS(Binarization);

} // namespace caffe
