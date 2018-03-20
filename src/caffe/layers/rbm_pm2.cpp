#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/rbm_layers.hpp"
#include "caffe/util/mlg_rng.hpp"

namespace caffe {

template <typename Dtype>
void RBMPM2Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	RBMLayer<Dtype>::LayerSetUp(bottom,top);

	const int repTimes = this->layer_param_.rbm_param().rbm_pm_param().batch_repeats();
	this->persistent = this->layer_param_.rbm_param().rbm_pm_param().persistent();

	if(this->persistent){
		vector<int> X1SShape(2);
		X1SShape[0] = this->M_*repTimes;
		X1SShape[1] = this->K_;
		X1_chain.Reshape(X1SShape);
		caffe_rng_uniform(X1_chain.count(), Dtype(0.), Dtype(1.), X1_chain.mutable_cpu_data());
	}
}

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
