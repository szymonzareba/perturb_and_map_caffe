#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/rbm_layers.hpp"
#include "caffe/util/mlg_rng.hpp"

namespace caffe {

template <typename Dtype>
void RBMPPM1Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	RBMLayer<Dtype>::LayerSetUp(bottom,top);

	const int repTimes = this->layer_param_.rbm_param().rbm_pm_param().batch_repeats();

	// tmp X1S for gibbs sampler
	vector<int> X1SShape(2);
	X1SShape[0] = this->M_*repTimes;
	X1SShape[1] = this->K_;
	X1Chain.Reshape(X1SShape);
	caffe_rng_uniform(X1Chain.count(), Dtype(0.), Dtype(1.), X1Chain.mutable_cpu_data());

	// tmp H1S for gibbs sampler
	vector<int> H1SShape(2);
	H1SShape[0] = this->M_*repTimes;
	H1SShape[1] = this->N_;
	H1Chain.Reshape(H1SShape);
	caffe_rng_uniform(H1Chain.count(), Dtype(0.), Dtype(1.), H1Chain.mutable_cpu_data());

}

template <typename Dtype>
void RBMPPM1Layer<Dtype>::gradient_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom){
	NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(RBMPPM1Layer);
#endif

INSTANTIATE_CLASS(RBMPPM1Layer);
REGISTER_LAYER_CLASS(RBMPPM1);

} // namespace caffe
