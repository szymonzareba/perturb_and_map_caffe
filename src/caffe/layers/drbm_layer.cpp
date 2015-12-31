#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/drbm_layers.hpp"
#include <limits>

namespace caffe {

template <typename Dtype>
void DRBMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	const int axis = bottom[0]->CanonicalAxisIndex(
			this->layer_param_.drbm_param().axis());

	K_ = bottom[0]->count(axis);
	M_ = bottom[0]->count(0, axis);

	layer_sizes.push_back(K_);

	for(int i = 0; i < this->layer_param_.drbm_param().num_hidden_size(); i++)
	{
		layer_sizes.push_back(this->layer_param_.drbm_param().num_hidden(i));
	}

	if(layer_sizes.size() <= 1){
		LOG(FATAL) << "Empty num_hidden" << std::endl;
	}



	// Check if we need to set up the weights
	if (this->blobs_.size() > 0) {
		LOG(INFO) << "Skipping parameter initialization";
	} else {
		this->blobs_.resize(2*layer_sizes.size()-1);

		vector<int> weight_shape(2);
		vector<int> bias_shape(2);

		// add visible biases, first weights and first hidden biases
		// visible biases
		bias_shape[0] = K_;
		bias_shape[1] = 1;
		Blob<Dtype>* vis_bias = new Blob<Dtype>(bias_shape);
		shared_ptr<Filler<Dtype> > vis_bias_filler(GetFiller<Dtype>(this->layer_param_.drbm_param().bias_filler(0)));
		vis_bias_filler->Fill(vis_bias);
		this->blobs_[0].reset(vis_bias);
		biases.push_back(vis_bias);

		// add weights and biases, layer by layer
		for(int i = 1; i < layer_sizes.size(); i++)
		{
			//weights
			weight_shape[0] = layer_sizes[i];
			weight_shape[1] = layer_sizes[i-1];
			Blob<Dtype>* next_weights = new Blob<Dtype>(weight_shape);
			shared_ptr<Filler<Dtype> > weights_filler(GetFiller<Dtype>(this->layer_param_.drbm_param().weight_filler(i-1)));
			weights_filler->Fill(next_weights);
			this->blobs_[2*i-1].reset(next_weights);
			weights.push_back(next_weights);

			//biases
			bias_shape[0] = layer_sizes[i];
			bias_shape[1] = 1;
			Blob<Dtype>* next_bias = new Blob<Dtype>(bias_shape);
			shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(this->layer_param_.drbm_param().bias_filler(i)));
			bias_filler->Fill(next_bias);
			this->blobs_[2*i].reset(next_bias);
			biases.push_back(next_bias);
		}
	}

	// verify
	for(int i = 0; i < this->blobs_.size(); i++)
	{
		vector<int> shape = this->blobs_[i]->shape();
		LOG(INFO) << i << " blob : " << shape[0] << " " << shape[1] << std::endl;
	}

	for(int i = 0; i < this->biases.size(); i++)
	{
		vector<int> shape = this->biases[i]->shape();
		LOG(INFO) << i << " bias : " << shape[0] << " " << shape[1] << std::endl;
	}

	for(int i = 0; i < this->weights.size(); i++)
	{
		vector<int> shape = this->weights[i]->shape();
		LOG(INFO) << i << " weight : " << shape[0] << " " << shape[1] << std::endl;
	}

	// set placeholders for probs and samples
	vector<int> tmp_shape(2);

	for(int i = 0; i < layer_sizes.size(); i++)
	{
		tmp_shape[0] = M_;
		tmp_shape[1] = layer_sizes[i];

		Blob<Dtype>* hid_prob_0 = new Blob<Dtype>(tmp_shape);
		probs_0.push_back(hid_prob_0);

		Blob<Dtype>* hid_state_0 = new Blob<Dtype>(tmp_shape);
		states_0.push_back(hid_state_0);

		Blob<Dtype>* hid_prob_1 = new Blob<Dtype>(tmp_shape);
		probs_1.push_back(hid_prob_1);

		Blob<Dtype>* hid_state_1 = new Blob<Dtype>(tmp_shape);
		states_1.push_back(hid_state_1);
	}

	for(int i = 0; i < probs_0.size(); i++)
	{
		vector<int> shape = probs_0[i]->shape();
		LOG(INFO) << i << " layer_tmp : " << shape[0] << " " << shape[1] << std::endl;
	}

	tmp_shape[0] = M_;
	tmp_shape[1] = 1;

	ones_m.Reshape(tmp_shape);
	caffe_set(ones_m.count(), (Dtype)1., ones_m.mutable_cpu_data());

	// Loss layers output a scalar; 0 axes.
	vector<int> loss_shape(0);
	top[2]->Reshape(loss_shape);

	// Calculate gradient, yes!
	this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void DRBMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	// Figure out the dimensions
	const int axis = bottom[0]->CanonicalAxisIndex(
			this->layer_param_.drbm_param().axis());
	const int new_K = bottom[0]->count(axis);
	CHECK_EQ(K_, new_K)
		<< "Input size incompatible with inner product parameters.";
	// The first "axis" dimensions are independent inner products; the total
	// number of these is M_, the product over these dimensions.
	//int oldM = M_;
	M_ = bottom[0]->count(0, axis);

	// The top shape will be the bottom shape with the flattened axes dropped,
	// and replaced by a single axis with dimension num_output (N_).
	vector<int> top_shape = bottom[0]->shape();
	top_shape.resize(axis + 1);
	top_shape[axis] = layer_sizes[layer_sizes.size()-1];
	top[0]->Reshape(top_shape);
	top[1]->Reshape(top_shape);

	//LOG(INFO) << "New top size : " << top_shape[axis] << std::endl;
}

template <typename Dtype>
void DRBMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	NOT_IMPLEMENTED;
	//TODO calculate forward pass, give value to all sublayers
}

template <typename Dtype>
void DRBMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	NOT_IMPLEMENTED;
	//TODO calculate reconstructions and likelihood
}

template <typename Dtype>
Dtype DRBMLayer<Dtype>::ll_cpu() {
	NOT_IMPLEMENTED;
}

template <typename Dtype>
void DRBMLayer<Dtype>::gradient_cpu() {
	NOT_IMPLEMENTED;
	//TODO calculate and set blob_diffs
}

template <typename Dtype>
DRBMLayer<Dtype>::~DRBMLayer()
{
	for(int i = 0; i < probs_0.size(); i++)
	{
		delete probs_0[i];
		delete states_0[i];
		delete probs_1[i];
		delete states_1[i];
	}
}

#ifdef CPU_ONLY
STUB_GPU(DRBMLayer);
#endif

INSTANTIATE_CLASS(DRBMLayer);
REGISTER_LAYER_CLASS(DRBM);

} // namespace caffe
