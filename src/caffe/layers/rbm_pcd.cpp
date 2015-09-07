#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/rbm_layers.hpp"

namespace caffe {

template <typename Dtype>
void RBMPCDLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	RBMLayer<Dtype>::LayerSetUp(bottom,top);

	if(this->layer_param_.rbm_param().rbm_pcd_param().gibbs_chains() > this->M_)
	{
		LOG(FATAL) << "gibbs chains > minibatch" << std::endl;
	}

	if(this->M_ % this->layer_param_.rbm_param().rbm_pcd_param().gibbs_chains() != 0)
	{
		LOG(FATAL) << "minibatch % gibbs chains != 0" << std::endl;
	}

	// tmp X1S for gibbs sampler
	vector<int> X1SShape(2);
	X1SShape[0] = this->layer_param_.rbm_param().rbm_pcd_param().gibbs_chains();
	X1SShape[1] = this->K_;
	X1Chain_.Reshape(X1SShape);
	caffe_rng_uniform(X1Chain_.count(), Dtype(0.), Dtype(1.), X1Chain_.mutable_cpu_data());

	// tmp H1S for gibbs sampler
	vector<int> H1SShape(2);
	H1SShape[0] = this->layer_param_.rbm_param().rbm_pcd_param().gibbs_chains();
	H1SShape[1] = this->N_;
	H1Chain_.Reshape(H1SShape);
	caffe_rng_uniform(H1Chain_.count(), Dtype(0.), Dtype(1.), H1Chain_.mutable_cpu_data());
}



template <typename Dtype>
void RBMPCDLayer<Dtype>::gradient_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom){

	Dtype scalar =  1. / this->M_;
	const Dtype* X0S = bottom[0]->cpu_data();
	const Dtype* H0S = top[0]->cpu_data();

	Dtype* X1Chain = this->X1Chain_.mutable_cpu_data();
	Dtype* H1Chain = this->H1Chain_.mutable_cpu_data();

	Dtype* X1S = this->X1S_.mutable_cpu_data();
	Dtype* H1S = this->H1S_.mutable_cpu_data();


	int chainNum;
	if(this->layer_param_.rbm_param().rbm_pcd_param().gibbs_chains() == -1)
	{
		chainNum = this->M_;
	}
	else
	{
		chainNum = this->layer_param_.rbm_param().rbm_pcd_param().gibbs_chains();
	}


    for(int gibbsStep = 0; gibbsStep < this->layer_param_.rbm_param().rbm_pcd_param().gibbs_steps(); gibbsStep++){

    	// X1S = 1 * H1S * W + 0 * X1S
    	// [m,k] = 1 * [m,n] * [n,k] + 0 * [m,k]
    	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
    			chainNum, this->K_, this->N_,
    			(Dtype)1., H1Chain, this->blobs_[0]->cpu_data(),
    			(Dtype)0., X1Chain);

    	// X1S = 1 * bm * b + 1 * X1S
    	// [m,k] = 1 * [m,1] * [1,k] + 1 * [m,k]
    	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
    			chainNum, this->K_, 1,
    			(Dtype)1., this->ones_m_.cpu_data(), this->blobs_[1]->cpu_data(),
    			(Dtype)1., X1Chain);

    	for(int i = 0; i < this->X1Chain_.count(); i++){
    		X1Chain[i] = sigmoid_cpu(X1Chain[i]);
    	}

    	// sample
    	sample_cpu(this->X1Chain_.count(), X1Chain);

    	// H1S = 1 * X1S * W(T) + 0 * H1S
    	// [m,n] = 1 * [m,k] * [k,n] + 0 * [m,n]
    	// OK
    	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
    			chainNum, this->N_, this->K_,
    			(Dtype)1., X1Chain, this->blobs_[0]->cpu_data(),
    			(Dtype)0., H1Chain);

    	// H1S = 1 * cm * c + 1 * H1S
    	// [m,n] = 1 * [m,1] * [1,n] + 1 * [m,n]
    	// OK
    	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
    			chainNum, this->N_, 1,
    			(Dtype)1., this->ones_m_.cpu_data(), this->blobs_[2]->cpu_data(),
    			(Dtype)1., H1Chain);

    	for(int i = 0; i < this->H1Chain_.count(); i++){
    		H1Chain[i] = sigmoid_cpu(H1Chain[i]);
    	}

    	// sample
    	sample_cpu(this->H1Chain_.count(), H1Chain);
    }

    int repeats = this->M_ / chainNum;
    replicate_data_cpu(repeats, &this->X1Chain_, &this->X1S_);
    replicate_data_cpu(repeats, &this->H1Chain_, &this->H1S_);

	if (this->param_propagate_down_[0]) {
	// calculate gradient with respect to W : x0S'*H0S - x1S'*H1S / M
		Blob<Dtype> tmp1(this->blobs_[0]->shape());
		Blob<Dtype> tmp2(this->blobs_[0]->shape());

		// dW1 = 1 * H(T) * X + 0 * dW1
		// [n,k] = 1 * [n,m] * [m,k] + 0 * [n,k]
    	caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			this->N_, this->K_, this->M_,
    			(Dtype)1., H0S, X0S,
    			(Dtype)0., tmp1.mutable_cpu_data());

		// dW2 = 1 * H(T) * X + 0 * dW1
		// [n,k] = 1 * [n,m] * [m,k] + 0 * [n,k]
    	caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			this->N_, this->K_, this->M_,
    			(Dtype)1., H1S, X1S,
    			(Dtype)0., tmp2.mutable_cpu_data());

    	caffe_sub(tmp1.count(), tmp1.cpu_data(), tmp2.cpu_data(), this->blobs_[0]->mutable_cpu_diff());
    	caffe_scal<Dtype>(this->blobs_[0]->count(), scalar, this->blobs_[0]->mutable_cpu_diff());
	}

	if (this->param_propagate_down_[1]) {
	// calculate gradient with respect to b : avg( x0s - x1s ) : c * ones(size(c,2),1) / M
		Blob<Dtype> tmp(bottom[0]->shape());
		caffe_sub(bottom[0]->count(), X0S, X1S, tmp.mutable_cpu_data());

		// [k,1] = 1 * [k,m] * [m,1] + 0 * [k,1]
    	caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			this->K_, 1, this->M_,
    			(Dtype)1., tmp.cpu_data(), this->ones_m_.cpu_data(),
    			(Dtype)0., this->blobs_[1]->mutable_cpu_diff());

    	// dB = dB / M
    	caffe_scal<Dtype>(this->blobs_[1]->count(), scalar, this->blobs_[1]->mutable_cpu_diff());
	}

	if (this->param_propagate_down_[2]) {
	// calculate gradient with respect to c : avg( h0s - h1s ) : c * ones(size(c,2),1) / M
		Blob<Dtype> tmp(top[0]->shape());
		caffe_sub(top[0]->count(), H0S, H1S, tmp.mutable_cpu_data());

		// dC = 1 * ones(T) * tmp + 0 * dC
		// [n,1] = 1 * [n,m] * [m,1] *  + 0 * [1,n]
    	caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			this->N_, 1, this->M_,
    			(Dtype)1., tmp.cpu_data(), this->ones_m_.cpu_data(),
    			(Dtype)0., this->blobs_[2]->mutable_cpu_diff());

    	// dC = dC / M
    	caffe_scal<Dtype>(this->blobs_[2]->count(), scalar, this->blobs_[2]->mutable_cpu_diff());
	}
}

#ifdef CPU_ONLY
STUB_GPU(RBMPCDLayer);
#endif

INSTANTIATE_CLASS(RBMPCDLayer);
REGISTER_LAYER_CLASS(RBMPCD);

} // namespace caffe
