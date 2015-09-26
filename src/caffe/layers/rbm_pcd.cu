#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/rbm_layers.hpp"

namespace caffe {

template <typename Dtype>
void RBMPCDLayer<Dtype>::gradient_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

	Dtype scalar = -1. / this->M_;

	const Dtype* X0SData = bottom[0]->gpu_data();
	const Dtype* H0Data = this->H0.gpu_data();
	const Dtype* H0SData = top[0]->gpu_data();

	Dtype* X1SCData = this->X1Chain.mutable_gpu_data();
	Dtype* H1SCData = this->H1Chain.mutable_gpu_data();

	const Dtype* W = this->blobs_[0]->gpu_data();
	const Dtype* b = this->blobs_[1]->gpu_data();
	const Dtype* c = this->blobs_[2]->gpu_data();

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
    	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
    			this->M_, this->K_, this->N_,
    			(Dtype)1., H1SCData, W,
    			(Dtype)0., X1SCData);

    	// X1S = 1 * bm * b + 1 * X1S
    	// [m,k] = 1 * [m,1] * [1,k] + 1 * [m,k]
    	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
    			this->M_, this->K_, 1,
    			(Dtype)1., this->ones_m_.gpu_data(), b,
    			(Dtype)1., X1SCData);


    	this->sigmoid_gpu(this->X1Chain.count(), X1SCData);
    	this->sample_gpu(this->X1Chain.count(), X1SCData);

    	// H1S = 1 * X1S * W(T) + 0 * H1S
    	// [m,n] = 1 * [m,k] * [k,n] + 0 * [m,n]
    	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
    			this->M_, this->N_, this->K_,
    			(Dtype)1., X1SCData, W,
    			(Dtype)0., H1SCData);

    	// H1S = 1 * cm(T) * c + 1 * H1S
    	// [m,n] = 1 * [m,1] * [1,n] + 1 * [m,n]
    	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
    			this->M_, this->N_, 1,
    			(Dtype)1., this->ones_m_.gpu_data(), c,
    			(Dtype)1., H1SCData);

    	this->sigmoid_gpu(this->H1Chain.count(), H1SCData);
    	this->sample_gpu(this->H1Chain.count(), H1SCData);
    }


    int repeats = this->M_ / chainNum;
    this->replicate_data_gpu(repeats, &this->X1Chain, &this->X1S);
    this->replicate_data_gpu(repeats, &this->H1Chain, &this->H1S);

	Dtype* X1SData = this->X1S.mutable_gpu_data();
	Dtype* H1SData = this->H1S.mutable_gpu_data();


	if (this->param_propagate_down_[0]) {
	// calculate gradient with respect to W : x0S'*H0S - x1S'*H1S / M
		Blob<Dtype> tmp1(this->blobs_[0]->shape());
		caffe_gpu_set(tmp1.count(), (Dtype)0, tmp1.mutable_gpu_data());

		Blob<Dtype> tmp2(this->blobs_[0]->shape());
		caffe_gpu_set(tmp2.count(), (Dtype)0, tmp2.mutable_gpu_data());

		// dW1 = 1 * X(T) * H + 0 * dW1
		// [n,k] = 1 * [n,m] * [m,k] + 0 * [n,k]
    	caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			this->N_, this->K_, this->M_,
    			(Dtype)1., H0SData, X0SData,
    			(Dtype)0., tmp1.mutable_gpu_data());

		// dW2 = 1 * X(T) * H + 0 * dW1
		// [n,k] = 1 * [n,m] * [m,k] + 0 * [n,k]
    	caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			this->N_, this->K_, this->M_,
    			(Dtype)1., H1SData, X1SData,
    			(Dtype)0., tmp2.mutable_gpu_data());

    	caffe_gpu_sub(tmp1.count(), tmp1.gpu_data(), tmp2.gpu_data(), this->blobs_[0]->mutable_gpu_diff());
    	caffe_gpu_scal<Dtype>(this->blobs_[0]->count(), scalar, this->blobs_[0]->mutable_gpu_diff());
	}

	if (this->param_propagate_down_[1]) {
	// calculate gradient with respect to b : avg( x0s - x1s ) : c * ones(size(c,2),1) / M
		Blob<Dtype> tmp(bottom[0]->shape());
		caffe_gpu_set(tmp.count(), (Dtype)0, tmp.mutable_gpu_data());
		caffe_gpu_sub(bottom[0]->count(), X0SData, X1SData, tmp.mutable_gpu_data());

		// dB = 1 * ones(T) * tmp + 0 * dB
		// [k,1] = 1 * [k,m] * [m,1]
    	caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			this->K_, 1, this->M_,
    			(Dtype)1., tmp.gpu_data(), this->ones_m_.gpu_data(),
    			(Dtype)0., this->blobs_[1]->mutable_gpu_diff());

    	// dB = dB / M
    	caffe_gpu_scal<Dtype>(this->blobs_[1]->count(), scalar, this->blobs_[1]->mutable_gpu_diff());
	}

	if (this->param_propagate_down_[2]) {
	// calculate gradient with respect to c : avg( h0s - h1s ) : c * ones(size(c,2),1) / M
		Blob<Dtype> tmp(top[0]->shape());
		caffe_gpu_set(tmp.count(), (Dtype)0, tmp.mutable_gpu_data());
		caffe_gpu_sub(tmp.count(), H0SData, H1SData, tmp.mutable_gpu_data());

		// dC = 1 * ones(T) * tmp + 0 * dC
		// [n,1] = 1 * [n,m] * [m,1]
    	caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			this->N_, 1, this->M_,
    			(Dtype)1., tmp.gpu_data(), this->ones_m_.gpu_data(),
    			(Dtype)0., this->blobs_[2]->mutable_gpu_diff());

    	// dC = dC / M
    	caffe_gpu_scal<Dtype>(this->blobs_[2]->count(), scalar, this->blobs_[2]->mutable_gpu_diff());
	}
}

template void RBMPCDLayer<float>::gradient_gpu( \
     const std::vector<Blob<float>*>& top, \
     const std::vector<bool>& propagate_down, \
     const std::vector<Blob<float>*>& bottom);

template void RBMPCDLayer<double>::gradient_gpu( \
     const std::vector<Blob<double>*>& top, \
     const std::vector<bool>& propagate_down, \
     const std::vector<Blob<double>*>& bottom);

} // namespace caffe
