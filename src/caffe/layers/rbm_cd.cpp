#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/rbm_layers.hpp"

namespace caffe {

template <typename Dtype>
void RBMCDLayer<Dtype>::gradient_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom){
/*
	Dtype scalar = 1. / this->M_;

	const Dtype* W = this->blobs_[0]->cpu_data();
	const Dtype* b = this->blobs_[1]->cpu_data();
	const Dtype* c = this->blobs_[2]->cpu_data();

	const Dtype* X0SData = bottom[0]->cpu_data();
	const Dtype* H0Data = H0->cpu_data();
	const Dtype* H0SData = H0S->cpu_data();

	Dtype* X1Data = X1->mutable_cpu_data();
	Dtype* X1SData = X1S->mutable_cpu_data();
	Dtype* H1Data = H1->mutable_cpu_data();
	Dtype* H1SData = H1S->mutable_cpu_data();

	caffe_copy(top[0]->count(), H0SData, H1SData);

    for(int gibbsStep = 0; gibbsStep < this->layer_param_.rbm_param().rbm_cd_param().gibbs_steps(); gibbsStep++){

    	// X1S = 1 * H1S * W + 0 * X1S
    	// [m,k] = 1 * [m,n] * [n,k] + 0 * [m,k]
    	// OK
    	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
    			this->M_, this->K_, this->N_,
    			(Dtype)1., H1SData, W,
    			(Dtype)0., X1Data);

    	// X1S = 1 * bm * b + 1 * X1S
    	// [m,k] = 1 * [m,1] * [1,k] + 1 * [m,k]
    	// OK
    	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
    			this->M_, this->K_, 1,
    			(Dtype)1., this->ones_m_.cpu_data(), b,
    			(Dtype)1., X1Data);

    	for(int i = 0; i < this->X1S.count(); i++){
    		X1Data[i] = sigmoid_cpu(X1Data[i]);
    	}

    	// sample
    	sample_cpu(this->X1S.count(), X1Data, X1SData);

    	// H1S = 1 * X1S * W(T) + 0 * H1S
    	// [m,n] = 1 * [m,k] * [k,n] + 0 * [m,n]
    	// OK
    	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
    			this->M_, this->N_, this->K_,
    			(Dtype)1., X1SData, W,
    			(Dtype)0., H1Data);

    	// H1S = 1 * cm * c + 1 * H1S
    	// [m,n] = 1 * [m,1] * [1,n] + 1 * [m,n]
    	// OK
    	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
    			this->M_, this->N_, 1,
    			(Dtype)1., this->ones_m_.cpu_data(), c,
    			(Dtype)1., H1Data);

    	for(int i = 0; i < this->H1S.count(); i++){
    		H1Data[i] = sigmoid_cpu(H1Data[i]);
    	}

    	// sample
    	sample_cpu(this->H1S.count(), H1Data, H1SData);
    }

	if (this->param_propagate_down_[0]) {
	// calculate gradient with respect to W : x0S'*H0S - x1S'*H1S / M
		Blob<Dtype> tmp1(this->blobs_[0]->shape());
		Blob<Dtype> tmp2(this->blobs_[0]->shape());

		// dW1 = 1 * H(T) * X + 0 * dW1
		// [n,k] = 1 * [n,m] * [m,k] + 0 * [n,k]
    	caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			this->N_, this->K_, this->M_,
    			(Dtype)1., H0SData, X0SData,
    			(Dtype)0., tmp1.mutable_cpu_data());

		// dW2 = 1 * H(T) * X + 0 * dW1
		// [n,k] = 1 * [n,m] * [m,k] + 0 * [n,k]
    	caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			this->N_, this->K_, this->M_,
    			(Dtype)1., H1SData, X1SData,
    			(Dtype)0., tmp2.mutable_cpu_data());

    	caffe_sub(tmp1.count(), tmp1.cpu_data(), tmp2.cpu_data(), this->blobs_[0]->mutable_cpu_diff());
    	caffe_scal<Dtype>(this->blobs_[0]->count(), scalar, this->blobs_[0]->mutable_cpu_diff());
	}

	if (this->param_propagate_down_[1]) {
	// calculate gradient with respect to b : avg( x0s - x1s ) : c * ones(size(c,2),1) / M
		Blob<Dtype> tmp(bottom[0]->shape());
		caffe_sub(bottom[0]->count(), X0SData, X1SData, tmp.mutable_cpu_data());

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
		caffe_sub(top[0]->count(), H0SData, H1SData, tmp.mutable_cpu_data());

		// dC = 1 * ones(T) * tmp + 0 * dC
		// [n,1] = 1 * [n,m] * [m,1] *  + 0 * [1,n]
    	caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			this->N_, 1, this->M_,
    			(Dtype)1., tmp.cpu_data(), this->ones_m_.cpu_data(),
    			(Dtype)0., this->blobs_[2]->mutable_cpu_diff());

    	// dC = dC / M
    	caffe_scal<Dtype>(this->blobs_[2]->count(), scalar, this->blobs_[2]->mutable_cpu_diff());
	}
	*/
}

#ifdef CPU_ONLY
STUB_GPU(RBMCDLayer);
#endif

INSTANTIATE_CLASS(RBMCDLayer);
REGISTER_LAYER_CLASS(RBMCD);

} // namespace caffe
