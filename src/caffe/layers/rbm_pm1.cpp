#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/rbm_layers.hpp"
#include "caffe/util/mlg_rng.hpp"

namespace caffe {

template <typename Dtype>
void RBMPM1Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	RBMLayer<Dtype>::LayerSetUp(bottom,top);

	const int repTimes = this->layer_param_.rbm_param().rbm_pm_param().batch_repeats();
	this->persistent = this->layer_param_.rbm_param().rbm_pm_param().persistent();

	if(this->persistent){
		vector<int> X1SShape(2);
		X1SShape[0] = this->M_*repTimes;
		X1SShape[1] = this->K_;
		X1Chain.Reshape(X1SShape);
		caffe_rng_uniform(X1Chain.count(), Dtype(0.), Dtype(1.), X1Chain.mutable_cpu_data());
	}
}


template <typename Dtype>
void RBMPM1Layer<Dtype>::gradient_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom){
	NOT_IMPLEMENTED;
	/*
	//create tmp for parameters
    vector<int> bias_shape(2);

    bias_shape[0] = this->M_;
    bias_shape[1] = this->K_;


	Blob<Dtype> bTmp;
	bTmp.Reshape( bias_shape );
	//caffe_set(bTmp.count(), Dtype(0), bTmp.mutable_cpu_data());

	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
			this->M_, this->K_, 1,
			(Dtype)1., this->ones_m_.cpu_data(), this->blobs_[1]->cpu_data(),
			(Dtype)0., bTmp.mutable_cpu_data());

    bias_shape[0] = this->M_;
    bias_shape[1] = this->N_;


	Blob<Dtype> cTmp;
	cTmp.Reshape( bias_shape );
	//caffe_set(cTmp.count(), Dtype(0), cTmp.mutable_cpu_data());

	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
			this->M_, this->N_, 1,
			(Dtype)1., this->ones_m_.cpu_data(), this->blobs_[2]->cpu_data(),
			(Dtype)0., cTmp.mutable_cpu_data());



	// perturb parameters
	Blob<Dtype> ra;

	ra.ReshapeLike(bTmp);
	Dtype* bTmpMutable = bTmp.mutable_cpu_data();
	MLGRNG<Dtype>::getInstance().mlg_cpu_gumbel(ra.count(), ra.mutable_cpu_data());

	for(int i = 0; i < bTmp.count(); i++){
		bTmpMutable[i] = bTmpMutable[i] + 1 * ra.cpu_data()[i];
	}

	ra.ReshapeLike(cTmp);
	Dtype* cTmpMutable = cTmp.mutable_cpu_data();
	MLGRNG<Dtype>::getInstance().mlg_cpu_gumbel(ra.count(), ra.mutable_cpu_data());

	for(int i = 0; i < cTmp.count(); i++){
		cTmpMutable[i] = cTmpMutable[i] + 1 * ra.cpu_data()[i];
	}


	Dtype scalar =  1. / this->M_;
	const Dtype* X0S = bottom[0]->cpu_data();
	const Dtype* H0S = top[0]->cpu_data();

	Dtype* X1S = this->X1S.mutable_cpu_data();
	Dtype* H1S = this->H1S.mutable_cpu_data();

	caffe_copy(bottom[0]->count(), X0S, X1S);
	caffe_copy(top[0]->count(), H0S, H1S);

	find_map_cpu(&(this->X1S_), &(this->H1S_), &bTmp, &cTmp, this->blobs_[0].get());

	X1S = this->X1S.mutable_cpu_data();
	H1S = this->H1S.mutable_cpu_data();

	if (this->param_propagate_down_[0]) {
		Blob<Dtype> tmp1(this->blobs_[0]->shape());
		Blob<Dtype> tmp2(this->blobs_[0]->shape());

    	caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			this->N_, this->K_, this->M_,
    			(Dtype)1., H0S, X0S,
    			(Dtype)0., tmp1.mutable_cpu_data());

    	caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			this->N_, this->K_, this->M_,
    			(Dtype)1., H1S, X1S,
    			(Dtype)0., tmp2.mutable_cpu_data());

    	caffe_sub(tmp1.count(), tmp1.cpu_data(), tmp2.cpu_data(), this->blobs_[0]->mutable_cpu_diff());
    	caffe_scal<Dtype>(this->blobs_[0]->count(), scalar, this->blobs_[0]->mutable_cpu_diff());
	}

	if (this->param_propagate_down_[1]) {
		Blob<Dtype> tmp(bottom[0]->shape());

		caffe_sub(bottom[0]->count(), X0S, X1S, tmp.mutable_cpu_data());
    	caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			this->K_, 1, this->M_,
    			(Dtype)1., tmp.cpu_data(), this->ones_m_.cpu_data(),
    			(Dtype)0., this->blobs_[1]->mutable_cpu_diff());

    	caffe_scal<Dtype>(this->blobs_[1]->count(), scalar, this->blobs_[1]->mutable_cpu_diff());
	}

	if (this->param_propagate_down_[2]) {
		Blob<Dtype> tmp(top[0]->shape());

		caffe_sub(top[0]->count(), H0S, H1S, tmp.mutable_cpu_data());
    	caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			this->N_, 1, this->M_,
    			(Dtype)1., tmp.cpu_data(), this->ones_m_.cpu_data(),
    			(Dtype)0., this->blobs_[2]->mutable_cpu_diff());

    	caffe_scal<Dtype>(this->blobs_[2]->count(), scalar, this->blobs_[2]->mutable_cpu_diff());
	}
*/
}

#ifdef CPU_ONLY
STUB_GPU(RBMPM1Layer);
#endif

INSTANTIATE_CLASS(RBMPM1Layer);
REGISTER_LAYER_CLASS(RBMPM1);

} // namespace caffe
