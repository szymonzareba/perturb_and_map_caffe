#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/rbm_layers.hpp"

namespace caffe {

template <typename Dtype>
void RBMPM1Layer<Dtype>::gradient_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom){

	const int repTimes = this->layer_param_.rbm_param().rbm_pm_param().batch_repeats();
	const Dtype pertStr = this->layer_param_.rbm_param().rbm_pm_param().pert_str();

	Blob<Dtype> repX;
	replicate_data_gpu(repTimes, bottom[0], &repX);

	Blob<Dtype> repH;
	replicate_data_gpu(repTimes, top[0], &repH);

	//create ones
	vector<int> ones_shape(2);
	ones_shape[0] = this->M_ * repTimes;
	ones_shape[1] = 1;

	Blob<Dtype> ones;
	ones.Reshape(ones_shape);
	caffe_gpu_set(ones.count(), Dtype(1), ones.mutable_gpu_data());

	//create tmp for parameters
    vector<int> bias_shape(2);

    bias_shape[0] = this->M_ * repTimes;
    bias_shape[1] = this->K_;


	Blob<Dtype> bTmp;
	bTmp.Reshape( bias_shape );

	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
			this->M_ * repTimes, this->K_, 1,
			(Dtype)1., ones.gpu_data(), this->blobs_[1]->gpu_data(),
			(Dtype)0., bTmp.mutable_gpu_data());

    bias_shape[0] = this->M_ * repTimes;
    bias_shape[1] = this->N_;


	Blob<Dtype> cTmp;
	cTmp.Reshape( bias_shape );

	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
			this->M_ * repTimes, this->N_, 1,
			(Dtype)1., ones.gpu_data(), this->blobs_[2]->gpu_data(),
			(Dtype)0., cTmp.mutable_gpu_data());



	// perturb parameters
	Blob<Dtype> ra;

	ra.ReshapeLike(bTmp);
	Dtype* bTmpMutable = bTmp.mutable_gpu_data();
	MLGRNG<Dtype>::getInstance().mlg_gpu_gumbel(ra.count(), ra.mutable_gpu_data());
	caffe_gpu_scal<Dtype>(ra.count(), pertStr, ra.mutable_gpu_data());
	caffe_gpu_add(bTmp.count(), bTmpMutable, ra.gpu_data(), bTmpMutable);

	ra.ReshapeLike(cTmp);
	Dtype* cTmpMutable = cTmp.mutable_gpu_data();
	MLGRNG<Dtype>::getInstance().mlg_gpu_gumbel(ra.count(), ra.mutable_gpu_data());
	caffe_gpu_scal<Dtype>(ra.count(), pertStr, ra.mutable_gpu_data());
	caffe_gpu_add(bTmp.count(), cTmpMutable, ra.gpu_data(), cTmpMutable);


	Dtype scalar =  -1. / ( this->M_ * repTimes );

	Blob<Dtype> X1;
	X1.ReshapeLike(repX);

	Blob<Dtype> H1;
	H1.ReshapeLike(repH);

	const Dtype* X0S = repX.gpu_data();
	const Dtype* H0S = repH.gpu_data();
	Dtype* X1S = X1.mutable_gpu_data();
	Dtype* H1S = H1.mutable_gpu_data();

	caffe_copy(repX.count(), X0S, X1S);
	caffe_copy(repH.count(), H0S, H1S);

	find_map_gpu(&X1, &H1, &bTmp, &cTmp, this->blobs_[0].get());

	X1S = X1.mutable_gpu_data();
	H1S = H1.mutable_gpu_data();

	if (this->param_propagate_down_[0]) {
		Blob<Dtype> tmp1(this->blobs_[0]->shape());
		Blob<Dtype> tmp2(this->blobs_[0]->shape());

    	caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			this->N_, this->K_, this->M_ * repTimes,
    			(Dtype)1., H0S, X0S,
    			(Dtype)0., tmp1.mutable_gpu_data());

    	caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			this->N_, this->K_, this->M_ * repTimes,
    			(Dtype)1., H1S, X1S,
    			(Dtype)0., tmp2.mutable_gpu_data());

    	caffe_gpu_sub(tmp1.count(), tmp1.gpu_data(), tmp2.gpu_data(), this->blobs_[0]->mutable_gpu_diff());
    	caffe_gpu_scal<Dtype>(this->blobs_[0]->count(), scalar, this->blobs_[0]->mutable_gpu_diff());
	}

	if (this->param_propagate_down_[1]) {
		Blob<Dtype> tmp(repX.shape());

		caffe_gpu_sub(tmp.count(), X0S, X1S, tmp.mutable_gpu_data());
    	caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			this->K_, 1, this->M_ * repTimes,
    			(Dtype)1., tmp.gpu_data(), ones.gpu_data(),
    			(Dtype)0., this->blobs_[1]->mutable_gpu_diff());

    	caffe_gpu_scal<Dtype>(this->blobs_[1]->count(), scalar, this->blobs_[1]->mutable_gpu_diff());
	}

	if (this->param_propagate_down_[2]) {
		Blob<Dtype> tmp(repH.shape());

		caffe_gpu_sub(tmp.count(), H0S, H1S, tmp.mutable_gpu_data());
    	caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			this->N_, 1, this->M_ * repTimes,
    			(Dtype)1., tmp.gpu_data(), ones.gpu_data(),
    			(Dtype)0., this->blobs_[2]->mutable_gpu_diff());

    	caffe_gpu_scal<Dtype>(this->blobs_[2]->count(), scalar, this->blobs_[2]->mutable_gpu_diff());
	}
}

template
void RBMPM1Layer<float>::gradient_gpu(const vector<Blob<float>*>& top, \
      const vector<bool>& propagate_down, \
      const vector<Blob<float>*>& bottom);

template
void RBMPM1Layer<double>::gradient_gpu(const vector<Blob<double>*>& top, \
      const vector<bool>& propagate_down, \
      const vector<Blob<double>*>& bottom);

} // namespace caffe
