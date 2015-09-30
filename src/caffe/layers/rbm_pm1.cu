#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/rbm_layers.hpp"

namespace caffe {

template <typename Dtype>
void RBMPM1Layer<Dtype>::gradient_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom){

		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(this->blobs_[0]->count(), this->blobs_[0]->gpu_data())) LOG(INFO) << "W not finite" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(this->blobs_[1]->count(), this->blobs_[1]->gpu_data())) LOG(INFO) << "b not finite" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(this->blobs_[2]->count(), this->blobs_[2]->gpu_data())) LOG(INFO) << "c not finite" << std::endl;

		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(this->blobs_[0]->count(), this->blobs_[0]->gpu_data())) LOG(INFO) << "W not in float range" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(this->blobs_[1]->count(), this->blobs_[1]->gpu_data())) LOG(INFO) << "b not in float range" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(this->blobs_[2]->count(), this->blobs_[2]->gpu_data())) LOG(INFO) << "c not in float range" << std::endl;

	const int repTimes = this->layer_param_.rbm_param().rbm_pm_param().batch_repeats();
	const Dtype pertStr = this->layer_param_.rbm_param().rbm_pm_param().pert_str();

	Blob<Dtype> repX;
	this->replicate_data_gpu(repTimes, bottom[0], &repX);

	Blob<Dtype> repH;
	switch(this->layer_param_.rbm_param().rbm_pm_param().map_method()){
		case RBMPMLayer<Dtype>::CoordinateDescent:
		{
			this->replicate_data_gpu(repTimes, top[0], &repH);
		}
		break;
		case RBMPMLayer<Dtype>::GreedyEnergyOptimization:
		{
			this->replicate_data_gpu(repTimes, top[0], &repH);
		}
		break;
		case RBMPMLayer<Dtype>::FreeEnergyGradientDescent:
		{
			this->replicate_data_gpu(repTimes, &this->H0, &repH);
		}
		break;
		case RBMPMLayer<Dtype>::NegativeGreedyEnergyOptimization:
		{
			this->replicate_data_gpu(repTimes, top[0], &repH);
		}
		break;
		case RBMPMLayer<Dtype>::NegativeFreeEnergyGradientDescent:
		{
			this->replicate_data_gpu(repTimes, &this->H0, &repH);
		}
		break;
	}

		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(repX.count(), repX.gpu_data())) LOG(INFO) << "repX not finite" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(this->H0.count(), this->H0.gpu_data())) LOG(INFO) << "H0 not finite" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(top[0]->count(), top[0]->gpu_data())) LOG(INFO) << "H0S not finite" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(repH.count(), repH.gpu_data())) LOG(INFO) << "repH not finite" << std::endl;

		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(repX.count(), repX.gpu_data(),0,1)) LOG(INFO) << "repX not in range" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(this->H0.count(), this->H0.gpu_data(),0,1)) LOG(INFO) << "H0 not in range" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(top[0]->count(), top[0]->gpu_data(),0,1)) LOG(INFO) << "H0S not in range" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(repH.count(), repH.gpu_data(),0,1)) LOG(INFO) << "repH not in range" << std::endl;

		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(repX.count(), repX.gpu_data())) LOG(INFO) << "repX not in float range" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(this->H0.count(), this->H0.gpu_data())) LOG(INFO) << "H0 not in float range" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(top[0]->count(), top[0]->gpu_data())) LOG(INFO) << "H0S not in float range" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(repH.count(), repH.gpu_data())) LOG(INFO) << "repH not in float range" << std::endl;

	//create ones
	vector<int> ones_shape(2);
	ones_shape[0] = this->M_ * repTimes;
	ones_shape[1] = 1;

	Blob<Dtype> ones;
	ones.Reshape(ones_shape);
	caffe_gpu_set(ones.count(), Dtype(1), ones.mutable_gpu_data());

		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(ones.count(), ones.gpu_data())) LOG(INFO) << "ones not finite" << std::endl;

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

    	if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(bTmp.count(), bTmp.gpu_data())) LOG(INFO) << "bTmp not finite" << std::endl;
    	if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(bTmp.count(), bTmp.gpu_data())) LOG(INFO) << "bTmp not in float range" << std::endl;

	Blob<Dtype> cTmp;
	cTmp.Reshape( bias_shape );

	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
			this->M_ * repTimes, this->N_, 1,
			(Dtype)1., ones.gpu_data(), this->blobs_[2]->gpu_data(),
			(Dtype)0., cTmp.mutable_gpu_data());

		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(cTmp.count(), cTmp.gpu_data())) LOG(INFO) << "cTmp not finite" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(cTmp.count(), cTmp.gpu_data())) LOG(INFO) << "cTmp not in float range" << std::endl;

	// perturb parameters
	Blob<Dtype> ra;

	ra.ReshapeLike(bTmp);
	Dtype* bTmpMutable = bTmp.mutable_gpu_data();
	MLGRNG<Dtype>::getInstance().mlg_gpu_gumbel(ra.count(), ra.mutable_gpu_data());
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(ra.count(), ra.gpu_data())) LOG(INFO) << "ra not finite" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(ra.count(), ra.gpu_data())) LOG(INFO) << "ra not in float range" << std::endl;
	caffe_gpu_scal<Dtype>(ra.count(), pertStr, ra.mutable_gpu_data());
	caffe_gpu_add(bTmp.count(), bTmpMutable, ra.gpu_data(), bTmpMutable);

		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(bTmp.count(), bTmp.gpu_data())) LOG(INFO) << "bTmp not finite" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(bTmp.count(), bTmp.gpu_data())) LOG(INFO) << "bTmprepH not in float range" << std::endl;

	ra.ReshapeLike(cTmp);
	Dtype* cTmpMutable = cTmp.mutable_gpu_data();
	MLGRNG<Dtype>::getInstance().mlg_gpu_gumbel(ra.count(), ra.mutable_gpu_data());
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(ra.count(), ra.gpu_data())) LOG(INFO) << "ra not finite" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(ra.count(), ra.gpu_data())) LOG(INFO) << "ra not in float range" << std::endl;
	caffe_gpu_scal<Dtype>(ra.count(), pertStr, ra.mutable_gpu_data());
	caffe_gpu_add(cTmp.count(), cTmpMutable, ra.gpu_data(), cTmpMutable);

		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(cTmp.count(), cTmp.gpu_data())) LOG(INFO) << "cTmp not finite" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(cTmp.count(), cTmp.gpu_data())) LOG(INFO) << "cTmp not in float range" << std::endl;

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

		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(X1.count(), X1.gpu_data())) LOG(INFO) << "X1 not finite" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(H1.count(), H1.gpu_data())) LOG(INFO) << "H1 not finite" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(X1.count(), X1.gpu_data())) LOG(INFO) << "X1 not in float range" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(H1.count(), H1.gpu_data())) LOG(INFO) << "H1 not in float range" << std::endl;

	this->find_map_gpu(&X1, &H1, &bTmp, &cTmp, this->blobs_[0].get());

		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(X1.count(), X1.gpu_data())) LOG(INFO) << "X1 not finite" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(H1.count(), H1.gpu_data())) LOG(INFO) << "H1 not finite" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(X1.count(), X1.gpu_data())) LOG(INFO) << "X1 not in float range" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(H1.count(), H1.gpu_data())) LOG(INFO) << "H1 not in float range" << std::endl;

	switch(this->layer_param_.rbm_param().rbm_pm_param().map_method()){
		case RBMPMLayer<Dtype>::CoordinateDescent:
		{
			X1S = X1.mutable_gpu_data();
			H1S = H1.mutable_gpu_data();
		}
		break;
		case RBMPMLayer<Dtype>::GreedyEnergyOptimization:
		{
			X1S = X1.mutable_gpu_data();
			H1S = H1.mutable_gpu_data();
		}
		break;
		case RBMPMLayer<Dtype>::FreeEnergyGradientDescent:
		{
			X1S = X1.mutable_gpu_data();
			H1S = H1.mutable_gpu_data();

			//////////////////////////// CblasNoTrans !!!!!!!!!!!!!!!!!!!!!!!!!
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
					this->M_ * repTimes, this->N_, this->K_,
					(Dtype)1., X1S, this->blobs_[0]->gpu_data(),
					(Dtype)0., H1S);

			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
					this->M_ * repTimes, this->N_, 1,
					(Dtype)1., ones.gpu_data(), this->blobs_[2]->gpu_data(),
					(Dtype)1., H1S);

			this->sigmoid_gpu(repH.count(), H1S);
		}
		break;
		case RBMPMLayer<Dtype>::NegativeGreedyEnergyOptimization:
		{
			X1S = X1.mutable_gpu_data();
			H1S = H1.mutable_gpu_data();
		}
		break;
		case RBMPMLayer<Dtype>::NegativeFreeEnergyGradientDescent:
		{
			X1S = X1.mutable_gpu_data();
			H1S = H1.mutable_gpu_data();

			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
					this->M_ * repTimes, this->N_, this->K_,
					(Dtype)1., X1S, this->blobs_[0]->gpu_data(),
					(Dtype)0., H1S);

			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
					this->M_ * repTimes, this->N_, 1,
					(Dtype)1., ones.gpu_data(), this->blobs_[2]->gpu_data(),
					(Dtype)1., H1S);

			this->sigmoid_gpu(repH.count(), H1S);
		}
		break;
	}

		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(X1.count(), X1.gpu_data())) LOG(INFO) << "X1 not finite" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(H1.count(), H1.gpu_data())) LOG(INFO) << "H1 not finite" << std::endl;

		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(X1.count(), X1.gpu_data(),0,1)) LOG(INFO) << "X1 not in range" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(H1.count(), H1.gpu_data(),0,1)) LOG(INFO) << "H1 not in range" << std::endl;

		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(X1.count(), X1.gpu_data())) LOG(INFO) << "X1 not in float range" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(H1.count(), H1.gpu_data())) LOG(INFO) << "H1 not in float range" << std::endl;

	// set gradient scale
	Dtype scalar =  -1. / ( this->M_ * repTimes );

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
    		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(tmp1.count(), tmp1.gpu_data())) LOG(INFO) << "tmp1 not finite" << std::endl;
    		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(tmp2.count(), tmp2.gpu_data())) LOG(INFO) << "tmp2 not finite" << std::endl;
    		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(tmp1.count(), tmp1.gpu_data())) LOG(INFO) << "tmp1 not in float range" << std::endl;
    		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(tmp2.count(), tmp2.gpu_data())) LOG(INFO) << "tmp2 not in float range" << std::endl;
    	caffe_gpu_sub(tmp1.count(), tmp1.gpu_data(), tmp2.gpu_data(), this->blobs_[0]->mutable_gpu_diff());
    	caffe_gpu_scal<Dtype>(this->blobs_[0]->count(), scalar, this->blobs_[0]->mutable_gpu_diff());

    		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(this->blobs_[0]->count(), this->blobs_[0]->gpu_diff())) LOG(INFO) << "dW not finite" << std::endl;
    		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(this->blobs_[0]->count(), this->blobs_[0]->gpu_diff(), -1, 1)) LOG(INFO) << "dW not in range" << std::endl;
    		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(this->blobs_[0]->count(), this->blobs_[0]->gpu_diff())) LOG(INFO) << "dW not in float range" << std::endl;
	}

	if (this->param_propagate_down_[1]) {
		Blob<Dtype> tmp(repX.shape());

		caffe_gpu_sub(tmp.count(), X0S, X1S, tmp.mutable_gpu_data());
			if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(tmp.count(), tmp.gpu_data())) LOG(INFO) << "tmp not finite" << std::endl;
			if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(tmp.count(), tmp.gpu_data())) LOG(INFO) << "tmp not in float range" << std::endl;
    	caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			this->K_, 1, this->M_ * repTimes,
    			(Dtype)1., tmp.gpu_data(), ones.gpu_data(),
    			(Dtype)0., this->blobs_[1]->mutable_gpu_diff());

    	caffe_gpu_scal<Dtype>(this->blobs_[1]->count(), scalar, this->blobs_[1]->mutable_gpu_diff());

    		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(this->blobs_[1]->count(), this->blobs_[1]->gpu_diff())) LOG(INFO) << "db not finite" << std::endl;
    		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(this->blobs_[1]->count(), this->blobs_[1]->gpu_diff(), -1, 1)) LOG(INFO) << "db not in range" << std::endl;
    		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(this->blobs_[1]->count(), this->blobs_[1]->gpu_diff())) LOG(INFO) << "db not in float range" << std::endl;
	}

	if (this->param_propagate_down_[2]) {
		Blob<Dtype> tmp(repH.shape());

		caffe_gpu_sub(tmp.count(), H0S, H1S, tmp.mutable_gpu_data());
			if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(tmp.count(), tmp.gpu_data())) LOG(INFO) << "tmp not finite" << std::endl;
			if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(tmp.count(), tmp.gpu_data())) LOG(INFO) << "tmp not in float range" << std::endl;
    	caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			this->N_, 1, this->M_ * repTimes,
    			(Dtype)1., tmp.gpu_data(), ones.gpu_data(),
    			(Dtype)0., this->blobs_[2]->mutable_gpu_diff());

    	caffe_gpu_scal<Dtype>(this->blobs_[2]->count(), scalar, this->blobs_[2]->mutable_gpu_diff());

    		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(this->blobs_[2]->count(), this->blobs_[2]->gpu_diff())) LOG(INFO) << "dc not finite" << std::endl;
    		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(this->blobs_[2]->count(), this->blobs_[2]->gpu_diff(), -1, 1)) LOG(INFO) << "dc not in range" << std::endl;
    		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(this->blobs_[2]->count(), this->blobs_[2]->gpu_diff())) LOG(INFO) << "dc not in float range" << std::endl;
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
