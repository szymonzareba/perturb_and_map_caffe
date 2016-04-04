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

	Blob<Dtype> X0;
	Blob<Dtype> H0;

	this->replicate_data_gpu(repTimes, bottom[0], &X0);

	switch(this->layer_param_.rbm_param().rbm_pm_param().map_method()){
		case RBMPMLayer<Dtype>::CoordinateDescent:
		{
			this->replicate_data_gpu(repTimes, &this->H0, &H0);
			this->sample_gpu(H0.count(), H0.mutable_gpu_data());
		}
		break;
		case RBMPMLayer<Dtype>::FreeEnergyGradientDescent:
		{
			this->replicate_data_gpu(repTimes, &this->H0, &H0);
		}
		break;
		case RBMPMLayer<Dtype>::FreeEnergyGradientDescentEta2:
		{
			this->replicate_data_gpu(repTimes, &this->H0, &H0);
		}
		break;
		case RBMPMLayer<Dtype>::GreedyEnergyOptimization:
		{
			this->replicate_data_gpu(repTimes, &this->H0, &H0);
			this->sample_gpu(H0.count(), H0.mutable_gpu_data());
		}
		break;
		case RBMPMLayer<Dtype>::RandomizedGreedyEnergyOptimization:
		{
			this->replicate_data_gpu(repTimes, &this->H0, &H0);
			this->sample_gpu(H0.count(), H0.mutable_gpu_data());
		}
		break;
		default:
		{
			NOT_IMPLEMENTED;
		}
	}



		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(X0.count(), X0.gpu_data())) LOG(INFO) << "repX not finite" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(this->H0.count(), this->H0.gpu_data())) LOG(INFO) << "H0 not finite" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(top[0]->count(), top[0]->gpu_data())) LOG(INFO) << "H0S not finite" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(H0.count(), H0.gpu_data())) LOG(INFO) << "repH not finite" << std::endl;

		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(X0.count(), X0.gpu_data(),0,1)) LOG(INFO) << "repX not in range" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(this->H0.count(), this->H0.gpu_data(),0,1)) LOG(INFO) << "H0 not in range" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(top[0]->count(), top[0]->gpu_data(),0,1)) LOG(INFO) << "H0S not in range" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(H0.count(), H0.gpu_data(),0,1)) LOG(INFO) << "repH not in range" << std::endl;

		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(X0.count(), X0.gpu_data())) LOG(INFO) << "repX not in float range" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(this->H0.count(), this->H0.gpu_data())) LOG(INFO) << "H0 not in float range" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(top[0]->count(), top[0]->gpu_data())) LOG(INFO) << "H0S not in float range" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(H0.count(), H0.gpu_data())) LOG(INFO) << "repH not in float range" << std::endl;

	//create ones
	vector<int> ones_shape(2);
	ones_shape[0] = this->M_ * repTimes;
	ones_shape[1] = 1;

	Blob<Dtype> ones;
	ones.Reshape(ones_shape);
	caffe_gpu_set(ones.count(), (Dtype)1., ones.mutable_gpu_data());

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

	if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(bTmp.count(), bTmp.gpu_data())) LOG(INFO) << "bTmp not finite" << std::endl;
	if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(bTmp.count(), bTmp.gpu_data())) LOG(INFO) << "bTmp not in float range" << std::endl;

    bias_shape[0] = this->M_ * repTimes;
    bias_shape[1] = this->N_;

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
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(bTmp.count(), bTmp.gpu_data())) LOG(INFO) << "bTmp not in float range" << std::endl;

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
	Blob<Dtype> H1;

	X1.ReshapeLike(X0);
	H1.ReshapeLike(H0);

	const Dtype* X0_data = X0.gpu_data();
	const Dtype* H0_data = H0.gpu_data();

	Dtype* X1_data = X1.mutable_gpu_data();
	Dtype* H1_data = H1.mutable_gpu_data();

	if(this->persistent){
		// init from chain
		caffe_copy(this->X1_chain.count(), X1_chain.gpu_data(), X1_data);
	}
	else{
		// init from data
		caffe_copy(X0.count(), X0_data, X1_data);
	}

	// init h1 perturbed
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
			this->M_ * repTimes, this->N_, this->K_,
			(Dtype)1., X1_data, this->blobs_[0]->gpu_data(),
			(Dtype)0., H1_data);

	caffe_gpu_add(cTmp.count(), H1_data, cTmp.gpu_data(), H1_data);
	this->sigmoid_gpu(H1.count(), H1_data);

		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(X1.count(), X1.gpu_data())) LOG(INFO) << "X1 not finite" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(H1.count(), H1.gpu_data())) LOG(INFO) << "H1 not finite" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(X1.count(), X1.gpu_data())) LOG(INFO) << "X1 not in float range" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(H1.count(), H1.gpu_data())) LOG(INFO) << "H1 not in float range" << std::endl;

	this->find_map_gpu(&X1, &H1, &bTmp, &cTmp, this->blobs_[0].get());

	/// Recalculate expectation H1 | X1
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
			this->M_ * repTimes, this->N_, this->K_,
			(Dtype)1., X1_data, this->blobs_[0]->gpu_data(),
			(Dtype)0., H1_data);

	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
			this->M_ * repTimes, this->N_, 1,
			(Dtype)1., ones.gpu_data(), this->blobs_[2]->gpu_data(),
			(Dtype)1., H1_data);

	this->sigmoid_gpu(H1.count(), H1_data);

	///
	//this->sample_gpu(H1.count(), H1_data);
	///

		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(X1.count(), X1.gpu_data())) LOG(INFO) << "X1 not finite" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(H1.count(), H1.gpu_data())) LOG(INFO) << "H1 not finite" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(X1.count(), X1.gpu_data())) LOG(INFO) << "X1 not in float range" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(H1.count(), H1.gpu_data())) LOG(INFO) << "H1 not in float range" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(X1.count(), X1.gpu_data(),0,1)) LOG(INFO) << "X1 not in range" << std::endl;
		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(H1.count(), H1.gpu_data(),0,1)) LOG(INFO) << "H1 not in range" << std::endl;


	if(this->persistent)
	{
		// save to chain
		caffe_copy(this->X1_chain.count(), X1_data, X1_chain.mutable_gpu_data());
	}

	// set gradient scale
	Dtype scalar =  -1. / ( this->M_ * repTimes );

	if (this->param_propagate_down_[0]) {
		Blob<Dtype> tmp_w_grad_1(this->blobs_[0]->shape());
		Blob<Dtype> tmp_w_grad_2(this->blobs_[0]->shape());

    	caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			this->N_, this->K_, this->M_ * repTimes,
    			(Dtype)1., H0_data, X0_data,
    			(Dtype)0., tmp_w_grad_1.mutable_gpu_data());

    	caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			this->N_, this->K_, this->M_ * repTimes,
    			(Dtype)1., H1_data, X1_data,
    			(Dtype)0., tmp_w_grad_2.mutable_gpu_data());

    		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(tmp_w_grad_1.count(), tmp_w_grad_1.gpu_data())) LOG(INFO) << "tmp_w_grad_1 not finite" << std::endl;
    		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(tmp_w_grad_2.count(), tmp_w_grad_2.gpu_data())) LOG(INFO) << "tmp_w_grad_2 not finite" << std::endl;
    		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(tmp_w_grad_1.count(), tmp_w_grad_1.gpu_data())) LOG(INFO) << "tmp_w_grad_1 not in float range" << std::endl;
    		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(tmp_w_grad_2.count(), tmp_w_grad_2.gpu_data())) LOG(INFO) << "tmp_w_grad_2 not in float range" << std::endl;

    	caffe_gpu_sub(tmp_w_grad_1.count(), tmp_w_grad_1.gpu_data(), tmp_w_grad_2.gpu_data(), this->blobs_[0]->mutable_gpu_diff());
    	caffe_gpu_scal<Dtype>(this->blobs_[0]->count(), scalar, this->blobs_[0]->mutable_gpu_diff());

    		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(this->blobs_[0]->count(), this->blobs_[0]->gpu_diff())) LOG(INFO) << "dW not finite" << std::endl;
    		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(this->blobs_[0]->count(), this->blobs_[0]->gpu_diff(), -1, 1)) LOG(INFO) << "dW not in range" << std::endl;
    		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(this->blobs_[0]->count(), this->blobs_[0]->gpu_diff())) LOG(INFO) << "dW not in float range" << std::endl;
	}

	if (this->param_propagate_down_[1]) {
		Blob<Dtype> tmp_b_grad(X0.shape());

		caffe_gpu_sub(tmp_b_grad.count(), X0_data, X1_data, tmp_b_grad.mutable_gpu_data());
			if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(tmp_b_grad.count(), tmp_b_grad.gpu_data())) LOG(INFO) << "tmp_b_grad not finite" << std::endl;
			if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(tmp_b_grad.count(), tmp_b_grad.gpu_data())) LOG(INFO) << "tmp_b_grad not in float range" << std::endl;
    	caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			this->K_, 1, this->M_ * repTimes,
    			(Dtype)1., tmp_b_grad.gpu_data(), ones.gpu_data(),
    			(Dtype)0., this->blobs_[1]->mutable_gpu_diff());

    	caffe_gpu_scal<Dtype>(this->blobs_[1]->count(), scalar, this->blobs_[1]->mutable_gpu_diff());

    		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(this->blobs_[1]->count(), this->blobs_[1]->gpu_diff())) LOG(INFO) << "db not finite" << std::endl;
    		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(this->blobs_[1]->count(), this->blobs_[1]->gpu_diff(), -1, 1)) LOG(INFO) << "db not in range" << std::endl;
    		if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(this->blobs_[1]->count(), this->blobs_[1]->gpu_diff())) LOG(INFO) << "db not in float range" << std::endl;
	}

	if (this->param_propagate_down_[2]) {
		Blob<Dtype> tmp_c_grad(H0.shape());

		caffe_gpu_sub(tmp_c_grad.count(), H0_data, H1_data, tmp_c_grad.mutable_gpu_data());
			if(MLGASSERT<Dtype>::getInstance().mlg_gpu_finite(tmp_c_grad.count(), tmp_c_grad.gpu_data())) LOG(INFO) << "tmp_c_grad not finite" << std::endl;
			if(MLGASSERT<Dtype>::getInstance().mlg_gpu_range(tmp_c_grad.count(), tmp_c_grad.gpu_data())) LOG(INFO) << "tmp_c_grad not in float range" << std::endl;
    	caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			this->N_, 1, this->M_ * repTimes,
    			(Dtype)1., tmp_c_grad.gpu_data(), ones.gpu_data(),
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
