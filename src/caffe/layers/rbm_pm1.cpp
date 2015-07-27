#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/rbm_layers.hpp"
#include "caffe/util/mlg_rng.hpp"

namespace caffe {

template <typename Dtype>
void RBMPM1Layer<Dtype>::gradient_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom){

	//create tmp for parameters
    vector<int> bias_shape(2);
/*
    bias_shape[0] = 5;
    bias_shape[1] = 5;
    Blob<Dtype> testBlob;
    testBlob.Reshape( bias_shape );

    LOG(INFO) << "BEFORE : " << std::endl;
	for(int i = 0; i < testBlob.count(); i++){
		LOG(INFO) << testBlob.cpu_data()[i] << std::endl;
	}

	MLGRNG<Dtype>::getInstance().mlg_cpu_gumbel(testBlob.count(), testBlob.mutable_cpu_data());

    LOG(INFO) << "FIRST : " << std::endl;
	for(int i = 0; i < testBlob.count(); i++){
		LOG(INFO) << testBlob.cpu_data()[i] << std::endl;
	}

	MLGRNG<Dtype>::getInstance().mlg_cpu_gumbel(testBlob.count(), testBlob.mutable_cpu_data());

    LOG(INFO) << "SECOND : " << std::endl;
	for(int i = 0; i < testBlob.count(); i++){
		LOG(INFO) << testBlob.cpu_data()[i] << std::endl;
	}
*/




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
/*
	Dtype* H0S = top[0]->mutable_cpu_data();

	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
			  this->M_, this->N_, this->K_,
			  (Dtype)1., X0S, this->blobs_[0]->cpu_data(),
			  (Dtype)0., H0S);

	for(int i = 0; i < top[0]->count(); i++){
		H0S[i] = H0S[i] + cTmp.cpu_data()[i];
	}
	for(int i = 0; i < top[0]->count(); i++){
		H0S[i] = sigmoid_cpu(H0S[i]);
	}

	sample_cpu(top[0]->count(), H0S);
*/

	Dtype* X1S = this->X1S_.mutable_cpu_data();
	Dtype* H1S = this->H1S_.mutable_cpu_data();

	caffe_copy(bottom[0]->count(), X0S, X1S);
	caffe_copy(top[0]->count(), H0S, H1S);

	find_map_cpu(&(this->X1S_), &(this->H1S_), &bTmp, &cTmp, this->blobs_[0].get());

	//caffe_copy(bottom[0]->count(), X0S, X1S);
	//caffe_copy(top[0]->count(), H0S, H1S);

	X1S = this->X1S_.mutable_cpu_data();
	H1S = this->H1S_.mutable_cpu_data();
/*
	LOG(INFO) << "XOS : "
			  << X0S[0] << " "
			  << X0S[1] << " "
			  << X0S[2] << " "
			  << X0S[3] << " "
			  << X0S[4] << std::endl;

	LOG(INFO) << "X1S : "
			  << X1S[0] << " "
			  << X1S[1] << " "
			  << X1S[2] << " "
			  << X1S[3] << " "
			  << X1S[4] << std::endl;

	LOG(INFO) << "H0S : "
			  << H0S[0] << " "
			  << H0S[1] << " "
			  << H0S[2] << " "
			  << H0S[3] << " "
			  << H0S[4] << std::endl;

	LOG(INFO) << "H1S : "
			  << H1S[0] << " "
			  << H1S[1] << " "
			  << H1S[2] << " "
			  << H1S[3] << " "
			  << H1S[4] << std::endl;
*/

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

}

#ifdef CPU_ONLY
STUB_GPU(RBMPM1Layer);
#endif

INSTANTIATE_CLASS(RBMPM1Layer);
REGISTER_LAYER_CLASS(RBMPM1);

} // namespace caffe
