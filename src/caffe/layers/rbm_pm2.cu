#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/rbm_layers.hpp"
#include "caffe/util/mlg_math.hpp"

namespace caffe {

template <typename Dtype>
__global__ void fill_W_N_g_K_kernel(const int wSize, const int N, const int K, const int* range, Dtype* w){
	CUDA_KERNEL_LOOP(index,wSize){
		int a = (index % K);
		int b = (index - a) / K;
		if(range[a] == b){
			w[index] = 1.;
		}
	}
}

template <typename Dtype>
__global__ void fill_W_K_g_N_kernel(const int wSize, const int N, const int K, const int* range, Dtype* w){
	CUDA_KERNEL_LOOP(index,wSize){
		int a = (index % K);
		int b = (index - a) / K;
		if(range[b] == a){
			w[index] = 1.;
		}
	}
}

template <typename Dtype>
void RBMPM2Layer<Dtype>::find_w_mask_gpu(Blob<Dtype>* W){

	switch(this->layer_param_.rbm_param().rbm_pm_param().rbm_pm2_param().w_pert_param()){
	case RBMPM2Layer::Random:
	{
		int* range;
		if(this->N_ > this->K_){
			cudaMalloc((void**) &range, this->N_ * sizeof(int));
			MLGRNG<Dtype>::getInstance().mlg_gpu_permutation(this->N_, range);
			//MLGRNG<Dtype>::getInstance().mlg_gpu_range(this->N_, 0, this->N_, range);
			fill_W_N_g_K_kernel<Dtype><<<CAFFE_GET_BLOCKS(W->count()), CAFFE_CUDA_NUM_THREADS>>>(
					W->count(), this->N_, this->K_,
					range, W->mutable_gpu_data());
			CUDA_POST_KERNEL_CHECK;

		}else{
			cudaMalloc((void**) &range, this->K_ * sizeof(int));
			MLGRNG<Dtype>::getInstance().mlg_gpu_permutation(this->K_, range);
			//MLGRNG<Dtype>::getInstance().mlg_gpu_range(this->K_, 0, this->K_, range);
			fill_W_K_g_N_kernel<Dtype><<<CAFFE_GET_BLOCKS(W->count()), CAFFE_CUDA_NUM_THREADS>>>(
					W->count(), this->N_, this->K_,
					range, W->mutable_gpu_data());
			CUDA_POST_KERNEL_CHECK;
		}
		cudaFree(range);
	}
	break;
	default:
		LOG(INFO) << "Perturbation not defined" << std::endl;
	}

}

template <typename Dtype>
void RBMPM2Layer<Dtype>::gradient_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom){

	const int repTimes = this->layer_param_.rbm_param().rbm_pm_param().batch_repeats();
	const Dtype pertStr = this->layer_param_.rbm_param().rbm_pm_param().pert_str();


	// replicate data
	Blob<Dtype> repX;
	replicate_data_gpu(repTimes, bottom[0], &repX);

	// replicate hidden
	Blob<Dtype> repH;
	replicate_data_gpu(repTimes, top[0], &repH);



	vector<int> shape_vector(2);



	//create ones
	shape_vector[0] = this->M_ * repTimes;
	shape_vector[1] = 1;

	Blob<Dtype> ones_m_rep;
	ones_m_rep.Reshape(shape_vector);
	caffe_gpu_set(ones_m_rep.count(), Dtype(1), ones_m_rep.mutable_gpu_data());



	//create tmp for parameters
	// tmp for b bias
	shape_vector[0] = this->M_ * repTimes;
	shape_vector[1] = this->K_;

	Blob<Dtype> bTmp;
	bTmp.Reshape( shape_vector );

	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
			this->M_ * repTimes, this->K_, 1,
			(Dtype)1., ones_m_rep.gpu_data(), this->blobs_[1]->gpu_data(),
			(Dtype)0., bTmp.mutable_gpu_data());



	// tmp for c bias
	shape_vector[0] = this->M_ * repTimes;
	shape_vector[1] = this->N_;

	Blob<Dtype> cTmp;
	cTmp.Reshape( shape_vector );

	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
			this->M_ * repTimes, this->N_, 1,
			(Dtype)1., ones_m_rep.gpu_data(), this->blobs_[2]->gpu_data(),
			(Dtype)0., cTmp.mutable_gpu_data());



	// tmp for W params
	Blob<Dtype> wTmp;
	wTmp.Reshape(this->blobs_[0]->shape());
	caffe_copy(wTmp.count(), this->blobs_[0]->gpu_data(), wTmp.mutable_gpu_data());



	// generate perturbation masks
	// mask for W
	Blob<Dtype> wMask;
	wMask.Reshape(this->blobs_[0]->shape());
	caffe_gpu_set(wMask.count(), (Dtype) 0., wMask.mutable_gpu_data());
	find_w_mask_gpu(&wMask);



	// mask for b
	Blob<Dtype> bMask;
	bMask.Reshape(this->blobs_[1]->shape());
	caffe_gpu_set(bMask.count(), (Dtype) 0., bMask.mutable_gpu_data());

	shape_vector[0] = this->N_;
	shape_vector[1] = 1;

	Blob<Dtype> ones_b;
	ones_b.Reshape(shape_vector);
	caffe_gpu_set(ones_b.count(), (Dtype) 1., ones_b.mutable_gpu_data());

	// bMask = wMask' * ones_b
	// [K,1] = [K,N] * [N,1]
	caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
			this->K_, 1, this->N_,
			(Dtype)1., wMask.gpu_data(), ones_b.gpu_data(),
			(Dtype)0., bMask.mutable_gpu_data());

	negate_kernel<Dtype><<<CAFFE_GET_BLOCKS(bMask.count()), CAFFE_CUDA_NUM_THREADS>>>(
			bMask.count(), bMask.mutable_gpu_data());
	CUDA_POST_KERNEL_CHECK;

	Blob<Dtype> bMaskRep;
	bMaskRep.ReshapeLike(bTmp);

	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
			this->M_ * repTimes, this->K_, 1,
			(Dtype)1., ones_m_rep.gpu_data(), bMask.gpu_data(),
			(Dtype)0., bMaskRep.mutable_gpu_data());



	// mask for c
	Blob<Dtype> cMask;
	cMask.Reshape(this->blobs_[2]->shape());
	caffe_gpu_set(cMask.count(), (Dtype) 0., cMask.mutable_gpu_data());

	shape_vector[0] = this->K_;
	shape_vector[1] = 1;

	Blob<Dtype> ones_c;
	ones_c.Reshape(shape_vector);
	caffe_gpu_set(ones_c.count(), (Dtype) 1., ones_c.mutable_gpu_data());

	// cMask = wMask * ones_c
	// [N,1] = [N,K] * [K,1]
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
			this->N_, 1, this->K_,
			(Dtype)1., wMask.gpu_data(), ones_c.gpu_data(),
			(Dtype)0., cMask.mutable_gpu_data());

	negate_kernel<Dtype><<<CAFFE_GET_BLOCKS(cMask.count()), CAFFE_CUDA_NUM_THREADS>>>(
			cMask.count(), cMask.mutable_gpu_data());
	CUDA_POST_KERNEL_CHECK;

	Blob<Dtype> cMaskRep;
	cMaskRep.ReshapeLike(cTmp);

	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
			this->M_ * repTimes, this->N_, 1,
			(Dtype)1., ones_m_rep.gpu_data(), cMask.gpu_data(),
			(Dtype)0., cMaskRep.mutable_gpu_data());



	// perturb parameters
	Blob<Dtype> ra;

	// perturb W
	ra.ReshapeLike(wTmp);
	MLGRNG<Dtype>::getInstance().mlg_gpu_gumbel(ra.count(), ra.mutable_gpu_data());
	caffe_gpu_scal<Dtype>(ra.count(), pertStr, ra.mutable_gpu_data());
	add_with_mask_kernel<Dtype><<<CAFFE_GET_BLOCKS(wTmp.count()), CAFFE_CUDA_NUM_THREADS>>>(
			wTmp.count(),
			wTmp.mutable_gpu_data(),
			wMask.gpu_data(), ra.gpu_data(),
			wTmp.mutable_gpu_data());
	CUDA_POST_KERNEL_CHECK;



	// perturb b
	ra.ReshapeLike(bTmp);
	MLGRNG<Dtype>::getInstance().mlg_gpu_gumbel(ra.count(), ra.mutable_gpu_data());
	caffe_gpu_scal<Dtype>(ra.count(), pertStr, ra.mutable_gpu_data());
	add_with_mask_kernel<Dtype><<<CAFFE_GET_BLOCKS(bTmp.count()), CAFFE_CUDA_NUM_THREADS>>>(
			bTmp.count(),
			bTmp.mutable_gpu_data(),
			bMaskRep.gpu_data(), ra.gpu_data(),
			bTmp.mutable_gpu_data());
	CUDA_POST_KERNEL_CHECK;



	// perturb c
	ra.ReshapeLike(cTmp);
	MLGRNG<Dtype>::getInstance().mlg_gpu_gumbel(ra.count(), ra.mutable_gpu_data());
	caffe_gpu_scal<Dtype>(ra.count(), pertStr, ra.mutable_gpu_data());
	add_with_mask_kernel<Dtype><<<CAFFE_GET_BLOCKS(cTmp.count()), CAFFE_CUDA_NUM_THREADS>>>(
			cTmp.count(),
			cTmp.mutable_gpu_data(),
			cMaskRep.gpu_data(), ra.gpu_data(),
			cTmp.mutable_gpu_data());
	CUDA_POST_KERNEL_CHECK;



	// find MAP
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

	find_map_gpu(&X1, &H1, &bTmp, &cTmp, &wTmp);

	X1S = X1.mutable_gpu_data();
	H1S = H1.mutable_gpu_data();



	// set gradient scale
	Dtype scalar =  -1. / ( this->M_ * repTimes );



	// calculate gradients
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
    			(Dtype)1., tmp.gpu_data(), ones_m_rep.gpu_data(),
    			(Dtype)0., this->blobs_[1]->mutable_gpu_diff());

    	caffe_gpu_scal<Dtype>(this->blobs_[1]->count(), scalar, this->blobs_[1]->mutable_gpu_diff());
	}

	if (this->param_propagate_down_[2]) {
		Blob<Dtype> tmp(repH.shape());

		caffe_gpu_sub(tmp.count(), H0S, H1S, tmp.mutable_gpu_data());
    	caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			this->N_, 1, this->M_ * repTimes,
    			(Dtype)1., tmp.gpu_data(), ones_m_rep.gpu_data(),
    			(Dtype)0., this->blobs_[2]->mutable_gpu_diff());

    	caffe_gpu_scal<Dtype>(this->blobs_[2]->count(), scalar, this->blobs_[2]->mutable_gpu_diff());
	}
}

template
void RBMPM2Layer<float>::gradient_gpu(const vector<Blob<float>*>& top, \
      const vector<bool>& propagate_down, \
      const vector<Blob<float>*>& bottom);

template
void RBMPM2Layer<double>::gradient_gpu(const vector<Blob<double>*>& top, \
      const vector<bool>& propagate_down, \
      const vector<Blob<double>*>& bottom);

template
void RBMPM2Layer<float>::find_w_mask_gpu(Blob<float>* W);

template
void RBMPM2Layer<double>::find_w_mask_gpu(Blob<double>* W);

} // namespace caffe
