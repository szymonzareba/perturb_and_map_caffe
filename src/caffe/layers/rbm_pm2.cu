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
__global__ void sample_distribution(const int n, const int randomID, const Dtype* random, const Dtype* cumulative, Dtype* mask){
	CUDA_KERNEL_LOOP(index,n){
		Dtype min;
		Dtype max;

		if (index == 0){
			min = 0;
		} else {
			min = cumulative[index -1];
		}

		if(index == n-1){
			max = 1;
		} else {
			max = cumulative[index];
		}

		if (min < random[randomID] && random[randomID] <= max) {
			mask[index] = 1.0;
		} else {
			mask[index] = 0.0;
		}
	}
}

template <typename Dtype>
__global__ void substitute_column(const int n, const int columnIndex, const int columnSize, const Dtype* toCopy, Dtype* dst){
	CUDA_KERNEL_LOOP(index,n){
		dst[columnIndex*columnSize + index] = toCopy[index];
	}
}

template <typename Dtype>
__global__ void divide_elementwise_kernel(const int n, const Dtype* scalar, Dtype* dst){
	CUDA_KERNEL_LOOP(index,n){
		dst[index] = dst[index] / scalar[0];
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
	case RBMPM2Layer::ExpProb:
	{
		vector<int> shape(2);

		// tmp b - bias for observable
		shape[0] = this->K_;
		shape[1] = 1;
		Blob<Dtype> mb(shape);
		caffe_gpu_set(mb.count(), (Dtype)1.0, mb.mutable_gpu_data());

		// tmp c - bias for latent
		shape[0] = this->N_;
		shape[1] = 1;
		Blob<Dtype> mc(shape);
		caffe_gpu_set(mc.count(), (Dtype)1.0, mc.mutable_gpu_data());

		// tmp maxk W
		Blob<Dtype> maskW(this->blobs_[0]->shape());

		// tmp exp(W)
		Blob<Dtype> probsW(this->blobs_[0]->shape());
		caffe_gpu_exp(this->blobs_[0]->count(), this->blobs_[0]->gpu_data(), probsW.mutable_gpu_data());

		// ones_k_1
		shape[0] = this->K_;
		shape[1] = 1;
		Blob<Dtype> ones_k_1(shape);
		caffe_gpu_set(ones_k_1.count(), (Dtype)1.0, ones_k_1.mutable_gpu_data());

		// ones_n_1
		shape[0] = this->N_;
		shape[1] = 1;
		Blob<Dtype> ones_n_1(shape);
		caffe_gpu_set(ones_n_1.count(), (Dtype)1.0, ones_n_1.mutable_gpu_data());

		// tmp for sum of one column/row
		shape[0] = 1;
		shape[1] = 1;
		Blob<Dtype> sumProbsW(shape);

		// tmp for mask updates
		Blob<Dtype> amc(mc.shape());
		Blob<Dtype> amb(mb.shape());

		// latent < observable
		if(this->N_ < this->K_){
			// tmp for one column/row of probsW to analyse in current loop
			// tmp for cumulative sum of selected column/row
			shape[0] = this->K_;
			shape[1] = 1;
			Blob<Dtype> subProbsW(shape);
			Blob<Dtype> cumProbsW(shape);

			// tmp for random variables
			// tmp for order
			shape[0] = this->N_;
			shape[1] = 1;
			Blob<Dtype> randoms(shape);
			MLGRNG<Dtype>::getInstance().mlg_gpu_uniform(randoms.count(), randoms.mutable_gpu_data());


			int* order = new int[this->N_];
			MLGRNG<Dtype>::getInstance().mlg_cpu_permutation(this->N_, order);


			for(int i=0; i < this->N_; i++)
			{
				// get current column/row id
				int indx = order[i];

				// set current mask
				caffe_gpu_set(amc.count(), (Dtype)0.0, amc.mutable_gpu_data());
				amc.mutable_cpu_data()[indx] = 1;

				caffe_gpu_set(amb.count(), (Dtype)0.0, amb.mutable_gpu_data());

				// put zeros where connection is already made
				// repmat mb to maskW
				// [N,K] = [N,1] x [1,K]
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
						this->N_, this->K_, 1,
						(Dtype)1., ones_n_1.gpu_data(), mb.gpu_data(),
						(Dtype)0., maskW.mutable_gpu_data());

				// inplace multiply probW = probW .* maskW
				caffe_gpu_mul(probsW.count(), probsW.gpu_data(), maskW.gpu_data(), probsW.mutable_gpu_data());

				// extract current column/row from probsW
				// [K,1] = [K,N] x [N,1]
				caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
						this->K_, 1, this->N_,
						(Dtype)1., probsW.gpu_data(), amc.gpu_data(),
						(Dtype)0., subProbsW.mutable_gpu_data());

				/*
				cumulative_sum_kernel<<<CAFFE_GET_BLOCKS(subProbsW.count()), CAFFE_CUDA_NUM_THREADS>>>(
						subProbsW.count(), subProbsW.gpu_data(), cumProbsW.mutable_gpu_data());
				CUDA_POST_KERNEL_CHECK;
				*/

				efficientScan(subProbsW.count(), subProbsW.mutable_gpu_data(), cumProbsW.mutable_gpu_data());

				// [1,1] = [1,K] x [K,1]
				caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
						1, 1, this->K_,
						(Dtype)1., subProbsW.gpu_data(), ones_k_1.gpu_data(),
						(Dtype)0., sumProbsW.mutable_gpu_data());

				divide_elementwise_kernel<Dtype><<<CAFFE_GET_BLOCKS(cumProbsW.count()), CAFFE_CUDA_NUM_THREADS>>>(
						cumProbsW.count(), sumProbsW.gpu_data(), cumProbsW.mutable_gpu_data());
				CUDA_POST_KERNEL_CHECK;

				sample_distribution<Dtype><<<CAFFE_GET_BLOCKS(cumProbsW.count()), CAFFE_CUDA_NUM_THREADS>>>(
						cumProbsW.count(), i, randoms.gpu_data(), cumProbsW.gpu_data(), amb.mutable_gpu_data());
				CUDA_POST_KERNEL_CHECK;

				substitute_column<Dtype><<<CAFFE_GET_BLOCKS(amb.count()), CAFFE_CUDA_NUM_THREADS>>>(
						amb.count(), indx, this->K_, amb.gpu_data(), W->mutable_gpu_data());
				CUDA_POST_KERNEL_CHECK;

				caffe_gpu_sub(mb.count(), mb.gpu_data(), amb.gpu_data(), mb.mutable_gpu_data());
				caffe_gpu_sub(mc.count(), mc.gpu_data(), amc.gpu_data(), mc.mutable_gpu_data());
			}
			delete order;
		} else {
			NOT_IMPLEMENTED;
		}
	}
	break;
	case RBMPM2Layer::AbsExpProb:
	{
		vector<int> shape(2);

		// tmp b - bias for observable
		shape[0] = this->K_;
		shape[1] = 1;
		Blob<Dtype> mb(shape);
		caffe_gpu_set(mb.count(), (Dtype)1.0, mb.mutable_gpu_data());

		// tmp c - bias for latent
		shape[0] = this->N_;
		shape[1] = 1;
		Blob<Dtype> mc(shape);
		caffe_gpu_set(mc.count(), (Dtype)1.0, mc.mutable_gpu_data());

		// tmp maxk W
		Blob<Dtype> maskW(this->blobs_[0]->shape());

		// tmp exp(W)
		Blob<Dtype> probsW(this->blobs_[0]->shape());
		caffe_gpu_abs(this->blobs_[0]->count(), this->blobs_[0]->gpu_data(), probsW.mutable_gpu_data());
		caffe_gpu_exp(this->blobs_[0]->count(), probsW.gpu_data(), probsW.mutable_gpu_data());

		// ones_k_1
		shape[0] = this->K_;
		shape[1] = 1;
		Blob<Dtype> ones_k_1(shape);
		caffe_gpu_set(ones_k_1.count(), (Dtype)1.0, ones_k_1.mutable_gpu_data());

		// ones_n_1
		shape[0] = this->N_;
		shape[1] = 1;
		Blob<Dtype> ones_n_1(shape);
		caffe_gpu_set(ones_n_1.count(), (Dtype)1.0, ones_n_1.mutable_gpu_data());

		// tmp for sum of one column/row
		shape[0] = 1;
		shape[1] = 1;
		Blob<Dtype> sumProbsW(shape);

		// tmp for mask updates
		Blob<Dtype> amc(mc.shape());
		Blob<Dtype> amb(mb.shape());

		// latent < observable
		if(this->N_ < this->K_){
			// tmp for one column/row of probsW to analyse in current loop
			// tmp for cumulative sum of selected column/row
			shape[0] = this->K_;
			shape[1] = 1;
			Blob<Dtype> subProbsW(shape);
			Blob<Dtype> cumProbsW(shape);

			// tmp for random variables
			// tmp for order
			shape[0] = this->N_;
			shape[1] = 1;
			Blob<Dtype> randoms(shape);
			MLGRNG<Dtype>::getInstance().mlg_gpu_uniform(randoms.count(), randoms.mutable_gpu_data());


			int* order = new int[this->N_];
			MLGRNG<Dtype>::getInstance().mlg_cpu_permutation(this->N_, order);


			for(int i=0; i < this->N_; i++)
			{
				// get current column/row id
				int indx = order[i];

				// set current mask
				caffe_gpu_set(amc.count(), (Dtype)0.0, amc.mutable_gpu_data());
				amc.mutable_cpu_data()[indx] = 1;

				caffe_gpu_set(amb.count(), (Dtype)0.0, amb.mutable_gpu_data());

				// put zeros where connection is already made
				// repmat mb to maskW
				// [N,K] = [N,1] x [1,K]
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
						this->N_, this->K_, 1,
						(Dtype)1., ones_n_1.gpu_data(), mb.gpu_data(),
						(Dtype)0., maskW.mutable_gpu_data());

				// inplace multiply probW = probW .* maskW
				caffe_gpu_mul(probsW.count(), probsW.gpu_data(), maskW.gpu_data(), probsW.mutable_gpu_data());

				// extract current column/row from probsW
				// [K,1] = [K,N] x [N,1]
				caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
						this->K_, 1, this->N_,
						(Dtype)1., probsW.gpu_data(), amc.gpu_data(),
						(Dtype)0., subProbsW.mutable_gpu_data());

				cumulative_sum_kernel<<<CAFFE_GET_BLOCKS(subProbsW.count()), CAFFE_CUDA_NUM_THREADS>>>(
						subProbsW.count(), subProbsW.gpu_data(), cumProbsW.mutable_gpu_data());
				CUDA_POST_KERNEL_CHECK;

				// [1,1] = [1,K] x [K,1]
				caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
						1, 1, this->K_,
						(Dtype)1., subProbsW.gpu_data(), ones_k_1.gpu_data(),
						(Dtype)0., sumProbsW.mutable_gpu_data());

				divide_elementwise_kernel<Dtype><<<CAFFE_GET_BLOCKS(cumProbsW.count()), CAFFE_CUDA_NUM_THREADS>>>(
						cumProbsW.count(), sumProbsW.gpu_data(), cumProbsW.mutable_gpu_data());
				CUDA_POST_KERNEL_CHECK;

				sample_distribution<Dtype><<<CAFFE_GET_BLOCKS(cumProbsW.count()), CAFFE_CUDA_NUM_THREADS>>>(
						cumProbsW.count(), i, randoms.gpu_data(), cumProbsW.gpu_data(), amb.mutable_gpu_data());
				CUDA_POST_KERNEL_CHECK;

				substitute_column<Dtype><<<CAFFE_GET_BLOCKS(amb.count()), CAFFE_CUDA_NUM_THREADS>>>(
						amb.count(), indx, this->K_, amb.gpu_data(), W->mutable_gpu_data());
				CUDA_POST_KERNEL_CHECK;

				caffe_gpu_sub(mb.count(), mb.gpu_data(), amb.gpu_data(), mb.mutable_gpu_data());
				caffe_gpu_sub(mc.count(), mc.gpu_data(), amc.gpu_data(), mc.mutable_gpu_data());
			}
			delete order;
		} else {
			NOT_IMPLEMENTED;
		}
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
		default:
		{
			NOT_IMPLEMENTED;
		}
	}

	//create ones
	vector<int> ones_m_shape(2);
	ones_m_shape[0] = this->M_ * repTimes;
	ones_m_shape[1] = 1;

	Blob<Dtype> ones_m_1;
	ones_m_1.Reshape(ones_m_shape);
	caffe_gpu_set(ones_m_1.count(), (Dtype)1., ones_m_1.mutable_gpu_data());

	//create tmp for parameters
	vector<int> bias_shape(2);

	bias_shape[0] = this->M_ * repTimes;
	bias_shape[1] = this->K_;

	Blob<Dtype> bTmp;
	bTmp.Reshape( bias_shape );
	
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
			this->M_ * repTimes, this->K_, 1,
			(Dtype)1., ones_m_1.gpu_data(), this->blobs_[1]->gpu_data(),
			(Dtype)0., bTmp.mutable_gpu_data());

    bias_shape[0] = this->M_ * repTimes;
    bias_shape[1] = this->N_;


	Blob<Dtype> cTmp;
	cTmp.Reshape( bias_shape );

	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
			this->M_ * repTimes, this->N_, 1,
			(Dtype)1., ones_m_1.gpu_data(), this->blobs_[2]->gpu_data(),
			(Dtype)0., cTmp.mutable_gpu_data());


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

	vector<int> shape_vector(2);
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

	negate_0_1_kernel<Dtype><<<CAFFE_GET_BLOCKS(bMask.count()), CAFFE_CUDA_NUM_THREADS>>>(
			bMask.count(), bMask.mutable_gpu_data());
	CUDA_POST_KERNEL_CHECK;

	Blob<Dtype> bMaskRep;
	bMaskRep.ReshapeLike(bTmp);

	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
			this->M_ * repTimes, this->K_, 1,
			(Dtype)1., ones_m_1.gpu_data(), bMask.gpu_data(),
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

	negate_0_1_kernel<Dtype><<<CAFFE_GET_BLOCKS(cMask.count()), CAFFE_CUDA_NUM_THREADS>>>(
			cMask.count(), cMask.mutable_gpu_data());
	CUDA_POST_KERNEL_CHECK;

	Blob<Dtype> cMaskRep;
	cMaskRep.ReshapeLike(cTmp);

	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
			this->M_ * repTimes, this->N_, 1,
			(Dtype)1., ones_m_1.gpu_data(), cMask.gpu_data(),
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

	MLGRNG<Dtype>::getInstance().mlg_gpu_gumbel(ra.count(), ra.mutable_gpu_data());
	caffe_gpu_scal<Dtype>(ra.count(), -pertStr, ra.mutable_gpu_data());
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

	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
			this->M_ * repTimes, this->N_, this->K_,
			(Dtype)1., X1_data, this->blobs_[0]->gpu_data(),
			(Dtype)0., H1_data);

	caffe_gpu_add(cTmp.count(), H1_data, cTmp.gpu_data(), H1_data);
	this->sigmoid_gpu(H1.count(), H1_data);

	this->find_map_gpu(&X1, &H1, &bTmp, &cTmp, &wTmp);

	/// Recalculate expectation H1 | X1
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
			this->M_ * repTimes, this->N_, this->K_,
			(Dtype)1., X1_data, this->blobs_[0]->gpu_data(),
			(Dtype)0., H1_data);

	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
			this->M_ * repTimes, this->N_, 1,
			(Dtype)1., ones_m_1.gpu_data(), this->blobs_[2]->gpu_data(),
			(Dtype)1., H1_data);

	this->sigmoid_gpu(H1.count(), H1_data);

	if(this->persistent)
	{
		// save to chain
		caffe_copy(this->X1_chain.count(), X1_data, X1_chain.mutable_gpu_data());
	}

	// set gradient scale
	Dtype scalar =  -1. / ( this->M_ * repTimes );

	// calculate gradients
	if (this->param_propagate_down_[0]) {
		Blob<Dtype> tmp1(this->blobs_[0]->shape());
		Blob<Dtype> tmp2(this->blobs_[0]->shape());

    	caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			this->N_, this->K_, this->M_ * repTimes,
    			(Dtype)1., H0_data, X0_data,
    			(Dtype)0., tmp1.mutable_gpu_data());

    	caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			this->N_, this->K_, this->M_ * repTimes,
    			(Dtype)1., H1_data, X1_data,
    			(Dtype)0., tmp2.mutable_gpu_data());

    	caffe_gpu_sub(tmp1.count(), tmp1.gpu_data(), tmp2.gpu_data(), this->blobs_[0]->mutable_gpu_diff());
    	caffe_gpu_scal<Dtype>(this->blobs_[0]->count(), scalar, this->blobs_[0]->mutable_gpu_diff());
	}

	if (this->param_propagate_down_[1]) {
		Blob<Dtype> tmp(X0.shape());

		caffe_gpu_sub(tmp.count(), X0_data, X1_data, tmp.mutable_gpu_data());
    	caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			this->K_, 1, this->M_ * repTimes,
    			(Dtype)1., tmp.gpu_data(), ones_m_1.gpu_data(),
    			(Dtype)0., this->blobs_[1]->mutable_gpu_diff());

    	caffe_gpu_scal<Dtype>(this->blobs_[1]->count(), scalar, this->blobs_[1]->mutable_gpu_diff());
	}

	if (this->param_propagate_down_[2]) {
		Blob<Dtype> tmp(H0.shape());

		caffe_gpu_sub(tmp.count(), H0_data, H1_data, tmp.mutable_gpu_data());
    	caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    			this->N_, 1, this->M_ * repTimes,
    			(Dtype)1., tmp.gpu_data(), ones_m_1.gpu_data(),
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
