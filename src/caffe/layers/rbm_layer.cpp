#include <vector>
#include <limits>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/rbm_layers.hpp"

namespace caffe {

template <typename Dtype>
void RBMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_hidden = this->layer_param_.rbm_param().num_hidden();
  N_ = num_hidden;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.rbm_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  M_ = bottom[0]->count(0, axis);

  LOG(INFO) << "k : " << K_ << std::endl;
  LOG(INFO) << "m : " << M_ << std::endl;
  LOG(INFO) << "n : " << N_ << std::endl;
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
	this->blobs_.resize(3);
    // Intialize the weight
    vector<int> weight_shape(2);
    weight_shape[0] = N_;
    weight_shape[1] = K_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.rbm_param().w_filler()));
    weight_filler->Fill(this->blobs_[0].get());

    // Bias weights for inputs
    vector<int> bias_b_shape(2);
    bias_b_shape[0] = K_;
    bias_b_shape[1] = 1;
    this->blobs_[1].reset(new Blob<Dtype>(bias_b_shape));
    shared_ptr<Filler<Dtype> > bias_b_filler(GetFiller<Dtype>(
    		this->layer_param_.rbm_param().b_filler()));
    bias_b_filler->Fill(this->blobs_[1].get());


    // Bias weights for outputs
    vector<int> bias_c_shape(2);
    bias_c_shape[0] = N_;
    bias_c_shape[1] = 1;
    this->blobs_[2].reset(new Blob<Dtype>(bias_c_shape));
    shared_ptr<Filler<Dtype> > bias_c_filler(GetFiller<Dtype>(
    		this->layer_param_.rbm_param().c_filler()));
    bias_c_filler->Fill(this->blobs_[2].get());


    // tmp X1S for gibbs sampler
    vector<int> XShape(2);
    XShape[0] = M_;
    XShape[1] = K_;

    X1S.Reshape(XShape);
    caffe_rng_uniform(X1S.count(), Dtype(0.), Dtype(1.), X1S.mutable_cpu_data());

    // tmp H1S for gibbs sampler
    vector<int> HShape(2);
    HShape[0] = M_;
    HShape[1] = N_;

    H0.Reshape(HShape);
    caffe_rng_uniform(H0.count(), Dtype(0.), Dtype(1.), H0.mutable_cpu_data());

    H1S.Reshape(HShape);
    caffe_rng_uniform(H1S.count(), Dtype(0.), Dtype(1.), H1S.mutable_cpu_data());

  }  // parameter initialization

  // Set up the bias b multiplier
  vector<int> ones_m_shape(2);
  ones_m_shape[0] = M_;
  ones_m_shape[1] = 1;
  ones_m_.Reshape(ones_m_shape);
  caffe_set(M_, Dtype(1), ones_m_.mutable_cpu_data());

  // Loss layers output a scalar; 0 axes.
  vector<int> loss_shape(0);
  top[2]->Reshape(loss_shape);

  // Calculate gradient, yes!
  this->param_propagate_down_.resize(this->blobs_.size(), true);

}

template <typename Dtype>
void RBMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.rbm_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  //int oldM = M_;
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);

  top[1]->Reshape(top_shape);

  //if(oldM != M_)
  //{
	  // Set up the bias b multiplier
	  vector<int> ones_m_shape(2);
	  ones_m_shape[0] = M_;
	  ones_m_shape[1] = 1;
	  ones_m_.Reshape(ones_m_shape);
	  caffe_set(M_, Dtype(1), ones_m_.mutable_cpu_data());
  //}
}

template <typename Dtype>
void RBMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* X0SData = bottom[0]->cpu_data();
  Dtype* H0Data = this->H0.mutable_cpu_data();
  Dtype* H0SData = top[0]->mutable_cpu_data();

  const Dtype* W = this->blobs_[0]->cpu_data();
  const Dtype* b = this->blobs_[1]->cpu_data();
  const Dtype* c = this->blobs_[2]->cpu_data();

  // H  = 1 * X * W(T) + 0 * H
  // top_data = 1 * bottom_data * weight(T) + 0 * top_data
  // [m,n] = 1 * [m,k] * [k,n] + 0 * [m,n]
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
		  M_, N_, K_,
		  (Dtype)1., X0SData, W,
		  (Dtype)0., H0Data);

  // H = 1 * cM * C + 1 * H
  // top_data = 1 * bias_c_multiplier * c + 1 * top_data
  // [m,n] = 1 * [m,1] * [1,n] + 1 * [m,n]
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
		  M_, N_, 1,
		  (Dtype)1., ones_m_.cpu_data(), c,
		  (Dtype)1., H0Data);

  for(int i = 0; i < top[0]->count(); i++){
	  H0Data[i] = sigmoid_cpu(H0Data[i]);
  }

  sample_cpu(top[0]->count(), H0Data, H0SData);

  top[2]->mutable_cpu_data()[0] = ll_cpu(top, bottom);
}

template <typename Dtype>
void RBMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  // calculate gradient
  gradient_cpu(top, propagate_down, bottom);

  if(bottom.size() == 2) // layer has second input for reconstruction
  {
	  // calculate layer backward output
	  const Dtype* H1SData = top[1]->cpu_data();
	  Dtype* X1SData = bottom[1]->mutable_cpu_data();

	  const Dtype* W = this->blobs_[0]->cpu_data();
	  const Dtype* b = this->blobs_[1]->cpu_data();
	  const Dtype* c = this->blobs_[2]->cpu_data();

	  // X = 1 * H * W + 0 * X
	  // bottom_data = 1 * top_data * weights + 0 * bottom_data
	  // [m,k] = 1 * [m,n] * [n,k] + 0 * [m,k]
	  // OK
	  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
			  M_, K_, N_,
	  		  (Dtype)1., H1SData, W,
	  		  (Dtype)0., X1SData);


	  // X = 1 * bM * b + 1 * X
	  // bottom_data = 1 * bias_b_multiplier * b + 1 * bottom_data
	  // [m,k] = 1 * [m,1] * [1,k] + 1 * [m,k]
	  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
	  		  M_, K_, 1,
	  		  (Dtype)1., ones_m_.cpu_data(), b,
	  		  (Dtype)1., X1SData);

	  for(int i = 0; i < bottom[0]->count(); i++){
		  X1SData[i] = sigmoid_cpu(X1SData[i]);
	  }

	  // sample binary x
	  sample_cpu(bottom[0]->count(), X1SData);
  }

}

template <typename Dtype>
void RBMLayer<Dtype>::gradient_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

	  if (this->param_propagate_down_[0]) {
		// calculate gradient with respect to W
		  LOG(FATAL) << this->type()
				  << " Layer cannot calculate W gradient.";
	  }

	  if (this->param_propagate_down_[1]) {
		// calculate gradient with respect to b
		  LOG(FATAL) << this->type()
				  << " Layer cannot calculate b gradient.";
	  }

	  if (this->param_propagate_down_[2]) {
		// calculate gradient with respect to c
		  LOG(FATAL) << this->type()
				  << " Layer cannot calculate c gradient.";
	  }
}

template <typename Dtype>
Dtype RBMLayer<Dtype>::ll_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom)
{
	NOT_IMPLEMENTED;

	Dtype loss = 0;

	switch(this->layer_param_.rbm_param().llaprox())
	{
		case RBMLayer::AIS:
		{}
		break;
		case RBMLayer::RAIS:
		{}
		break;
		case RBMLayer::CSL:
		{}
		break;
		case RBMLayer::REC:
		{
			Blob<Dtype> xTmp;
			xTmp.ReshapeLike(this->X1S);

			const Dtype* xData = bottom[0]->cpu_data();
			const Dtype* hData = top[0]->cpu_data();
			Dtype* xTmpData = xTmp.mutable_cpu_data();

			const Dtype* W = this->blobs_[0]->cpu_data();
			const Dtype* b = this->blobs_[1]->cpu_data();

				caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
						M_, K_, N_,
						(Dtype)1., hData, W,
						(Dtype)0., xTmpData);

				caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
						M_, K_, 1,
						(Dtype)1., ones_m_.cpu_data(), b,
						(Dtype)1., xTmpData);

				for(int i = 0; i < xTmp.count(); i++){
					xTmpData[i] = sigmoid_cpu(xTmpData[i]);
				}

				sample_cpu(xTmp.count(), xTmpData);
				caffe_sub<Dtype>(xTmp.count(), xData, xTmpData, xTmp.mutable_cpu_data());

			Dtype r = caffe_cpu_asum<Dtype>(xTmp.count(), xTmp.cpu_data());
			r = r / (Dtype) xTmp.count();
			loss = r;

		}
		break;
		default:
		{
			LOG(INFO) << "No such ll approx";
		}
		break;
	}



	return loss;
}

template <typename Dtype>
void RBMLayer<Dtype>::sample_cpu(int N, Dtype* mat)
{
    vector<int> shape(1, N);
    Blob<Dtype> randoms(shape);

	MLGRNG<Dtype>::getInstance().mlg_cpu_uniform(randoms.count(), randoms.mutable_cpu_data());

    const Dtype* rand_data = randoms.cpu_data();

    for (int i = 0; i < N; ++i) {
    	if(mat[i] < rand_data[i]) {
    		mat[i] = 1;
    	}
    	else {
    		mat[i] = 0;
    	}
    }
}

template <typename Dtype>
void RBMLayer<Dtype>::sample_cpu(int N, const Dtype* src, Dtype* dst)
{
    vector<int> shape(1, N);
    Blob<Dtype> randoms(shape);

	MLGRNG<Dtype>::getInstance().mlg_cpu_uniform(randoms.count(), randoms.mutable_cpu_data());

    const Dtype* rand_data = randoms.cpu_data();

    for (int i = 0; i < N; ++i) {
    	if(src[i] < rand_data[i]) {
    		dst[i] = 1;
    	}
    	else {
    		dst[i] = 0;
    	}
    }
}

template <typename Dtype>
void RBMLayer<Dtype>::replicate_data_cpu(const int N, const int R, const Dtype* X, Dtype* repX){
	NOT_IMPLEMENTED;
}

template <typename Dtype>
void RBMLayer<Dtype>::replicate_data_cpu(const int N, Blob<Dtype>* X, Blob<Dtype>* repX){
	NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(RBMLayer);
#endif

INSTANTIATE_CLASS(RBMLayer);
REGISTER_LAYER_CLASS(RBM);

} // namespace caffe
