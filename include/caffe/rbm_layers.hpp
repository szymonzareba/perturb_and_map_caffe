#ifndef MLG_RBM_LAYERS_HPP_
#define MLG_RBM_LAYERS_HPP_

#ifndef CPU_ONLY
#include <curand.h>
#include <curand_kernel.h>
#endif

#include "caffe/util/mlg_rng.hpp"

namespace caffe {


/**
 * @brief RBM layer for Forward and Backward calculation
 */
template <typename Dtype>
class RBMLayer : public Layer<Dtype> {
 public:
  explicit RBMLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "RBM Simple"; }
  //virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 3; }

  enum { AIS, RAIS, CSL, REC };

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void gradient_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void gradient_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual Dtype ll_cpu(const vector<Blob<Dtype>*>& top,
		  const vector<Blob<Dtype>*>& bottom);
  virtual Dtype ll_gpu(const vector<Blob<Dtype>*>& top,
		  const vector<Blob<Dtype>*>& bottom);
  virtual void sample_cpu(int N, Dtype* mat);
  virtual void sample_gpu(int N, Dtype* mat);
  virtual inline Dtype sigmoid_cpu(Dtype x){ return 1. / (1. + exp(-x)); }
  virtual void sigmoid_gpu(int count, Dtype* data);

  // minibatch size
  int M_;
  // input dimention
  int K_;
  // output dimention
  int N_;

  Blob<Dtype> ones_m_;

  // tmp values for gibbs sampler
  // use this for PersistentCD (!)
  Blob<Dtype> X1S_;
  Blob<Dtype> H1S_;


  #ifndef CPU_ONLY
  Blob<Dtype> randomContainer;
  #endif



};

/**
 * @brief RBM layer for Forward and Backward calculation.
 * Includes Contrastive Divergence gradient calculation.
 */
template <typename Dtype>
class RBMCDLayer : public RBMLayer<Dtype> {
 public:
  explicit RBMCDLayer(const LayerParameter& param)
      : RBMLayer<Dtype>(param) {}
  virtual inline const char* type() const { return "RBM CD"; }

 protected:
  virtual void gradient_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void gradient_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

/**
 * @brief RBM layer for Forward and Backward calculation.
 * Includes Persistent Contrastive Divergence gradient calculation.
 */
template <typename Dtype>
class RBMPCDLayer : public RBMLayer<Dtype> {
 public:
  explicit RBMPCDLayer(const LayerParameter& param)
      : RBMLayer<Dtype>(param) {}
  virtual inline const char* type() const { return "RBM CD"; }

 protected:
  virtual void gradient_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void gradient_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

/**
 * @brief RBM layer for Forward and Backward calculation.
 * Includes Perturb and MAP basic routines
 */
template <typename Dtype>
class RBMPMLayer : public RBMLayer<Dtype> {
 public:
  explicit RBMPMLayer(const LayerParameter& param)
      : RBMLayer<Dtype>(param) {}
  virtual inline const char* type() const { return "RBM PM"; }
  enum { CoordinateDescent, FreeEnergyGradientDescent };

 protected:
  virtual void find_map_cpu(Blob<Dtype>* X, Blob<Dtype>* H, Blob<Dtype>* b, Blob<Dtype>* c, Blob<Dtype>* W);
  virtual void find_map_gpu(Blob<Dtype>* X, Blob<Dtype>* H, Blob<Dtype>* b, Blob<Dtype>* c, Blob<Dtype>* W);
  virtual void replicate_data_cpu(const int N, Blob<Dtype>* X, Blob<Dtype>* repX);
  virtual void replicate_data_gpu(const int N, Blob<Dtype>* X, Blob<Dtype>* repX);
};

/**
 * @brief RBM layer for Forward and Backward calculation.
 * Includes Perturb and MAP order 1 basic routines
 */
template <typename Dtype>
class RBMPM1Layer : public RBMPMLayer<Dtype> {
 public:
  explicit RBMPM1Layer(const LayerParameter& param)
      : RBMPMLayer<Dtype>(param) {}
  virtual inline const char* type() const { return "RBM PM1"; }

 protected:
  virtual void gradient_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void gradient_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

/**
 * @brief RBM layer for Forward and Backward calculation.
 * Includes Perturb and MAP order 2 basic routines
 */
template <typename Dtype>
class RBMPM2Layer : public RBMPMLayer<Dtype> {
 public:
  explicit RBMPM2Layer(const LayerParameter& param)
      : RBMPMLayer<Dtype>(param) {}
  virtual inline const char* type() const { return "RBM PM1"; }

  enum { Random };

 protected:
  virtual void gradient_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void gradient_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void find_w_mask_gpu(Blob<Dtype>* W);
  virtual void find_w_mask_cpu(Blob<Dtype>* W);
};

} // namespace Caffe

#endif  // MLG_RBM_LAYERS_HPP_
