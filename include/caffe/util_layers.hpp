#ifndef MLG_UTIL_LAYERS_HPP_
#define MLG_UTIL_LAYERS_HPP_

#ifndef CPU_ONLY
#include <curand.h>
#include <curand_kernel.h>
#endif

#include "caffe/util/mlg_rng.hpp"
#include "caffe/util/mlg_math.hpp"

namespace caffe {

/**
 * @brief Util layer for data binarization
 */
template <typename Dtype>
class BinarizationLayer : public Layer<Dtype> {
 public:
  explicit BinarizationLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "Binarization"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

}
#endif  // MLG_UTIL_LAYERS_HPP_
