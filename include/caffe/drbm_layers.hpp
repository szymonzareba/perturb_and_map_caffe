#ifndef MLG_DRBM_LAYERS_HPP_
#define MLG_DRBM_LAYERS_HPP_


#include "caffe/util/mlg_rng.hpp"
#include "caffe/util/mlg_assert.hpp"


namespace caffe {

template <typename Dtype>
class DRBMLayer : public Layer<Dtype> {
 public:
  explicit DRBMLayer<Dtype> (const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "DRBM Simple"; }
  //virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 3; }
  virtual ~DRBMLayer();

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void gradient_cpu();
  virtual void gradient_gpu();
  virtual Dtype ll_cpu();
  virtual Dtype ll_gpu();

  void sigmoid_gpu(int count, Dtype* data);
  void sample_gpu(int N, Dtype* mat);
  void sample_gpu(int N, const Dtype* src, Dtype* dst);

  Blob<Dtype> randomContainer;

  vector<int> layer_sizes;
  vector<Blob<Dtype>*> biases;
  vector<Blob<Dtype>*> weights;

  int K_;
  int M_;

  vector<Blob<Dtype>*> probs_0;
  vector<Blob<Dtype>*> states_0;

  vector<Blob<Dtype>*> probs_1;
  vector<Blob<Dtype>*> states_1;

  Blob<Dtype> ones_m;
};


template <typename Dtype>
class DRBMMeanFieldLayer : public DRBMLayer<Dtype> {
 public:
  explicit DRBMMeanFieldLayer(const LayerParameter& param)
      : DRBMLayer<Dtype>(param) {}
  virtual inline const char* type() const { return "DRBM Mean Field"; }

 protected:
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

};

template <typename Dtype>
class DRBMPMLayer : public DRBMLayer<Dtype> {
 public:
  explicit DRBMPMLayer(const LayerParameter& param)
      : DRBMLayer<Dtype>(param) {}
  virtual inline const char* type() const { return "DRBM PM"; }

 protected:
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void optimizeEnergy_cpu();
  virtual void optimizeEnergy_gpu();
  virtual void generatePerturbations_cpu() = 0;
  virtual void generatePerturbations_gpu() = 0;
};

template <typename Dtype>
class DRBMPM1Layer : public DRBMPMLayer<Dtype> {
 public:
  explicit DRBMPM1Layer(const LayerParameter& param)
      : DRBMPMLayer<Dtype>(param) {}
  virtual inline const char* type() const { return "DRBM PM1"; }

 protected:
  virtual void generatePerturbations_cpu();
  virtual void generatePerturbations_gpu();
};

}
#endif  // MLG_DRBM_LAYERS_HPP_
