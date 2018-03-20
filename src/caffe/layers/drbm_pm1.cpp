#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/drbm_layers.hpp"

namespace caffe {

template <typename Dtype>
void DRBMPM1Layer<Dtype>::generatePerturbations_cpu(){
	NOT_IMPLEMENTED;
	//TODO generate perturbations
}

template void DRBMPM1Layer<float>::generatePerturbations_cpu();
template void DRBMPM1Layer<double>::generatePerturbations_cpu();

#ifdef CPU_ONLY
STUB_GPU(DRBMPM1Layer);
#endif

INSTANTIATE_CLASS(DRBMPM1Layer);
REGISTER_LAYER_CLASS(DRBMPM1);

} // namespace caffe
