#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/drbm_layers.hpp"
#include "caffe/util/mlg_math.hpp"

namespace caffe {

template <typename Dtype>
void DRBMPM1Layer<Dtype>::generatePerturbations_gpu() {
	NOT_IMPLEMENTED;
	//TODO generate perturbations
}

template void DRBMPM1Layer<float>::generatePerturbations_gpu();
template void DRBMPM1Layer<double>::generatePerturbations_gpu();

//INSTANTIATE_LAYER_GPU_FUNCS(DRBMPM1Layer);

} // namespace caffe
