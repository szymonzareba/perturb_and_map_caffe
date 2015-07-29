#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/rbm_layers.hpp"

namespace caffe {

template <typename Dtype>
void RBMPMLayer<Dtype>::find_map_cpu(Blob<Dtype>* X, Blob<Dtype>* H, Blob<Dtype>* b, Blob<Dtype>* c, Blob<Dtype>* W){

	switch(this->layer_param_.rbm_param().rbm_pm_param().map_method()){
		case RBMPMLayer::CoordinateDescent:
		{
			Dtype* XS = X->mutable_cpu_data();
			Dtype* HS = H->mutable_cpu_data();
			int descentSteps = this->layer_param_.rbm_param().rbm_pm_param().coordinate_descent_param().descent_steps();

			caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
					  this->M_, this->N_, this->K_,
					  (Dtype)1., XS, W->cpu_data(),
					  (Dtype)0., HS);

			caffe_add(H->count(), HS, c->cpu_data(), HS);

			for(int i = 0; i < H->count(); i++){
				if(HS[i] > (Dtype)0.){
					HS[i] = (Dtype) 1.;
				}else{
					HS[i] = (Dtype) 0.;
				}
			}

			for(int descent = 0; descent < descentSteps; descent++){

				caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
						this->M_, this->K_, this->N_,
						(Dtype)1., HS, W->cpu_data(),
						(Dtype)0., XS);


				caffe_add(X->count(), XS, b->cpu_data(), XS);

				for(int i = 0; i < X->count(); i++){
					if(XS[i] > (Dtype)0.){
						XS[i] = (Dtype) 1.;
					}else{
						XS[i] = (Dtype) 0.;
					}
				}

				caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
						this->M_, this->N_, this->K_,
						(Dtype)1., XS, W->cpu_data(),
						(Dtype)0., HS);

				caffe_add(H->count(), HS, c->cpu_data(), HS);

				for(int i = 0; i < H->count(); i++){
					if(HS[i] > (Dtype)0.){
						HS[i] = (Dtype) 1.;
					}else{
						HS[i] = (Dtype) 0.;
					}
				}

			}
		}
		break;
		case RBMPMLayer::FreeEnergyGradientDescent:
		{
			NOT_IMPLEMENTED;
		}
		break;
		default:
		{
			LOG(INFO) << "No such MAP method";
		}
	}
}

template <typename Dtype>
void RBMPMLayer<Dtype>::replicate_data_cpu(const int N, Blob<Dtype>* X, Blob<Dtype>* repX){
	NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(RBMPMLayer);
#endif

INSTANTIATE_CLASS(RBMPMLayer);
REGISTER_LAYER_CLASS(RBMPM);

} // namespace caffe
