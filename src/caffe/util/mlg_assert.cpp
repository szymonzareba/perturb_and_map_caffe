#include "caffe/util/mlg_assert.hpp"


namespace caffe {

template <typename Dtype>
MLGASSERT<Dtype>::MLGASSERT(){
	work = true;
}

template <typename Dtype>
MLGASSERT<Dtype>::~MLGASSERT(){
}

template MLGASSERT<float>::MLGASSERT();
template MLGASSERT<double>::MLGASSERT();

template MLGASSERT<float>::~MLGASSERT();
template MLGASSERT<double>::~MLGASSERT();


}
