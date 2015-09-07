#ifndef MLG_UTIL_MATH_H_
#define MLG_UTIL_MATH_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit

#include "glog/logging.h"

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"

namespace caffe {


template <typename Dtype>
__global__ void sample_ge0_kernel(const int n, Dtype* y);

template <typename Dtype>
__global__ void sample_ge0_5_kernel(const int n, Dtype* y);

template <typename Dtype>
__global__ void add_scaled_kernel(const int n, const Dtype alpha, const Dtype* a, const Dtype beta, const Dtype* b, Dtype* x);

template <typename Dtype>
__global__ void relax_0_1_kernel(const int n, Dtype* x);

template <typename Dtype>
__global__ void replicate_kernel(const int xcount, const int repxcount, const Dtype* x, Dtype* repx);

template <typename Dtype>
__global__ void negate_kernel(const int n, Dtype* x);

template <typename Dtype>
__global__ void add_with_mask_kernel(const int n, const Dtype* a, const Dtype* bMask, const Dtype* b, Dtype* x);

template <typename Dtype>
__global__ void binarization_kernel(const int count, const Dtype threshold, const Dtype* x, Dtype* y);
}  // namespace caffe

#endif  // MLG_UTIL_MATH__H_
