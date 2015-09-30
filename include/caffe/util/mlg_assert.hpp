#ifndef MLG_ASSERT_HPP_
#define MLG_ASSERT_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"

#ifndef CPU_ONLY
#include <curand.h>
#include <curand_kernel.h>
#endif

namespace caffe {

template <typename Dtype>
class MLGASSERT {
	private:
		static MLGASSERT<Dtype> * pInstance;
		MLGASSERT();
		MLGASSERT(const MLGASSERT<Dtype>& rs){
			pInstance = rs.pInstance;
			work = rs.work;
		}
		MLGASSERT<Dtype>& operator = (const MLGASSERT<Dtype>& rs) {
			if (this != &rs) {
				pInstance = rs.pInstance;
			}

			return *this;
		}
		~MLGASSERT ();

	public:
		static MLGASSERT<Dtype>& getInstance(){
			static MLGASSERT<Dtype> theInstance;
			pInstance = &theInstance;
			return *pInstance;
		}
		bool work;
		//void mlg_cpu_uniform(const int N, Dtype* data);
		bool mlg_gpu_finite(const int N, const Dtype* data);
		bool mlg_gpu_range(const int N, const Dtype* data);
		bool mlg_gpu_range(const int N, const Dtype* data, const Dtype min, const Dtype max);
};

template<typename Dtype>
MLGASSERT<Dtype>* MLGASSERT<Dtype>::pInstance = NULL;

}  // namespace caffe

#endif  // MLG_ASSERT_HPP_
