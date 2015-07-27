#ifndef MLG_RNG_HPP_
#define MLG_RNG_HPP_

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
class MLGRNG {
	private:
		static MLGRNG<Dtype> * pInstance;
		MLGRNG();
		MLGRNG(const MLGRNG<Dtype>& rs){
			stateCount = rs.stateCount;
			states = rs.states;
			pInstance = rs.pInstance;
		}
		MLGRNG<Dtype>& operator = (const MLGRNG<Dtype>& rs) {
			if (this != &rs) {
				pInstance = rs.pInstance;
			}

			return *this;
		}
		~MLGRNG ();

#ifndef CPU_ONLY
		int stateCount;
		curandState_t* states;
#endif

	public:
		static MLGRNG<Dtype>& getInstance(){
			static MLGRNG<Dtype> theInstance;
			pInstance = &theInstance;
			return *pInstance;
		}

		void mlg_cpu_uniform(const int N, Dtype* data);
		void mlg_gpu_uniform(const int N, Dtype* data);

		void mlg_cpu_gumbel(const int N, Dtype* data);
		void mlg_gpu_gumbel(const int N, Dtype* data);
};

template<typename Dtype>
MLGRNG<Dtype>* MLGRNG<Dtype>::pInstance = NULL;

}  // namespace caffe

#endif  // MLG_RNG_HPP_
