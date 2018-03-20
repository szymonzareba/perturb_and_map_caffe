#ifndef MLG_RNG_HPP_
#define MLG_RNG_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"

#include <stdio.h>
#include <stdlib.h>


#ifndef CPU_ONLY
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#endif

#define MLG_MIN_UNI (Dtype)(0.0000001)
#define MLG_MAX_UNI (Dtype)(1.-MLG_MIN_UNI)

namespace caffe {

template <typename Dtype>
class MLGRNG {
	private:
		static MLGRNG<Dtype> * pInstance;
		MLGRNG();
		MLGRNG(const MLGRNG<Dtype>& rs){
			//stateCount = rs.stateCount;
			//states = rs.states;
			pInstance = rs.pInstance;
			gen = rs.gen;
			this->initialized = rs.initialized;
		}
		MLGRNG<Dtype>& operator = (const MLGRNG<Dtype>& rs) {
			if (this != &rs) {
				pInstance = rs.pInstance;
			}

			return *this;
		}
		~MLGRNG ();

#ifndef CPU_ONLY
		//int stateCount;
		//curandState_t* states;
		curandGenerator_t gen;
		bool initialized;

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

		void mlg_cpu_permutation(const int N, int* data);
		void mlg_gpu_permutation(const int N, int* data);

		void mlg_cpu_range(const int N, const int min, const int max, int* data);
		void mlg_gpu_range(const int N, const int min, const int max, int* data);
};

template<typename Dtype>
MLGRNG<Dtype>* MLGRNG<Dtype>::pInstance = NULL;

}  // namespace caffe

#endif  // MLG_RNG_HPP_
