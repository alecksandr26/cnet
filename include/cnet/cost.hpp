#ifndef COST_INCLUDED
#define COST_INCLUDED

#include "mat.hpp"

#include <cstddef>

namespace cnet {
	namespace cost {
		template<class T>
		class cost {
		public:
			virtual mat<T> operator()(const mat<T> *A, const mat<T> *Y,
						  std::size_t in_size) const = 0;
			virtual mat<T> derivate(const mat<T> &A, const mat<T> &Y, std::size_t in_size) const = 0;
		};
		

		template<class T>
		class mse : public cost<T> {
		public:
			mat<T> operator()(const mat<T> *A, const mat<T> *Y, std::size_t in_size) const override;
			mat<T> derivate(const mat<T> &A, const mat<T> &Y, std::size_t in_size) const override;
		};
	}
}

#endif


