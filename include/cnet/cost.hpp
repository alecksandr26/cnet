#ifndef COST_INCLUDED
#define COST_INCLUDED

#include "mat.hpp"

#include <cstddef>

namespace cnet {
	namespace cost {
		template<class T>
		class cost_func {
		public:
			virtual mat<T> func(const mat<T> *A, const mat<T> *Y, std::size_t in_size) const = 0;
			virtual mat<T> dfunc_da(const mat<T> &A, const mat<T> &Y, std::size_t in_size) const = 0;
		};
		

		template<class T>
		class mse : public cost_func<T> {
		public:
			mat<T> func(const mat<T> *A, const mat<T> *Y, std::size_t in_size) const override;
			mat<T> dfunc_da(const mat<T> &A, const mat<T> &Y, std::size_t in_size) const override;
		};
	}
}

#endif


