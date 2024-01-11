#ifndef COST_FUNC_INCLUDED
#define COST_FUNC_INCLUDED

#include <cstddef>

// #include "variable.hpp"

namespace cnet::cfuncs {
	// using namespace variable;
	// class Error : public Var {
	// 	// Nothing here yet
	// };
	
	template<typename T>
	T Mse(const T *A, const T *Y, std::size_t N);

	template<typename T>
	T MseDerivate(const T &A, const T &Y, std::size_t N);
}

#endif


