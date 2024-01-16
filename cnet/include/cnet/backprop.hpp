#ifndef BACK_PROP_INCLUDED
#define BACK_PROP_INCLUDED

#include <cstddef>
#include "dtypes.hpp"

namespace cnet::weights::backprop {
	using namespace cnet;
	using namespace dtypes;
	
	// Get the derivate error with respect the input from the previous layer
	template<typename T>
	void get_derror_dinput(const T *dE, const T *W, T *dI, std::size_t rows, std::size_t cols);
	
	// fit backprop receives n * 1 dimension error and n * 1 dimension weighted derivate
	
	// Fit backpropagation:
	// a = f(z), where z = b + w_1,k * i_1 + ... + w_n,k * i_n + ..., f is the activation function,
	// a is the actual neuron and i is the input
	// dE = d(e)/d(z) = d(e)/d(a) * d(a)/d(z),
	// Depends on the type of weights is going to receive one derivate or the another
	// dW = d(z)/d(w_k) = i_k, if is a normal weight
	// dW = d(z)/d(b_k) = 1,  if it is a bias
	
	template<typename T>
	void fit_weights(const T *dE, const T *dW, T *W,
			 std::size_t rows, std::size_t cols, float64 lr);

}

#endif
