#ifndef WEIGHTS_INCLUDED
#define WEIGHTS_INCLUDED

#include <ostream>

#include "dtypes.hpp"
#include "mat.hpp"
#include "variable.hpp"

namespace cnet::weights {
	using namespace dtypes;
	using namespace mathops;
	using namespace variable;
	
	class Weights : public Var {
	public:		
		std::size_t get_num_params(void) const;
		
		// For the moment just this options
		
		// To compute the derivate error for the previos layer
		// a = f(z), where z = b + w_1 * i_1 + ... + w_n * i_n + ..., f is the activation function,
		// dZ = d(e)/d(z) = d(e)/d(a) * d(a)/d(z),
		// I is the input from which we want its derivate, basically d(e)/d(i_k)
		// Var get_derror_dinput(const Var &dZ, Shape in) const;
		// Mat<float32> get_derror_dinput(const Mat<float32> &dZ, Shape in) const;
		// Mat<float64> get_derror_dinput(const Mat<float64> &dZ, Shape in) const;
		
		// Fit backpropagation:
		// a = f(z), where z = b + w_1 * i_1 + ... + w_n * i_n + ..., f is the activation function,
		// a is the actual neuron and i is the input
		// dZ = d(e)/d(z) = d(e)/d(a) * d(a)/d(z),
		// Depends on the type of weights is going to receive one derivate or the another
		// dW = d(z)/d(w_k) = i_k, if is a normal weight
		// dW = d(z)/d(b_k) = 1,  if it is a bias
		Weights &fit(const Mat<float32> &dZ, const Mat<float32> &dW, float64 lr);
		Weights &fit(const Mat<float64> &dZ, const Mat<float64> &dW, float64 lr);
		Weights &fit(const Var &dZ, const Var &dW, float64 lr);

		// Not sure to add the weights in that way 
		// Weights &add_weights(CnetDtype dtype, Shape shape);
		// Weights &add_weights(CnetDtype dtype, const std::string &name, Shape shape);
	};
}

#endif



