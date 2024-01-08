#include "cnet/cost.hpp"

#include "cnet/mat.hpp"

#include <cstddef>
#include <iostream>

template<class T>
cnet::mat<T> cnet::cost::mse<T>::operator()(const cnet::mat<T> *A, const cnet::mat<T> *Y,
					    std::size_t in_size) const
{
	cnet::mat<T> cost(A[0].get_rows(), A[0].get_cols(), 0.0);

	for (std::size_t i = 0; i < in_size; i++)
		cost += (A[i] - Y[i]) ^ (A[i] - Y[i]);
	;

	// We need to implmenet the / for the mat operation
	return cost * (1.0 / in_size);
}

template<class T>
cnet::mat<T> cnet::cost::mse<T>::derivate(const cnet::mat<T> &A, const cnet::mat<T> &Y,
					  std::size_t in_size) const
{
	return (A - Y) * (2.0 / in_size);
}

template class cnet::cost::cost<double>;
template class cnet::cost::mse<double>;
