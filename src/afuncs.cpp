
#include <cstddef>
#include <cmath>
#include <complex>
#include <iostream>

#include "../include/cnet/afuncs.hpp"

template<typename T>
cnet::mat<T> &cnet::sigmoid(cnet::mat<T> &m)
{
	for (std::size_t i = 0; i < m.get_n_rows(); i++)
		for (std::size_t j = 0; j < m.get_n_cols(); j++)
			m(i, j) = (1.0 / (1 + exp(- m(i, j))));
	return m;
}

template<typename T>
cnet::mat<T> &cnet::relu(cnet::mat<T> &m)
{
	const long double epsilon = 1e-9;
    
	for (std::size_t i = 0; i < m.get_n_rows(); i++)
		for (std::size_t j = 0; j < m.get_n_cols(); j++)
			m(i, j) = (m(i, j) >= epsilon) ? m(i, j) : 0.0;
    
	return m;
}

template cnet::mat<double> &cnet::relu(cnet::mat<double> &m);
template cnet::mat<double> &cnet::sigmoid(cnet::mat<double> &m);


