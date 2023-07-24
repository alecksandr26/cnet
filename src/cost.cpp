
#include <cstddef>

#include "../include/cnet/mat.hpp"
#include "../include/cnet/ann.hpp"
#include "../include/cnet/cost.hpp"

template<typename T>
long double cnet::mse(cnet::ann<T> &ann, cnet::mat<T> *input,
		      cnet::mat<T> *output, std::size_t train_size)
{
	long double cost = 0.0;

	for (std::size_t i = 0; i < train_size; i++) {
		cnet::mat<T> Y = ann.feedforward(input[i]);
		cnet::mat<T> dif_Y = output[i] - Y;
		cnet::mat<T> C = dif_Y * dif_Y.transpose();
		cost += cnet::grand_sum(C);
	}
	
	return cost;
}


template long double cnet::mse(cnet::ann<double> &ann, cnet::mat<double> *input,
			       cnet::mat<double> *output, std::size_t train_size);




