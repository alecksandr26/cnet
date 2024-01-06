#include "cnet/mat.hpp"
#include "cnet/afunc.hpp"
#include "cnet/layer.hpp"
#include "cnet/ann.hpp"
#include "cnet/cost.hpp"

#include <iostream>
#include <limits>
#include <memory>

template<class T>
cnet::ann<T>::ann(const cnet::nn_arch *arch, std::size_t l, std::unique_ptr<afunc::afunc<T>> &&func)
{
	l_ = l;
	layers_ = new layer::dense<T>[l];
	
	std::size_t in, out;
	for (std::size_t i = 1; i <= l_; i++) {
		in  = arch[i - 1];
		out = arch[i];
		
		layers_[i - 1].set_units(out);
		layers_[i - 1].set_afunc(std::move(func));
		layers_[i - 1].build(in);
	}
}

template<class T>
cnet::ann<T>::~ann(void)
{
	delete[] layers_;
}

template<class T>
std::size_t cnet::ann<T>::get_layers(void)
{
	return l_;
}

template<class T>
cnet::mat<T> cnet::ann<T>::feedforward(const cnet::mat<T> &X)
{
	cnet::mat<T> in = X, out;

	for (std::size_t i = 0; i < l_; i++) {
		out = layers_[i](in);
		in  = out;
	}

	return out;
}

template<class T>
void cnet::ann<T>::fit(mat<T> *input, mat<T> *output, std::size_t train_size, double lr,
		       std::size_t nepochs)
{
	while (nepochs--) {
		double	     c, dc;
		const double h = 0.00001;	 // epsilon

		// for (std::size_t l = 0; l < l_; l++) {
		// 	for (std::size_t i = 0; i < layers_[l].W_.get_rows(); i++)
		// 		for (std::size_t j = 0; j < layers_[l].W_.get_cols(); j++) {
		// 			c = cnet::mse(*this, input, output, train_size);
		// 			layers_[l].W_(i, j) += h;
		// 			dc = cnet::mse(*this, input, output, train_size);
		// 			layers_[l].W_(i, j) -= h;
		// 			layers_[l].W_(i, j) -= lr * (dc - c) / h;
		// 		}

		// 	for (std::size_t i = 0; i < layers_[l].B_.get_rows(); i++)
		// 		for (std::size_t j = 0; j < layers_[l].B_.get_cols(); j++) {
		// 			c = cnet::mse(*this, input, output, train_size);
		// 			layers_[l].B_(i, j) += h;
		// 			dc = cnet::mse(*this, input, output, train_size);
		// 			layers_[l].B_(i, j) -= h;
		// 			layers_[l].B_(i, j) -= lr * (dc - c) / h;
		// 		}
		// }
	}
}

template class cnet::ann<double>;
