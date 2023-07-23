
#include <cstddef>
#include <complex>
#include "../include/cnet/mat.hpp"
#include "../include/cnet/layer.hpp"
#include "../include/cnet/afuncs.hpp"

template<class T>
cnet::layer<T>::layer(std::size_t in, std::size_t out, enum AFUNC_TYPE afunc)
{
	in_ = in;
	out_ = out;
	afunc_ = afunc;
	W_.rsize(out, in);
	B_.rsize(out, 1);
	
	cnet::rand_mat(W_, 0.0, 1.0);
	cnet::rand_mat(B_, 0.0, 1.0);
}

template<class T>
cnet::mat<T> cnet::layer<T>::feedforward(cnet::mat<T> &X)
{
	if (X.get_n_cols() != 1 || X.get_n_rows() != W_.get_n_cols())
		throw std::invalid_argument("invalid argument: Matrices has different sizes");
	
	cnet::mat<T> A = (W_ * X) + B_;

	switch ((int) afunc_) {	
	case cnet::CNET_RELU:
		cnet::relu(A);
		break;

	case cnet::CNET_SIGMOID:
		cnet::sigmoid(A);
		break;
	}
	
	return A;
}

template class cnet::mat<double>;
template class cnet::layer<double>;
