
#include <cstddef>
#include <complex>
#include "../include/cnet/mat.hpp"
#include "../include/cnet/layer.hpp"
#include "../include/cnet/afunc.hpp"

template<class T>
cnet::layer<T>::layer(std::size_t in, std::size_t out)
{
	in_ = in;
	out_ = out;
	afunc_ = cnet::CNET_NONE;
	W_.resize(out, in);
	B_.resize(out, 1);

	rand_range(0.0, 1.0);
}


template<class T>
cnet::layer<T>::layer(std::size_t in, std::size_t out, enum AFUNC_TYPE afunc)
{
	in_ = in;
	out_ = out;
	afunc_ = afunc;
	W_.resize(out, in);
	B_.resize(out, 1);
	
	rand_range(0.0, 1.0);
}

template<class T>
cnet::layer<T>::layer(void)
{
	
}

template<class T>
void cnet::layer<T>::mod(std::size_t in, std::size_t out, enum AFUNC_TYPE afunc)
{
	in_ = in;
	out_ = out;
	afunc_ = afunc;
	W_.resize(out, in);
	B_.resize(out, 1);

	rand_range(0.0, 1.0);
}

template<class T>
void cnet::layer<T>::mod(std::size_t in, std::size_t out)
{
	in_ = in;
	out_ = out;
	W_.resize(out, in);
	B_.resize(out, 1);
	rand_range(0.0, 1.0);
}

template<class T>
void cnet::layer<T>::rand_range(T a, T b)
{
	cnet::rand_mat(W_, a, b);
	cnet::rand_mat(B_, a, b);
}

template<class T>
cnet::mat<T> cnet::layer<T>::feedforward(cnet::mat<T> &X)
{
	if (X.get_cols() != 1 || X.get_rows() != W_.get_cols())
		throw std::invalid_argument("invalid argument: Matrices has different sizes");
	
	cnet::mat<T> A = W_ * X + B_;

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
