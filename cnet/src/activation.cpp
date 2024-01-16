#include "utils/avx.hpp"
#include "utils/raw_mat.hpp"

#include "cnet/dtypes.hpp"
#include "cnet/variable.hpp"
#include "cnet/mat.hpp"
#include "cnet/activation.hpp"

#include <cstddef>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <functional>

// To optimize the matrix operations and the parallelization of it
#include <immintrin.h>
#include <omp.h>

using namespace std;
using namespace cnet;
using namespace dtypes;
using namespace activation;
using namespace mathops;
using namespace utils;
using namespace variable;

// Trick to get template pointer functions
// https://stackoverflow.com/questions/4573941/c-function-pointer-to-the-template-function-pointer

template<class T>
struct StLinear {
	static T Linear(const T &X)
	{
		return X;
	}
};

template<typename T>
T cnet::activation::Linear(const T &X)
{
	return StLinear<T>::Linear(X);
}

template<class T>
struct StLinearDerivate {
	static T LinearDerivate(const T &X)
	{
		return T(X.get_shape(), 1.0);
	}
};

template<>
Var StLinearDerivate<Var>::LinearDerivate(const Var &X)
{
	switch (X.get_dtype()) {
	case FLOAT_32_DTYPE:
		return Var(X.get_shape(), (float32) 1.0);
	case FLOAT_64_DTYPE:
		return Var(X.get_shape(), (float64) 1.0);
	default:
		throw invalid_argument("invalid argument: Invalid variable datatype");
		break;		
	}
}

template<typename T>
T cnet::activation::LinearDerivate(const T &X)
{
	return StLinearDerivate<T>::LinearDerivate(X);
}

template<typename T>
static void sigmoid_helper(const Mat<T> &X, Mat<T> &Y)
{
	size_t rows = X.get_rows();
	size_t cols = X.get_cols();
	
	Y.resize(rows, cols);
	
	T *x_allocated_mat = X.get_allocated_mat();
	T *y_allocated_mat = Y.get_allocated_mat();
	size_t n		= rows * cols;
	size_t n_ite_8	= n - (n % 8);
	size_t n_ite_4	= n - (n % 4);

	switch (n) {
	case 1: y_allocated_mat[0] = (1.0 / (1.0 + exp(-x_allocated_mat[0]))); break;
	case 2:
		y_allocated_mat[0] = (1.0 / (1.0 + exp(-x_allocated_mat[0])));
		y_allocated_mat[1] = (1.0 / (1.0 + exp(-x_allocated_mat[1])));
		break;
	case 3:
		y_allocated_mat[0] = (1.0 / (1.0 + exp(-x_allocated_mat[0])));
		y_allocated_mat[1] = (1.0 / (1.0 + exp(-x_allocated_mat[1])));
		y_allocated_mat[2] = (1.0 / (1.0 + exp(-x_allocated_mat[2])));
		break;
	default:
#pragma omp parallel for
		for (size_t i = 0; i < n_ite_8; i += 8)
			sigmoid_avx_8(&y_allocated_mat[i], &x_allocated_mat[i]);

#pragma omp parallel for
		for (size_t i = n_ite_8; i < n_ite_4; i += 4)
			sigmoid_avx_4(&y_allocated_mat[i], &x_allocated_mat[i]);

		for (size_t i = n_ite_4; i < n; i++)
			y_allocated_mat[i] = (1.0 / (1 + exp(-x_allocated_mat[i])));
		break;

	}
}

template<typename T>
static void sigmoid_derivate_helper(const Mat<T> &X, Mat<T> &Y)
{
	size_t rows = X.get_rows();
	size_t cols = X.get_cols();

	// Alloc the matrix
	Y.resize(rows, cols);
	
	T *x_allocated_mat = X.get_allocated_mat();
	T *y_allocated_mat = Y.get_allocated_mat();
	size_t n		= rows * cols;
	size_t n_ite_8	= n - (n % 8);
	size_t n_ite_4	= n - (n % 4);

	switch (n) {
	case 1:
		y_allocated_mat[0] =
			(1.0 / (2.0 + exp(-x_allocated_mat[0]) + exp(x_allocated_mat[0])));
		break;
	case 2:
		y_allocated_mat[0] =
			(1.0 / (2.0 + exp(-x_allocated_mat[0]) + exp(x_allocated_mat[0])));
		y_allocated_mat[1] =
			(1.0 / (2.0 + exp(-x_allocated_mat[1]) + exp(x_allocated_mat[1])));
		break;
	case 3:
		y_allocated_mat[0] =
			(1.0 / (2.0 + exp(-x_allocated_mat[0]) + exp(x_allocated_mat[0])));
		y_allocated_mat[1] =
			(1.0 / (2.0 + exp(-x_allocated_mat[1]) + exp(x_allocated_mat[1])));
		y_allocated_mat[2] =
			(1.0 / (2.0 + exp(-x_allocated_mat[2]) + exp(x_allocated_mat[2])));
		break;
	default:
#pragma omp parallel for
		for (size_t i = 0; i < n_ite_8; i += 8)
			derivate_sigmoid_avx_8(&y_allocated_mat[i], &x_allocated_mat[i]);
		
#pragma omp parallel for
		for (size_t i = n_ite_8; i < n_ite_4; i += 4)
			derivate_sigmoid_avx_4(&y_allocated_mat[i], &x_allocated_mat[i]);
		
		for (size_t i = n_ite_4; i < n; i++)
			y_allocated_mat[i] = (1.0 / (2.0 + exp(-x_allocated_mat[i]) +
						     exp(x_allocated_mat[i])));

		break;
	}
}

template<class T>
struct StSigmoid {
	static T Sigmoid(const T &X)
	{
		T Y(X.get_shape());
		sigmoid_helper(X, Y);
		return Y;
	}
};

template<>
Var StSigmoid<Var>::Sigmoid(const Var &X)
{
	Var Y(X.get_shape(), X.get_dtype());
	
	switch (X.get_dtype()) {
	case FLOAT_32_DTYPE:
		sigmoid_helper(X.get_cmf32(), Y.get_mf32());
		break;
	case FLOAT_64_DTYPE:
		sigmoid_helper(X.get_cmf64(), Y.get_mf64());
		break;
	default:
		throw runtime_error("runtime error: Invalid datatype");
		break;
	}

	return Y;
}

template<typename T>
T cnet::activation::Sigmoid(const T &X)
{
	return StSigmoid<T>::Sigmoid(X);
}


template<class T>
struct StSigmoidDerivate {
	static T SigmoidDerivate(const T &X)
	{
		T Y(X.get_shape());
		sigmoid_derivate_helper(X, Y);
		return Y;
	}
};

template<>
Var StSigmoidDerivate<Var>::SigmoidDerivate(const Var &X)
{
	Var Y(X.get_shape(), X.get_dtype());
	
	switch (X.get_dtype()) {
	case FLOAT_32_DTYPE:
		sigmoid_derivate_helper(X.get_cmf32(), Y.get_mf32());
		break;
	case FLOAT_64_DTYPE:
		sigmoid_derivate_helper(X.get_cmf64(), Y.get_mf64());
		break;
	default:
		throw runtime_error("runtime error: Invalid datatype");
		break;
	}

	return Y;
}


template<typename T>
T cnet::activation::SigmoidDerivate(const T &X)
{
	return StSigmoidDerivate<T>::SigmoidDerivate(X);
}

template<typename T>
static void relu_helper(const Mat<T> &X, Mat<T> &Y)
{
	size_t rows = X.get_rows();
	size_t cols = X.get_cols();
	
	Y.resize(rows, cols);

	T *x_allocated_mat = X.get_allocated_mat();
	T *y_allocated_mat = Y.get_allocated_mat();
	size_t n	     = rows * cols;
	size_t n_ite_8  = n - (n % 8);
	size_t n_ite_4  = n - (n % 4);

	switch (n) {
	case 1: y_allocated_mat[0] = max(x_allocated_mat[0], (T) 0.0); break;
	case 2:
		y_allocated_mat[0] = max(x_allocated_mat[0], (T) 0.0);
		y_allocated_mat[1] = max(x_allocated_mat[1], (T) 0.0);
		break;
	case 3:
		y_allocated_mat[0] = max(x_allocated_mat[0], (T) 0.0);
		y_allocated_mat[1] = max(x_allocated_mat[1], (T) 0.0);
		y_allocated_mat[2] = max(x_allocated_mat[2], (T) 0.0);
		break;
	default:
#pragma omp parallel for
		for (size_t i = 0; i < n_ite_8; i += 8)
			relu_avx_8(&y_allocated_mat[i], &x_allocated_mat[i]);
		
#pragma omp parallel for
		for (size_t i = n_ite_8; i < n_ite_4; i += 4)
			relu_avx_4(&y_allocated_mat[i], &x_allocated_mat[i]);
		
		for (size_t i = n_ite_4; i < n; i++)
			y_allocated_mat[i] = max(x_allocated_mat[i], (T) 0.0);

		break;
	}
}

template<typename T>
static void relu_derivate_helper(const Mat<T> &X, Mat<T> &Y)
{
	// super funny but yea the derivate of relu_helper is `1`
	Y = Mat<T>(X.get_shape(), (T) 1.0);
}

template<class T>
struct StRelu {
	static T Relu(const T &X)
	{
		T Y(X.get_shape());
		relu_helper(X, Y);
		return Y;
	}
};

template<>
Var StRelu<Var>::Relu(const Var &X)
{
	Var Y(X.get_shape(), X.get_dtype());
	
	switch (X.get_dtype()) {
	case FLOAT_32_DTYPE:
		relu_helper(X.get_cmf32(), Y.get_mf32());
		break;
	case FLOAT_64_DTYPE:
		relu_helper(X.get_cmf64(), Y.get_mf64());
		break;
	default:
		throw runtime_error("runtime error: Invalid datatype");
		break;
	}

	return Y;
}

template<typename T>
T cnet::activation::Relu(const T &X)
{
	return StRelu<T>::Relu(X);
}

template<class T>
struct StReluDerivate {
	static T ReluDerivate(const T &X)
	{
		return StLinearDerivate<T>::LinearDerivate(X);
	}
};

template<typename T>
T cnet::activation::ReluDerivate(const T &X)
{
	return StReluDerivate<T>::ReluDerivate(X);
}

template Mat<float32> cnet::activation::Linear(const Mat<float32> &X);
template Mat<float64> cnet::activation::Linear(const Mat<float64> &X);
template Var cnet::activation::Linear(const Var &X);

template class StLinear<Mat<float32>>;
template class StLinear<Mat<float64>>;
template class StLinear<Var>;

template Mat<float32> cnet::activation::LinearDerivate(const Mat<float32> &X);
template Mat<float64> cnet::activation::LinearDerivate(const Mat<float64> &X);
template Var cnet::activation::LinearDerivate(const Var &X);

template class StLinearDerivate<Mat<float32>>;
template class StLinearDerivate<Mat<float64>>;
template class StLinearDerivate<Var>;

template Mat<float32> cnet::activation::Sigmoid(const Mat<float32> &X);
template Mat<float64> cnet::activation::Sigmoid(const Mat<float64> &X);
template Var cnet::activation::Sigmoid(const Var &X);

template class StSigmoid<Mat<float32>>;
template class StSigmoid<Mat<float64>>;
template class StSigmoid<Var>;

template Mat<float32> cnet::activation::SigmoidDerivate(const Mat<float32> &X);
template Mat<float64> cnet::activation::SigmoidDerivate(const Mat<float64> &X);
template Var cnet::activation::SigmoidDerivate(const Var &X);

template class StSigmoidDerivate<Mat<float32>>;
template class StSigmoidDerivate<Mat<float64>>;
template class StSigmoidDerivate<Var>;

template Mat<float32> cnet::activation::Relu(const Mat<float32> &X);
template Mat<float64> cnet::activation::Relu(const Mat<float64> &X);
template Var cnet::activation::Relu(const Var &X);

template class StRelu<Mat<float32>>;
template class StRelu<Mat<float64>>;
template class StRelu<Var>;

template Mat<float32> cnet::activation::ReluDerivate(const Mat<float32> &X);
template Mat<float64> cnet::activation::ReluDerivate(const Mat<float64> &X);
template Var cnet::activation::ReluDerivate(const Var &X);

template class StReluDerivate<Mat<float32>>;
template class StReluDerivate<Mat<float64>>;
template class StReluDerivate<Var>;

template<class T>
void cnet::activation::Afunc<T>::alloc_afunc(const string &afunc_name)
{
	static unordered_map<string, function<T(const T &)>> afunc_map = {
		{"Linear", StLinear<T>::Linear},
		{"Sigmoid", StSigmoid<T>::Sigmoid},
		{"Relu", StRelu<T>::Relu}
	};

	static unordered_map<string, function<T(const T &)>> afunc_derivate_map = {
		{"Linear", StLinearDerivate<T>::LinearDerivate},
		{"Sigmoid", StSigmoidDerivate<T>::SigmoidDerivate},
		{"Relu", StReluDerivate<T>::ReluDerivate}
	};

	auto it = afunc_map.find(afunc_name);
	if (it == afunc_map.end())
		throw invalid_argument("invalid argument: Invalid activation function name");
	afunc_ = it->second;

	auto it2 = afunc_derivate_map.find(afunc_name);
	if (it2 == afunc_derivate_map.end())
		throw invalid_argument("invalid argument: Invalid activation function name");
	afunc_derivate_ = it2->second;
}

template class cnet::activation::Afunc<Mat<float32>>;
template class cnet::activation::Afunc<Mat<float64>>;
template class cnet::activation::Afunc<Var>;

template static void sigmoid_helper(const Mat<float64> &X, Mat<float64> &Y);
template static void sigmoid_helper(const Mat<float32> &X, Mat<float32> &Y);
template static void sigmoid_derivate_helper(const Mat<float64> &X, Mat<float64> &Y);
template static void sigmoid_derivate_helper(const Mat<float32> &X, Mat<float32> &Y);

template static void relu_helper(const Mat<float64> &X, Mat<float64> &Y);
template static void relu_helper(const Mat<float32> &X, Mat<float32> &Y);
template static void relu_derivate_helper(const Mat<float64> &X, Mat<float64> &Y);
template static void relu_derivate_helper(const Mat<float32> &X, Mat<float32> &Y);

