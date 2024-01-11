
#include "cnet/dtypes.hpp"
#include "cnet/mat.hpp"
#include "cnet/variable.hpp"
#include "cnet/cfuncs.hpp"

#include <cstddef>

using namespace std;
using namespace cnet;
using namespace dtypes;
using namespace mathops;
using namespace variable;
using namespace cfuncs;

template<typename T>
T cnet::cfuncs::Mse(const T *A, const T *Y, size_t N)
{
	if (N == 0)
		throw invalid_argument("invalid argument: Invalid Empty set");
	
	T E(A[0].get_shape(), 0.0);
	for (size_t i = 0; i < N; i++)
		E += (A[i] - Y[i]) ^ (A[i] - Y[i]);
	return E * (1.0 / N);
}

template<>
Var cnet::cfuncs::Mse(const Var *A, const Var *Y, size_t N)
{
	if (N == 0)
		throw invalid_argument("invalid argument: Invalid Empty set");

	Var E;
	
	switch (A[0].get_dtype()) {
	case FLOAT_32_DTYPE:
		E.resize(A[0].get_shape(), (float32) 0.0);
		break;
	case FLOAT_64_DTYPE:
		E.resize(A[0].get_shape(), (float64) 0.0);
		break;
	default:
		throw invalid_argument("invalid argument: Invalid variable datatype");
		break;		
	}
	
	for (size_t i = 0; i < N; i++)
		E += (A[i] - Y[i]) ^ (A[i] - Y[i]);

	switch (E.get_dtype()) {
	case FLOAT_32_DTYPE:
		return E * ((float32) 1.0 / N);
		break;
	case FLOAT_64_DTYPE:
		return E * ((float64) 1.0 / N);
		break;
	default:
		throw runtime_error("invalid datatype: Invalid variable datatype");
		break;				
	}
}

template<typename T>
T cnet::cfuncs::MseDerivate(const T &A, const T &Y, std::size_t N)
{
	if (N == 0)
		throw invalid_argument("invalid argument: Invalid Empty set");
	return (A - Y) * (2.0 / N);
}

template Mat<float32> cnet::cfuncs::Mse(const Mat<float32> *A, const Mat<float32> *Y, size_t N);
template Mat<float64> cnet::cfuncs::Mse(const Mat<float64> *A, const Mat<float64> *Y, size_t N);

template Mat<float32> cnet::cfuncs::MseDerivate(const Mat<float32> &A, const Mat<float32> &Y, size_t N);
template Mat<float64> cnet::cfuncs::MseDerivate(const Mat<float64> &A, const Mat<float64> &Y, size_t N);

template Var cnet::cfuncs::MseDerivate(const Var &A, const Var &Y, size_t N);
