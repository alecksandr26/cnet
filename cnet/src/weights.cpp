#include "cnet/dtypes.hpp"
#include "cnet/mat.hpp"
#include "cnet/variable.hpp"
#include "cnet/weights.hpp"
#include "cnet/backprop.hpp"

using namespace std;
using namespace cnet;
using namespace dtypes;
using namespace mathops;
using namespace variable;
using namespace weights;

size_t cnet::weights::Weights::get_weights(void) const
{
	return get_rows() * get_cols();
}

// To compute the derivate error for the previos layer
// a = f(z), where z = b + w_1 * i_1 + ... + w_n * i_n + ..., f is the activation function,
// dE = d(e)/d(z) = d(e)/d(a) * d(a)/d(z),
// I is the input from which we want its derivate, basically d(e)/d(i_k)
Var cnet::weights::Weights::get_derror_dinput(const Var &dE, Shape in) const
{
	Var dI(in);
	Shape s = get_shape();
	
	switch (dtype_) {
	case FLOAT_32_DTYPE: {
		Mat<float32> &M = dI.get_mf32();
		backprop::get_derror_dinput(dE.get_cmf32().get_allocated_mat(), M_.f32.get_allocated_mat(),
				  M.get_allocated_mat(), s.rows, s.cols);
		break;
	}
	case FLOAT_64_DTYPE: {
		Mat<float64> &M = dI.get_mf64();
		backprop::get_derror_dinput(dE.get_cmf64().get_allocated_mat(), M_.f64.get_allocated_mat(),
					    M.get_allocated_mat(), s.rows, s.cols);
		break;
	}
	default:
		throw runtime_error("runtime error: Invalid datatype");
		break;
	}
	
	return dI;
}

Mat<float32> cnet::weights::Weights::get_derror_dinput(const Mat<float32> &dE, Shape in) const
{
	Mat<float32> dI(in);
	Shape s = get_shape();
	backprop::get_derror_dinput(dE.get_allocated_mat(), M_.f32.get_allocated_mat(),
				    dI.get_allocated_mat(), s.rows, s.cols);
	return dI;
}

Mat<float64> cnet::weights::Weights::get_derror_dinput(const Mat<float64> &dE, Shape in) const
{
	Mat<float64> dI(in);
	Shape s = get_shape();
	backprop::get_derror_dinput(dE.get_allocated_mat(), M_.f64.get_allocated_mat(),
				    dI.get_allocated_mat(), s.rows, s.cols);
	return dI;
}

Weights &cnet::weights::Weights::fit(const Var &dE, const Var &dW, float64 lr)
{
	switch (dtype_) {
	case FLOAT_32_DTYPE: {
		Mat<float32> &W = get_mf32();
		backprop::fit_weights(dE.get_cmf32().get_allocated_mat(), dW.get_cmf32().get_allocated_mat(),
				     W.get_allocated_mat(), W.get_rows(), W.get_cols(), lr);
		break;
	}
	case FLOAT_64_DTYPE: {
		Mat<float64> &W = get_mf64();
		backprop::fit_weights(dE.get_cmf64().get_allocated_mat(), dW.get_cmf64().get_allocated_mat(),
				     W.get_allocated_mat(), W.get_rows(), W.get_cols(), lr);
		break;
	}
	default:
		throw runtime_error("runtime error: Invalid datatype");
		break;
	}

	return *this;
}

Weights &cnet::weights::Weights::fit(const Mat<float32> &dE, const Mat<float32> &dW, float64 lr)
{
	Mat<float32> &W = get_mf32();
	backprop::fit_weights(dE.get_allocated_mat(), dW.get_allocated_mat(),
			     W.get_allocated_mat(), W.get_rows(), W.get_cols(), lr);
	return *this;
}

Weights &cnet::weights::Weights::fit(const Mat<float64> &dE, const Mat<float64> &dW, float64 lr)
{
	Mat<float64> &W = get_mf64();
	backprop::fit_weights(dE.get_allocated_mat(), dW.get_allocated_mat(),
			     W.get_allocated_mat(), W.get_rows(), W.get_cols(), lr);
	return *this;
}

Weights &cnet::weights::Weights::add_weights(CnetDtype dtype, Shape shape)
{
	switch (dtype) {
	case FLOAT_32_DTYPE:
		M_.f32.resize(shape);
		break;
	case FLOAT_64_DTYPE:
		M_.f64.resize(shape);
		break;
	default:
		throw runtime_error("runtime error: Invalid datatype");
		break;
	}
	
	dtype_ = dtype;
	
	return *this;
}

Weights &cnet::weights::Weights::add_weights(CnetDtype dtype, const string &name, Shape shape)
{
	switch (dtype) {
	case FLOAT_32_DTYPE:
		M_.f32.resize(shape);
		break;
	case FLOAT_64_DTYPE:
		M_.f64.resize(shape);
		break;
	default:
		throw runtime_error("runtime error: Invalid datatype");
		break;
	}
	
	dtype_ = dtype;
	set_name(name);
	
	return *this;
}
