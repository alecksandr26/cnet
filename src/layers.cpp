#include <cstddef>
#include <functional>

#include "cnet/mat.hpp"
#include "cnet/variable.hpp"
#include "cnet/afuncs.hpp"
#include "cnet/weights.hpp"
#include "cnet/layers.hpp"

// #define DEFAULT_AMOUNT_OF_BATCHES 4
using namespace std;
using namespace cnet;
using namespace mathops;
using namespace variable;
using namespace layers;
using namespace afuncs;
using namespace weights;

// Base class to construct the ohter layers
cnet::layers::Layer::Layer(void)
{
	in_ = {0, 0};
	out_ = {0, 0};
	trainable_ = built_ = false;
	// The default value
	dtype_ = FLOAT_32_DTYPE;
}

cnet::layers::Layer::~Layer(void)
{
	
}

cnet::layers::Layer::Layer(bool trainable, Shape in, Shape out)
{
	trainable_ = trainable;
	in_ = in;
	out_ = out;
	// The default value
	dtype_ = FLOAT_32_DTYPE;
	built_ = false;
}

cnet::layers::Layer::Layer(bool trainable, std::size_t in, std::size_t out)
{
	trainable_ = trainable;
	in_ = {in, 1};
	out_ = {out, 1};
	// The default value
	dtype_ = FLOAT_32_DTYPE;
	built_ = false;
}

cnet::layers::Layer::Layer(bool trainable, Shape in, Shape out, CnetDtype dtype)
{
	trainable_ = trainable;
	in_ = in;
	out_ = out;
	dtype_ = dtype;
	built_ = false;
}

cnet::layers::Layer::Layer(bool trainable, std::size_t in, std::size_t out, CnetDtype dtype)
{
	trainable_ = trainable;
	in_ = {in, 1};
	out_ = {out, 1};
	dtype_ = dtype;
	built_ = false;
}

size_t cnet::layers::Layer::get_in_size(void) const
{
	return in_.rows * in_.cols;
}

size_t cnet::layers::Layer::get_out_size(void) const
{
	return out_.rows * out_.cols;
}

Shape cnet::layers::Layer::get_in_shape(void) const
{
	return in_;
}

Shape cnet::layers::Layer::get_out_shape(void) const
{
	return out_;
}

CnetDtype cnet::layers::Layer::get_dtype(void) const
{
	return dtype_;
}

bool cnet::layers::Layer::is_built(void) const
{
	return built_;
}

// It may be will require rebuild 
Layer &cnet::layers::Layer::set_in_size(size_t in_size)
{
	in_ = {in_size, 1};

	if (built_)
		build(in_);
	
	return *this;
}

Layer &cnet::layers::Layer::set_in_shape(Shape in)
{
	in_ = in;
	if (built_)
		build(in_);
	
	return *this;
}

// Dense layer it is a normal NN of the type Y = Act(W * X + B), where X is one dimension
// matrix
cnet::layers::Dense::Dense(void) : Layer(true, 0, 0)
{
	units_			   = 0;
	use_bias_		   = true;
	afunc_name_ = default_afunc_name;
}

cnet::layers::Dense::~Dense(void)
{
	
}

cnet::layers::Dense::Dense(size_t units) : Layer(true, 0, units)
{
	units_			   = units;
	use_bias_		   = true;
	afunc_name_ = default_afunc_name;
}

cnet::layers::Dense::Dense(size_t units, CnetDtype dtype) : Layer(true, 0, units, dtype)
{
	units_			   = units;
	use_bias_		   = true;
	afunc_name_ = default_afunc_name;
}

cnet::layers::Dense::Dense(size_t units, const string &afunc_name) : Layer(true, 0, units)
{
	units_			   = units;
	use_bias_		   = true;
	afunc_name_ = afunc_name;
}

cnet::layers::Dense::Dense(size_t units, const string &afunc_name,
			   CnetDtype dtype) : 	Layer(true, 0, units, dtype)
{
	units_			   = units;
	use_bias_		   = true;
	afunc_name_ = afunc_name;
}

size_t cnet::layers::Dense::get_units(void) const
{
	return units_;
}

size_t cnet::layers::Dense::get_weights(void) const
{
	return W_.get_weights();
}

Shape cnet::layers::Dense::get_weights_shape(void) const
{
	return W_.get_shape();
}

size_t cnet::layers::Dense::get_biases(void) const
{
	return B_.get_weights();
}

Shape cnet::layers::Dense::get_biases_shape(void) const
{
	return B_.get_shape();
}

bool cnet::layers::Dense::use_bias(void) const
{
	return use_bias_;
}

const string &cnet::layers::Dense::get_afunc_name(void) const
{
	return afunc_name_;
}

const Var &cnet::layers::Dense::get_cmat_weights(void) const
{
	return W_;
}

const Var &cnet::layers::Dense::get_cmat_biases(void) const
{
	return B_;
}

Weights &cnet::layers::Dense::get_mat_weights(void)
{
	return W_;
}

Weights &cnet::layers::Dense::get_mat_biases(void)
{
	return B_;
}

// To compute the derivate error for the previos layer
// a = f(z), where z = b + w_1 * i_1 + ... + w_n * i_n + ..., f is the activation function,
// dE = d(e)/d(a)
// I is the input from which we want its derivate, basically d(e)/d(i_k)
Error cnet::layers::Dense::get_derror_dinput(const Error &dE) const
{
	// d(a)/d(z)
	Var dZ = afunc_.afunc_derivate_(Z_);
	return static_cast<Error>(W_.get_derror_dinput(dE ^ dZ, in_));
}

Mat<float32> cnet::layers::Dense::get_derror_dinput(const Mat<float32> &dE) const
{
	// d(a)/d(z)
	Mat<float32> dZ = afunc_.afunc_derivate_(Z_).get_cmf32();
	return W_.get_derror_dinput(dE ^ dZ, in_);
}

Mat<float64> cnet::layers::Dense::get_derror_dinput(const Mat<float64> &dE) const
{
	Mat<float64> dZ = afunc_.afunc_derivate_(Z_).get_cmf64();
	return W_.get_derror_dinput(dE ^ dZ, in_);
}

Output cnet::layers::Dense::operator()(const Input &X)
{
	if (!built_)
		build(X.get_shape());
	
	Z_ = W_ * X;
	if (use_bias_)
		Z_ += B_;
	return static_cast<Output>(afunc_.afunc_(Z_));
}

Mat<float32> cnet::layers::Dense::operator()(const Mat<float32> &X)
{
	if (!built_)
		build(X.get_shape());

	Z_ = W_ * Var(X);
	if (use_bias_)
		Z_ += B_;
	return afunc_.afunc_(Z_).get_mf32();
}

Mat<float64> cnet::layers::Dense::operator()(const Mat<float64> &X)
{
	if (!built_)
		build(X.get_shape());

	Z_ = W_ * Var(X);
	if (use_bias_)
		Z_ += B_;
	return afunc_.afunc_(Z_).get_mf64();
}

Dense &cnet::layers::Dense::build(size_t in_size)
{
	if (units_ == 0)
		throw runtime_error("invalid layer: Layer is not initlized");
	if (in_size == 0)
		throw invalid_argument("invalid argument: Invalid input size Dense layer only supports n x 1 dimension input");
	
	in_ = {in_size, 1};
	W_.add_weights(dtype_, default_weights_name, {units_, in_size});
	if (use_bias_)
		B_.add_weights(dtype_, default_biases_name, {units_, 1});

	afunc_.alloc_afunc(afunc_name_);
	built_ = true;

	return *this;
}

Dense &cnet::layers::Dense::build(Shape in)
{
	if (units_ == 0)
		throw runtime_error("invalid layer: Layer is not initlized");
	if (in.cols > 1)
		throw invalid_argument("invalid argument: Dense layer only supports n x 1 dimension input");
	if (in.rows == 0)
		throw invalid_argument("invalid argument: Invalid input size Dense layer only supports n x 1 dimension input");
	
	in_ = in;
	W_.add_weights(dtype_, default_weights_name, {units_, in_.rows});
	if (use_bias_)
		B_.add_weights(dtype_, default_biases_name, {units_, 1});
	
	afunc_.alloc_afunc(afunc_name_);
	built_ = true;
	
	return *this;
}

// It may be will require rebuild 
Dense &cnet::layers::Dense::set_units(size_t units)
{
	units_ = units;
	built_ = false;
	return *this;
}

// It may be will require rebuild 
Dense &cnet::layers::Dense::set_use_bias(bool use_bias)
{
	use_bias_ = use_bias;
	built_ = false;
	return *this;
}

Dense &cnet::layers::Dense::set_afunc(const std::string &afunc_name)
{
	afunc_name_ = afunc_name;
	built_ = false;
	return *this;
}

Dense &cnet::layers::Dense::rand_uniform_range(float64 a, float64 b)
{
	if (!built_)
		build(in_);
	
	switch (dtype_) {
	case FLOAT_32_DTYPE:
		W_.rand_uniform_range((float32) a, (float32) b);
		if (use_bias_)
			B_.rand_uniform_range((float32) a, (float32) b);
		break;
	case FLOAT_64_DTYPE:
		W_.rand_uniform_range(a, b);
		if (use_bias_)
			B_.rand_uniform_range(a, b);
		break;
	default:
		runtime_error("runtime error: Invalid data dtype");
		break;
	}

	return *this;
}


// Fit backpropagation:
// a = f(z), where z = b + w_1 * i_1 + ... + w_n * i_n + ..., f is the activation function,
// a is the actual neuron and i is the input
// dE = d(e)/d(a)
// I is the input of this layer
Dense &cnet::layers::Dense::fit(const Mat<float32> &dE, const Mat<float32> &I, float64 lr)
{
	// d(a)/d(z)
	Mat<float32> dZ = afunc_.afunc_derivate_(Z_).get_cmf32();
	
	// dE * dZ = d(e)/d(z) = d(e)/d(a) * d(a)/d(z)
	// I = d(z)/d(w)
	// 1 = d(z)/d(b)
	W_.fit(dE ^ dZ, I, lr);
	if (use_bias_)
		B_.fit(dE ^ dZ, Mat<float32>(I.get_shape(), 1.0), lr);
	return *this;
}

Dense &cnet::layers::Dense::fit(const Mat<float64> &dE, const Mat<float64> &I, float64 lr)
{
	Mat<float64> dZ = afunc_.afunc_derivate_(Z_).get_cmf64();
	
	W_.fit(dE ^ dZ, I, lr);
	if (use_bias_)
		B_.fit(dE ^ dZ, Mat<float64>(I.get_shape(), 1.0), lr);
	return *this;
}

Dense &cnet::layers::Dense::fit(const Error &dE, const Input &I, float64 lr)
{
	const Var &dZ = afunc_.afunc_derivate_(Z_);
	
	W_.fit(dE * dZ, I, lr);
	if (use_bias_) {
		switch (B_.get_dtype()) {
		case FLOAT_32_DTYPE:
			B_.fit(dE * dZ, Var(I.get_shape(), (float32) 1.0), lr);
			break;
		case FLOAT_64_DTYPE:
			B_.fit(dE * dZ, Var(I.get_shape(),(float64) 1.0), lr);
			break;
		default:
			throw runtime_error("runtime error: Invalid datatype");
			break;			
		}

	}
	return *this;
}
