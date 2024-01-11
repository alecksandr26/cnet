#include "cnet/dtypes.hpp"
#include "cnet/mat.hpp"
#include "cnet/variable.hpp"

#include <type_traits>

using namespace std;
using namespace cnet;
using namespace variable;
using namespace mathops;
using namespace dtypes;

cnet::variable::Var::Var(void)
{
	dtype_ = FLOAT_32_DTYPE;
	has_name_ = false;
}

cnet::variable::Var::~Var(void)
{
	
}

cnet::variable::Var::Var(Shape shape)
{
	dtype_ = FLOAT_32_DTYPE;
	M_.f32.resize(shape);
	has_name_ = false;
}

cnet::variable::Var::Var(Shape shape, CnetDtype dtype)
{
	dtype_ = dtype;
	switch (dtype) {
	case FLOAT_32_DTYPE:
		M_.f32.resize(shape);
		break;
	case FLOAT_64_DTYPE:
		M_.f64.resize(shape);
		break;
	default:
		throw invalid_argument("invalid argument: Invalid datatype");
		break;
	}

	has_name_ = false;
}

cnet::variable::Var::Var(size_t rows, size_t cols)
{
	dtype_ = FLOAT_32_DTYPE;
	M_.f32.resize(rows, cols);
	has_name_ = false;
}

cnet::variable::Var::Var(size_t rows, size_t cols, CnetDtype dtype)
{
	dtype_ = dtype;
	switch (dtype) {
	case FLOAT_32_DTYPE:
		M_.f32.resize(rows, cols);
		break;
	case FLOAT_64_DTYPE:
		M_.f64.resize(rows, cols);
		break;
	default:
		throw invalid_argument("invalid argument: Invalid datatype");
		break;
	}
	has_name_ = false;
}

cnet::variable::Var::Var(Shape shape, float32 init_val)
{
	dtype_ = FLOAT_32_DTYPE;
	M_.f32.resize(shape, init_val);
	has_name_ = false;
}

cnet::variable::Var::Var(size_t rows, size_t cols, float32 init_val)
{
	dtype_ = FLOAT_32_DTYPE;
	M_.f32.resize(rows, cols, init_val);
	has_name_ = false;
}

cnet::variable::Var::Var(Shape shape, float64 init_val)
{
	dtype_ = FLOAT_64_DTYPE;
	M_.f64.resize(shape, init_val);
	has_name_ = false;
}

cnet::variable::Var::Var(size_t rows, size_t cols, float64 init_val)
{
	dtype_ = FLOAT_64_DTYPE;
	M_.f64.resize(rows, cols, init_val);
	has_name_ = false;
}

cnet::variable::Var::Var(const Mat<float32> &M)
{
	dtype_ = FLOAT_32_DTYPE;
	M_.f32(M);
	has_name_ = false;
}

cnet::variable::Var::Var(const Mat<float64> &M)
{
	dtype_ = FLOAT_64_DTYPE;
	M_.f64(M);
	has_name_ = false;
}

cnet::variable::Var::Var(const Var &V)
{
	dtype_ = V.get_dtype();
	switch (dtype_) {
	case FLOAT_32_DTYPE:
		M_.f32(V.get_cmf32());
		break;
	case FLOAT_64_DTYPE:
		M_.f64(V.get_cmf64());
		break;
	default:
		throw invalid_argument("invalid argument: Invalid datatype");
		break;
	}
	has_name_ = false;
}

Shape cnet::variable::Var::get_shape(void) const
{
	return M_.f32.get_shape();
}

size_t cnet::variable::Var::get_rows(void) const
{
	return M_.f32.get_rows();
}

size_t cnet::variable::Var::get_cols(void) const
{
	return M_.f32.get_cols();
}

CnetDtype cnet::variable::Var::get_dtype(void) const
{
	return dtype_;
}

const string &cnet::variable::Var::get_name(void) const
{
	return name_;
}

const Mat<float32> &cnet::variable::Var::get_cmf32(void) const
{
	if (dtype_ != FLOAT_32_DTYPE)
		throw runtime_error("runtime error: Invalid datatype");
	return M_.f32;
}

const Mat<float64> &cnet::variable::Var::get_cmf64(void) const
{
	if (dtype_ != FLOAT_64_DTYPE)
		throw runtime_error("runtime error: Invalid datatype");
	return M_.f64;
}

Mat<float32> &cnet::variable::Var::get_mf32(void)
{
	if (dtype_ != FLOAT_32_DTYPE)
		throw runtime_error("runtime error: Invalid datatype");
	return M_.f32;
}

Mat<float64> &cnet::variable::Var::get_mf64(void)
{
	if (dtype_ != FLOAT_64_DTYPE)
		throw runtime_error("invalid var: Invalid datatype");
	return M_.f64;
}


Var &cnet::variable::Var::set_name(const string &name)
{
	name_ = name;
	has_name_ = true;
	return *this;
}

// Use better uniform function
Var &cnet::variable::Var::rand_uniform_range(float32 a, float32 b)
{
	if (dtype_ != FLOAT_32_DTYPE)
		throw runtime_error("runtime error: Invalid datatype");

	M_.f32.rand(a, b);
	
	return *this;
}

Var &cnet::variable::Var::rand_uniform_range(float64 a, float64 b)
{
	if (dtype_ != FLOAT_64_DTYPE)
		throw runtime_error("runtime error: Invalid datatype");
	
	M_.f64.rand(a, b);
	return *this;
}


float32 &cnet::variable::Var::at_mf32(std::size_t i, std::size_t j)
{
	if (dtype_ != FLOAT_32_DTYPE)
		throw runtime_error("runtime error: Invalid datatype");
	return M_.f32(i, j);
}

float64 &cnet::variable::Var::at_mf64(std::size_t i, std::size_t j)
{
	if (dtype_ != FLOAT_64_DTYPE)
		throw runtime_error("runtime error: Invalid datatype");
	return M_.f64(i, j);
}

Var &cnet::variable::Var::resize(size_t rows, size_t cols)
{
	switch (dtype_) {
	case FLOAT_32_DTYPE:
		M_.f32.resize(rows, cols);
		break;
	case FLOAT_64_DTYPE:
		M_.f64.resize(rows, cols);
		break;
	default:
		throw invalid_argument("invalid argument: Invalid datatype");
		break;
	}

	return *this;
}
Var &cnet::variable::Var::resize(Shape shape)
{
	switch (dtype_) {
	case FLOAT_32_DTYPE:
		M_.f32.resize(shape);
		break;
	case FLOAT_64_DTYPE:
		M_.f64.resize(shape);
		break;
	default:
		throw invalid_argument("invalid argument: Invalid datatype");
		break;
	}

	return *this;
}

Var &cnet::variable::Var::resize(size_t rows, size_t cols, float32 init_val)
{
	if (dtype_ != FLOAT_32_DTYPE) {
		M_.f32.~Mat<float32>();
		M_.f32 = Mat<float32>(rows, cols, init_val);
	} else
		M_.f32.resize(rows, cols, init_val);
	
	dtype_ = FLOAT_32_DTYPE;

	return *this;
}

Var &cnet::variable::Var::resize(size_t rows, size_t cols, float64 init_val)
{
	if (dtype_ != FLOAT_64_DTYPE) {
		M_.f64.~Mat<float64>();
		M_.f64 = Mat<float64>(rows, cols, init_val);
	} else
		M_.f64.resize(rows, cols, init_val);
	dtype_ = FLOAT_64_DTYPE;
	
	return *this;
}

Var &cnet::variable::Var::resize(Shape shape, float32 init_val)
{
	if (dtype_ != FLOAT_32_DTYPE) {
		M_.f32.~Mat<float32>();
		M_.f32 = Mat<float32>(shape, init_val);
	} else
		M_.f32.resize(shape, init_val);
	
	dtype_ = FLOAT_32_DTYPE;
	
	return *this;
}

Var &cnet::variable::Var::resize(Shape shape, float64 init_val)
{
	if (dtype_ != FLOAT_64_DTYPE) {
		M_.f64.~Mat<float64>();
		M_.f64 = Mat<float64>(shape, init_val);
	} else
		M_.f64.resize(shape, init_val);
	dtype_ = FLOAT_64_DTYPE;
	
	return *this;
}

Var &cnet::variable::Var::operator()(const Mat<float32> &M)
{
	if (dtype_ != FLOAT_32_DTYPE) {
		M_.f32.~Mat<float32>();
		M_.f32(M);
	} else
		M_.f32(M);
	dtype_ = FLOAT_32_DTYPE;

	return *this;
}

Var &cnet::variable::Var::operator()(const Mat<float64> &M)
{
	if (dtype_ != FLOAT_64_DTYPE) {
		M_.f64.~Mat<float64>();
		M_.f64(M);
	} else
		M_.f64(M);
	dtype_ = FLOAT_64_DTYPE;
	
	return *this;
}


void cnet::variable::Var::operator=(const Var &V)
{
	
	switch (V.get_dtype()) {
	case FLOAT_32_DTYPE: {
		(*this)(V.get_cmf32());
		break;
	}
	case FLOAT_64_DTYPE: {
		(*this)(V.get_cmf64());
		break;
	}
	default:
		throw invalid_argument("invalid argument: Invalid datatype");
		break;
	}
}

void cnet::variable::Var::operator=(const Mat<float32> &M)
{
	if (dtype_ != FLOAT_32_DTYPE) {
		M_.f32.~Mat<float32>();
		M_.f32(M);
	} else
		M_.f32(M);
	dtype_ = FLOAT_32_DTYPE;
}

void cnet::variable::Var::operator=(const Mat<float64> &M)
{
	if (dtype_ != FLOAT_64_DTYPE) {
		M_.f64.~Mat<float64>();
		M_.f64(M);
	} else
		M_.f64(M);
	dtype_ = FLOAT_64_DTYPE;
}

Var cnet::variable::Var::operator*(const Var &V) const
{
	// Try to cast the matrices to be able to multiply different matrices datatypes
	if (V.get_dtype() != dtype_)
		throw invalid_argument("invalid argument: Variables don't have the same datatype");
	
	switch (dtype_) {
	case FLOAT_32_DTYPE:
		return Var(M_.f32 * V.get_cmf32());
		break;
	case FLOAT_64_DTYPE:
		return Var(M_.f64 * V.get_cmf64());
		break;
	default:
		throw invalid_argument("invalid argument: Invalid datatype");
		break;
	}
}

void cnet::variable::Var::operator*=(const Var &V)
{
	// Try to cast
	if (V.get_dtype() != dtype_)
		throw invalid_argument("invalid argument: Variables don't have the same datatype");

	switch (dtype_) {
	case FLOAT_32_DTYPE:
		(*this)(M_.f32 * V.get_cmf32());
		break;
	case FLOAT_64_DTYPE:
		(*this)(M_.f64 * V.get_cmf64());
		break;
	default:
		throw invalid_argument("invalid argument: Invalid datatype");
		break;
	}
}

Var cnet::variable::Var::operator+(const Var &V) const
{
	// Try to cast the matrices to be able to multiply different matrices datatypes
	if (V.get_dtype() != dtype_)
		throw invalid_argument("invalid argument: Variables don't have the same datatype");
	
	switch (dtype_) {
	case FLOAT_32_DTYPE:
		return Var(M_.f32 + V.get_cmf32());
		break;
	case FLOAT_64_DTYPE:
		return Var(M_.f64 + V.get_cmf64());
		break;
	default:
		throw invalid_argument("invalid argument: Invalid datatype");
		break;
	}
}

void cnet::variable::Var::operator+=(const Var &V)
{
	// Try to cast
	if (V.get_dtype() != dtype_)
		throw invalid_argument("invalid argument: Variables don't have the same datatype");

	switch (dtype_) {
	case FLOAT_32_DTYPE:
		(*this)(M_.f32 + V.get_cmf32());
		break;
	case FLOAT_64_DTYPE:
		(*this)(M_.f64 + V.get_cmf64());
		break;
	default:
		throw invalid_argument("invalid argument: Invalid datatype");
		break;
	}
}


Var cnet::variable::Var::operator-(const Var &V) const
{
	// Try to cast the matrices to be able to multiply different matrices datatypes
	if (V.get_dtype() != dtype_)
		throw invalid_argument("invalid argument: Variables don't have the same datatype");
	
	switch (dtype_) {
	case FLOAT_32_DTYPE:
		return Var(M_.f32 - V.get_cmf32());
		break;
	case FLOAT_64_DTYPE:
		return Var(M_.f64 - V.get_cmf64());
		break;
	default:
		throw invalid_argument("invalid argument: Invalid datatype");
		break;
	}
}

void cnet::variable::Var::operator-=(const Var &V)
{
	// Try to cast
	if (V.get_dtype() != dtype_)
		throw invalid_argument("invalid argument: Variables don't have the same datatype");

	switch (dtype_) {
	case FLOAT_32_DTYPE:
		(*this)(M_.f32 - V.get_cmf32());
		break;
	case FLOAT_64_DTYPE:
		(*this)(M_.f64 - V.get_cmf64());
		break;
	default:
		throw invalid_argument("invalid argument: Invalid datatype");
		break;
	}
}



Var cnet::variable::Var::operator^(const Var &V) const
{
	// Try to cast the matrices to be able to multiply different matrices datatypes
	if (V.get_dtype() != dtype_)
		throw invalid_argument("invalid argument: Variables don't have the same datatype");
	
	switch (dtype_) {
	case FLOAT_32_DTYPE:
		return Var(M_.f32 ^ V.get_cmf32());
		break;
	case FLOAT_64_DTYPE:
		return Var(M_.f64 ^ V.get_cmf64());
		break;
	default:
		throw invalid_argument("invalid argument: Invalid datatype");
		break;
	}
}

void cnet::variable::Var::operator^=(const Var &V)
{
	// Try to cast
	if (V.get_dtype() != dtype_)
		throw invalid_argument("invalid argument: Variables don't have the same datatype");

	switch (dtype_) {
	case FLOAT_32_DTYPE:
		(*this)(M_.f32 ^ V.get_cmf32());
		break;
	case FLOAT_64_DTYPE:
		(*this)(M_.f64 ^ V.get_cmf64());
		break;
	default:
		throw invalid_argument("invalid argument: Invalid datatype");
		break;
	}
}

Var cnet::variable::Var::operator*(float32 val) const
{
	// Try to cast the matrices to be able to multiply different matrices datatypes
	if (FLOAT_32_DTYPE != dtype_)
		throw invalid_argument("invalid argument: Variables don't have the same datatype");
	
	
	return Var(M_.f32 * val);
}

void cnet::variable::Var::operator*=(float32 val)
{
	// Try to cast the matrices to be able to multiply different matrices datatypes
	if (FLOAT_32_DTYPE != dtype_)
		throw invalid_argument("invalid argument: Variables don't have the same datatype");

	(*this)(M_.f32 * val);
}

Var cnet::variable::Var::operator*(float64 val) const
{
	// Try to cast the matrices to be able to multiply different matrices datatypes
	if (FLOAT_64_DTYPE != dtype_)
		throw invalid_argument("invalid argument: Variables don't have the same datatype");
	
	
	return Var(M_.f64 * val);
}

void cnet::variable::Var::operator*=(float64 val)
{
	// Try to cast the matrices to be able to multiply different matrices datatypes
	if (FLOAT_64_DTYPE != dtype_)
		throw invalid_argument("invalid argument: Variables don't have the same datatype");

	(*this)(M_.f64 * val);
}
