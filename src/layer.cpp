#include "cnet/mat.hpp"
#include "cnet/afunc.hpp"
#include "cnet/layer.hpp"

#include <memory>
#include <complex>
#include <cstddef>
#include <unordered_map>
#include <functional>
#include <utility>
#include <cassert>

#include <immintrin.h> // For AVX2 intrinsics
#include <omp.h>

#define DEFAULT_AMOUNT_OF_BATCHES 4

// This is super shitty we need to improve this thing
template<typename T=double>
static std::unique_ptr<cnet::afunc::afunc<T>> create_afunc(const std::string &name)
{
	static const std::unordered_map<std::string,
					std::function<std::unique_ptr<cnet::afunc::afunc<T>>()>> afunc_map = {
		{
			"linear",
			[]() {
				return std::make_unique<cnet::afunc::linear<T>>();
			}
		},
		{
			"sigmoid",
			[]() {
				return std::make_unique<cnet::afunc::sigmoid<T>>();
			}
		},
		{
			"relu",
			[]() {
				return std::make_unique<cnet::afunc::relu<T>>();
			}
		},
	};

	auto it = afunc_map.find(name);
	if (it == afunc_map.end()) {
		// Handle unknown activation function type
		throw std::runtime_error("Unknown activation function: " + name);
	}
	
	return it->second();
}

template<class T>
cnet::layer::dense<T>::dense(std::size_t units)
{
	units_   = units;
	func_ = std::make_unique<cnet::afunc::linear<T>>();
	in_ = 0;
	cnet::layer::layer<T>::built_ = false;
	use_bias_ = true;
}

template<class T>
cnet::layer::dense<T>::dense(std::size_t units, const std::string &afunc_name)
{
	units_   = units;
	func_ = create_afunc(afunc_name);
	in_ = 0;
	cnet::layer::layer<T>::built_ = false;
	use_bias_ = true;
}

template<class T>
cnet::layer::dense<T>::dense(void)
{
	units_ = in_ = 0;
	func_ = NULL;
	cnet::layer::layer<T>::built_ = false;
	use_bias_ = true;
}

template<class T>
cnet::layer::dense<T>::~dense(void)
{
	units_ = in_ = 0;
	cnet::layer::layer<T>::built_ = false;
	func_ = NULL;
}

template<class T>
void cnet::layer::dense<T>::build(std::size_t in_size)
{
	if (units_ == 0)
		throw std::invalid_argument("invalid layer: Layer is not initlized");

	in_ = in_size;
	W_.resize(units_, in_);
	
	if (use_bias_)
		B_.resize(units_, 1);
	
	// By default initi the values between 0.0 to 1.0
	W_.rand(0.0, 1.0);
	if (use_bias_)
		B_.rand(0.0, 1.0);
	
	cnet::layer::layer<T>::built_ = true;
}

template<class T>
void cnet::layer::dense<T>::build(std::size_t in_size, T init_val)
{
	if (units_ == 0)
		throw std::invalid_argument("invalid layer: Layer is not initlized");

	in_ = in_size;
	W_.resize(units_, in_, init_val);
	if (use_bias_)
		B_.resize(units_, 1, init_val);
	cnet::layer::layer<T>::built_ = true;
}

template<class T>
void cnet::layer::dense<T>::rand_range(T a, T b)
{
	W_.rand(a, b);
	
	if (use_bias_)
		B_.rand(a, b);
}

template<class T>
cnet::mat<T> cnet::layer::dense<T>::operator()(const cnet::mat<T> &X)
{
	// Build the layer
	if (!cnet::layer::layer<T>::built_)
		build(X.get_rows());
	
	if (X.get_cols() != 1 || X.get_rows() != W_.get_cols())
		throw std::invalid_argument("invalid argument: Matrices has different sizes");
	
	// The feedforward operation
	if (!use_bias_)
		return (*func_)(W_ * X);
	
	return (*func_)(W_ * X + B_);
}

template<class T>
void cnet::layer::dense<T>::fit_backprop(const cnet::mat<T> &E, double lr, const cnet::mat<T> &A)
{
	if (units_ == 0)
		throw std::invalid_argument("invalid layer: Layer is not initlized");
	
	if (E.get_cols() != 1 || A.get_cols() != 1 || A.get_rows() != W_.get_cols())
		throw std::invalid_argument("invalid argument: Matrices has different sizes");
	
	// Update the bias by error
	cnet::mat<T> df_dx = func_->derivate(W_ * A + B_);
	if (use_bias_)
		B_ -= df_dx ^ (E * lr);

	std::size_t rows = W_.get_rows(), cols = W_.get_cols();
	std::size_t n = rows * cols;
	
	T *w_mat = W_.get_mat_alloc();
	T *e_mat = E.get_mat_alloc();
	T *df_dx_mat = df_dx.get_mat_alloc();
	T *a_mat = A.get_mat_alloc();
	
	switch (n) {
	case 1: {
		vec2double v = _mm_set_pd(lr, e_mat[0]);
		vec2double v2 = _mm_set_pd(df_dx_mat[0], a_mat[0]);
		vec2double v3 = _mm_mul_pd(v, v2);
		
		w_mat[0] -= v3[0] * v3[1];
		break;
	}
	case 2: {
		int index = 1 / cols;
		vec4double v = _mm256_set_pd(lr, e_mat[0], lr, e_mat[index]);
		vec4double v2 = _mm256_set_pd(df_dx_mat[0], a_mat[0], df_dx_mat[index], a_mat[index]);
		vec4double v3 = _mm256_mul_pd(v, v2);
		
		w_mat[0] -= v3[0] * v3[1];
		w_mat[1] -= v3[1] * v3[2];
		break;
		
	}
	case 3: {
		int index = 1 / cols;
		vec4double v = _mm256_set_pd(lr, e_mat[0], lr, e_mat[index]);
		vec4double v2 = _mm256_set_pd(df_dx_mat[0], a_mat[0], df_dx_mat[index], a_mat[index]);
		vec4double v3 = _mm256_mul_pd(v, v2);
		
		w_mat[0] -= v2[0] * v3[1];
		w_mat[1] -= v2[1] * v3[2];

		vec2double v4 = _mm_set_pd(lr, e_mat[0]);
		vec2double v5 = _mm_set_pd(df_dx_mat[0], a_mat[0]);
		vec2double v6 = _mm_mul_pd(v4, v5);
		w_mat[2] -= v6[0] * v6[1];
		break;
	}
	default: {
				
		vec4double lr_vec_4 = _mm256_set_pd(lr, lr, lr, lr);
		vec2double lr_vec_2 = _mm_set_pd(lr, lr);
		
		std::size_t cols_ite_4 = cols - (cols % 4);
		std::size_t cols_ite_2 = cols - (cols % 2);

#pragma omp parallel for
		for (std::size_t i = 0; i < rows; i++) {
			// In the case 4 by 4 cols
			vec4double E_vec_4 = _mm256_set_pd(e_mat[i], e_mat[i],
							   e_mat[i], e_mat[i]);
			vec4double df_dx_vec_4 = _mm256_set_pd(df_dx_mat[i], df_dx_mat[i],
							       df_dx_mat[i], df_dx_mat[i]);
			// lr * E
			vec4double first_part_vec_4 = _mm256_mul_pd(lr_vec_4, E_vec_4);
			
			// In the case 2 by 2 cols
			vec2double E_vec_2 = _mm_set_pd(e_mat[i], e_mat[i]);
			vec2double df_dx_vec_2 = _mm_set_pd(df_dx_mat[i], df_dx_mat[i]);
			// lr * E
			vec2double first_part_vec_2 = _mm_mul_pd(lr_vec_2, E_vec_2);

			// In the case one by one cols
			vec2double v = _mm_set_pd(lr, e_mat[i]);
			
			for (std::size_t j = 0; j < cols_ite_4; j += 4) {
				vec4double W_vec = _mm256_loadu_pd(&w_mat[i * cols + j]);
				vec4double A_vec = _mm256_loadu_pd(&a_mat[j]);
				// W -= lr * E * df_dx * A
				W_vec = _mm256_sub_pd(W_vec, _mm256_mul_pd(first_part_vec_4,
									   _mm256_mul_pd(df_dx_vec_4,
											 A_vec)));
				_mm256_storeu_pd(&w_mat[i * cols + j], W_vec);
			}

			for (std::size_t j = cols_ite_4; j < cols_ite_2; j += 2) {
				vec2double W_vec = _mm_loadu_pd(&w_mat[i * cols + j]);
				vec2double A_vec = _mm_loadu_pd(&a_mat[j]);
				
				vec2double sub = _mm_mul_pd(first_part_vec_2,
							    _mm_mul_pd(df_dx_vec_2, A_vec));
				// W -= lr * E * df_dx * A
				W_vec = _mm_sub_pd(W_vec, sub);
				_mm_storeu_pd(&w_mat[i * cols + j], W_vec);
			}

			for (std::size_t j = cols_ite_2; j < cols; j++) {
				vec2double v2 = _mm_set_pd(df_dx_mat[i], a_mat[j]);
				vec2double v3 = _mm_mul_pd(v, v2);
				w_mat[i * cols + j] -= v3[0] * v3[1];
			}
		}
		break;
	}
	}
	
	// // Update the weights by error
	// for (std::size_t i = 0; i < W_.get_rows(); i++)
	// 	for (std::size_t j = 0; j < W_.get_cols(); j++)
	// 		W_(i, j) -= lr * E(i, 0) * df_dx(i, 0) * A(j, 0);
}

template<class T>
std::size_t cnet::layer::dense<T>::get_units(void) const
{
	return units_;
}

template<class T>
std::size_t cnet::layer::dense<T>::get_out_size(void) const
{
	return units_;
}

template<class T>
std::size_t cnet::layer::dense<T>::get_in_size(void) const
{
	return in_;
}

template<class T>
std::size_t cnet::layer::dense<T>::get_use_bias(void) const
{
	return use_bias_;
}

template<class T>
cnet::mat<T> cnet::layer::dense<T>::get_weights(void) const
{
	return W_;
}

template<class T>
cnet::mat<T> cnet::layer::dense<T>::get_biases(void) const
{
	return B_;
}


template<class T>
void cnet::layer::dense<T>::set_units(std::size_t units)
{
	units_ = units;
	
	// Needs to rebuild the layer
	cnet::layer::layer<T>::built_ = false;
}

template<class T>
void cnet::layer::dense<T>::set_afunc(const std::string &afunc_name)
{
	func_ = create_afunc(afunc_name);
}

template<class T>
void cnet::layer::dense<T>::set_use_bias(bool use_bias)
{
	use_bias_ = use_bias;
}



template<class T>
cnet::layer::input<T>::input(void)
{
	cnet::layer::layer<T>::built_ = false;
	batches_ = out_ = in_shape_.second = in_shape_.first = 0;
	A_ = NULL;
}

template<class T>
cnet::layer::input<T>::~input(void)
{
	if (A_)
		delete A_;
}


template<class T>
cnet::layer::input<T>::input(std::size_t in_size)
{
	cnet::layer::layer<T>::built_ = false;
	out_ = in_shape_.first = in_size; // Rows
	in_shape_.second = 1;	    // Cols
	batches_ = DEFAULT_AMOUNT_OF_BATCHES;
	A_ = NULL;
}

template<class T>
cnet::layer::input<T>::input(std::size_t in_size, std::size_t batches)
{
	cnet::layer::layer<T>::built_ = false;
	out_ = in_shape_.first = in_size; // Rows
	in_shape_.second = 1;	    // Cols
	batches_ = batches;
	A_ = NULL;

	assert(0 && "Not implemented yet");
}

template<class T>
cnet::layer::input<T>::input(std::pair<std::size_t, std::size_t> in_shape)
{
	cnet::layer::layer<T>::built_ = false;
	in_shape_.first = in_shape.first;            // Rows
	in_shape_.second = in_shape.second;	    // Cols
	batches_ = DEFAULT_AMOUNT_OF_BATCHES;

	out_ = in_shape_.first * in_shape_.second;
	A_ = NULL;
	
	assert(0 && "Not implemented yet");
}

template<class T>
cnet::layer::input<T>::input(std::pair<std::size_t, std::size_t> in_shape, std::size_t batches)
{
	cnet::layer::layer<T>::built_ = false;
	in_shape_.first = in_shape.first;            // Rows
	in_shape_.second = in_shape.second;	    // Cols
	batches_ = batches;

	out_ = in_shape_.first * in_shape_.second;
	A_ = NULL;

	assert(0 && "Not implemented yet");
}

template<class T>
cnet::mat<T> cnet::layer::input<T>::operator()(const cnet::mat<T> &X)
{
	// Here it will transform the input to a one dimension output
	// For the moment just run like this
	
	return X;
}

template<class T>
void cnet::layer::input<T>::build(void)
{
	if (out_ == 0)
		throw std::invalid_argument("invalid layer: Layer is not initlized");

	// Alloc the matrices by the batch size
	
	cnet::layer::layer<T>::built_ = true;
	
}

template<class T>
std::size_t cnet::layer::input<T>::get_out_size(void) const
{
	return out_;
}

template<class T>
std::size_t cnet::layer::input<T>::get_in_size(void) const
{
	return in_shape_.first * in_shape_.second;;
}

template<class T>
std::pair<std::size_t, std::size_t> cnet::layer::input<T>::get_in_shape(void) const
{
	return in_shape_;
}

template class cnet::layer::layer<double>;
template class cnet::layer::trainable_layer<double>;
template class cnet::layer::nontrainable_layer<double>;
template class cnet::layer::dense<double>;
template class cnet::layer::input<double>;
template static std::unique_ptr<cnet::afunc::afunc<double>> create_afunc(const std::string &name);
