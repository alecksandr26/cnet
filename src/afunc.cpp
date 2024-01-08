#include "cnet/afunc.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>

// To optimize the matrix operations and the parallelization of it
#include <immintrin.h>	      // For AVX2 intrinsics
#include <omp.h>

// my own implementation of _mm256_exp_pd
inline cnet::vec4double _mm256_exp_pd(cnet::vec4double vec)
{
	return _mm256_set_pd(std::exp(vec[0]), std::exp(vec[1]), std::exp(vec[2]),
			     std::exp(vec[3]));
}

template<class T>
cnet::mat<T> cnet::afunc::linear<T>::operator()(const cnet::mat<T> &X) const
{
	return cnet::mat<T>(X);
}

template<class T>
cnet::mat<T> cnet::afunc::linear<T>::derivate(const cnet::mat<T> &X) const
{
	return cnet::mat<T>(X.get_rows(), X.get_cols(), 1.0);
}

template<class T>
cnet::mat<T> cnet::afunc::sigmoid<T>::operator()(const cnet::mat<T> &X) const
{
	std::size_t row_ = X.get_rows();
	std::size_t col_ = X.get_cols();

	cnet::mat<T> Y(row_, col_);

	T *x_mat_alloc = X.get_mat_alloc();
	T *y_mat_alloc = Y.get_mat_alloc();

	std::size_t n		= row_ * col_;
	std::size_t n_ite_8	= n - (n % 8);
	std::size_t n_ite_4	= n - (n % 4);
	vec4double  vec_one	= _mm256_set_pd(1.0, 1.0, 1.0, 1.0);
	vec4double  vec_neg_one = _mm256_set_pd(-1.0, -1.0, -1.0, -1.0);

	switch (n) {
	case 1: y_mat_alloc[0] = (1.0 / (1.0 + exp(-x_mat_alloc[0]))); break;
	case 2:
		y_mat_alloc[0] = (1.0 / (1.0 + exp(-x_mat_alloc[0])));
		y_mat_alloc[1] = (1.0 / (1.0 + exp(-x_mat_alloc[1])));
		break;
	case 3:
		y_mat_alloc[0] = (1.0 / (1.0 + exp(-x_mat_alloc[0])));
		y_mat_alloc[1] = (1.0 / (1.0 + exp(-x_mat_alloc[1])));
		y_mat_alloc[2] = (1.0 / (1.0 + exp(-x_mat_alloc[2])));
		break;
	case 4:
	case 5:
	case 6:
	case 7:
#pragma omp parallel for
		for (std::size_t i = 0; i < n_ite_4; i += 4) {
			// Load 4 elements from the matrix
			vec4double vec = _mm256_loadu_pd(&x_mat_alloc[i]);

			// Compute sigmoid for each element
			vec = _mm256_div_pd(
				vec_one,
				_mm256_add_pd(vec_one, _mm256_exp_pd(_mm256_mul_pd(
							       vec_neg_one, vec))));

			// Store the result back to the matrix
			_mm256_storeu_pd(&y_mat_alloc[i],
					 _mm256_set_pd(vec[0], vec[1], vec[2], vec[3]));
		}

		for (std::size_t i = n_ite_4; i < n; i++)
			y_mat_alloc[i] = (1.0 / (1 + exp(-x_mat_alloc[i])));

		break;
	default:
#pragma omp parallel for
		for (std::size_t i = 0; i < n_ite_8; i += 8) {
			// Load 4 elements from the matrix
			vec4double vec = _mm256_loadu_pd(&x_mat_alloc[i]);

			// Compute sigmoid for each element
			vec = _mm256_div_pd(
				vec_one,
				_mm256_add_pd(vec_one, _mm256_exp_pd(_mm256_mul_pd(
							       vec_neg_one, vec))));

			// Store the result back to the matrix
			_mm256_storeu_pd(&y_mat_alloc[i],
					 _mm256_set_pd(vec[0], vec[1], vec[2], vec[3]));

			// Load 4 elements from the matrix
			vec = _mm256_loadu_pd(&x_mat_alloc[i + 4]);

			// Compute sigmoid for each element
			vec = _mm256_div_pd(
				vec_one,
				_mm256_add_pd(vec_one, _mm256_exp_pd(_mm256_mul_pd(
							       vec_neg_one, vec))));

			// Store the result back to the matrix
			_mm256_storeu_pd(&y_mat_alloc[i + 4],
					 _mm256_set_pd(vec[0], vec[1], vec[2], vec[3]));
		}

		for (std::size_t i = n_ite_8; i < n; i++)
			y_mat_alloc[i] = (1.0 / (1 + exp(-x_mat_alloc[i])));

		break;
	}

	return Y;
}

template<class T>
cnet::mat<T> cnet::afunc::sigmoid<T>::derivate(const cnet::mat<T> &X) const
{
	std::size_t row_ = X.get_rows();
	std::size_t col_ = X.get_cols();

	cnet::mat<T> Y(row_, col_);

	T *x_mat_alloc = X.get_mat_alloc();
	T *y_mat_alloc = Y.get_mat_alloc();

	std::size_t n		= row_ * col_;
	std::size_t n_ite_8	= n - (n % 8);
	std::size_t n_ite_4	= n - (n % 4);
	vec4double  vec_one	= _mm256_set_pd(1.0, 1.0, 1.0, 1.0);
	vec4double  vec_two	= _mm256_set_pd(2.0, 2.0, 2.0, 2.0);
	vec4double  vec_neg_one = _mm256_set_pd(-1.0, -1.0, -1.0, -1.0);

	switch (n) {
	case 1:
		y_mat_alloc[0] =
			(1.0 / (2.0 + exp(-x_mat_alloc[0]) + exp(x_mat_alloc[0])));
		break;
	case 2:
		y_mat_alloc[0] =
			(1.0 / (2.0 + exp(-x_mat_alloc[0]) + exp(x_mat_alloc[0])));
		y_mat_alloc[1] =
			(1.0 / (2.0 + exp(-x_mat_alloc[1]) + exp(x_mat_alloc[1])));
		break;
	case 3:
		y_mat_alloc[0] =
			(1.0 / (2.0 + exp(-x_mat_alloc[0]) + exp(x_mat_alloc[0])));
		y_mat_alloc[1] =
			(1.0 / (2.0 + exp(-x_mat_alloc[1]) + exp(x_mat_alloc[1])));
		y_mat_alloc[2] =
			(1.0 / (2.0 + exp(-x_mat_alloc[2]) + exp(x_mat_alloc[2])));
		break;
	case 4:
	case 5:
	case 6:
	case 7:
#pragma omp parallel for
		for (std::size_t i = 0; i < n_ite_4; i += 4) {
			// Load 4 elements from the matrix
			vec4double vec = _mm256_loadu_pd(&x_mat_alloc[i]);

			// Compute derivate of sigmoid for each element
			vec4double exp_neg_x =
				_mm256_exp_pd(_mm256_mul_pd(vec_neg_one, vec));
			vec4double exp_x = _mm256_exp_pd(vec);

			vec4double denominator =
				_mm256_add_pd(_mm256_add_pd(vec_two, exp_neg_x), exp_x);
			vec = _mm256_div_pd(vec_one, denominator);

			// Store the result back to the matrix
			_mm256_storeu_pd(&y_mat_alloc[i],
					 _mm256_set_pd(vec[0], vec[1], vec[2], vec[3]));
		}

		for (std::size_t i = n_ite_4; i < n; i++)
			y_mat_alloc[i] = (1.0 / (2.0 + exp(-x_mat_alloc[i]) +
						 exp(x_mat_alloc[i])));

		break;
	default:
#pragma omp parallel for
		for (std::size_t i = 0; i < n_ite_8; i += 8) {
			// Load 4 elements from the matrix
			vec4double vec = _mm256_loadu_pd(&x_mat_alloc[i]);

			// Compute derivate of sigmoid for each element
			vec4double exp_neg_x =
				_mm256_exp_pd(_mm256_mul_pd(vec_neg_one, vec));
			vec4double exp_x = _mm256_exp_pd(vec);

			vec4double denominator =
				_mm256_add_pd(_mm256_add_pd(vec_two, exp_neg_x), exp_x);
			vec = _mm256_div_pd(vec_one, denominator);

			// Store the result back to the matrix
			_mm256_storeu_pd(&y_mat_alloc[i],
					 _mm256_set_pd(vec[0], vec[1], vec[2], vec[3]));

			// Load 4 elements from the matrix
			vec = _mm256_loadu_pd(&x_mat_alloc[i + 4]);

			// Compute derivate of sigmoid for each element
			exp_neg_x = _mm256_exp_pd(_mm256_mul_pd(vec_neg_one, vec));
			exp_x	  = _mm256_exp_pd(vec);

			denominator =
				_mm256_add_pd(_mm256_add_pd(vec_two, exp_neg_x), exp_x);
			vec = _mm256_div_pd(vec_one, denominator);

			// Store the result back to the matrix
			_mm256_storeu_pd(&y_mat_alloc[i + 4],
					 _mm256_set_pd(vec[0], vec[1], vec[2], vec[3]));
		}

		for (std::size_t i = n_ite_8; i < n; i++)
			y_mat_alloc[i] = (1.0 / (2.0 + exp(-x_mat_alloc[i]) +
						 exp(x_mat_alloc[i])));

		break;
	}

	return Y;
}

template<class T>
cnet::mat<T> cnet::afunc::relu<T>::operator()(const cnet::mat<T> &X) const
{
	std::size_t row_ = X.get_rows();
	std::size_t col_ = X.get_cols();

	cnet::mat<T> Y(row_, col_);

	T *x_mat_alloc = X.get_mat_alloc();
	T *y_mat_alloc = Y.get_mat_alloc();

	std::size_t n	     = row_ * col_;
	std::size_t n_ite_8  = n - (n % 8);
	std::size_t n_ite_4  = n - (n % 4);
	vec4double  vec_zero = _mm256_setzero_pd();

	switch (n) {
	case 1: y_mat_alloc[0] = std::max(x_mat_alloc[0], 0.0); break;
	case 2:
		y_mat_alloc[0] = std::max(x_mat_alloc[0], 0.0);
		y_mat_alloc[1] = std::max(x_mat_alloc[1], 0.0);
		break;
	case 3:
		y_mat_alloc[0] = std::max(x_mat_alloc[0], 0.0);
		y_mat_alloc[1] = std::max(x_mat_alloc[1], 0.0);
		y_mat_alloc[2] = std::max(x_mat_alloc[2], 0.0);
		break;
	case 4:
	case 5:
	case 6:
	case 7:
#pragma omp parallel for
		for (std::size_t i = 0; i < n_ite_4; i += 4) {
			// Load 4 elements from the matrix
			vec4double vec = _mm256_loadu_pd(&x_mat_alloc[i]);

			// Compute relu for each element
			vec = _mm256_max_pd(vec, vec_zero);

			// Store the result back to the matrix
			_mm256_storeu_pd(&y_mat_alloc[i], vec);
		}

		for (std::size_t i = n_ite_4; i < n; i++)
			y_mat_alloc[i] = std::max(x_mat_alloc[i], 0.0);

		break;
	default:
#pragma omp parallel for
		for (std::size_t i = 0; i < n_ite_8; i += 8) {
			// Load 4 elements from the matrix
			vec4double vec = _mm256_loadu_pd(&x_mat_alloc[i]);

			// Compute relu for each element
			vec = _mm256_max_pd(vec, vec_zero);

			// Store the result back to the matrix
			_mm256_storeu_pd(&y_mat_alloc[i], vec);

			// Load 4 elements from the matrix
			vec = _mm256_loadu_pd(&x_mat_alloc[i + 4]);

			// Compute relu for each element
			vec = _mm256_max_pd(vec, vec_zero);

			// Store the result back to the matrix
			_mm256_storeu_pd(&y_mat_alloc[i + 4], vec);
		}

		for (std::size_t i = n_ite_8; i < n; i++)
			y_mat_alloc[i] = std::max(x_mat_alloc[i], 0.0);

		break;
	}

	return Y;
}

template<class T>
cnet::mat<T> cnet::afunc::relu<T>::derivate(const cnet::mat<T> &X) const
{
	// super funny but yea the derivate of relu is `1`
	return cnet::mat<T>(X.get_rows(), X.get_cols(), 1.0);
}

template class cnet::afunc::afunc<double>;
template class cnet::afunc::linear<double>;
template class cnet::afunc::sigmoid<double>;
template class cnet::afunc::relu<double>;
