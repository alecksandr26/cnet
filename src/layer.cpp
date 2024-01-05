#include "cnet/mat.hpp"
#include "cnet/afunc.hpp"
#include "cnet/layer.hpp"

#include <memory>
#include <complex>
#include <cstddef>

#include <immintrin.h> // For AVX2 intrinsics
#include <omp.h>

template<class T>
cnet::model::layer<T>::layer(std::size_t in, std::size_t out)
{
	in_    = in;
	out_   = out;
	func_ = NULL;

	// Alloc the matrices
	W_.resize(out, in, 0.0);
	B_.resize(out, 1, 0.0);
}

template<class T>
cnet::model::layer<T>::layer(std::size_t in, std::size_t out, std::unique_ptr<afunc::act_func<T>> &&func)
{
	in_    = in;
	out_   = out;
	func_ = std::move(func);
	
	// Alloc the matrices
	W_.resize(out, in, 0.0);
	B_.resize(out, 1, 0.0);
}

template<class T>
cnet::model::layer<T>::layer(void)
{
	out_ = in_ = 0;
	func_ = NULL;
}

template<class T>
void cnet::model::layer<T>::mod(std::size_t in, std::size_t out, std::unique_ptr<afunc::act_func<T>> &&func)
{
	in_    = in;
	out_   = out;
	func_ = std::move(func);
	
	// Realloc the matrices
	W_.resize(out, in, 0.0);
	B_.resize(out, 1, 0.0);
}

template<class T>
void cnet::model::layer<T>::mod(std::size_t in, std::size_t out)
{
	in_  = in;
	out_ = out;

	// Realloc the matrices
	W_.resize(out, in, 0.0);
	B_.resize(out, 1, 0.0);
}

template<class T>
void cnet::model::layer<T>::mod(std::unique_ptr<afunc::act_func<T>> &&func)
{
	func_ = std::move(func);
}

template<class T>
void cnet::model::layer<T>::rand_range(T a, T b)
{
	W_.rand(a, b);
	B_.rand(a, b);
}

template<class T>
cnet::mat<T> cnet::model::layer<T>::feedforward(const cnet::mat<T> &X) const
{
	if (X.get_cols() != 1 || X.get_rows() != W_.get_cols())
		throw std::invalid_argument("invalid argument: Matrices has different sizes");
	
	// The feedforward operation
	return func_->func(W_ * X + B_);
}

template<class T>
void cnet::model::layer<T>::fit_back_prog(const cnet::mat<T> &E, double lr, const cnet::mat<T> &A)
{
	if (E.get_cols() != 1 || A.get_cols() != 1 || A.get_rows() != W_.get_cols())
		throw std::invalid_argument("invalid argument: Matrices has different sizes");
	
	// Update the bias by error
	cnet::mat<T> df_dx = func_->dfunc_dx(W_ * A + B_);
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

template class cnet::model::layer<double>;
