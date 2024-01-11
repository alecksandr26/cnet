#include <cassert>
#include <cstdlib>
#include <cstring>

#include "cnet/dtypes.hpp"
#include "cnet/utils_avx.hpp"
#include "cnet/utils_mat.hpp"

#include <immintrin.h>
#include <omp.h>

using namespace cnet::dtypes;
using namespace cnet::mathops::utils;
using namespace std;

void  *cnet::mathops::utils::alloc_mem_matrix(size_t n, size_t item_size)
{
	assert(n && item_size);
	
	return aligned_alloc(item_size, n * item_size);
}

void cnet::mathops::utils::free_mem_matrix(void *ptr)
{
	assert(ptr);
	free(ptr);
}

template<typename T>
void cnet::mathops::utils::cp_raw_mat(T *dst_mat, T *src_mat,
			     size_t rows, size_t cols,
			     size_t dst_cols, size_t src_cols)
{
	assert(dst_mat && src_mat);

	size_t cols_ite_8 = cols - (cols % 8);
	size_t cols_ite_4 = cols - (cols % 4);
	size_t cols_ite_2 = cols - (cols % 2);

#pragma omp parallel for
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols_ite_8; j += 8)
			mov_avx_8(&dst_mat[i * dst_cols + j], &src_mat[i * src_cols + j]);

		for (size_t j = cols_ite_8; j < cols_ite_4; j += 4)
			mov_avx_4(&dst_mat[i * dst_cols + j], &src_mat[i * src_cols + j]);
		
		for (size_t j = cols_ite_4; j < cols_ite_2; j += 2)
			mov_avx_2(&dst_mat[i * dst_cols + j], &src_mat[i * src_cols + j]);

		for (size_t j = cols_ite_2; j < cols; j++)
			dst_mat[i * dst_cols + j] = src_mat[i * src_cols + j];
	}
		
}

template<>
void cnet::mathops::utils::mul_sqr_raw_mat(float64 *A, float64 *B, float64 *C, std::size_t n)
{
	size_t nb =  (n + 3) / 4;

	vec4d *vec_a = (vec4d *) alloc_mem_matrix(n * nb, sizeof(vec4d));
	memset(vec_a, 0, sizeof(vec4d) * n * nb);
	vec4d *vec_b = (vec4d *) alloc_mem_matrix(n * nb, sizeof(vec4d));
	memset(vec_b, 0, sizeof(vec4d) * n * nb);

#pragma omp parallel for collapse(2)
	for (size_t i = 0; i < n; i++)
		for (size_t j = 0; j < n; j++) {
			vec_a[i * nb + j / 4][j % 4] = A[i * n + j];
			vec_b[i * nb + j / 4][j % 4] = B[j * n + i]; // Transpose the matrix
		}

#pragma omp parallel for collapse(2)	
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < n; j++) {
			vec4d s = _mm256_setzero_pd();

			for (size_t k = 0; k < nb; k++)
				s = _mm256_add_pd(s, _mm256_mul_pd(vec_a[i * nb + k],
								   vec_b[j * nb + k]));
			
			C[i * n + j] = hsum_avx256(s);
		}
	}

	free(vec_a);
	free(vec_b);
}

template<>
void cnet::mathops::utils::mul_sqr_raw_mat(float32 *A, float32 *B, float32 *C, std::size_t n)
{
	size_t nb =  (n + 7) / 8;

	vec8f *vec_a = (vec8f *) alloc_mem_matrix(n * nb, sizeof(vec8f));
	memset(vec_a, 0, sizeof(vec8f) * n * nb);
	vec8f *vec_b = (vec8f *) alloc_mem_matrix(n * nb, sizeof(vec8f));
	memset(vec_b, 0, sizeof(vec8f) * n * nb);

	
#pragma omp parallel for collapse(2)
	for (size_t i = 0; i < n; i++)
		for (size_t j = 0; j < n; j++) {
			vec_a[i * nb + j / 8][j % 8] = A[i * n + j];
			vec_b[i * nb + j / 8][j % 8] = B[j * n + i]; // Transpose the matrix
		}


#pragma omp parallel for collapse(2)
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < n; j++) {
			vec8f s = _mm256_setzero_ps();
			
			for (size_t k = 0; k < nb; k++)
				s = _mm256_add_ps(s, _mm256_mul_ps(vec_a[i * nb + k],
								   vec_b[j * nb + k]));
			C[i * n + j] = hsum_avx256(s);
		}
	}


	free(vec_a);
	free(vec_b);
}

template<typename T>
void cnet::mathops::utils::init_raw_mat(T *A, size_t rows, size_t cols, T init_val)
{
	size_t n		= rows * cols;
	size_t n_ite_8	= n - (n % 8);
	size_t n_ite_4	= n - (n % 4);

	switch (n) {
	case 1: A[0] = init_val; break;
	case 2:
		A[0] = init_val;
		A[1] = init_val;
		break;
	case 3:
		A[0] = init_val;
		A[1] = init_val;
		A[2] = init_val;
		break;
	default:
#pragma omp parallel for		
		for (std::size_t i = 0; i < n_ite_8; i += 8)
			store_val_avx_8(&A[i], init_val);
		
#pragma omp parallel for		
		for (std::size_t i = n_ite_8; i < n_ite_4; i += 4)
			store_val_avx_4(&A[i], init_val);

		for (std::size_t i = n_ite_4; i < n; i++)
			A[i] = init_val;

		break;
	}
}


template<typename T>
void cnet::mathops::utils::add_raw_mat(T *A, T *B, size_t rows, size_t cols)
{
	size_t n	    = rows * cols;
	size_t n_ite_8 = n - (n % 8);
	size_t n_ite_4 = n - (n % 4);
	
	switch (n) {
	case 1:
		A[0] += B[0];
		break;
	case 2:
		A[0] += B[0];
		A[1] += B[1];
		break;
	case 3:
		A[0] += B[0];
		A[1] += B[1];
		A[2] += B[2];
		break;
	default:
#pragma omp parallel for
		for (size_t i = 0; i < n_ite_8; i += 8)
			add_avx_8(&A[i], &B[i]);
		
#pragma omp parallel for		
		for (size_t i = n_ite_8; i < n_ite_4; i += 4)
			add_avx_4(&A[i], &B[i]);
		
		for (size_t i = n_ite_4; i < n; i++)
			A[i] += B[i];
		
		break;
	}
}

template<typename T>
void cnet::mathops::utils::sub_raw_mat(T *A, T *B, size_t rows, size_t cols)
{
	size_t n	    = rows * cols;
	size_t n_ite_8 = n - (n % 8);
	size_t n_ite_4 = n - (n % 4);
	
	switch (n) {
	case 1:
		A[0] -= B[0];
		break;
	case 2:
		A[0] -= B[0];
		A[1] -= B[1];
		break;
	case 3:
		A[0] -= B[0];
		A[1] -= B[1];
		A[2] -= B[2];
		break;
	default:
#pragma omp parallel for
		for (size_t i = 0; i < n_ite_8; i += 8)
			sub_avx_8(&A[i], &B[i]);
		
#pragma omp parallel for		
		for (size_t i = n_ite_8; i < n_ite_4; i += 4)
			sub_avx_4(&A[i], &B[i]);
		
		for (size_t i = n_ite_4; i < n; i++)
			A[i] -= B[i];
		
		break;
	}
}

template<typename T>
void cnet::mathops::utils::hardmard_mul_raw_mat(T *A, T *B, size_t rows, size_t cols)
{
	size_t n	    = rows * cols;
	size_t n_ite_8 = n - (n % 8);
	size_t n_ite_4 = n - (n % 4);
	
	switch (n) {
	case 1:
		A[0] *= B[0];
		break;
	case 2:
		A[0] *= B[0];
		A[1] *= B[1];
		break;
	case 3:
		A[0] *= B[0];
		A[1] *= B[1];
		A[2] *= B[2];
		break;
	default:
#pragma omp parallel for
		for (size_t i = 0; i < n_ite_8; i += 8)
			mul_avx_8(&A[i], &B[i]);
		
#pragma omp parallel for		
		for (size_t i = n_ite_8; i < n_ite_4; i += 4)
			mul_avx_4(&A[i], &B[i]);
		
		for (size_t i = n_ite_4; i < n; i++)
			A[i] *= B[i];
		
		break;
	}
}

template<typename T>
void cnet::mathops::utils::scalar_mul_raw_mat(T *A, T b, size_t rows, size_t cols)
{
	size_t n	    = rows * cols;
	size_t n_ite_8 = n - (n % 8);
	size_t n_ite_4 = n - (n % 4);
	
	switch (n) {
	case 1:
		A[0] *= b;
		break;
	case 2:
		A[0] *= b;
		A[1] *= b;
		break;
	case 3:
		A[0] *= b;
		A[1] *= b;
		A[2] *= b;
		break;
	default:
#pragma omp parallel for
		for (size_t i = 0; i < n_ite_8; i += 8)
			mul_avx_8(&A[i], b);
		
#pragma omp parallel for		
		for (size_t i = n_ite_8; i < n_ite_4; i += 4)
			mul_avx_4(&A[i], b);
		
		for (size_t i = n_ite_4; i < n; i++)
			A[i] *= b;
		
		break;
	}
}

template<>
float64 cnet::mathops::utils::grand_sum_raw_mat(float64 *A, size_t rows, size_t cols)
{
	size_t n	    = rows * cols;
	size_t n_ite_8 = n - (n % 8);
	size_t n_ite_4 = n - (n % 4);
	
	float64 res = 0.0;
	vec4d vec4 = _mm256_setzero_pd();
	
	switch (n) {
	case 1:
		res = A[0];
		break;
	case 2:
		res = A[0] + A[1];
		break;
	case 3:
		res = A[0] + A[1] + A[2];
		break;
	default:
		for (size_t i = 0; i < n_ite_8; i += 8) {
			sum_avx256(vec4, &A[i]);
			sum_avx256(vec4, &A[i + 4]);
		}
		
		for (size_t i = n_ite_8; i < n_ite_4; i += 4)
			sum_avx256(vec4, &A[i]);

		res = hsum_avx256(vec4);
		
		for (size_t i = n_ite_4; i < n; i++)
			res += A[i];
		
		break;
	}

	return res;
}

template<>
float32 cnet::mathops::utils::grand_sum_raw_mat(float32 *A, size_t rows, size_t cols)
{
	size_t n	    = rows * cols;
	size_t n_ite_8 = n - (n % 8);
	size_t n_ite_4 = n - (n % 4);

	float32 res = 0.0;

	vec8f vec8 = _mm256_setzero_ps();
	vec4f vec4 = _mm_setzero_ps();
	
	switch (n) {
	case 1:
		res = A[0];
		break;
	case 2:
		res = A[0] + A[1];
		break;
	case 3:
		res = A[0] + A[1] + A[2];
		break;
	default:
		for (size_t i = 0; i < n_ite_8; i += 8)
			sum_avx256(vec8, &A[i]);

		res = hsum_avx256(vec8);
		
		for (size_t i = n_ite_8; i < n_ite_4; i += 4)
			sum_avx128(vec4, &A[i]);

		res += hsum_avx128(vec4);
		
		for (size_t i = n_ite_4; i < n; i++)
			res += A[i];
		
		break;
	}

	return res;
}

template void cnet::mathops::utils::cp_raw_mat(float64 *dst_mat, float64 *src_mat,
					       size_t rows, size_t cols,
					       size_t dst_cols, size_t src_cols);
template void cnet::mathops::utils::cp_raw_mat(float32 *dst_mat, float32 *src_mat,
					       size_t rows, size_t cols,
					       size_t dst_cols, size_t src_cols);
template void cnet::mathops::utils::init_raw_mat(float64 *A, size_t rows, size_t cols, float64 init_val);
template void cnet::mathops::utils::init_raw_mat(float32 *A, size_t rows, size_t cols, float32 init_val);
template void cnet::mathops::utils::add_raw_mat(float64 *A, float64 *B, size_t rows, size_t cols);
template void cnet::mathops::utils::add_raw_mat(float32 *A, float32 *B, size_t rows, size_t cols);
template void cnet::mathops::utils::sub_raw_mat(float64 *A, float64 *B, size_t rows, size_t cols);
template void cnet::mathops::utils::sub_raw_mat(float32 *A, float32 *B, size_t rows, size_t cols);
template void cnet::mathops::utils::hardmard_mul_raw_mat(float64 *A, float64 *B, size_t rows, size_t cols);
template void cnet::mathops::utils::hardmard_mul_raw_mat(float32 *A, float32 *B, size_t rows, size_t cols);
template void cnet::mathops::utils::scalar_mul_raw_mat(float64 *A, float64 b, size_t rows, size_t cols);
template void cnet::mathops::utils::scalar_mul_raw_mat(float32 *A, float32 b, size_t rows, size_t cols);
