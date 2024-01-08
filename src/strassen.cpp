#include "cnet/mat.hpp"
#include "cnet/utils_avx.hpp"
#include "cnet/utils_mat.hpp"
#include "cnet/strassen.hpp"

#include <type_traits>
#include <cassert>
#include <cstring>
#include <immintrin.h>
#include <omp.h>

using namespace cnet;
using namespace cnet::utils;
using namespace cnet::strassen;
using namespace std;

#define MIN_ELEMENTS_TO_MAT_MUL_STRASSEN 128

static size_t fast_log2(uint64_t n)
{
#define S(k)					\
	if (n >= (UINT64_C(1) << k)) {		\
		i += k;				\
		n >>= k;			\
	}

	int i = -(n == 0);
	S(32);
	S(16);
	S(8);
	S(4);
	S(2);
	S(1);
	return i;

#undef S	
}

static int precomputed_pow_2_n[] = {
	1,	  2,	    4,	      8,	 16,	    32,	      64,      128,
	256,	  512,	    1024,     2048,	 4096,	    8192,     16384,   32768,
	65536,	  131072,   262144,   524288,	 1048576,   2097152,  4194304, 8388608,
	16777216, 33554432, 67108864, 134217728, 268435456, 536870912};

size_t cnet::strassen::pad_size(size_t a_rows, size_t a_cols, size_t b_rows, size_t b_cols)
{
	size_t max_dimension = max(max(a_rows, b_rows), max(a_cols, b_cols));
	return precomputed_pow_2_n[fast_log2(max_dimension) + 1];
}

template<typename T>
T *cnet::strassen::pad_mat(const Mat<T> &A, size_t n)
{
	T *padded_mat, *allocated_mat = A.get_allocated_mat();
	
	padded_mat = (T *) alloc_mem_matrix(n * n, sizeof(T));
	memset(padded_mat, (T) 0.0, sizeof(T) * n * n);
	
	size_t rows = A.get_rows();
	size_t cols = A.get_cols();

	switch (n) {
	case 1:
	case 2:
	case 4:
		for (size_t i = 0; i < rows; i++)
			for (size_t j = 0; j < cols; j++)
				padded_mat[i * n + j] = allocated_mat[i * cols + j];
		break;
	default:
		cp_raw_mat(padded_mat, allocated_mat, rows, cols, n, cols);
		break;
	}
	

	return padded_mat;
}



template<typename T>
void cnet::strassen::mat_mul(T *A, T *B, T *C, size_t n, size_t N)
{
	if (n <= MIN_ELEMENTS_TO_MAT_MUL_STRASSEN) {
		mul_sqr_raw_mat(A, B, C, n);
		return;
	}

	std::size_t k = n / 2;

	// C11 = A11 B11 + A12 B21
	mat_mul(A, B, C, k, N);
	mat_mul(A + k, B + k * N, C, k, N);

	// C12 = A11 B12 + A12 B22
	mat_mul(A, B + k, C + k, k, N);
	mat_mul(A + k, B + k * N + k, C + k, k, N);

	// C21 = A21 B11 + A22 B21
	mat_mul(A + k * N, B, C + k * N, k, N);
	mat_mul(A + k * N + k, B + k * N, C + k * N, k, N);

	// C22 = A21 B12 + A22 B22
	mat_mul(A + k * N, B + k, C + k * N + k, k, N);
	mat_mul(A + k * N + k, B + k * N + k, C + k * N + k, k, N);
}

template double *cnet::strassen::pad_mat(const Mat<double> &A, size_t n);
template float *cnet::strassen::pad_mat(const Mat<float> &A, size_t n);
template void cnet::strassen::mat_mul(double *, double *, double *, size_t, size_t);
template void cnet::strassen::mat_mul(float *, float *, float *, size_t, size_t);

// Previos strassen algorithm
#if 0
template<typename T>
void strassen_mat_mul(T *A, T *B, T *C, std::size_t n, std::size_t N)
{
	if (n <= MAT_VEC_SIZE) {
		switch (n) {
		case 1: C[0] = A[0] * B[0]; break;
		case 2:
			for (std::size_t i = 0; i < n; i++)
				for (std::size_t j = 0; j < n; j++) {
					a[i * N_B + j / VEC_4_DOUBLE_SIZE]
					 [j % VEC_4_DOUBLE_SIZE] = A[i * N + j];
					b[i * N_B + j / VEC_4_DOUBLE_SIZE]
					 [j % VEC_4_DOUBLE_SIZE] =
						 B[j * N + i];	      // Transpose
				}
			break;
		case 4:
			for (std::size_t i = 0; i < n; i++) {
				for (std::size_t j = 0; j < n; j += VEC_4_DOUBLE_SIZE) {
					cnet::vec4double vecA =
						_mm256_loadu_pd(&A[i * N + j]);
					_mm256_storeu_pd(
						&a[i * N_B + j / VEC_4_DOUBLE_SIZE][0],
						vecA);

					cnet::vec4double vecB = _mm256_set_pd(
						B[(j + 3) * N + i], B[(j + 2) * N + i],
						B[(j + 1) * N + i], B[(j + 0) * N + i]);
					_mm256_storeu_pd(
						&b[i * N_B + j / VEC_4_DOUBLE_SIZE][0],
						vecB);
				}
			}
			break;

		default:
#pragma omp parallel for collapse(2)
			for (std::size_t i = 0; i < n; i += 2) {
				for (std::size_t j = 0; j < n;
				     j += 2 * VEC_4_DOUBLE_SIZE) {
					cnet::vec4double vecA1 =
						_mm256_loadu_pd(&A[i * N + j]);
					cnet::vec4double vecA2 = _mm256_loadu_pd(
						&A[i * N + j + VEC_4_DOUBLE_SIZE]);
					cnet::vec4double vecA3 =
						_mm256_loadu_pd(&A[(i + 1) * N + j]);
					cnet::vec4double vecA4 = _mm256_loadu_pd(
						&A[(i + 1) * N + j + VEC_4_DOUBLE_SIZE]);
					
					_mm256_storeu_pd(
						&a[i * N_B + j / VEC_4_DOUBLE_SIZE][0],
						vecA1);
					_mm256_storeu_pd(&a[i * N_B +
							    (j + VEC_4_DOUBLE_SIZE) /
								    VEC_4_DOUBLE_SIZE][0],
							 vecA2);
					_mm256_storeu_pd(&a[(i + 1) * N_B +
							    j / VEC_4_DOUBLE_SIZE][0],
							 vecA3);
					_mm256_storeu_pd(&a[(i + 1) * N_B +
							    (j + VEC_4_DOUBLE_SIZE) /
								    VEC_4_DOUBLE_SIZE][0],
							 vecA4);

					cnet::vec4double vecB1 = _mm256_set_pd(
						B[(j + 3) * N + i], B[(j + 2) * N + i],
						B[(j + 1) * N + i], B[(j + 0) * N + i]);

					cnet::vec4double vecB2 = _mm256_set_pd(
						B[(j + 3 + VEC_4_DOUBLE_SIZE) * N + i],
						B[(j + 2 + VEC_4_DOUBLE_SIZE) * N + i],
						B[(j + 1 + VEC_4_DOUBLE_SIZE) * N + i],
						B[(j + VEC_4_DOUBLE_SIZE) * N + i]);

					cnet::vec4double vecB3 =
						_mm256_set_pd(B[(j + 3) * N + i + 1],
							      B[(j + 2) * N + i + 1],
							      B[(j + 1) * N + i + 1],
							      B[(j + 0) * N + i + 1]);

					cnet::vec4double vecB4 = _mm256_set_pd(
						B[(j + 3 + VEC_4_DOUBLE_SIZE) * N + i +
						  1],
						B[(j + 2 + VEC_4_DOUBLE_SIZE) * N + i +
						  1],
						B[(j + 1 + VEC_4_DOUBLE_SIZE) * N + i +
						  1],
						B[(j + VEC_4_DOUBLE_SIZE) * N + i + 1]);
					_mm256_storeu_pd(
						&b[i * N_B + j / VEC_4_DOUBLE_SIZE][0],
						vecB1);
					_mm256_storeu_pd(&b[i * N_B +
							    (j + VEC_4_DOUBLE_SIZE) /
								    VEC_4_DOUBLE_SIZE][0],
							 vecB2);
					_mm256_storeu_pd(&b[(i + 1) * N_B +
							    j / VEC_4_DOUBLE_SIZE][0],
							 vecB3);
					_mm256_storeu_pd(&b[(i + 1) * N_B +
							    (j + VEC_4_DOUBLE_SIZE) /
								    VEC_4_DOUBLE_SIZE][0],
							 vecB4);
				}
			}
			break;
		}

		switch (n) {
		case 2:
		case 4:
			for (std::size_t i = 0; i < n; i++) {
				for (std::size_t j = 0; j < n; j++) {
					cnet::vec4double s = _mm256_setzero_pd();
					for (std::size_t k = 0; k < N_B; k++)
						s = _mm256_add_pd(
							s, _mm256_mul_pd(a[i * N_B + k],
									 b[j * N_B + k]));
					// Horizontal summation and store the result
					C[i * N + j] = hsum_double_avx256(s);
				}
			}
			break;
		default:
#pragma omp parallel for collapse(2)
			for (std::size_t i = 0; i < n; i += 2) {
				for (std::size_t j = 0; j < n; j += 2) {
					// initialize the accumulators
					cnet::vec4double s1 = _mm256_setzero_pd();
					cnet::vec4double s2 = _mm256_setzero_pd();
					cnet::vec4double s3 = _mm256_setzero_pd();
					cnet::vec4double s4 = _mm256_setzero_pd();

					for (std::size_t k = 0; k < N_B; k++) {
						s1 = _mm256_add_pd(
							s1,
							_mm256_mul_pd(a[i * N_B + k],
								      b[j * N_B + k]));
						s2 = _mm256_add_pd(
							s2,
							_mm256_mul_pd(
								a[i * N_B + k],
								b[(j + 1) * N_B + k]));
						s3 = _mm256_add_pd(
							s3, _mm256_mul_pd(
								    a[(i + 1) * N_B + k],
								    b[j * N_B + k]));
						s4 = _mm256_add_pd(
							s4,
							_mm256_mul_pd(
								a[(i + 1) * N_B + k],
								b[(j + 1) * N_B + k]));
					}

					// Horizontal summation and store the results
					C[i * N + j]	       = hsum_double_avx256(s1);
					C[i * N + j + 1]       = hsum_double_avx256(s2);
					C[(i + 1) * N + j]     = hsum_double_avx256(s3);
					C[(i + 1) * N + j + 1] = hsum_double_avx256(s4);
				}
			}
			break;
		}

		return;
	}

	std::size_t k = n / 2;

	// C11 = A11 B11 + A12 B21
	strassen_mat_mul(A, B, C, k, N, a, b);
	strassen_mat_mul(A + k, B + k * N, C, k, N, a, b);

	// C12 = A11 B12 + A12 B22
	strassen_mat_mul(A, B + k, C + k, k, N, a, b);
	strassen_mat_mul(A + k, B + k * N + k, C + k, k, N, a, b);

	// C21 = A21 B11 + A22 B21
	strassen_mat_mul(A + k * N, B, C + k * N, k, N, a, b);
	strassen_mat_mul(A + k * N + k, B + k * N, C + k * N, k, N, a, b);

	// C22 = A21 B12 + A22 B22
	strassen_mat_mul(A + k * N, B + k, C + k * N + k, k, N, a, b);
	strassen_mat_mul(A + k * N + k, B + k * N + k, C + k * N + k, k, N, a, b);
}

#endif
