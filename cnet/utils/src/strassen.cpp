#include "utils/avx.hpp"
#include "utils/raw_mat.hpp"
#include "utils/strassen.hpp"

#include <type_traits>
#include <cassert>
#include <cstring>
#include <cstdint>

using namespace std;
using namespace cnet::mathops;
using namespace utils;
using namespace strassen;

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

size_t cnet::mathops::strassen::pad_size(size_t a_rows, size_t a_cols, size_t b_rows, size_t b_cols)
{
	size_t max_dimension = max(max(a_rows, b_rows), max(a_cols, b_cols));
	return precomputed_pow_2_n[fast_log2(max_dimension) + 1];
}

template<typename T>
T *cnet::mathops::strassen::pad_mat(const T *A, size_t rows, size_t cols, size_t n)
{
	T *padded_mat = (T *) alloc_mem_matrix(n * n, sizeof(T));
	memset(padded_mat, (T) 0.0, sizeof(T) * n * n);
	
	switch (n) {
	case 1:
	case 2:
	case 4:
		for (size_t i = 0; i < rows; i++)
			for (size_t j = 0; j < cols; j++)
				padded_mat[i * n + j] = A[i * cols + j];
		break;
	default:
		cp_raw_mat(padded_mat, A, rows, cols, n, cols);
		break;
	}
	
	return padded_mat;
}

template<typename T>
void cnet::mathops::strassen::mat_mul(const T *A, const T *B, T *C, size_t n, size_t N)
{
	if (n <= MIN_ELEMENTS_TO_MAT_MUL_STRASSEN) {
		mul_sqr_raw_mat(A, B, C, n);
		return;
	}

	size_t k = n / 2;

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

template double *cnet::mathops::strassen::pad_mat(const double *A, size_t rows, size_t cols, size_t n);
template float *cnet::mathops::strassen::pad_mat(const float *A, size_t rows, size_t cols, size_t n);
template void cnet::mathops::strassen::mat_mul(const double *A, const double *B, double *C, size_t n, size_t N);
template void cnet::mathops::strassen::mat_mul(const float *A, const float *B, float *C, size_t n, size_t N);
