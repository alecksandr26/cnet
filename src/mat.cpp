#include <cassert>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>

// To optimize the matrix multiplication and the parallelization of it
#include "cnet/mat.hpp"

#include <immintrin.h>
#include <omp.h>

#define MAT_VEC_SIZE 128

#if __AVX512F__
#define VEC_SIZE 8
// number of 8-element vectors
#else
#define VEC_SIZE 4
// number of 4-element vectors
#endif

constexpr std::size_t N_B = (MAT_VEC_SIZE + VEC_SIZE - 1) / VEC_SIZE;

// For the multiplication process
omp_lock_t strassen_mat_mul_lock;

int precomputed_pow_2_n[] = {
	1,	  2,	    4,	      8,	 16,	    32,	      64,      128,
	256,	  512,	    1024,     2048,	 4096,	    8192,     16384,   32768,
	65536,	  131072,   262144,   524288,	 1048576,   2097152,  4194304, 8388608,
	16777216, 33554432, 67108864, 134217728, 268435456, 536870912};

inline double hsum_double_avx256(cnet::vec4double v)
{
	__m128d vlow  = _mm256_castpd256_pd128(v);
	__m128d vhigh = _mm256_extractf128_pd(v, 1);	    // high 128

	vlow = _mm_add_pd(vlow, vhigh);	       // reduce down to 128

	__m128d high64 = _mm_unpackhi_pd(vlow, vlow);

	return _mm_cvtsd_f64(_mm_add_sd(vlow, high64));	       // reduce to scalar
}

inline double hsum_double_avx512(cnet::vec8double v)
{
	__m256d v256_low  = _mm512_castpd512_pd256(v);
	__m256d v256_high = _mm512_extractf64x4_pd(v, 1);	 // Extract high 256 bits

	// Horizontal summation within each 256-bit lane
	__m256d sum256 = _mm256_add_pd(v256_low, v256_high);

	// Extract the lower 128 bits
	__m128d sum128 = _mm256_castpd256_pd128(sum256);

	// Perform horizontal summation within the lower 128 bits
	__m128d high64 = _mm_unpackhi_pd(sum128, sum128);
	sum128	       = _mm_add_pd(sum128, high64);

	return _mm_cvtsd_f64(sum128);	     // Reduce to scalar
}

inline void alloc_mem_matrix(void **mat_, std::size_t n, std::size_t item_size)
{
	assert(mat_ && n && item_size);

	(*mat_) = std::aligned_alloc(item_size, n * item_size);
	// (*mat_) = std::calloc(n, item_size);
}

inline void free_mem_matrix(void *ptr)
{
	assert(ptr);
	std::free(ptr);
}

static std::size_t fast_log2(uint64_t n)
{
#define S(k)                           \
	if (n >= (UINT64_C(1) << k)) { \
		i += k;                \
		n >>= k;               \
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

void print_mat_debug(double *m, std::size_t n)
{
	for (std::size_t i = 0; i < n; i++) {
		for (std::size_t j = 0; j < n; j++)
			std::cout << m[i * n + j] << " ";
		std::cout << std::endl;
	}
}

// TODO: figure out how to do this with complex numbers
template<typename T>
void pad_matrix_to_power_2(const cnet::mat<T> &A, T **p_a, std::size_t n)
{
	assert(p_a != NULL);

	alloc_mem_matrix((void **) p_a, n * n, sizeof(T));

	std::memset(*p_a, 0.0, sizeof(T) * n * n);

	// TODO: Needs to support complex numbers
	std::size_t rows = A.get_rows();
	std::size_t cols = A.get_cols();

	switch (n) {
	case 1:
	case 2:
	case 4:
		for (std::size_t i = 0; i < rows; i++)
			for (std::size_t j = 0; j < cols; j++)
				(*p_a)[i * n + j] = A(i, j);

		break;
	default: {
		std::size_t cols_mod	= cols % VEC_SIZE;
		std::size_t cols_to_ite = cols - cols_mod;

		std::size_t rows_mod	= rows % 2;
		std::size_t rows_to_ite = rows - rows_mod;

#pragma omp parallel for
		for (std::size_t i = 0; i < rows_to_ite; i += 2) {
			for (std::size_t j = 0; j < cols_to_ite; j += VEC_SIZE) {
				cnet::vec4double vectorA = _mm256_loadu_pd(&A(i, j));
				_mm256_storeu_pd(&(*p_a)[i * n + j], vectorA);

				cnet::vec4double vectorB = _mm256_loadu_pd(&A(i + 1, j));
				_mm256_storeu_pd(&(*p_a)[(i + 1) * n + j], vectorB);
			}

			for (std::size_t j = cols_to_ite; j < cols; j++) {
				(*p_a)[i * n + j]	= A(i, j);
				(*p_a)[(i + 1) * n + j] = A(i + 1, j);
			}
		}
		
#pragma omp parallel for		
		for (std::size_t i = rows_to_ite; i < rows; i++)
			for (std::size_t j = 0; j < cols; j++)
				(*p_a)[i * n + j] = A(i, j);

		break;
	}
	}
}

template<typename T>
#if __AVX512F__
void strassen_mat_mul(T *A, T *B, T *C, std::size_t n, std::size_t N, cnet::vec8double *a,
		      cnet::vec8double *b)
#else
void strassen_mat_mul(T *A, T *B, T *C, std::size_t n, std::size_t N, cnet::vec4double *a,
		      cnet::vec4double *b)
#endif
{
	if (n <= MAT_VEC_SIZE) {
		switch (n) {
		case 1: C[0] = A[0] * B[0]; break;
		case 2:
			for (std::size_t i = 0; i < n; i++)
				for (std::size_t j = 0; j < n; j++) {
					a[i * N_B + j / VEC_SIZE][j % VEC_SIZE] =
						A[i * N + j];
					b[i * N_B + j / VEC_SIZE][j % VEC_SIZE] =
						B[j * N + i];	     // Transpose
				}
			break;
		case 4:
			for (std::size_t i = 0; i < n; i++) {
				for (std::size_t j = 0; j < n; j += VEC_SIZE) {
					cnet::vec4double vecA =
						_mm256_loadu_pd(&A[i * N + j]);
					_mm256_storeu_pd(&a[i * N_B + j / VEC_SIZE][0],
							 vecA);

					cnet::vec4double vecB = _mm256_set_pd(
						B[(j + 3) * N + i], B[(j + 2) * N + i],
						B[(j + 1) * N + i], B[(j + 0) * N + i]);
					_mm256_storeu_pd(&b[i * N_B + j / VEC_SIZE][0],
							 vecB);
				}
			}
			break;

		default:
#pragma omp parallel for collapse(2)
			for (std::size_t i = 0; i < n; i += 2) {
				for (std::size_t j = 0; j < n; j += 2 * VEC_SIZE) {
					cnet::vec4double vecA1 =
						_mm256_loadu_pd(&A[i * N + j]);
					cnet::vec4double vecA2 =
						_mm256_loadu_pd(&A[i * N + j + VEC_SIZE]);
					cnet::vec4double vecA3 =
						_mm256_loadu_pd(&A[(i + 1) * N + j]);
					cnet::vec4double vecA4 = _mm256_loadu_pd(
						&A[(i + 1) * N + j + VEC_SIZE]);
					_mm256_storeu_pd(&a[i * N_B + j / VEC_SIZE][0],
							 vecA1);
					_mm256_storeu_pd(&a[i * N_B +
							    (j + VEC_SIZE) / VEC_SIZE][0],
							 vecA2);
					_mm256_storeu_pd(
						&a[(i + 1) * N_B + j / VEC_SIZE][0],
						vecA3);
					_mm256_storeu_pd(&a[(i + 1) * N_B +
							    (j + VEC_SIZE) / VEC_SIZE][0],
							 vecA4);

					cnet::vec4double vecB1 = _mm256_set_pd(
						B[(j + 3) * N + i], B[(j + 2) * N + i],
						B[(j + 1) * N + i], B[(j + 0) * N + i]);

					cnet::vec4double vecB2 = _mm256_set_pd(
						B[(j + 3 + VEC_SIZE) * N + i],
						B[(j + 2 + VEC_SIZE) * N + i],
						B[(j + 1 + VEC_SIZE) * N + i],
						B[(j + VEC_SIZE) * N + i]);

					cnet::vec4double vecB3 =
						_mm256_set_pd(B[(j + 3) * N + i + 1],
							      B[(j + 2) * N + i + 1],
							      B[(j + 1) * N + i + 1],
							      B[(j + 0) * N + i + 1]);

					cnet::vec4double vecB4 = _mm256_set_pd(
						B[(j + 3 + VEC_SIZE) * N + i + 1],
						B[(j + 2 + VEC_SIZE) * N + i + 1],
						B[(j + 1 + VEC_SIZE) * N + i + 1],
						B[(j + VEC_SIZE) * N + i + 1]);
					_mm256_storeu_pd(&b[i * N_B + j / VEC_SIZE][0],
							 vecB1);
					_mm256_storeu_pd(&b[i * N_B +
							    (j + VEC_SIZE) / VEC_SIZE][0],
							 vecB2);
					_mm256_storeu_pd(
						&b[(i + 1) * N_B + j / VEC_SIZE][0],
						vecB3);
					_mm256_storeu_pd(&b[(i + 1) * N_B +
							    (j + VEC_SIZE) / VEC_SIZE][0],
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

template<class T>
cnet::mat<T>::mat(std::size_t rows, std::size_t cols)
{
	if (rows == 0 || cols == 0)
		throw std::invalid_argument(
			"invalid argument: row and col can't be zero");

	row_ = rows;
	col_ = cols;

	alloc_mem_matrix((void **) &mat_, col_ * row_, sizeof(T));
#if __AVX512F__
	alloc_mem_matrix((void **) &vec_mat_alloc, MAT_VEC_SIZE * MAT_VEC_SIZE,
			 sizeof(cnet::vec8double));
	std::memset(vec_mat_alloc, 0,
		    sizeof(cnet::vec8double) * MAT_VEC_SIZE * MAT_VEC_SIZE);
#else
	alloc_mem_matrix((void **) &vec_mat_alloc, MAT_VEC_SIZE * MAT_VEC_SIZE,
			 sizeof(cnet::vec4double));
	std::memset(vec_mat_alloc, 0,
		    sizeof(cnet::vec4double) * MAT_VEC_SIZE * MAT_VEC_SIZE);
#endif
}

template<class T>
cnet::mat<T>::mat(std::size_t rows, std::size_t cols, T initial)
{
	if (rows == 0 || cols == 0)
		throw std::invalid_argument(
			"invalid argument: row and col can't be zero");

	row_ = rows;
	col_ = cols;

	alloc_mem_matrix((void **) &mat_, col_ * row_, sizeof(T));
#if __AVX512F__
	alloc_mem_matrix((void **) &vec_mat_alloc, MAT_VEC_SIZE * MAT_VEC_SIZE,
			 sizeof(cnet::vec8double));
	std::memset(vec_mat_alloc, 0,
		    sizeof(cnet::vec8double) * MAT_VEC_SIZE * MAT_VEC_SIZE);
#else
	alloc_mem_matrix((void **) &vec_mat_alloc, MAT_VEC_SIZE * MAT_VEC_SIZE,
			 sizeof(cnet::vec4double));
	std::memset(vec_mat_alloc, 0,
		    sizeof(cnet::vec4double) * MAT_VEC_SIZE * MAT_VEC_SIZE);
#endif

	for (std::size_t i = 0; i < row_; i++)
		for (std::size_t j = 0; j < col_; j++)
			mat_[i * col_ + j] = initial;
}

template<class T>
cnet::mat<T>::mat(std::initializer_list<std::initializer_list<T>> m)
{
	if (m.size() == 0) throw std::invalid_argument("invalid argument: Empty matrix");

	std::size_t n = m.begin()->size();
	for (std::size_t i = 1; i < m.size(); i++)
		if (n != (m.begin() + i)->size())
			throw std::invalid_argument(
				"invalid argument: Invalid structure of the matrix");

	col_ = n;
	row_ = m.size();

	alloc_mem_matrix((void **) &mat_, col_ * row_, sizeof(T));
#if __AVX512F__
	alloc_mem_matrix((void **) &vec_mat_alloc, MAT_VEC_SIZE * MAT_VEC_SIZE,
			 sizeof(cnet::vec8double));
	std::memset(vec_mat_alloc, 0,
		    sizeof(cnet::vec8double) * MAT_VEC_SIZE * MAT_VEC_SIZE);
#else
	alloc_mem_matrix((void **) &vec_mat_alloc, MAT_VEC_SIZE * MAT_VEC_SIZE,
			 sizeof(cnet::vec4double));
	std::memset(vec_mat_alloc, 0,
		    sizeof(cnet::vec4double) * MAT_VEC_SIZE * MAT_VEC_SIZE);
#endif

	for (std::size_t i = 0; i < row_; i++)
		for (std::size_t j = 0; j < col_; j++)
			mat_[i * col_ + j] = *((m.begin() + i)->begin() + j);
}

template<class T>
cnet::mat<T>::mat(const cnet::mat<T> &m)
{
	col_ = m.get_cols();
	row_ = m.get_rows();

	alloc_mem_matrix((void **) &mat_, col_ * row_, sizeof(T));

#if __AVX512F__
	alloc_mem_matrix((void **) &vec_mat_alloc, MAT_VEC_SIZE * MAT_VEC_SIZE,
			 sizeof(cnet::vec8double));
	std::memset(vec_mat_alloc, 0,
		    sizeof(cnet::vec8double) * MAT_VEC_SIZE * MAT_VEC_SIZE);
#else
	alloc_mem_matrix((void **) &vec_mat_alloc, MAT_VEC_SIZE * MAT_VEC_SIZE,
			 sizeof(cnet::vec4double));
	std::memset(vec_mat_alloc, 0,
		    sizeof(cnet::vec4double) * MAT_VEC_SIZE * MAT_VEC_SIZE);
#endif

	for (std::size_t i = 0; i < row_; i++)
		for (std::size_t j = 0; j < col_; j++)
			mat_[i * col_ + j] = m(i, j);
}

template<class T>
cnet::mat<T>::mat(void)
{
	col_ = row_   = 0;
	mat_	      = NULL;
	vec_mat_alloc = NULL;
#if __AVX512F__
	alloc_mem_matrix((void **) &vec_mat_alloc, MAT_VEC_SIZE * MAT_VEC_SIZE,
			 sizeof(cnet::vec8double));
	std::memset(vec_mat_alloc, 0,
		    sizeof(cnet::vec8double) * MAT_VEC_SIZE * MAT_VEC_SIZE);
#else
	alloc_mem_matrix((void **) &vec_mat_alloc, MAT_VEC_SIZE * MAT_VEC_SIZE,
			 sizeof(cnet::vec4double));
	std::memset(vec_mat_alloc, 0,
		    sizeof(cnet::vec4double) * MAT_VEC_SIZE * MAT_VEC_SIZE);
#endif
}

template<class T>
cnet::mat<T>::~mat(void)
{
	if (mat_) free_mem_matrix((void *) mat_);
	if (vec_mat_alloc) free_mem_matrix((void *) vec_mat_alloc);

	mat_	      = NULL;
	vec_mat_alloc = NULL;
	col_ = row_ = 0;
}

template<class T>
void cnet::mat<T>::resize(std::size_t rows, std::size_t cols)
{
	if (rows == row_ && cols == col_) return;

	if (mat_ != NULL) free_mem_matrix((void *) mat_);

	col_ = cols;
	row_ = rows;

	alloc_mem_matrix((void **) &mat_, col_ * row_, sizeof(T));
}

template<class T>
void cnet::mat<T>::resize(std::size_t rows, std::size_t cols, T initial)
{
	if (!(rows == row_ && cols == col_)) {
		if (mat_) free_mem_matrix((void *) mat_);
		alloc_mem_matrix((void **) &mat_, cols * rows, sizeof(T));
	}

	col_ = cols;
	row_ = rows;

	for (std::size_t i = 0; i < row_; i++)
		for (std::size_t j = 0; j < col_; j++)
			mat_[i * col_ + j] = initial;
}

template<class T>
std::size_t cnet::mat<T>::get_rows(void) const
{
	return row_;
}

template<class T>
std::size_t cnet::mat<T>::get_cols(void) const
{
	return col_;
}

template<class T>
cnet::mat<T> cnet::mat<T>::transpose(void)
{
	cnet::mat<T> R(col_, row_);

	for (std::size_t i = 0; i < row_; i++)
		for (std::size_t j = 0; j < col_; j++)
			R(j, i) = mat_[i * col_ + j];

	return R;
}

template<class T>
T *cnet::mat<T>::get_mat_alloc(void) const
{
	if (mat_ == NULL)
		throw std::out_of_range("uninitialized mat: Matrix is uninitialized");
	return mat_;
}

template<class T>
#if __AVX512F__
cnet::vec8double *cnet::mat<T>::get_vec_mat_alloc(void) const
#else
cnet::vec4double *cnet::mat<T>::get_vec_mat_alloc(void) const
#endif
{
	if (vec_mat_alloc == NULL)
		throw std::out_of_range(
			"uninitialized mat: Mat vec allocation is uninitialized");
	return vec_mat_alloc;
}

template<class T>
T &cnet::mat<T>::operator()(std::size_t i, std::size_t j) const
{
	if (i >= row_ || j >= col_)
		throw std::out_of_range("out of range: Matrix subscript out of bounds");
	return mat_[i * col_ + j];
}

template<class T>
cnet::mat<T> cnet::mat<T>::operator+(const cnet::mat<T> &B)
{
	if (col_ != B.get_cols() || row_ != B.get_rows())
		throw std::invalid_argument(
			"invalid argument: Matrices have different sizes");

	// Alloc the matrix
	cnet::mat<T> C(row_, col_);

	T *c_mat_alloc = C.get_mat_alloc();
	T *b_mat_alloc = B.get_mat_alloc();

	// By default it will increment in 4 by 4
	std::size_t col_inc = 4;
	switch (col_) {
	case 1:
		col_inc = 1;
		break;
	case 2: case 3:
		col_inc = 2;
		break;
	}

	std::size_t row_inc = 4;
	switch (row_) {
	case 1:
		row_inc = 1;
		break;
	case 2: case 3:
		row_inc = 2;
		break;
	}
	
	std::size_t col_mod = col_ % col_inc;
	std::size_t col_ite = col_ - col_mod;
	std::size_t row_mod = row_ % row_inc;
	std::size_t row_ite = row_ - row_mod;

	switch (row_inc) {
	case 1:
		switch (col_inc) {
		case 1:
#pragma omp parallel for collapse(2)
			for (std::size_t i = 0; i < row_; i++)
				for (std::size_t j = 0; j < col_; j++)
					c_mat_alloc[i * col_ + j] = mat_[i * col_ + j]
						+ b_mat_alloc[i * col_ + j];
			break;
		case 2: {
#pragma omp parallel for
			for (std::size_t i = 0; i < row_; i++) {
				for (std::size_t j = 0; j < col_ite; j += 2) {
					vec2double a = _mm_set_pd(mat_[i * col_ + j + 1], mat_[i * col_ + j]);
					vec2double b = _mm_set_pd(b_mat_alloc[i * col_ + j + 1],
								  b_mat_alloc[i * col_ + j]);
					vec2double result = _mm_add_pd(a, b);
					_mm_storeu_pd(&c_mat_alloc[i * col_ + j], result);
				}

				for (std::size_t j = col_ite; j < col_; j++)
					c_mat_alloc[i * col_ + j] = mat_[i * col_ + j]
						+ b_mat_alloc[i * col_ + j];
			}
		}
			break;
		default:
#pragma omp parallel for
			for (std::size_t i = 0; i < row_; i++) {
				for (std::size_t j = 0; j < col_ite; j += 4) {
					vec4double a = _mm256_set_pd(mat_[i * col_ + j + 3],
								     mat_[i * col_ + j + 2],
								     mat_[i * col_ + j + 1],
								     mat_[i * col_ + j]);
					vec4double b = _mm256_set_pd(b_mat_alloc[i * col_ + j + 3],
								     b_mat_alloc[i * col_ + j + 2],
								     b_mat_alloc[i * col_ + j + 1],
								     b_mat_alloc[i * col_ + j]);
					vec4double result = _mm256_add_pd(a, b);
					_mm256_storeu_pd(&c_mat_alloc[i * col_ + j], result);
				}

				for (std::size_t j = col_ite; j < col_; j++)
					c_mat_alloc[i * col_ + j] = mat_[i * col_ + j]
						+ b_mat_alloc[i * col_ + j];
			}
			
			break;
		}
		break;
	case 2:
		switch (col_inc) {
		case 1:
#pragma omp parallel for collapse(2)
			for (std::size_t i = 0; i < row_ite; i += 2)
				for (std::size_t j = 0; j < col_; j++)
					for (std::size_t k = 0; k < 2; k++)
						c_mat_alloc[(i + k) * col_ + j] = mat_[(i + k) * col_ + j]
							+ b_mat_alloc[(i + k) * col_ + j];
			break;
		case 2:
#pragma omp parallel for
			for (std::size_t i = 0; i < row_ite; i += 2) {
				for (std::size_t j = 0; j < col_ite; j += 2) {
					for (std::size_t k = 0; k < 2; k++) {
						vec2double a = _mm_set_pd(mat_[(i + k) * col_ + j + 1],
									  mat_[(i + k) * col_ + j]);
						vec2double b = _mm_set_pd(b_mat_alloc[(i + k) * col_ + j + 1],
									  b_mat_alloc[(i + k) * col_ + j]);
						vec2double result = _mm_add_pd(a, b);
						_mm_storeu_pd(&c_mat_alloc[(i + k) * col_ + j], result);
					}
				}

				for (std::size_t j = col_ite; j < col_; j++)
					for (std::size_t k = 0; k < 2; k++)
						c_mat_alloc[(i + k) * col_ + j] = mat_[(i + k) * col_ + j]
							+ b_mat_alloc[(i + k) * col_ + j];
			}
			break;
		default:
#pragma omp parallel for
			for (std::size_t i = 0; i < row_ite; i += 2) {
				for (std::size_t j = 0; j < col_ite; j += 4) {
					for (std::size_t k = 0; k < 2; k++) {
						vec4double a = _mm256_set_pd(mat_[(i + k) * col_ + j + 3],
								  mat_[(i + k) * col_ + j + 2],
								  mat_[(i + k) * col_ + j + 1],
								  mat_[(i + k) * col_ + j]);
						vec4double b = _mm256_set_pd(b_mat_alloc[(i + k) * col_
											 + j + 3],
								  b_mat_alloc[(i + k) * col_ + j + 2],
								  b_mat_alloc[(i + k) * col_ + j + 1],
								  b_mat_alloc[(i + k) * col_ + j]);
						vec4double result = _mm256_add_pd(a, b);
						_mm256_storeu_pd(&c_mat_alloc[(i + k) * col_ + j], result);
					}
				}

				for (std::size_t j = col_ite; j < col_; j++) {
					for (std::size_t k = 0; k < 2; k++) {
						c_mat_alloc[(i + k) * col_ + j] = mat_[(i + k) * col_ + j]
							+ b_mat_alloc[(i + k) * col_ + j];
					}
				}
			}
			break;
		}

#pragma omp parallel for collapse(2)
		for (std::size_t i = row_ite; i < row_; i++)
			for (std::size_t j = 0; j < col_; j++)
				c_mat_alloc[i * col_ + j] = mat_[i * col_ + j]
					+ b_mat_alloc[i * col_ + j];
		break;
	default:
		switch (row_inc) {
		case 1:
#pragma omp parallel for collapse(2)
			for (std::size_t i = 0; i < row_ite; i += 4)
				for (std::size_t j = 0; j < col_; j++)
					for (std::size_t k = 0; k < 4; k++)
						c_mat_alloc[(i + k) * col_ + j] = mat_[(i + k) * col_ + j]
							+ b_mat_alloc[(i + k) * col_ + j];
			break;
		case 2:
#pragma omp parallel for
			for (std::size_t i = 0; i < row_ite; i += 4) {
				for (std::size_t j = 0; j < col_; j += 2)
					for (std::size_t k = 0; k < 4; k++) {
						vec2double a = _mm_set_pd(mat_[(i + k) * col_ + j + 1],
									  mat_[(i + k) * col_ + j]);
						vec2double b = _mm_set_pd(b_mat_alloc[(i + k) * col_ + j + 1],
									  b_mat_alloc[(i + k) * col_ + j]);
						vec2double result = _mm_add_pd(a, b);
						_mm_storeu_pd(&c_mat_alloc[(i + k) * col_ + j], result);
					}
				
				for (std::size_t j = col_ite; j < col_; j++)
					for (std::size_t k = 0; k < 4; j++)
						c_mat_alloc[(i + k) * col_ + j] = mat_[(i + k) * col_ + j]
							+ b_mat_alloc[(i + k) * col_ + j];
			}
			
			break;
		default:
#pragma omp parallel for
			for (std::size_t i = 0; i < row_ite; i += 4) {
				for (std::size_t j = 0; j < col_; j += 4)
					for (std::size_t k = 0; k < 4; k++) {
						vec4double a = _mm256_set_pd(mat_[(i + k) * col_ + j + 3],
									     mat_[(i + k) * col_ + j + 2],
									     mat_[(i + k) * col_ + j + 1],
									     mat_[(i + k) * col_ + j]);
						vec4double b = _mm256_set_pd(b_mat_alloc[(i + k) * col_
											 + j + 3],
									     b_mat_alloc[(i + k) * col_
											 + j + 2],
									     b_mat_alloc[(i + k) * col_
											 + j + 1],
									     b_mat_alloc[(i + k) * col_ + j]);
						vec4double result = _mm256_add_pd(a, b);
						_mm256_storeu_pd(&c_mat_alloc[(i + k) * col_ + j], result);
					}
				
				for (std::size_t j = col_ite; j < col_; j++)
					for (std::size_t k = 0; k < 4; k++)
						c_mat_alloc[(i + k) * col_ + j] = mat_[(i + k) * col_ + j]
							+ b_mat_alloc[(i + k) * col_ + j];
			}
			
			break;
		}
#pragma omp parallel for collapse(2)
		for (std::size_t i = row_ite; i < row_; i++)
			for (std::size_t j = 0; j < col_; j++)
				c_mat_alloc[i * col_ + j] = mat_[i * col_ + j]
					+ b_mat_alloc[i * col_ + j];
		
		break;
	}

	return C;
}

template<class T>
cnet::mat<T> cnet::mat<T>::operator-(const cnet::mat<T> &B)
{
	if (col_ != B.get_cols() || row_ != B.get_rows())
		throw std::invalid_argument(
			"invalid argument: Matrices has different sizes");
	
	// Alloc the matrix
	cnet::mat<T> C(row_, col_);
	
	T *c_mat_alloc = C.get_mat_alloc();
	T *b_mat_alloc = B.get_mat_alloc();

	// By default it will increment in 4 by 4
	std::size_t col_inc = 4;
	switch (col_) {
	case 1:
		col_inc = 1;
		break;
	case 2: case 3:
		col_inc = 2;
		break;
	}

	std::size_t row_inc = 4;
	switch (row_) {
	case 1:
		row_inc = 1;
		break;
	case 2: case 3:
		row_inc = 2;
		break;
	}
	
	std::size_t col_mod = col_ % col_inc;
	std::size_t col_ite = col_ - col_mod;
	std::size_t row_mod = row_ % row_inc;
	std::size_t row_ite = row_ - row_mod;

	switch (row_inc) {
	case 1:
		switch (col_inc) {
		case 1:
#pragma omp parallel for collapse(2)
			for (std::size_t i = 0; i < row_; i++)
				for (std::size_t j = 0; j < col_; j++)
					c_mat_alloc[i * col_ + j] = mat_[i * col_ + j]
						- b_mat_alloc[i * col_ + j];
			break;
		case 2:
#pragma omp parallel for
			for (std::size_t i = 0; i < row_; i++) {
				for (std::size_t j = 0; j < col_ite; j += 2) {
					vec2double a = _mm_set_pd(mat_[i * col_ + j + 1], mat_[i * col_ + j]);
					vec2double b = _mm_set_pd(b_mat_alloc[i * col_ + j + 1],
								  b_mat_alloc[i * col_ + j]);
					vec2double result = _mm_sub_pd(a, b);  // Change here
					_mm_storeu_pd(&c_mat_alloc[i * col_ + j], result);
				}

				for (std::size_t j = col_ite; j < col_; j++)
					c_mat_alloc[i * col_ + j] = mat_[i * col_ + j]
						- b_mat_alloc[i * col_ + j];
			}
			break;
		default:
#pragma omp parallel for
			for (std::size_t i = 0; i < row_; i++) {
				for (std::size_t j = 0; j < col_ite; j += 4) {
					vec4double a = _mm256_set_pd(mat_[i * col_ + j + 3],
								     mat_[i * col_ + j + 2],
								     mat_[i * col_ + j + 1],
								     mat_[i * col_ + j]);
					vec4double b = _mm256_set_pd(b_mat_alloc[i * col_ + j + 3],
								     b_mat_alloc[i * col_ + j + 2],
								     b_mat_alloc[i * col_ + j + 1],
								     b_mat_alloc[i * col_ + j]);
					vec4double result = _mm256_sub_pd(a, b);  // Change here
					_mm256_storeu_pd(&c_mat_alloc[i * col_ + j], result);
				}

				for (std::size_t j = col_ite; j < col_; j++)
					c_mat_alloc[i * col_ + j] = mat_[i * col_ + j]
						- b_mat_alloc[i * col_ + j];
			}
			break;
		}
		break;
	case 2:
		switch (col_inc) {
		case 1:
#pragma omp parallel for collapse(2)
			for (std::size_t i = 0; i < row_ite; i += 2)
				for (std::size_t j = 0; j < col_; j++)
					for (std::size_t k = 0; k < 2; k++)
						c_mat_alloc[(i + k) * col_ + j] = mat_[(i + k) * col_ + j]
							- b_mat_alloc[(i + k) * col_ + j];
			break;
		case 2:
#pragma omp parallel for
			for (std::size_t i = 0; i < row_ite; i += 2) {
				for (std::size_t j = 0; j < col_ite; j += 2) {
					for (std::size_t k = 0; k < 2; k++) {
						vec2double a = _mm_set_pd(mat_[(i + k) * col_ + j + 1],
									  mat_[(i + k) * col_ + j]);
						vec2double b = _mm_set_pd(b_mat_alloc[(i + k) * col_ + j + 1],
									  b_mat_alloc[(i + k) * col_ + j]);
						vec2double result = _mm_sub_pd(a, b);
						_mm_storeu_pd(&c_mat_alloc[(i + k) * col_ + j], result);
					}
				}

				for (std::size_t j = col_ite; j < col_; j++)
					for (std::size_t k = 0; k < 2; k++)
						c_mat_alloc[(i + k) * col_ + j] = mat_[(i + k) * col_ + j]
							- b_mat_alloc[(i + k) * col_ + j];
			}
			break;
		default:
#pragma omp parallel for
			for (std::size_t i = 0; i < row_ite; i += 2) {
				for (std::size_t j = 0; j < col_ite; j += 4) {
					for (std::size_t k = 0; k < 2; k++) {
						vec4double a = _mm256_set_pd(mat_[(i + k) * col_ + j + 3],
									     mat_[(i + k) * col_ + j + 2],
									     mat_[(i + k) * col_ + j + 1],
									     mat_[(i + k) * col_ + j]);
						vec4double b = _mm256_set_pd(b_mat_alloc[(i + k) * col_
											 + j + 3],
									     b_mat_alloc[(i + k) * col_
											 + j + 2],
									     b_mat_alloc[(i + k) * col_
											 + j + 1],
									     b_mat_alloc[(i + k) * col_ + j]);
						vec4double result = _mm256_sub_pd(a, b);
						_mm256_storeu_pd(&c_mat_alloc[(i + k) * col_ + j], result);
					}
				}

				for (std::size_t j = col_ite; j < col_; j++) {
					for (std::size_t k = 0; k < 2; k++) {
						c_mat_alloc[(i + k) * col_ + j] = mat_[(i + k) * col_ + j]
							- b_mat_alloc[(i + k) * col_ + j];
					}
				}
			}
			break;
		}

#pragma omp parallel for collapse(2)
		for (std::size_t i = row_ite; i < row_; i++)
			for (std::size_t j = 0; j < col_; j++)
				c_mat_alloc[i * col_ + j] = mat_[i * col_ + j]
					- b_mat_alloc[i * col_ + j];
		break;
	default:
		switch (row_inc) {
		case 1:
#pragma omp parallel for collapse(2)
			for (std::size_t i = 0; i < row_ite; i += 4)
				for (std::size_t j = 0; j < col_; j++)
					for (std::size_t k = 0; k < 4; k++)
						c_mat_alloc[(i + k) * col_ + j] = mat_[(i + k) * col_ + j]
							- b_mat_alloc[(i + k) * col_ + j];
			break;
		case 2:
#pragma omp parallel for
			for (std::size_t i = 0; i < row_ite; i += 4) {
				for (std::size_t j = 0; j < col_; j += 2)
					for (std::size_t k = 0; k < 4; k++) {
						vec2double a = _mm_set_pd(mat_[(i + k) * col_ + j + 1],
									  mat_[(i + k) * col_ + j]);
						vec2double b = _mm_set_pd(b_mat_alloc[(i + k) * col_ + j + 1],
									  b_mat_alloc[(i + k) * col_ + j]);
						vec2double result = _mm_sub_pd(a, b);
						_mm_storeu_pd(&c_mat_alloc[(i + k) * col_ + j], result);
					}

				for (std::size_t j = col_ite; j < col_; j++)
					for (std::size_t k = 0; k < 4; k++)
						c_mat_alloc[(i + k) * col_ + j] = mat_[(i + k) * col_ + j]
							- b_mat_alloc[(i + k) * col_ + j];
			}

			break;
		default:
#pragma omp parallel for
			for (std::size_t i = 0; i < row_ite; i += 4) {
				for (std::size_t j = 0; j < col_; j += 4)
					for (std::size_t k = 0; k < 4; k++) {
						vec4double a = _mm256_set_pd(mat_[(i + k) * col_ + j + 3],
									     mat_[(i + k) * col_ + j + 2],
									     mat_[(i + k) * col_ + j + 1],
									     mat_[(i + k) * col_ + j]);
						vec4double b = _mm256_set_pd(b_mat_alloc[(i + k) * col_
											 + j + 3],
									     b_mat_alloc[(i + k) * col_
											 + j + 2],
									     b_mat_alloc[(i + k) * col_
											 + j + 1],
									     b_mat_alloc[(i + k) * col_ + j]);
						vec4double result = _mm256_sub_pd(a, b);
						_mm256_storeu_pd(&c_mat_alloc[(i + k) * col_ + j], result);
					}

				for (std::size_t j = col_ite; j < col_; j++)
					for (std::size_t k = 0; k < 4; k++)
						c_mat_alloc[(i + k) * col_ + j] = mat_[(i + k) * col_ + j]
							- b_mat_alloc[(i + k) * col_ + j];
			}

			break;
		}
		
#pragma omp parallel for collapse(2)
		for (std::size_t i = row_ite; i < row_; i++)
			for (std::size_t j = 0; j < col_; j++)
				c_mat_alloc[i * col_ + j] = mat_[i * col_ + j]
					- b_mat_alloc[i * col_ + j];
		break;
	}


	return C;
}

// This function needs to support complex variables
template<class T>
cnet::mat<T> cnet::mat<T>::operator*(const cnet::mat<T> &B)
{
	if (col_ != B.get_rows())
		throw std::invalid_argument("invalid argument: n cols != n rows");

	// Alloc padded matrices
	T *p_a, *p_b, *p_c;

	// Pad the matrices
	std::size_t max_dimension =
		std::max(std::max(row_, B.get_rows()), std::max(col_, B.get_cols()));
	std::size_t n = precomputed_pow_2_n[fast_log2(max_dimension) + 1];

	pad_matrix_to_power_2(*this, &p_a, n);
	pad_matrix_to_power_2(B, &p_b, n);

	alloc_mem_matrix((void **) &p_c, n * n, sizeof(T));
	std::memset(p_c, 0.0, sizeof(T) * n * n);

#if __AVX512F__
	cnet::vec8double *a = get_vec_mat_alloc();
	cnet::vec8double *b = B.get_vec_mat_alloc();
#else
	cnet::vec4double *a = get_vec_mat_alloc();
	cnet::vec4double *b = B.get_vec_mat_alloc();
#endif

	strassen_mat_mul(p_a, p_b, p_c, n, n, a, b);

	std::free(p_a);
	std::free(p_b);

	// Alloc the reult matrix
	std::size_t  cols = B.get_cols();
	std::size_t  rows = row_;
	cnet::mat<T> C(rows, cols);

	std::size_t cols_mod	= cols % VEC_SIZE;
	std::size_t cols_to_ite = cols - cols_mod;

	std::size_t rows_mod	= rows % 2;
	std::size_t rows_to_ite = rows - rows_mod;

	T *c_mat_alloc = C.get_mat_alloc();

#pragma omp parallel for
	for (std::size_t i = 0; i < rows_to_ite; i += 2) {
		for (std::size_t j = 0; j < cols_to_ite; j += VEC_SIZE) {
			cnet::vec4double vectorA = _mm256_loadu_pd(&p_c[i * n + j]);
			_mm256_storeu_pd(&c_mat_alloc[i * rows + j], vectorA);

			cnet::vec4double vectorB = _mm256_loadu_pd(&p_c[(i + 1) * n + j]);
			_mm256_storeu_pd(&c_mat_alloc[(i + 1) * rows + j], vectorB);
		}

		for (std::size_t j = cols_to_ite; j < cols; j++) {
			c_mat_alloc[i * rows + j]	= p_c[i * n + j];
			c_mat_alloc[(i + 1) * rows + j] = p_c[i * n + j];
		}
	}

	for (std::size_t i = rows_to_ite; i < rows; i++)
		for (std::size_t j = 0; j < cols; j++)
			c_mat_alloc[i * rows + j] = p_c[i * n + j];

	std::free(p_c);

	return C;
}

template<class T>
void cnet::mat<T>::operator=(std::initializer_list<std::initializer_list<T>> m)
{
	if (m.size() == 0) throw std::invalid_argument("invalid argument: Empty matrix");

	std::size_t n = m.begin()->size();
	for (std::size_t i = 1; i < m.size(); i++)
		if (n != (m.begin() + i)->size())
			throw std::invalid_argument(
				"invalid argument: Invalid structure of the matrix");

	resize(m.size(), n);
	for (std::size_t i = 0; i < m.size(); i++)
		for (std::size_t j = 0; j < n; j++)
			mat_[i * col_ + j] = *((m.begin() + i)->begin() + j);
}

template<class T>
void cnet::mat<T>::operator=(const cnet::mat<T> &B)
{
	resize(B.get_rows(), B.get_cols());

	// Vectorize this loop
	// for (std::size_t i = 0; i < B.get_rows(); i++)
	// 	for (std::size_t j = 0; j < B.get_cols(); j++)
	// 		mat_[i * col_ + j] = B(i, j);

	// Copy the matrices
	std::memcpy((void *) mat_, (void *) B.get_mat_alloc(),
		    sizeof(T) * B.get_cols() * B.get_rows());
}

void cnet::rand_mat(cnet::mat<double> &m, double a, double b)
{
	std::random_device		 rd;
	std::mt19937			 gen(rd());
	std::uniform_real_distribution<> dis(a, b);

	for (std::size_t i = 0; i < m.get_rows(); i++)
		for (std::size_t j = 0; j < m.get_cols(); j++)
			m(i, j) = dis(gen);
}

void cnet::rand_mat(cnet::mat<std::complex<double>> &m, double a, double b)
{
	std::random_device		 rd;
	std::mt19937			 gen(rd());
	std::uniform_real_distribution<> dis(a, b);

	for (std::size_t i = 0; i < m.get_rows(); i++)
		for (std::size_t j = 0; j < m.get_cols(); j++)
			m(i, j) = std::complex<double>(
				dis(gen), dis(gen));	    // rand + i * rand
}

template<typename T>
T cnet::grand_sum(cnet::mat<T> &m)
{
	T res = 0.0;

	for (std::size_t i = 0; i < m.get_rows(); i++)
		for (std::size_t j = 0; j < m.get_cols(); j++)
			res += m(i, j);
	return res;
}

template class cnet::mat<double>;
// Not yet for complex value
// template class cnet::mat<std::complex<double>>;

template double cnet::grand_sum(cnet::mat<double> &m);
#if __AVX512F__
template static void strassen_mat_mul(double *p_a, double *p_b, double *p_c,
				      std::size_t n, std::size_t N, cnet::vec8double *a,
				      cnet::vec8double *b);
#else
template static void strassen_mat_mul(double *p_a, double *p_b, double *p_c,
				      std::size_t n, std::size_t N, cnet::vec4double *a,
				      cnet::vec4double *b);
#endif
template static void pad_matrix_to_power_2(const cnet::mat<double> &A, double **p_a,
					   std::size_t n);
