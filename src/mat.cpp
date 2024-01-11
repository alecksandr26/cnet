#include "cnet/dtypes.hpp"
#include "cnet/mat.hpp"
#include "cnet/utils_mat.hpp"
#include "cnet/utils_avx.hpp"
#include "cnet/strassen.hpp"

#include <cassert>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>

// To optimize the matrix multiplication and the parallelization of it
#include <immintrin.h>	      // For AVX2 intrinsics
#include <omp.h>

using namespace cnet::dtypes;
using namespace cnet::mathops;
using namespace utils;
using namespace strassen;
using namespace std;

#ifndef NDEBUG
static void print_mat_debug(float64 *m, size_t n)
{
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < n; j++)
			cout << m[i * n + j] << " ";
		cout << endl;
	}
}

static void print_mat_debug(float32 *m, size_t n)
{
	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < n; j++)
			cout << m[i * n + j] << " ";
		cout << endl;
	}
}
#endif

template<Numeric T>
cnet::mathops::Mat<T>::Mat(size_t rows, size_t cols)
{
	if (rows == 0 || cols == 0)
		throw invalid_argument("invalid argument: invalid shape of matrix");

	shape_.rows = rows;
	shape_.cols = cols;
	mat_ = (T *) alloc_mem_matrix(rows * cols, sizeof(T));
}

template<Numeric T>
cnet::mathops::Mat<T>::Mat(Shape shape)
{
	if (shape.rows == 0 || shape.cols == 0)
		throw invalid_argument("invalid argument: invalid shape of matrix");

	shape_.rows = shape.rows;
	shape_.cols = shape.cols;
	mat_ = (T *) alloc_mem_matrix(shape.rows * shape.cols, sizeof(T));
}

template<Numeric T>
cnet::mathops::Mat<T>::Mat(size_t rows, size_t cols, T init_val)
{
	if (rows == 0 || cols == 0)
		throw invalid_argument("invalid argument: invalid shape of matrix");

	shape_.rows = rows;
	shape_.cols = cols;
	mat_ = (T *) alloc_mem_matrix(rows * cols, sizeof(T));
	
	init_raw_mat(mat_, rows, cols, init_val);
}

template<Numeric T>
cnet::mathops::Mat<T>::Mat(Shape shape, T init_val)
{
	if (shape.rows == 0 || shape.cols == 0)
		throw invalid_argument("invalid argument: invalid shape of matrix");

	shape_.rows = shape.rows;
	shape_.cols = shape.cols;
	mat_ = (T *) alloc_mem_matrix(shape.rows * shape.cols, sizeof(T));
	
	init_raw_mat(mat_, shape.rows, shape.cols, init_val);
}

template<Numeric T>
cnet::mathops::Mat<T>::Mat(const initializer_list<initializer_list<T>> &M)
{
	if (M.size() == 0) throw invalid_argument("invalid argument: Empty initializer structure");

	size_t n = M.begin()->size();
	for (size_t i = 1; i < M.size(); i++)
		if (n != (M.begin() + i)->size())
			throw invalid_argument("invalid argument: Invalid structure of the matrix");

	shape_.cols = n;
	shape_.rows = M.size();
	mat_ = (T *) alloc_mem_matrix(shape_.rows * shape_.cols, sizeof(T));
	
	for (size_t i = 0; i < shape_.rows; i++)
		for (size_t j = 0; j < shape_.cols; j++)
			mat_[i * n + j] = *((M.begin() + i)->begin() + j);
}

// It doens't compile 
// template<Numeric T>
// cnet::mathops::Mat<T>::Mat(const initializer_list<T> &M)
// {
// 	if (M.size() == 0) throw invalid_argument("invalid argument: Empty initializer structure");

// 	size_t n = M.size();
	
// 	shape_.cols = n;
// 	shape_.rows = 1;
// 	mat_ = (T *) alloc_mem_matrix(shape_.rows * shape_.cols, sizeof(T));
	
// 	for (size_t i = 0; i < shape_.cols; i++)
// 		mat_[i] = *(M.begin() + i);
// }

template<Numeric T>
cnet::mathops::Mat<T>::Mat(const Mat<T> &M)
{
	T *allocated_mat = M.get_allocated_mat();
	
	shape_.cols = M.get_cols();
	shape_.rows = M.get_rows();
	mat_ = (T *) alloc_mem_matrix(shape_.rows * shape_.cols, sizeof(T));
	
	cp_raw_mat(mat_, allocated_mat, shape_.rows,
		   shape_.cols, shape_.cols, shape_.cols);
}

template<Numeric T>
void cnet::mathops::Mat<T>::operator()(const Mat<T> &M)
{
	if (mat_) free_mem_matrix((void *) mat_);

	T *allocated_mat = M.get_allocated_mat();
	
	shape_.cols = M.get_cols();
	shape_.rows = M.get_rows();
	mat_ = (T *) alloc_mem_matrix(shape_.rows * shape_.cols, sizeof(T));
	
	cp_raw_mat(mat_, allocated_mat, shape_.rows,
		   shape_.cols, shape_.cols, shape_.cols);
}

template<Numeric T>
cnet::mathops::Mat<T>::Mat(void)
{
	shape_.cols = shape_.rows   = 0;
	mat_	      = NULL;
}

template<Numeric T>
cnet::mathops::Mat<T>::~Mat(void)
{
	if (mat_) free_mem_matrix((void *) mat_);

	mat_	      = NULL;
	shape_.cols = shape_.rows   = 0;
}

template<Numeric T>
Mat<T> &cnet::mathops::Mat<T>::resize(size_t rows, size_t cols)
{
	if (rows == 0 || cols == 0)
		throw invalid_argument("invalid argument: invalid shape of matrix");
	if (rows == shape_.rows && cols == shape_.cols) return *this;
	if (mat_ != NULL) free_mem_matrix((void *) mat_);

	shape_.rows = rows;
	shape_.cols = cols;
	mat_ = (T *) alloc_mem_matrix(rows * cols, sizeof(T));

	return *this;
}

template<Numeric T>
Mat<T> &cnet::mathops::Mat<T>::resize(size_t rows, size_t cols, T init_val)
{
	resize(rows, cols);
	init_raw_mat(mat_, rows, cols, init_val);
	return *this;
}


template<Numeric T>
Mat<T> &cnet::mathops::Mat<T>::resize(Shape shape)
{
	if (shape.rows == 0 || shape.cols == 0)
		throw invalid_argument("invalid argument: invalid shape of matrix");
	if (shape.rows == shape_.rows && shape.cols == shape_.cols) return *this;
	if (mat_ != NULL) free_mem_matrix((void *) mat_);

	shape_ = shape;
	mat_ = (T *) alloc_mem_matrix(shape.rows * shape.cols, sizeof(T));

	return *this;
}

template<Numeric T>
Mat<T> &cnet::mathops::Mat<T>::resize(Shape shape, T init_val)
{
	resize(shape);
	init_raw_mat(mat_, shape.rows, shape.cols, init_val);
	return *this;
}

template<Numeric T>
size_t cnet::mathops::Mat<T>::get_rows(void) const
{
	return shape_.rows;
}

template<Numeric T>
size_t cnet::mathops::Mat<T>::get_cols(void) const
{
	return shape_.cols;
}

template<Numeric T>
Shape cnet::mathops::Mat<T>::get_shape(void) const
{
	return shape_;
}

// TODO: Optimize this thin
template<Numeric T>
Mat<T> cnet::mathops::Mat<T>::transpose(void) const
{
	Mat<T> R(shape_);

	for (size_t i = 0; i < shape_.rows; i++)
		for (size_t j = 0; j < shape_.cols; j++)
			R(j, i) = mat_[i * shape_.cols + j];

	return R;
}

template<Numeric T>
Mat<T> &cnet::mathops::Mat<T>::transpose_(void)
{
	T *allocated_mat = (T *) alloc_mem_matrix(shape_.rows * shape_.cols, sizeof(T));

	for (size_t i = 0; i < shape_.rows; i++)
		for (size_t j = 0; j < shape_.cols; j++)
			allocated_mat[j * shape_.rows + i] = mat_[i * shape_.cols + j];

	// Swap the shape
	shape_ = {shape_.cols, shape_.cols};
	free(mat_);
	mat_ = allocated_mat;

	return *this;
}

template<Numeric T>
T *cnet::mathops::Mat<T>::get_allocated_mat(void) const
{
	if (mat_ == NULL)
		throw runtime_error("uninitialized Mat: Matrix is uninitialized");
	return mat_;
}

template<Numeric T>
T &cnet::mathops::Mat<T>::operator()(size_t i, size_t j) const
{
	if (i >= shape_.rows || j >= shape_.cols)
		throw out_of_range("out of range: Matrix subscript out of bounds");
	return mat_[i * shape_.cols + j];
}

template<Numeric T>
T &cnet::mathops::Mat<T>::operator[](size_t i) const
{
	if (i >= shape_.cols)
		throw out_of_range("out of range: Matrix as array subscript out of bounds");
	if (1 < shape_.rows)
		throw runtime_error("invalid shape of array: Matrix doesn't have a shape of array (n , 0)");
	
	return mat_[i];
}

template<Numeric T>
Mat<T> cnet::mathops::Mat<T>::operator+(const Mat<T> &B) const
{
	if (shape_.cols != B.get_cols() || shape_.rows != B.get_rows())
		throw invalid_argument("invalid argument: Matrices have different sizes");

	// Alloc the matrix
	Mat<T> C(shape_);

	T *c_allocated_mat = C.get_allocated_mat();
	T *b_allocated_mat = B.get_allocated_mat();
	
	cp_raw_mat(c_allocated_mat, mat_, shape_.rows,
		   shape_.cols, shape_.cols, shape_.cols);
	add_raw_mat(c_allocated_mat, b_allocated_mat, shape_.rows, shape_.cols);
	
	return C;
}

template<Numeric T>
void cnet::mathops::Mat<T>::operator+=(const Mat<T> &B)
{
	if (shape_.cols != B.get_cols() || shape_.rows != B.get_rows())
		throw invalid_argument("invalid argument: Matrices have different sizes");
	
	T *b_allocated_mat = B.get_allocated_mat();
	add_raw_mat(mat_, b_allocated_mat, shape_.rows, shape_.cols);
}

template<Numeric T>
Mat<T> cnet::mathops::Mat<T>::operator-(const Mat<T> &B) const
{
	if (shape_.cols != B.get_cols() || shape_.rows != B.get_rows())
		throw invalid_argument("invalid argument: Matrices have different sizes");

	// Alloc the matrix
	Mat<T> C(shape_);

	T *c_allocated_mat = C.get_allocated_mat();
	T *b_allocated_mat = B.get_allocated_mat();
	
	cp_raw_mat(c_allocated_mat, mat_, shape_.rows,
		   shape_.cols, shape_.cols, shape_.cols);
	sub_raw_mat(c_allocated_mat, b_allocated_mat, shape_.rows, shape_.cols);

	return C;
}

template<Numeric T>
void cnet::mathops::Mat<T>::operator-=(const Mat<T> &B)
{
	if (shape_.cols != B.get_cols() || shape_.rows != B.get_rows())
		throw std::invalid_argument("invalid argument: Matrices have different sizes");

	T *b_allocated_mat = B.get_allocated_mat();
	sub_raw_mat(mat_, b_allocated_mat, shape_.rows, shape_.cols);
}

// This function needs to support complex variables
template<Numeric T>
Mat<T> cnet::mathops::Mat<T>::operator*(const Mat<T> &B) const
{
	if (shape_.cols != B.get_rows())
		throw invalid_argument("invalid argument: Matrices have different sizes");

	T *padded_mat_a, *padded_mat_b, *padded_mat_c;
	
	size_t n = pad_size(shape_.rows, shape_.cols, B.get_rows(), B.get_cols());
	padded_mat_a = pad_mat(*this, n);
	padded_mat_b = pad_mat(B, n);

	padded_mat_c = (T *) alloc_mem_matrix(n * n, sizeof(T));
	
	// Needs to support complex values
	init_raw_mat(padded_mat_c, n, n, (T) 0.0);
	
	mat_mul(padded_mat_a, padded_mat_b, padded_mat_c, n, n);
	
	free(padded_mat_a);
	free(padded_mat_b);
	
	// Alloc the reult matrix
	size_t  cols	 = B.get_cols();
	size_t  rows	 = shape_.rows;
	Mat<T> C(rows, cols);
	T *c_allocated_mat = C.get_allocated_mat();

	cp_raw_mat(c_allocated_mat, padded_mat_c, rows, cols, cols, n);
	
	free(padded_mat_c);

	return C;
}

// This function needs to support complex variables
template<Numeric T>
void cnet::mathops::Mat<T>::operator*=(const Mat<T> &B)
{
	if (shape_.cols != B.get_rows())
		throw invalid_argument("invalid argument: Matrices have different sizes");

	T *padded_mat_a, *padded_mat_b, *padded_mat_c;
	
	size_t n = pad_size(shape_.rows, shape_.cols, B.get_rows(), B.get_cols());
	padded_mat_a = pad_mat(*this, n);
	padded_mat_b = pad_mat(B, n);

	padded_mat_c = (T *) alloc_mem_matrix(n * n, sizeof(T));
	
	// Needs to support complex values
	init_raw_mat(padded_mat_c, n, n, (T) 0.0);
	
	mat_mul(padded_mat_a, padded_mat_b, padded_mat_c, n, n);
	
	free(padded_mat_a);
	free(padded_mat_b);
	
	// Alloc the reult matrix
	size_t  cols	 = B.get_cols();
	size_t  rows	 = shape_.rows;
	resize(rows, cols);

	cp_raw_mat(mat_, padded_mat_c, rows, cols, cols, n);
	
	free(padded_mat_c);
}

template<Numeric T>
Mat<T> cnet::mathops::Mat<T>::operator^(const Mat<T> &B) const
{
	if (shape_.cols != B.get_cols() || shape_.rows != B.get_rows())
		throw invalid_argument("invalid argument: Matrices have different sizes");
	
	Mat<T> C(shape_);
	T *c_allocated_mat = C.get_allocated_mat();
	T *b_allocated_mat = B.get_allocated_mat();

	cp_raw_mat(c_allocated_mat, mat_, shape_.rows,
		   shape_.cols, shape_.cols, shape_.cols);
	hardmard_mul_raw_mat(c_allocated_mat, b_allocated_mat, shape_.rows, shape_.cols);

	return C;
}

template<Numeric T>
void cnet::mathops::Mat<T>::operator^=(const Mat<T> &B)
{
	if (shape_.cols != B.get_cols() || shape_.rows != B.get_rows())
		throw invalid_argument("invalid argument: Matrices have different sizes");

	T *b_allocated_mat = B.get_allocated_mat();
	hardmard_mul_raw_mat(mat_, b_allocated_mat, shape_.rows, shape_.cols);
}

template<Numeric T>
void cnet::mathops::Mat<T>::operator=(const initializer_list<initializer_list<T>> &M)
{
	if (M.size() == 0) throw invalid_argument("invalid argument: Empty initializer structure");

	size_t n = M.begin()->size();
	for (size_t i = 1; i < M.size(); i++)
		if (n != (M.begin() + i)->size())
			throw invalid_argument("invalid argument: Invalid structure of the matrix");
	
	resize(M.size(), n);
	
	for (size_t i = 0; i < shape_.rows; i++)
		for (size_t j = 0; j < shape_.cols; j++)
			mat_[i * n + j] = *((M.begin() + i)->begin() + j);
}


// template<Numeric T>
// void cnet::mathops::Mat<T>::operator=(const initializer_list<T> &M)
// {
// 	if (M.size() == 0) throw invalid_argument("invalid argument: Empty initializer structure");

// 	size_t n = M.size();
	
// 	shape_.cols = n;
// 	shape_.rows = 1;
// 	mat_ = (T *) alloc_mem_matrix(shape_.rows * shape_.cols, sizeof(T));
	
// 	for (size_t i = 0; i < shape_.cols; i++)
// 		mat_[i] = *(M.begin() + i);
// }

template<Numeric T>
void cnet::mathops::Mat<T>::operator=(const Mat<T> &M)
{
	T *allocated_mat = M.get_allocated_mat();
	
	resize(M.get_shape());
	cp_raw_mat(mat_, allocated_mat, shape_.rows,
		   shape_.cols, shape_.cols, shape_.cols);
}

template<Numeric T>
Mat<T> cnet::mathops::Mat<T>::operator*(T a) const
{
	if (shape_.rows == 0 || shape_.cols == 0)
		throw invalid_argument("invalid argument: invalid shape of matrix");
	
	// Alloc the matrix
	Mat<T> C(shape_);
	T *c_allocated_mat = C.get_allocated_mat();

	cp_raw_mat(c_allocated_mat, mat_, shape_.rows,
		   shape_.cols, shape_.cols, shape_.cols);
	scalar_mul_raw_mat(c_allocated_mat, a, shape_.rows, shape_.cols);
	
	return C;
}

template<Numeric T>
void cnet::mathops::Mat<T>::operator*=(T a)
{
	if (shape_.rows == 0 || shape_.cols == 0)
		throw invalid_argument("invalid argument: invalid shape of matrix");
	
	scalar_mul_raw_mat(mat_, a, shape_.rows, shape_.cols);
}

template<Numeric T>
Mat<T> &cnet::mathops::Mat<T>::rand(T a, T b)
{
	if (shape_.rows == 0 || shape_.cols == 0)
		throw invalid_argument("invalid argument: Invalid matrix");

	random_device		 rd;
	mt19937			 gen(rd());
	uniform_real_distribution<> dis(a, b);

#pragma omp parallel for collapse(2)
	for (size_t i = 0; i < shape_.rows; i++)
		for (size_t j = 0; j < shape_.cols; j++)
			mat_[i * shape_.cols + j] = dis(gen);
	return *this;
}

template<Numeric T>
T cnet::mathops::Mat<T>::grand_sum(void) const
{
	if (shape_.rows == 0 || shape_.cols == 0)
		throw invalid_argument("invalid argument: Invalid matrix");
	
	return grand_sum_raw_mat(mat_, shape_.rows, shape_.cols);
}

template class cnet::mathops::Mat<float32>;
template class cnet::mathops::Mat<float64>;

// Not yet for complex value
// template Numeric cnet::mathops::Mat<std::complex<float64>>;
// How we can deal the complex data type
// https://stackoverflow.com/questions/13636540/how-to-check-for-the-type-of-a-template-parameter
