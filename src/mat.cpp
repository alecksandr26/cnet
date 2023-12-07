#include <iostream>
#include <cassert>
#include <complex>
#include <stdexcept>
#include <random>

#include "cnet/mat.hpp"

template<class T>
cnet::mat<T>::mat(std::size_t row, std::size_t col)
{
	if (row == 0 || col == 0)
		throw std::invalid_argument("invalid argument: row and col can't be zero");
	
	row_ = row;
	col_ = col;
	mat_ = new T[row * col];
}

template<class T>
cnet::mat<T>::mat(std::size_t row, std::size_t col, T initial)
{
	if (row == 0 || col == 0)
		throw std::invalid_argument("invalid argument: row and col can't be zero");
	
	row_ = row;
	col_ = col;
	mat_ = new T[row * col];

	for (std::size_t i = 0; i < row_; i++)
		for (std::size_t j = 0; j < col_; j++)
			mat_[i * col_ + j] = initial;
}

template<class T>
cnet::mat<T>::mat(std::initializer_list<std::initializer_list<T>> m)
{
	if (m.size() == 0)
		throw std::invalid_argument("invalid argument: Empty matrix");

	std::size_t n = m.begin()->size();
	for (std::size_t i = 1; i < m.size(); i++)
		if (n != (m.begin() + i)->size())
			throw std::invalid_argument("invalid argument: Invalid structure of the matrix");
	
	col_ = n;
	row_ = m.size();
	mat_ = new T[col_ * row_];
	for (std::size_t i = 0; i < row_; i++)
		for (std::size_t j = 0; j < col_; j++)
			mat_[i * col_ + j] = *((m.begin() + i)->begin() + j);
}

template<class T>
cnet::mat<T>::mat(const cnet::mat<T> &m)
{	
	col_ = m.get_cols();
	row_ = m.get_rows();
	mat_ = new T[col_ * row_];
	for (std::size_t i = 0; i < row_; i++)
		for (std::size_t j = 0; j < col_; j++)
			mat_[i * col_ + j] = m(i, j);
}

template<class T>
cnet::mat<T>::mat(void)
{
	col_ = row_ = 0;
	mat_ = NULL;
}

template<class T>
cnet::mat<T>::~mat(void)
{
	delete[] mat_;
	mat_ = NULL;
	col_ = row_ = 0;
}

template<class T>
void cnet::mat<T>::resize(std::size_t rows, std::size_t cols)
{
	if (mat_ != NULL)
		delete[] mat_;

	col_ = cols;
	row_ = rows;
	mat_ = new T[col_ * row_];
}

template<class T>
std::size_t cnet::mat<T>::get_rows() const
{
	return row_;
}

template<class T>
std::size_t cnet::mat<T>::get_cols() const
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
		throw std::invalid_argument("invalid argument: Matrices have different sizes");

	cnet::mat<T> C(row_, col_);
	for (std::size_t i = 0; i < row_; i++)
		for (std::size_t j = 0; j < col_; j++)
			C(i, j) = mat_[i * col_ + j] + B(i, j);
	
	return C;
}

template<class T>
cnet::mat<T> cnet::mat<T>::operator-(const cnet::mat<T> &B)
{
	if (col_ != B.get_cols() || row_ != B.get_rows())
		throw std::invalid_argument("invalid argument: Matrices has different sizes");

	cnet::mat<T> C(row_, col_);
	for (std::size_t i = 0; i < row_; i++)
		for (std::size_t j = 0; j < col_; j++)
			C(i, j) = mat_[i * col_ + j] - B(i, j);
	
	return C;
}


// This function needs to support complex variables
template<class T>
cnet::mat<T> cnet::mat<T>::operator*(const cnet::mat<T> &B)
{
	if (col_ != B.get_rows())
		throw std::invalid_argument("invalid argument: n cols != n rows");
	
	cnet::mat<T> R(row_, B.get_cols());
	for (std::size_t i = 0; i < row_; i++)
		for (std::size_t j = 0; j < B.get_cols(); j++) {
			R(i, j) = 0.0; // We have an error here
			for (size_t k = 0; k < col_; k++)
				R(i, j) += mat_[i * col_ + k] * B(k, j);
		}

	
	return R;
}

template<class T>
void cnet::mat<T>::operator=(std::initializer_list<std::initializer_list<T>> m)
{
	if (m.size() == 0)
		throw std::invalid_argument("invalid argument: Empty matrix");

	std::size_t n = m.begin()->size();
	for (std::size_t i = 1; i < m.size(); i++)
		if (n != (m.begin() + i)->size())
			throw std::invalid_argument("invalid argument: Invalid structure of the matrix");

	resize(m.size(), n);
	for (std::size_t i = 0; i < m.size(); i++)
		for (std::size_t j = 0; j < n; j++)
			mat_[i * col_ + j] = *((m.begin() + i)->begin() + j);

}

template<class T>
void cnet::mat<T>::operator=(const cnet::mat<T> &B)
{
	resize(B.get_rows(), B.get_cols());
	
	for (std::size_t i = 0; i < B.get_rows(); i++)
		for (std::size_t j = 0; j < B.get_cols(); j++)
			mat_[i * col_ + j] = B(i, j);
}


void cnet::rand_mat(cnet::mat<double> &m, double a, double b)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(a, b);
	
	for (std::size_t i = 0; i < m.get_rows(); i++)
		for (std::size_t j = 0; j < m.get_cols(); j++)
			m(i, j) =  dis(gen);
}

void cnet::rand_mat(cnet::mat<std::complex<double>> &m, double a, double b)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(a, b);
	
	for (std::size_t i = 0; i < m.get_rows(); i++)
		for (std::size_t j = 0; j < m.get_cols(); j++)
			m(i, j) = std::complex<double>(dis(gen), dis(gen)); // rand + i * rand
}

template<typename T>
T cnet::grand_sum(cnet::mat<T> &m)
{
	T res = 0;
	
	for (std::size_t i = 0; i < m.get_rows(); i++)
		for (std::size_t j = 0; j < m.get_cols(); j++)
			res += m(i, j);
	return res;
}

template class cnet::mat<double>;
template class cnet::mat<std::complex<double>>;
template double cnet::grand_sum(cnet::mat<double> &m);
