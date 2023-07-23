#include <iostream>
#include <cassert>
#include <complex>
#include <stdexcept>
#include <random>

#include "../include/cnet/mat.hpp"

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
cnet::mat<T>::mat(std::initializer_list<std::initializer_list<T>> m)
{
	if (m.size() == 0)
		throw std::invalid_argument("invalid argument: Empty matrix");

	std::size_t n = m.begin()->size();
	for (std::size_t i = 1; i < m.size(); i++)
		if (n != (m.begin() + i)->size())
			throw std::invalid_argument("invalid argument: Invalid structure of the matrix");
	
	
	this->col_ = n;
	this->row_ = m.size();
	this->mat_ = new T[this->col_ * this->row_];
	for (std::size_t i = 0; i < m.size(); i++)
		for (std::size_t j = 0; j < n; j++)
			(*this)(i, j) = *((m.begin() + i)->begin() + j);
}

// template<class T>
// cnet::mat<T>::mat(cnet::mat<T> &m)
// {
// 	col_ = m.get_n_cols();
// 	row_ = m.get_n_rows();
// 	mat_ = new T[col_ * row_];
// 	for (std::size_t i = 0; i < row_; i++)
// 		for (std::size_t j = 0; j < col_; j++)
// 			(*this)(i, j) = m(i, j);
// }

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
void cnet::mat<T>::rsize(std::size_t rows, std::size_t cols)
{
	if (mat_ != NULL)
		delete[] mat_;

	col_ = cols;
	row_ = rows;
	mat_ = new T[col_ * row_];
}

template<class T>
std::size_t cnet::mat<T>::get_n_rows() const
{
	return row_;
}

template<class T>
std::size_t cnet::mat<T>::get_n_cols() const
{
	return col_;
}

template<class T>
T &cnet::mat<T>::operator()(std::size_t row, std::size_t col) const
{
	if (row >= row_ || col >= col_)
		throw std::out_of_range("out of range: Matrix subscript out of bounds");
	return mat_[row * col_ + col];
}

template<class T>
cnet::mat<T> cnet::mat<T>::operator+(const cnet::mat<T> &B)
{
	if (this->get_n_cols() != B.get_n_cols()
	    || this->get_n_rows() != B.get_n_rows())
		throw std::invalid_argument("invalid argument: Matrices has different sizes");

	cnet::mat<T> C(this->get_n_rows(), this->get_n_cols());
	for (std::size_t i = 0; i < this->get_n_rows(); i++)
		for (std::size_t j = 0; j < this->get_n_cols(); j++)
			C(i, j) = (*this)(i, j) + B(i, j);
	
	return C;
}

template<class T>
cnet::mat<T> cnet::mat<T>::operator*(const cnet::mat<T> &B)
{
	if (this->get_n_cols() != B.get_n_rows())
		throw std::invalid_argument("invalid argument: n cols != n rows");
	
	cnet::mat<T> C(this->get_n_rows(), B.get_n_cols());
	for (std::size_t i = 0; i < this->get_n_rows(); i++)
		for (std::size_t j = 0; j < B.get_n_cols(); j++) {
			C(i, j) = 0;
			for (size_t k = 0; k < this->get_n_cols(); k++)
				C(i, j) += (*this)(i, k) * B(k, j);
		}

	
	return C;
}


template<class T>
cnet::mat<T> cnet::mat<T>::operator=(std::initializer_list<std::initializer_list<T>> m)
{
	if (m.size() == 0)
		throw std::invalid_argument("invalid argument: Empty matrix");

	std::size_t n = m.begin()->size();
	for (std::size_t i = 1; i < m.size(); i++)
		if (n != (m.begin() + i)->size())
			throw std::invalid_argument("invalid argument: Invalid structure of the matrix");

	
	cnet::mat<T> C(m.size(), n);
	for (std::size_t i = 0; i < m.size(); i++)
		for (std::size_t j = 0; j < n; j++)
			C(i, j) = *((m.begin() + i)->begin() + j);

	return C;
}



void cnet::rand_mat(cnet::mat<double> &m, double a, double b)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(a, b);
	
	for (std::size_t i = 0; i < m.get_n_rows(); i++)
		for (std::size_t j = 0; j < m.get_n_cols(); j++)
			m(i, j) =  dis(gen);
}

void cnet::rand_mat(cnet::mat<std::complex<double>> &m, double a, double b)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(a, b);
	
	for (std::size_t i = 0; i < m.get_n_rows(); i++)
		for (std::size_t j = 0; j < m.get_n_cols(); j++)
			m(i, j) = std::complex<double>(dis(gen), dis(gen)); // rand + i * rand
}



template class cnet::mat<double>;
template class cnet::mat<std::complex<double>>;
