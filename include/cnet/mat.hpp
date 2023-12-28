/*
  @file mat.hpp
  @brief
  
  @author Erick Carrillo.
  @copyright Copyright (C) 2023, Erick Alejandro Carrillo López, All rights reserved.
  @license This project is released under the MIT License
*/

#ifndef MAT_INCLUDED
#define MAT_INCLUDED

#include <cstddef>
#include <complex>
#include <iomanip>
#include <vector>
#include <immintrin.h>

namespace cnet {
	typedef __m512d vec8double;
	typedef __m256d vec4double;
	
	template<class T>
	class mat {
	public:
		mat(std::size_t rows, std::size_t cols);
		mat(std::size_t rows, std::size_t cols, T initial);
		mat(std::initializer_list<std::initializer_list<T>> m);
		mat(const mat<T> &m);
		mat(void);
		~mat(void);
		
		void resize(std::size_t rows, std::size_t cols);
		void resize(std::size_t rows, std::size_t cols, T initial);
		std::size_t get_rows(void) const;
		std::size_t get_cols(void) const;
		mat<T> transpose(void);
		T *get_mat_alloc(void) const;
		
#if __AVX512F__
		cnet::vec8double *get_vec_mat_alloc(void) const;
#else
		cnet::vec4double *get_vec_mat_alloc(void) const;
#endif
		
		
		T &operator()(std::size_t row, std::size_t col) const;
		mat<T> operator+(const mat<T> &B);
		mat<T> operator-(const mat<T> &B);
		mat<T> operator*(const mat<T> &B);
		void operator=(std::initializer_list<std::initializer_list<T>> m);
		void operator=(const mat<T> &B);
		
		friend std::ostream &operator<<(std::ostream &os, const mat<T> &m) {
			os << "[";
			for (std::size_t i = 0; i < m.row_; i++) {
				if (i > 0) {
					os << " ";
				}
				os << "[";
				for (std::size_t j = 0; j < m.col_; j++) {
					os << m.mat_[i * m.col_ + j];
					if (j < m.col_ - 1) {
						os << "\t";
					}
				}
				os << "]";
				if (i < m.row_ - 1) {
					os << "\n";
				}
			}
			os << "]";

			return os;
		}

	private:
		std::size_t row_, col_;
		T *mat_;
#if __AVX512F__
		cnet::vec8double *vec_mat_alloc;
#else
		cnet::vec4double *vec_mat_alloc;
#endif
	};

	extern void rand_mat(mat<double> &m, double a, double b);
	extern void rand_mat(mat<std::complex<double>> &m, double a, double b);
	
	template<typename T>
	extern T grand_sum(mat<T> &m);
}

#endif
