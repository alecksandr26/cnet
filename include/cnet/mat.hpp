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

namespace cnet {
	template<class T> class mat {
	public:
		mat(std::size_t rows, std::size_t cols);
		mat(std::size_t rows, std::size_t cols, T initial);
		mat(std::initializer_list<std::initializer_list<T>> m);
		mat(const mat<T> &m);
		mat(void);
		~mat(void);
		
		void resize(std::size_t rows, std::size_t cols);
		std::size_t get_rows() const;
		std::size_t get_cols() const;
		mat<T> transpose(void);
		
		T &operator()(std::size_t row, std::size_t col) const;
		mat<T> operator+(const mat<T> &B);
		mat<T> operator-(const mat<T> &B);
		mat<T> operator*(const mat<T> &B);
		void operator=(std::initializer_list<std::initializer_list<T>> m);
		void operator=(const mat<T> &B);
		
		friend std::ostream &operator<<(std::ostream& os, const mat<T> &m)
		{
			for (std::size_t i = 0; i < m.get_rows(); i++) {
				for (std::size_t j = 0; j < m.get_cols(); j++) {
					if (j == 0)
						os << '|';
					os << std::fixed << std::setprecision(5) << m(i, j);
					if (j == m.get_cols() - 1)
						os << '|';
					else
						os << ' ';
					
				}
				os << '\n';
			}
			
			return os;
		}

	private:
		std::size_t row_, col_;
		T *mat_;
	};

	extern void rand_mat(mat<double> &m, double a, double b);
	extern void rand_mat(mat<std::complex<double>> &m, double a, double b);
	
	template<typename T>
	extern T grand_sum(mat<T> &m);
}

#endif
