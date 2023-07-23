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
		mat(std::initializer_list<std::initializer_list<T>> m);
		// mat(cnet::mat<T> &m);
		mat(void);
		~mat(void);

		void rsize(std::size_t rows, std::size_t cols);
		std::size_t get_n_rows() const;
		std::size_t get_n_cols() const;
		
		T &operator()(std::size_t row, std::size_t col) const;
		mat<T> operator+(const mat<T> &B);
		mat<T> operator*(const mat<T> &B);
		mat<T> operator=(std::initializer_list<std::initializer_list<T>> m);
		
		friend std::ostream &operator<<(std::ostream& os, const mat<T> &m)
		{
			for (std::size_t i = 0; i < m.get_n_rows(); i++) {
				for (std::size_t j = 0; j < m.get_n_cols(); j++) {
					if (j == 0)
						os << '|';
					os << std::fixed << std::setprecision(5) << m(i, j);
					if (j == m.get_n_cols() - 1)
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
}

#endif
