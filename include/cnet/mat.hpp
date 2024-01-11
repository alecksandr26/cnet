/*
  @file Mat.hpp
  @brief
  
  @author Erick Carrillo.
  @copyright Copyright (C) 2023, Erick Alejandro Carrillo López, All rights reserved.
  @license This project is released under the MIT License
*/

#ifndef MAT_INCLUDED
#define MAT_INCLUDED

#include <cstddef>
#include <ostream>
#include <type_traits>
#include <concepts>

#include "dtypes.hpp"

namespace cnet::mathops {
	using namespace cnet::dtypes;
	
	class Shape {
	public:
		std::size_t rows, cols;

		Shape() = default;
		
		constexpr Shape(const std::initializer_list<std::size_t> &l)
		{
			if (l.size() > 2)
				throw std::invalid_argument("invalid argument: Invalid list of initializer");
			rows = *(l.begin());
			cols = *(l.begin() + 1);
		}
		
		constexpr Shape(std::size_t init_rows, std::size_t init_cols)
		{
			rows = init_rows;
			cols = init_cols;
		}

		bool operator==(const Shape &s) { return rows == s.rows and cols == s.cols; }
		
		friend std::ostream& operator<<(std::ostream& os, const Shape &shape) {
			os << "shape=(rows=" << shape.rows << ", cols=" << shape.cols << ")";
			return os;
		}
	};

	template<typename T>
	concept Numeric = std::is_arithmetic_v<T>;
	
	template<Numeric T>
	class Mat {
	public:
		Mat(void);
		~Mat(void);
		
		Mat(std::size_t rows, std::size_t cols);
		Mat(Shape shape);
		Mat(std::size_t rows, std::size_t cols, T init_val);
		Mat(Shape shape, T init_val);
		Mat(const std::initializer_list<std::initializer_list<T>> &M);
		// Mat(const std::initializer_list<T> &M);
		Mat(const Mat<T> &M);
		void operator()(const Mat<T> &M);
		
		Mat<T> &resize(std::size_t rows, std::size_t cols);
		Mat<T> &resize(std::size_t rows, std::size_t cols, T init_val);
		Mat<T> &resize(Shape shape);
		Mat<T> &resize(Shape shape, T init_val);
		
		std::size_t get_rows(void) const;
		std::size_t get_cols(void) const;
		Shape get_shape(void) const;
		Mat<T> transpose(void) const;
		Mat<T> &transpose_(void);
		T *get_allocated_mat(void) const;
		Mat<T> &rand(T a, T b);
		T grand_sum(void) const;


		T &operator()(std::size_t row, std::size_t col) const;
		T &operator[](std::size_t i) const;
		void operator=(const std::initializer_list<std::initializer_list<T>> &M);
		// void operator=(const std::initializer_list<T> &M);
		void operator=(const Mat<T> &B);
		Mat<T> operator+(const Mat<T> &B) const;
		Mat<T> operator-(const Mat<T> &B) const;
		Mat<T> operator*(const Mat<T> &B) const;
		Mat<T> operator*(T a) const;
		Mat<T> operator^(const Mat<T> &B) const; // element-wise multiplication or Hadamard product
		void operator+=(const Mat<T> &B);
		void operator-=(const Mat<T> &B);
		void operator*=(const Mat<T> &B);
		void operator*=(T a);
		void operator^=(const Mat<T> &B); // element-wise multiplication or Hadamard product
		
		friend std::ostream &operator<<(std::ostream &os, const Mat<T> &M)
		{
			os << "Mat=(\n";
			os << "[";
			for (std::size_t i = 0; i < M.shape_.rows; i++) {
				if (i > 0)
					os << " ";
				os << "[";
				for (std::size_t j = 0; j < M.shape_.cols; j++) {
					os << M.mat_[i * M.shape_.cols + j];
					if (j < M.shape_.cols - 1)
						os << "\t";
				}
				os << "]";
				if (i < M.shape_.rows - 1)
					os << "\n";
			}
			os << "],\n" << M.shape_ << ", "
			   << ((std::is_same<T, float64>::value)
			       ? CnetDtype(FLOAT_64_DTYPE)
			       : CnetDtype(FLOAT_32_DTYPE)) << ", "
			   << "addrs=" << (void *) &M << ")";
			
			return os;
		}

	protected:
		Shape shape_;
		T *mat_;
	};
}

#endif
