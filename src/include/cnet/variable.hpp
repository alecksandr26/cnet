// This thing pretends to be the abstraction of my matrices

#ifndef VARIABLE_INCLUDED
#define VARIABLE_INCLUDED

#include <ostream>
#include "dtypes.hpp"
#include "mat.hpp"

namespace cnet::variable {
	using namespace mathops;
	using namespace dtypes;

	union Mats {
		Mat<float32> f32;
		Mat<float64> f64;

		// To initialized and deallocates the memory
		Mats() : f32() {} 
		~Mats() {f32.~Mat<float32>();}
		
		// Add other constructors if needed
		Mats(const Mat<float32>& m) : f32(m) {}
		Mats(const Mat<float64>& m) : f64(m) {}
	};
	
	class Var {
	public:
		Var(void);
		~Var(void);

		Var(size_t rows, size_t cols);
		Var(size_t rows, size_t cols, CnetDtype dtype);
		Var(Shape shape);
		Var(Shape shape, CnetDtype dtype);
		
		// Add initialzerz random initializers
		Var(size_t rows, size_t cols, float32 init_val);
		Var(Shape shape, float32 init_val);
		
		Var(size_t rows, size_t cols, float64 init_val);
		Var(Shape shape, float64 init_val);

		// Try list initialzerz 
		Var(const Mat<float32> &M);
		Var(const Mat<float64> &M);
		Var(const Var &V);

		Shape get_shape(void) const;
		std::size_t get_cols(void) const;
		std::size_t get_rows(void) const;
		CnetDtype get_dtype(void) const;
		
		const Mat<float32> &get_cmf32(void) const;
		const Mat<float64> &get_cmf64(void) const;
		const std::string &get_name(void) const;
		
		Var &set_name(const std::string &name);
		Var &rand_uniform_range(float32 a, float32 b);
		Var &rand_uniform_range(float64 a, float64 b);
		
		// Its better to fetch the matrices and do the required math operations that
		// doing it over this level of abstraction
		Mat<float32> &get_mf32(void);
		Mat<float64> &get_mf64(void);
		
		float32 &at_mf32(std::size_t i, std::size_t j);
		float64 &at_mf64(std::size_t i, std::size_t j);

		Var &resize(std::size_t rows, std::size_t cols);
		Var &resize(Shape shape);
		Var &resize(std::size_t rows, std::size_t cols, float64 init_val);
		Var &resize(Shape shape, float64 init_val);
		Var &resize(std::size_t rows, std::size_t cols, float32 init_val);
		Var &resize(Shape shape, float32 init_val);
		
		Var &operator()(const Mat<float32> &M);
		Var &operator()(const Mat<float64> &M);
		
		void operator=(const Var &V);
		void operator=(const Mat<float32> &M);
		void operator=(const Mat<float64> &M);
		
		Var operator*(const Var &V) const;
		void operator*=(const Var &V);
		
		Var operator+(const Var &V) const;
		void operator+=(const Var &V);

		Var operator-(const Var &V) const;
		void operator-=(const Var &V);
		
		Var operator^(const Var &V) const;
		void operator^=(const Var &V);

		Var operator*(float32 val) const;
		void operator*=(float32 val);
		
		Var operator*(float64 val) const;
		void operator*=(float64 val);

		friend std::ostream &operator<<(std::ostream &os, const Var &V)
		{
			if (V.has_name_)
				os << V.name_ << "=(";
			else
				os << "Var=(";
			switch (V.dtype_) {
			case FLOAT_32_DTYPE:
				os << V.M_.f32 << ",\n";
				break;
			case FLOAT_64_DTYPE:
				os << V.M_.f64 << ",\n";
				break;
			default:
				throw std::runtime_error("invalid var: Invalid datatype");
				break;
			}

			os << V.dtype_ << ", addrs=" << (void *) &V << ")";

			return os;
		}
		
	protected:
		CnetDtype dtype_;
		std::string name_;
		bool has_name_;
		Mats M_;

	};
}

#endif

