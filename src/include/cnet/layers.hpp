/*
  @file layers.hpp
  @brief
  
  @author Erick Carrillo.
  @copyright Copyright (C) 2023, Erick Alejandro Carrillo López, All rights reserved.
  @license This project is released under the MIT License
*/

#ifndef LAYERS_INCLUDED
#define LAYERS_INCLUDED

#include <cstddef>
#include <functional>
#include <ostream>

#include "dtypes.hpp"
#include "mat.hpp"
#include "variable.hpp"
#include "activation.hpp"
#include "weights.hpp"

namespace cnet::layers {
	using namespace dtypes;
	using namespace mathops;
	using namespace variable;
	using namespace activation;
	using namespace weights;
	
	
	// These objects inputs and outpus are thinked to be just notation and be matrices as n x 1 form
	class Input : public Var {
		// Nothing necessary here yet

		friend std::ostream &operator<<(std::ostream &os, const Input &V)
		{
			switch (V.dtype_) {
			case FLOAT_32_DTYPE:
				os << "Input=(" << V.M_.f32 << ",\ndtype=Mat<float32>, "
				   << "addrs=" << (void *) &V << ")";
				break;
			case FLOAT_64_DTYPE:
				os << "Input=(" << V.M_.f64 << ",\ndtype=Mat<float64>, "
				   << "addrs=" << (void *) &V << ")";
				break;
			default:
				throw std::runtime_error("invalid var: Invalid datatype");
				break;
			}

			return os;
		}
	};
	
	class Output : public Var {
		// Nothing necessary here yet

		// Overload the print functions
		friend std::ostream &operator<<(std::ostream &os, const Output &V)
		{
			switch (V.dtype_) {
			case FLOAT_32_DTYPE:
				os << "Output=(" << V.M_.f32 << ",\ndtype=Mat<float32>, "
				   << "addrs=" << (void *) &V << ")";
				break;
			case FLOAT_64_DTYPE:
				os << "Output=(" << V.M_.f64 << ",\ndtype=Mat<float64>, "
				   << "addrs=" << (void *) &V << ")";
				break;
			default:
				throw std::runtime_error("invalid var: Invalid datatype");
				break;
			}

			return os;
		}
	};

	class Error : public Var {
		// Nothing necessary here yet
	};

	// The base class to construct the other layers
	class Layer {
	public:
		Layer(void);
		~Layer(void);
		Layer(bool trainable, Shape in, Shape out);
		Layer(bool trainable, std::size_t in, std::size_t out);
		Layer(bool trainable, Shape in, Shape out, CnetDtype dtype);
		Layer(bool trainable, std::size_t in, std::size_t out, CnetDtype dtype);
				
		std::size_t get_in_size(void) const;
		Shape get_in_shape(void) const;
		std::size_t get_out_size(void) const;
		Shape get_out_shape(void) const;
		CnetDtype get_dtype(void) const;
		bool is_built(void) const;

		// FeedForward
		virtual Output operator()(const Input &X) = 0;
		virtual Mat<float32> operator()(const Mat<float32> &X) = 0;
		virtual Mat<float64> operator()(const Mat<float64> &X) = 0;

		// virtual Output operator()(const Input &X) const = 0; // Feedforward function
		virtual Layer &build(std::size_t in_size) = 0;
		virtual Layer &build(Shape in) = 0;

		// It may be will require rebuild 
		Layer &set_in_size(std::size_t in_size);
		Layer &set_in_shape(Shape shape);
		
	protected:
		bool built_, trainable_;
		Shape in_, out_;
		CnetDtype dtype_;
	};
	
	
	// Dense layer it is a normal NN of the type Y = Act(W * X + B), where X is one dimension
	// matrix
	class Dense : public Layer {
	public:
		constexpr static std::string default_weights_name = "Weights";
		constexpr static std::string default_biases_name = "Biases";
		constexpr static std::string default_afunc_name = "Linear";
		
		// By default it uses the basic Linear act function
		Dense(std::size_t units);
		Dense(std::size_t units, CnetDtype dtype);
		Dense(std::size_t units, const std::string &afunc_name);
		Dense(std::size_t units, const std::string &afunc_name, CnetDtype dtype);
		
		Dense(void);
		~Dense(void);
		
		std::size_t get_units(void) const;
		std::size_t get_weights(void) const;
		Shape get_weights_shape(void) const;
		std::size_t get_biases(void) const;
		Shape get_biases_shape(void) const;
		bool use_bias(void) const;
		const std::string &get_afunc_name(void) const;
		const Var &get_cmat_weights(void) const;
		const Var &get_cmat_biases(void) const;
		Weights &get_mat_weights(void);
		Weights &get_mat_biases(void);

		// To compute the error for the previos layer
		// a = f(z), where z = b + w_1 * i_1 + ... + w_n * i_n + ..., f is the activation function,
		// dE = d(e)/d(a)
		// I is the input from which we want its derivate, basically d(e)/d(i_k)
		Error get_derror_dinput(const Error &dA) const;
		Mat<float32> get_derror_dinput(const Mat<float32> &dA) const;
		Mat<float64> get_derror_dinput(const Mat<float64> &dA) const;
		
		// FeedForward
		Output operator()(const Input &X) override;
		Mat<float32> operator()(const Mat<float32> &X) override;
		Mat<float64> operator()(const Mat<float64> &X) override;
		
		Dense &build(std::size_t in_size) override;
		Dense &build(Shape in) override;

		// It may be will require rebuild 
		Dense &set_units(std::size_t units);
		Dense &set_use_bias(bool use_bias);
		Dense &set_afunc(const std::string &afunc_name);
		
		// TODO: Let the user to chose its own initializer function
		Dense &rand_uniform_range(float64 a, float64 b);

		// Fit backpropagation:
		// a = f(z), where z = b + w_1 * i_1 + ... + w_n * i_n + ..., f is the activation function,
		// a is the actual neuron and i is the input
		// dE = d(e)/d(a)
		// I is the input of this layer
		Dense &fit(const Error &dE, const Input &I, float64 lr);
		Dense &fit(const Mat<float32> &dE, const Mat<float32> &I, float64 lr);
		Dense &fit(const Mat<float64> &dE, const Mat<float64> &I, float64 lr);
			
		friend std::ostream &operator<<(std::ostream& os, const Dense &L)
		{
			os << "Dense=(\n";
			if (L.built_) {
				os << L.W_ << ",\n";
				os << L.B_ << ",\n";
			}

			os << "built=" << (L.built_ ? "true" : "false") << ", ";
			os << "in=(" << L.in_ << "), ";
			os << "out=(" << L.out_ << "), ";
			os << "units=" << L.units_ << ", ";
			os << L.dtype_ << ", ";
			os << "use_bias=" << (L.use_bias_ ? "true" : "false") << ", ";
			os << "activation=" << L.afunc_name_ << ")";

			return os;
		}
			
	private:
		bool use_bias_;
		std::size_t units_;
		std::string afunc_name_;
		Weights W_, B_;
		Var Z_;		
		Afunc<Var> afunc_;

	};
}

#endif

