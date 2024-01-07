/*
  @file dense.hpp
  @brief
  
  @author Erick Carrillo.
  @copyright Copyright (C) 2023, Erick Alejandro Carrillo López, All rights reserved.
  @license This project is released under the MIT License
*/

#ifndef LAYER_INCLUDED
#define LAYER_INCLUDED

#include <cstddef>
#include <utility>
#include <memory>

#include "mat.hpp"
#include "afunc.hpp"

namespace cnet {
	namespace layer {
		template<class T>
		class layer {
		public:
			bool trainable;
			
			virtual mat<T> operator()(const mat<T> &X) = 0; // Feedforward function
			virtual std::size_t get_in_size(void) const = 0;
			virtual std::size_t get_out_size(void) const = 0;
			
		protected:
			bool built_;
		};


		template<class T>
		class trainable_layer : public layer<T> {
		public:
			trainable_layer(void)
			{
				layer<T>::trainable = true;
			}
			
			virtual void build(std::size_t in_size) = 0;
			virtual void build(std::size_t in_size, T init_val) = 0;
		};


		template<class T>
		class nontrainable_layer : public layer<T> {
		public:
			nontrainable_layer(void)
			{
				layer<T>::trainable = false;
			}
			
			virtual void build(void) = 0;
		};
		

		// The input layer will flat the input to pass to a normal dense feedforward layer
		template<class T>
		class input : public nontrainable_layer<T> {
		public:
			input(void);
			input(std::size_t in_size); // Flattern input
			input(std::pair<std::size_t, std::size_t> in_shape); // Not Flattern input
			input(std::size_t in_size, std::size_t batches); // Flattern input
			
			// Not Flattern input
			input(std::pair<std::size_t, std::size_t> in_shape, std::size_t batches); 

			~input(void);

			std::size_t get_in_size(void) const override;
			std::size_t get_out_size(void) const override;
			std::pair<std::size_t, std::size_t> get_in_shape(void) const;
			
			mat<T> operator()(const mat<T> &X) override;
			// mat<T> operator()(const mat<T> *X);
			void build(void) override;
						
		private:
			std::pair<std::size_t, std::size_t> in_shape_;
			std::size_t out_, batches_;
			
			// If it needs to flat the inputs
			mat<T> *A_;
		};
		
		// Dense layer it is a normal NN of the type Y = Act(W * X + B), where X is one dimension
		// object
		template<class T>
		class dense : public trainable_layer<T> {
		public:
			static constexpr bool trainable = true;

			// By default it uses the basic linear act function
			dense(std::size_t units);
			dense(std::size_t units, const std::string &afunc_name);
			
			dense(void);
			~dense(void);
			void build(std::size_t in_size) override;
			void build(std::size_t in_size, T init_val) override;
			
			// TODO: Let the user to chose its own randomizer function
			void rand_range(T a, T b);

			// It will
			mat<T> operator()(const mat<T> &X) override; // Feedforwar function
			void fit_backprop(const mat<T> &E, double lr, const mat<T> &A);
			
			std::size_t get_units(void) const;
			std::size_t get_in_size(void) const;
			std::size_t get_out_size(void) const;
			std::size_t get_use_bias(void) const;
			cnet::mat<T> get_weights(void) const;
			cnet::mat<T> get_biases(void) const;
			
			void set_units(std::size_t units);
			void set_afunc(const std::string &afunc_name);
			void set_use_bias(bool use_bias);
			
			friend std::ostream &operator<<(std::ostream& os, const dense<T> &l)
			{
				os << "W = \n" << l.W_;
				if (l.use_bias_)
					os << "\nB = \n" << l.B_;
			
				return os;
			}

			
		private:
			bool use_bias_;
			std::size_t in_, units_;
			std::unique_ptr<cnet::afunc::afunc<T>> func_;
			mat<T> W_, B_;
		};


	}
}

#endif

