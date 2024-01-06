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
#include <memory>

#include "mat.hpp"
#include "afunc.hpp"
#include "model.hpp"
#include "cost.hpp"

namespace cnet {
	namespace layer {
		template<class T>
		class layer {
		public:
			virtual mat<T> operator()(const mat<T> &X) = 0; // Feedforward function
			virtual void build(std::size_t in_size) = 0;
			virtual void build(std::size_t in_size, T init_val) = 0;
		};

		
		// Dense layer it is a normal NN of the type Y = Act(W * X + B), where X is one dimension
		// object
		template<class T>
		class dense : public layer<T> {
		public:
			static constexpr bool trainable = true;

			// By default it uses the basic linear act function
			dense(std::size_t units);
			dense(std::size_t units, std::unique_ptr<afunc::afunc<T>> &&func);
			
			dense(void);
			~dense(void);
			void build(std::size_t in_size) override;
			void build(std::size_t in_size, T init_val) override;
			
			// TODO: Let the user to chose its own randomizer function
			void rand_range(T a, T b);
			
			mat<T> operator()(const mat<T> &X) override; // Feedforwar function
			void fit_backprop(const mat<T> &E, double lr, const mat<T> &A);
			
			std::size_t get_units(void) const;
			std::size_t get_in_size(void) const;
			std::size_t get_use_bias(void) const;
			
			void set_units(std::size_t units);
			void set_afunc(std::unique_ptr<afunc::afunc<T>> &&func);
			void set_use_bias(bool use_bias);
			
			friend std::ostream &operator<<(std::ostream& os, const dense<T> &l)
			{
				os << "W = \n" << l.W_;
				if (l.use_bias_)
					os << "\nB = \n" << l.B_;
			
				return os;
			}

			
		private:
			bool built_, use_bias_;
			std::size_t in_, units_;
			std::unique_ptr<afunc::afunc<T>> func_;
			mat<T> W_, B_;
		};


	}
}

#endif

