/*
  @file layer.hpp
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
	namespace model {
		template<class T>
		class layer : public model_abs<T> {
		public:
			layer(std::size_t in, std::size_t out);
			layer(std::size_t in, std::size_t out, std::unique_ptr<afunc::act_func<T>> &&func);
			layer(void);
		
			void mod(std::size_t in, std::size_t out,
				 std::unique_ptr<afunc::act_func<T>> &&func);
			void mod(std::size_t in, std::size_t out);
			void mod(std::unique_ptr<afunc::act_func<T>> &&func);
			
			void rand_range(T a, T b);
			mat<T> feedforward(const mat<T> &X) const override;
			void fit_backprog(const mat<T> &error, double lr, const mat<T> &A);
		
			friend std::ostream &operator<<(std::ostream& os, const layer<T> &l)
			{
				os << "W = \n" << l.W_;
				os << "\nB = \n" << l.B_;
			
				return os;
			}
			
			mat<T> W_, B_;
		
		private:
			
			std::size_t in_, out_;
			std::unique_ptr<afunc::act_func<T>> func_;
		};


	}
}

#endif

