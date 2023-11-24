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
#include "afunc.hpp"
#include "mat.hpp"

namespace cnet {
	template<class T>
	class layer {
	public:
		layer(std::size_t in, std::size_t out);
		layer(std::size_t in, std::size_t out, enum AFUNC_TYPE afunc);
		layer(void);
		
		void mod(std::size_t in, std::size_t out, enum AFUNC_TYPE afunc);
		void mod(std::size_t in, std::size_t out);
		void rand_range(T a, T b);
		mat<T> feedforward(mat<T> &X);

		friend std::ostream &operator<<(std::ostream& os, const layer<T> &l)
		{
			os << "W = \n" << l.W_;
			os << "B = \n" << l.B_;
			
			return os;
		}

		mat<T> W_, B_;
		
	private:
		std::size_t in_, out_;
		enum AFUNC_TYPE afunc_;
	};
}

#endif

