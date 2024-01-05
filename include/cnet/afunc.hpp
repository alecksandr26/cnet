/*
  @file afunc.hpp
  @brief
  
  @author Erick Carrillo.
  @copyright Copyright (C) 2023, Erick Alejandro Carrillo López, All rights reserved.
  @license This project is released under the MIT License
*/

#ifndef AFUNCS_INCLUDED
#define AFUNCS_INCLUDED

#include "mat.hpp"

namespace cnet {
	namespace afunc {
		
		// This naming class needs a refactor
		template<class T>
		class act_func {
		public:
			virtual mat<T> func(const mat<T> &X) const = 0;
			virtual mat<T> dfunc_dx(const mat<T> &X) const  = 0;
		};
		
		template<class T>
		class sigmoid : public act_func<T> {
		public:
			mat<T> func(const mat<T> &X) const override;
			mat<T> dfunc_dx(const mat<T> &X) const override;
		};


		template<class T>
		class relu : public act_func<T> {
		public:
			mat<T> func(const mat<T> &X) const override;
			mat<T> dfunc_dx(const mat<T> &X) const override;
		};
	}
}


#endif




