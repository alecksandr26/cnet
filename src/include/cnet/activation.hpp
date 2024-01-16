/*
  @file afunc.hpp
  @brief
  
  @author Erick Carrillo.
  @copyright Copyright (C) 2023, Erick Alejandro Carrillo López, All rights reserved.
  @license This project is released under the MIT License
*/

#ifndef ACTIVATION_INCLUDED
#define ACTIVATION_INCLUDED

#include <functional>

#include "variable.hpp"

namespace cnet::activation {
	using namespace variable;
	
	template<typename T>
	T Linear(const T &X);

	template<typename T>
	T LinearDerivate(const T &X);

	template<typename T>
	T Sigmoid(const T &X);

	template<typename T>
	T SigmoidDerivate(const T &X);

	template<typename T>
	T Relu(const T &X);
	
	template<typename T>
	T ReluDerivate(const T &X);

	template<class T>
	struct Afunc {
		Afunc() : afunc_(NULL), afunc_derivate_(NULL) {}
			
		void alloc_afunc(const std::string &afunc_name);
		
		std::function<T(const T &)> afunc_, afunc_derivate_;
	};
}

#endif




