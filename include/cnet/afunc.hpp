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
	enum AFUNC_TYPE {
		CNET_SIGMOID = 0,
		CNET_RELU,
		CNET_NONE
	};
	
	template<typename T>
	mat<T> &sigmoid(mat<T> &m);

	template<typename T>
	mat<T> &relu(mat<T> &m);
}


#endif




