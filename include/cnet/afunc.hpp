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
		class afunc {
		public:
			virtual Mat<T> operator()(const Mat<T> &X) const = 0;
			virtual Mat<T> derivate(const Mat<T> &X) const  = 0;
		};

		template<class T>
		class linear : public afunc<T> {
		public:
			Mat<T> operator()(const Mat<T> &X) const override;
			Mat<T> derivate(const Mat<T> &X) const override;
		};
		
		template<class T>
		class sigmoid : public afunc<T> {
		public:
			Mat<T> operator()(const Mat<T> &X) const override;
			Mat<T> derivate(const Mat<T> &X) const override;
		};


		template<class T>
		class relu : public afunc<T> {
		public:
			Mat<T> operator()(const Mat<T> &X) const override;
			Mat<T> derivate(const Mat<T> &X) const override;
		};
	}
}


#endif




