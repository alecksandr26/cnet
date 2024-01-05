/*
  @file ann.hpp
  @brief
  
  @author Erick Carrillo.
  @copyright Copyright (C) 2023, Erick Alejandro Carrillo López, All rights reserved.
  @license This project is released under the MIT License
*/


#ifndef ANN_INCLUDED
#define ANN_INCLUDED

#include <cstddef>

#include "mat.hpp"
#include "afunc.hpp"
#include "layer.hpp"

#define CNET_MAX_AMOUNT_LAYERS 1024

#define size_arch(arch) ((sizeof(arch) / sizeof(cnet::nn_arch)) - 1)

namespace cnet {
	typedef int nn_arch;
	
	template<class T>
	class ann {
	public:
		ann(const nn_arch *arch, std::size_t l, std::unique_ptr<afunc::act_func<T>> &&func);
		~ann(void);
		
		mat<T> feedforward(const mat<T> &X);
		void fit(mat<T> *input, mat<T> *output, std::size_t train_size, double lr, std::size_t nepochs);
		std::size_t get_layers(void);
		
		
		friend std::ostream &operator<<(std::ostream& os, const ann<T> &ann)
		{
			for (std::size_t l = 0; l < ann.l_; l++) {
				os << "layer: " << l << '\n';
				os << ann.layers_[l] << '\n';
			}
			return os;
		}
		
	private:
		std::size_t l_;
		model::layer<T> *layers_;
	};
}


#endif


