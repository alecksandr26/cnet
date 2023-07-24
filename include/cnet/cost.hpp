



#ifndef COST_INCLUDED
#define COST_INCLUDED

#include "ann.hpp"

namespace cnet {
	
	template<typename T>
	long double mse(ann<T> &ann, mat<T> *input,
			mat<T> *output, std::size_t train_size);
}

#endif


