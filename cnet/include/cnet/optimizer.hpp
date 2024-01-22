#ifndef OPTIMIZERS_INCLUDED
#define OPTIMIZERS_INCLUDED

#include <cstddef>

#include "dtypes.hpp"
#include "mat.hpp"
#include "variable.hpp"

namespace cnet::optimizers {
	using namespace dtypes;
	using namespace mathops;
	using namespace variable;

	class Optimizer {
		Optimizer(void);
		~Optimizer(void);
	};
}



/*
  Wegiths = Optimizer(Weights, error);
  
 */

#endif


