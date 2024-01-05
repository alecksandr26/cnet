#ifndef MODEL_INCLUDED
#define MODEL_INCLUDED

#include <memory>
#include "cost.hpp"

namespace cnet {
	namespace model {
		template<class T>
		class model_abs {
		public:
			virtual mat<T> feedforward(const mat<T> &X) const = 0;
			virtual void fit(const mat<T> *X, const mat<T> *Y, std::size_t in_size,
					 std::size_t epochs, double lr,
					 std::unique_ptr<cost::cost_func<T>> &&func) = 0;
		};
	};
};


#endif
