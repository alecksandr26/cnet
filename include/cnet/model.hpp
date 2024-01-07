#ifndef MODEL_INCLUDED
#define MODEL_INCLUDED

#include <memory>

#include "cost.hpp"
#include "layer.hpp"

#define MAX_AMOUNT_OF_LAYERS 100

namespace cnet {
	namespace model {
		class model {
		public:
			// To do feedforward of the model
			template<typename T>
			virtual mat<T> operator()(const mat<T> &X) = 0;
			
			// Back prop algorithm for the moment
			virtual void fit(const mat<T> *X, const mat<T> *Y, std::size_t in_size,
					 std::size_t epochs, double lr,
					 const std::string &cfunc_name) = 0;

			
			std::size_t get_num_layers(void);
			
		protected:
			layer::layer<T> **layers_;
			std::size_t num_layers_;
		};
		
		// I want to have a similar model as keras

		class sequential : public model {
		public:
			sequential(void);
			~sequential(void);

			template<typename T>
			sequential(std::initializer_list<layer::layer<T>> layers);
			// void add(layer::layer<T> &layer);

			template<typename T>
			mat<T> operator()(const mat<T> &X) override;
		private:
			
		};
		
	};
};


#endif
