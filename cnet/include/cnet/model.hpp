#ifndef MODEL_INCLUDED
#define MODEL_INCLUDED

#include <cstddef>

#include "dtypes.hpp"
#include "mat.hpp"
#include "layer.hpp"
#include "loss.hpp"
#include "optimizer.hpp"

namespace cnet::model {
	using namespace dtypes;
	using namespace mathops;
	using namespace variable;
	using namespace layers;
	
	class Model : public Layer {
	public:
		// Feedforward of the model
		virtual mat<float32> operator()(const mat<float32> &X) = 0;
		virtual mat<float64> operator()(const mat<float64> &X) = 0;
		virtual Output operator()(const Input &X) = 0;

		virtual void compile(const std::string &loss_name);
		
		// Back prop algorithm for the moment
		virtual void fit(const mat<float32> *X, const mat<float32> *Y, std::size_t size,
				 std::size_t epochs, float64 lr) = 0;
		
		virtual void fit(const mat<float64> *X, const mat<float64> *Y, std::size_t size,
				 std::size_t epochs, float64 lr) = 0;

		virtual void fit(const mat<float64> *X, const mat<float64> *Y, std::size_t size,
				 std::size_t epochs, float64 lr) = 0;

			
		std::size_t get_num_layers(void) const;
			
	protected:
		std::size_t nlayers_;
	};
		
	class Sequential : public Model {
	public:
		Sequential(void);
		~Sequential(void);
		
		Sequential(std::initializer_list<Layer &> layers);
	private:
		
	};
		
}


#endif
