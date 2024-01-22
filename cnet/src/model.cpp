#include "cnet/dyptes.hpp"
#include "cnet/model.hpp"
#include "cnet/layer.hpp"
#include "cnet/mat.hpp"

std::size_t cnet::model::model::get_num_layers(void) { return num_layers_; }

cnet::model::sequential::sequential(void)
{
	cnet::model::model<T>::layers_ =
		new cnet::layer::layer<T> *[MAX_AMOUNT_OF_LAYERS];
}

cnet::model::sequential::~sequential(void)
{
	delete cnet::model::model<T>::layers_;
}

cnet::model::sequential<T>::sequential(std::initializer_list<layer::layer<T>> layers)
{
	cnet::model::model<T>::layers_ =
		new cnet::layer::layer<T> *[MAX_AMOUNT_OF_LAYERS];
}

cnet::mat<T> cnet::model::sequential::operator()(const cnet::mat<T> &X)
{
	// NOnthing here yet
}

