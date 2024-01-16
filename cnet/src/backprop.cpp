#include "utils/avx.hpp"
#include "utils/raw_mat.hpp"

#include "cnet/dtypes.hpp"
#include "cnet/backprop.hpp"

#include <cassert>
#include <omp.h>

using namespace std;
using namespace cnet;
using namespace mathops;
using namespace utils;

// To compute the derivate error for the previos layer
// a = f(z), where z = b + w_1 * i_1 + ... + w_n * i_n + ..., f is the activation function,
// dE = d(e)/d(z) = d(e)/d(a) * d(a)/d(z),
// I is the input from which we want its derivate, basically d(e)/d(i_k)
template<>
void cnet::weights::backprop::get_derror_dinput(const float32 *dE, const float32 *W, float32 *dI,
						size_t rows, size_t cols)
{
	// Maybe not necessary but for be sure
	assert(dE && W && dI && "Can't receive null pointers");
	assert(rows && cols && "Can't be zero");
	
	// d(e)/d(i_k) = sum_q=1_to_n d(e)/d(a_q) * d(a_q)/d(z_q) * d(z_q)/d(i_K)
	// d(z_q)/d(i_K) = w_q,k
	size_t n = rows * cols;
	size_t n_ite_8 = n - (n % 8);
	size_t n_ite_4 = n - (n % 4);

	// We can optimize the access 
	for (size_t i = 0; i < n_ite_8; i += 8) {
		vec8f vde = _mm256_set_ps(dE[(i + 7) / cols], dE[(i + 6) / cols],
					  dE[(i + 5) / cols], dE[(i + 4) / cols],
					  dE[(i + 3) / cols], dE[(i + 2) / cols],
					  dE[(i + 1) / cols], dE[i / cols]);
		vec8f vw = _mm256_loadu_ps(&W[i]);
		
		vec8f vdi = _mm256_set_ps(dI[(i + 7) % cols], dI[(i + 6) % cols],
					  dI[(i + 5) % cols], dI[(i + 4) % cols],
					  dI[(i + 3) % cols], dI[(i + 2) % cols],
					  dI[(i + 1) % cols], dI[i % cols]);
		
		// d(e)/d(i) += d(e)/d(z) * d(z)/d(i)
		vdi = _mm256_add_ps(vdi, _mm256_mul_ps(vde, vw));
		
		// Alloc the results
		for (size_t j = 0; j < 8; j++)
			dI[j % cols] += vdi[j];
   	}
	
	for (size_t i = n_ite_8; i < n_ite_4; i += 4) {
		vec4f vde = _mm_set_ps(dE[(i + 3) / cols], dE[(i + 2) / cols],
					  dE[(i + 1) / cols], dE[i / cols]);
		vec4f vw = _mm_loadu_ps(&W[i]);
		
		vec4f vdi = _mm_set_ps(dI[(i + 3) % cols], dI[(i + 2) % cols],
					  dI[(i + 1) % cols], dI[i % cols]);
		
		// d(e)/d(i) += d(e)/d(z) * d(z)/d(i)
		vdi = _mm_add_ps(vdi, _mm_mul_ps(vde, vw));
		
		// Alloc the results
		for (size_t j = 0; j < 4; j++)
			dI[j % cols] += vdi[j];
   	}

	for (size_t i = n_ite_4; i < n; i++)
		dI[i % cols] += dE[i / cols] * W[i];
}

template<>
void cnet::weights::backprop::get_derror_dinput(const float64 *dE, const float64 *W, float64 *dI,
						  size_t rows, size_t cols)
{
	// Maybe not necessary but for be sure
	assert(dE && W && dI && "Can't receive null pointers");
	assert(rows && cols && "Can't be zero");
	
	size_t n = rows * cols;
	size_t n_ite_8 = n - (n % 8);
	size_t n_ite_4 = n - (n % 4);

	// We can optimize the access 
#pragma omp parallel for
	for (size_t i = 0; i < n_ite_8; i += 8) {
		vec4d vde = _mm256_set_pd(dE[(i + 3) / cols], dE[(i + 2) / cols],
					  dE[(i + 1) / cols], dE[i / cols]);
		vec4d vw = _mm256_loadu_pd(&W[i]);
		
		vec4d vdi = _mm256_set_pd(dI[(i + 3) % cols], dI[(i + 2) % cols],
					  dI[(i + 1) % cols], dI[i % cols]);
		
		// d(e)/d(i) += d(e)/d(z) * d(z)/d(i)
		vdi = _mm256_add_pd(vdi, _mm256_mul_pd(vde, vw));
		
		// Alloc the results
		for (size_t j = 0; j < 4; j++)
			dI[j % cols] += vdi[j];
		
		vde = _mm256_set_pd(dE[(i + 7) / cols], dE[(i + 6) / cols],
				    dE[(i + 5) / cols], dE[(i + 4) / cols]);
		
		vw = _mm256_loadu_pd(&W[i + 4]);

		vdi = _mm256_set_pd(dI[(i + 7) % cols], dI[(i + 6) % cols],
				    dI[(i + 5) % cols], dI[(i + 4) % cols]);
		// d(e)/d(i) += d(e)/d(z) * d(z)/d(i)
		vdi = _mm256_add_pd(vdi, _mm256_mul_pd(vde, vw));

		// Alloc the results
		for (size_t j = 4; j < 8; j++)
			dI[j % cols] += vdi[j];
   	}


#pragma omp parallel for
	for (size_t i = n_ite_8; i < n_ite_4; i += 4) {
		vec4d vde = _mm256_set_pd(dE[(i + 3) / cols], dE[(i + 2) / cols],
					  dE[(i + 1) / cols], dE[i / cols]);
		vec4d vw = _mm256_loadu_pd(&W[i]);
		
		vec4d vdi = _mm256_set_pd(dI[(i + 3) % cols], dI[(i + 2) % cols],
					  dI[(i + 1) % cols], dI[i % cols]);
		
		// d(e)/d(i) += d(e)/d(z) * d(z)/d(i)
		vdi = _mm256_add_pd(vdi, _mm256_mul_pd(vde, vw));
		
		// Alloc the results
		for (size_t j = 0; j < 4; j++)
			dI[j % cols] += vdi[j];
   	}
	
	for (size_t i = n_ite_4; i < n; i++)
		dI[i % cols] += dE[i / cols] * W[i];
}


// Fit backpropagation:
// a = f(z), where z = b + w_1 * i_1 + ... + w_n * i_n + ..., f is the activation function,
// a is the actual neuron and i is the input
// dE = d(e)/d(z) = d(e)/d(a) * d(a)/d(z),
// Depends on the type of weights is going to receive one derivate or the another
// dW = d(z)/d(w_k) = i_k, if is a normal weight
// dW = d(z)/d(b_k) = 1,  if it is a bias
template<>
void cnet::weights::backprop::fit_weights(const float32 *dE, const float32 *dW, float32 *W,
					  size_t rows, size_t cols, float64 lr)
{
	// Maybe not necessary but for be sure
	assert(dE && dW && W && "Can't receive null pointers");
	assert(rows && cols && "Can't be zero");
	
	// d(e)/d(w_k) = d(e)/d(z) * d(z)/d(w_k)
	// w_k -= lr * d(e)/d(w_k)

	// dE = rows * 1
	// dW = 1 * cols
	
	size_t n = rows * cols;
	size_t n_ite_8 = n - (n % 8);
	size_t n_ite_4 = n - (n % 4);

	vec8f v8lr = init_avx256((float32) lr);
	vec4f v4lr = init_avx128((float32) lr);

	// Do also simd instructions to access more fast to the correspondned values
	
#pragma omp parallel for
	for (size_t i = 0; i < n_ite_8; i += 8) {
		vec8f vde = _mm256_set_ps(dE[(i + 7) / cols], dE[(i + 6) / cols],
					 dE[(i + 5) / cols], dE[(i + 4) / cols],
					 dE[(i + 3) / cols], dE[(i + 2) / cols],
					 dE[(i + 1) / cols], dE[i / cols]);

		vec8f vdw = _mm256_set_ps(dW[(i + 7) % cols], dW[(i + 6) % cols],
					  dW[(i + 5) % cols], dW[(i + 4) % cols],
					  dW[(i + 3) % cols], dW[(i + 2) % cols],
					  dW[(i + 1) % cols], dW[i % cols]);
		
		vec8f vw = _mm256_loadu_ps(&W[i]);
		vw = _mm256_sub_ps(vw, _mm256_mul_ps(v8lr, _mm256_mul_ps(vde, vdw)));
		_mm256_storeu_ps(&W[i], vw);
	}
	
#pragma omp parallel for
	for (size_t i = n_ite_8; i < n_ite_4; i += 4) {
		vec4f vde = _mm_set_ps(dE[(i + 3) / cols], dE[(i + 2) / cols],
				       dE[(i + 1) / cols], dE[i / cols]);
		vec4f vdw = _mm_set_ps(dW[(i + 3) % cols], dW[(i + 2) % cols],
				       dW[(i + 1) % cols], dW[i % cols]);
		
		vec4f vw = _mm_loadu_ps(&W[i]);
		vw = _mm_sub_ps(vw, _mm_mul_ps(v4lr, _mm_mul_ps(vde, vdw)));
		_mm_storeu_ps(&W[i], vw);
	}
	
	for (size_t i = n_ite_4; i < n; i++)
		W[i] -= dE[i / cols] * dW[i % cols] * (float32) lr;
}

template<>
void cnet::weights::backprop::fit_weights(const float64 *dE, const float64 *dW, float64 *W,
					   size_t rows, size_t cols, float64 lr)
{
	// Maybe not necessary but for be sure
	assert(dE && dW && W && "Can't receive null pointers");
	assert(rows && cols &&  "Can't be zero");
	
	// d(e)/d(w_k) = d(e)/d(z) * d(z)/d(w_k)
	// w_k -= lr * d(e)/d(w_k)

	// dE = rows * 1
	// dW = 1 * cols
	
	size_t n = rows * cols;
	size_t n_ite_8 = n - (n % 8);
	size_t n_ite_4 = n - (n % 4);

	vec4d v4lr = init_avx256(lr);

#pragma omp parallel for
	for (size_t i = 0; i < n_ite_8; i += 8) {
		vec4d vde = _mm256_set_pd(dE[(i + 3) / cols], dE[(i + 2) / cols],
					  dE[(i + 1) / cols], dE[i / cols]);
		vec4d vdw = _mm256_set_pd(dW[(i + 3) % cols], dW[(i + 2) % cols],
					  dW[(i + 1) % cols], dW[i % cols]);

		vec4d vw = _mm256_loadu_pd(&W[i]);
		vw = _mm256_sub_pd(vw, _mm256_mul_pd(v4lr, _mm256_mul_pd(vde, vdw)));
		_mm256_storeu_pd(&W[i], vw);
		
		vde = _mm256_set_pd(dE[(i + 7) / cols], dE[(i + 6) / cols],
				    dE[(i + 5) / cols], dE[(i + 4) / cols]);
		vdw = _mm256_set_pd(dW[(i + 7) % cols], dW[(i + 6) % cols],
				    dW[(i + 5) % cols], dW[(i + 4) % cols]);

		vw = _mm256_loadu_pd(&W[i + 4]);
		
		vw = _mm256_sub_pd(vw, _mm256_mul_pd(v4lr, _mm256_mul_pd(vde, vdw)));
		_mm256_storeu_pd(&W[i + 4], vw);
	}


#pragma omp parallel for
	for (size_t i = n_ite_8; i < n_ite_4; i += 4) {
		vec4d vde = _mm256_set_pd(dE[(i + 3) / cols], dE[(i + 2) / cols],
				    dE[(i + 1) / cols], dE[i / cols]);
		vec4d vdw = _mm256_set_pd(dW[(i + 3) % cols], dW[(i + 2) % cols],
				    dW[(i + 1) % cols], dW[i % cols]);

		vec4d vw = _mm256_loadu_pd(&W[i]);
		vw = _mm256_sub_pd(vw, _mm256_mul_pd(v4lr, _mm256_mul_pd(vde, vdw)));
		_mm256_storeu_pd(&W[i], vw);
	}
	
	for (size_t i = n_ite_4; i < n; i++)
		W[i] -= dE[i / cols] * dW[i % cols] * lr;
}

