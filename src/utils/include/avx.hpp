#ifndef UTILS_AVX_INCLUDED
#define UTILS_AVX_INCLUDED

#include <cmath>
#include <immintrin.h>

namespace cnet::mathops::utils {
	typedef __m256d vec4d;
	typedef __m128d vec2d;
	typedef __m256 vec8f;
	typedef __m128 vec4f;

	// To move and store data more quickly
	inline void mov_avx256(double *dst, const double *src)
	{
		vec4d vec = _mm256_loadu_pd(src);
		_mm256_storeu_pd(dst, vec);
	}

	inline void mov_avx256(float *dst, const float *src)
	{
		vec8f vec = _mm256_loadu_ps(src);
		_mm256_storeu_ps(dst, vec);
	}

	inline void mov_avx128(double *dst, const double *src)
	{
		vec2d vec = _mm_loadu_pd(src);
		_mm_storeu_pd(dst, vec);
	}

	inline void mov_avx128(float *dst, const float *src)
	{
		vec4f vec = _mm_loadu_ps(src);
		_mm_storeu_ps(dst, vec);
	}

	inline void mov_avx_8(double *dst, const double *src)
	{
		mov_avx256(dst, src);
		mov_avx256(dst + 4, src + 4);
	}

	inline void mov_avx_8(float *dst, const float *src)
	{
		mov_avx256(dst, src);
	}

	inline void mov_avx_4(double *dst, const double *src)
	{
		mov_avx256(dst, src);
	}

	inline void mov_avx_4(float *dst, const float *src)
	{
		mov_avx128(dst, src);
	}

	inline void mov_avx_2(double *dst, const double *src)
	{
		mov_avx128(dst, src);
	}

	inline void mov_avx_2(float *dst, const float *src)
	{
		dst[0] = src[0];
		dst[1] = src[1];
	}

	inline vec4d init_avx256(double init_val)
	{
		return _mm256_set_pd(init_val, init_val, init_val, init_val);
	}

	inline vec8f init_avx256(float init_val)
	{
		return _mm256_set_ps(init_val, init_val, init_val, init_val,
				    init_val, init_val, init_val, init_val);
	}

	inline vec4f init_avx128(float init_val)
	{
		return _mm_set_ps(init_val, init_val, init_val, init_val);
	}

	inline void store_val_avx_8(double *dst, double val)
	{
		vec4d  vec = init_avx256(val);
		_mm256_storeu_pd(dst, vec);
		_mm256_storeu_pd(dst + 4, vec);
	}

	inline void store_val_avx_8(float *dst, float val)
	{
		vec8f  vec = init_avx256(val);
		_mm256_storeu_ps(dst, vec);
	}

	inline void store_val_avx_4(double *dst, double val)
	{
		vec4d  vec = init_avx256(val);
		_mm256_storeu_pd(dst, vec);
	}

	inline void store_val_avx_4(float *dst, float val)
	{
		vec4f  vec = init_avx128(val);
		_mm_storeu_ps(dst, vec);
	}

	// To do an horizontal sumation of the whole vectors
	inline double hsum_avx256(vec4d vec)
	{
		vec2d vlow  = _mm256_castpd256_pd128(vec);
		vec2d vhigh = _mm256_extractf128_pd(vec, 1);	    // high 128

		vlow = _mm_add_pd(vlow, vhigh);	       // reduce down to 128

		vec2d high64 = _mm_unpackhi_pd(vlow, vlow);
	
		return _mm_cvtsd_f64(_mm_add_sd(vlow, high64));	       // reduce to scalar
	}

	inline float hsum_avx256(vec8f vec)
	{
		// hiQuad = ( x7, x6, x5, x4 )
		vec4f hiQuad = _mm256_extractf128_ps(vec, 1);
		// loQuad = ( x3, x2, x1, x0 )
		vec4f loQuad = _mm256_castps256_ps128(vec);
		// sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
		vec4f sumQuad = _mm_add_ps(loQuad, hiQuad);
		// loDual = ( -, -, x1 + x5, x0 + x4 )
		vec4f loDual = sumQuad;
		// hiDual = ( -, -, x3 + x7, x2 + x6 )
		vec4f hiDual = _mm_movehl_ps(sumQuad, sumQuad);
		// sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
		vec4f sumDual = _mm_add_ps(loDual, hiDual);
		// lo = ( -, -, -, x0 + x2 + x4 + x6 )
		vec4f lo = sumDual;
		// hi = ( -, -, -, x1 + x3 + x5 + x7 )
		vec4f hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
		// sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
		vec4f sum = _mm_add_ss(lo, hi);
		return _mm_cvtss_f32(sum);
	}

	inline float hsum_avx128(vec4f vec)
	{
		vec4f vlow = vec;
		vec4f high64 = _mm_unpackhi_ps(vlow, vlow);
		vlow = _mm_add_ps(vlow, high64); // reduce to scalar
		return _mm_cvtss_f32(_mm_hadd_ps(vlow, vlow));
	}


	// To add several values more quickly
	inline void add_avx256(double *dst, const double *src)
	{
		vec4d vec_a = _mm256_loadu_pd(dst);
		vec4d vec_b = _mm256_loadu_pd(src);
		vec_a = _mm256_add_pd(vec_a, vec_b);
		_mm256_storeu_pd(dst, vec_a);
	}
	

	inline void add_avx256(float *dst, const float *src)
	{
		vec8f vec_a = _mm256_loadu_ps(dst);
		vec8f vec_b = _mm256_loadu_ps(src);
		vec_a = _mm256_add_ps(vec_a, vec_b);
		_mm256_storeu_ps(dst, vec_a);
	}
	
	inline void add_avx128(float *dst, const float *src)
	{
		vec4f vec_a = _mm_loadu_ps(dst);
		vec4f vec_b = _mm_loadu_ps(src);
		vec_a = _mm_add_ps(vec_a, vec_b);
		_mm_storeu_ps(dst, vec_a);
	}

	inline void add_avx_8(double *dst, const double *src)
	{
		add_avx256(dst, src);
		add_avx256(dst + 4, src + 4);
	}

	inline void add_avx_4(double *dst, const double *src)
	{
		add_avx256(dst, src);
	}

	inline void add_avx_8(float *dst, const float *src)
	{
		add_avx256(dst, src);
	}

	inline void add_avx_4(float *dst, const float *src)
	{
		add_avx128(dst, src);
	}
	

	// To sub several values more quickly
	inline void sub_avx256(double *dst, const double *src)
	{
		vec4d vec_a = _mm256_loadu_pd(dst);
		vec4d vec_b = _mm256_loadu_pd(src);
		vec_a = _mm256_sub_pd(vec_a, vec_b);
		_mm256_storeu_pd(dst, vec_a);
	}


	inline void sub_avx256(float *dst, const float *src)
	{
		vec8f vec_a = _mm256_loadu_ps(dst);
		vec8f vec_b = _mm256_loadu_ps(src);
		vec_a = _mm256_sub_ps(vec_a, vec_b);
		_mm256_storeu_ps(dst, vec_a);
	}
	
	inline void sub_avx128(float *dst, const float *src)
	{
		vec4f vec_a = _mm_loadu_ps(dst);
		vec4f vec_b = _mm_loadu_ps(src);
		vec_a = _mm_sub_ps(vec_a, vec_b);
		_mm_storeu_ps(dst, vec_a);
	}

	inline void sub_avx_8(double *dst, const double *src)
	{
		sub_avx256(dst, src);
		sub_avx256(dst + 4, src + 4);
	}

	inline void sub_avx_4(double *dst, const double *src)
	{
		sub_avx256(dst, src);
	}

	inline void sub_avx_8(float *dst, const float *src)
	{
		sub_avx256(dst, src);
	}

	inline void sub_avx_4(float *dst, const float *src)
	{
		sub_avx128(dst, src);
	}
	
	// To mul several values more quickly
	inline void mul_avx256(double *dst, const double *src)
	{
		vec4d vec_a = _mm256_loadu_pd(dst);
		vec4d vec_b = _mm256_loadu_pd(src);
		vec_a = _mm256_mul_pd(vec_a, vec_b);
		_mm256_storeu_pd(dst, vec_a);
	}


	inline void mul_avx256(float *dst, const float *src)
	{
		vec8f vec_a = _mm256_loadu_ps(dst);
		vec8f vec_b = _mm256_loadu_ps(src);
		vec_a = _mm256_mul_ps(vec_a, vec_b);
		_mm256_storeu_ps(dst, vec_a);
	}
	
	inline void mul_avx128(float *dst, const float *src)
	{
		vec4f vec_a = _mm_loadu_ps(dst);
		vec4f vec_b = _mm_loadu_ps(src);
		vec_a = _mm_mul_ps(vec_a, vec_b);
		_mm_storeu_ps(dst, vec_a);
	}


	inline void mul_avx_8(double *dst, const double *src)
	{
		mul_avx256(dst, src);
		mul_avx256(dst + 4, src + 4);
	}

	inline void mul_avx_4(double *dst, const double *src)
	{
		mul_avx256(dst, src);
	}

	inline void mul_avx_8(float *dst, const float *src)
	{
		mul_avx256(dst, src);
	}

	inline void mul_avx_4(float *dst, const float *src)
	{
		mul_avx128(dst, src);
	}

	// To do an scalar vector product more quickly
	inline void mul_avx256(double *dst, vec4d val)
	{
		vec4d vec_a = _mm256_loadu_pd(dst);
		vec_a = _mm256_mul_pd(vec_a, val);
		_mm256_storeu_pd(dst, vec_a);
	}


	inline void mul_avx256(float *dst, vec8f val)
	{
		vec8f vec_a = _mm256_loadu_ps(dst);
		vec_a = _mm256_mul_ps(vec_a, val);
		_mm256_storeu_ps(dst, vec_a);
	}
	
	inline void mul_avx128(float *dst, vec4f val)
	{
		vec4f vec_a = _mm_loadu_ps(dst);
		vec_a = _mm_mul_ps(vec_a, val);
		_mm_storeu_ps(dst, vec_a);
	}

	inline void mul_avx_8(double *dst, double val)
	{
		vec4d vec = init_avx256(val);
		mul_avx256(dst, vec);
		mul_avx256(dst + 4, vec);
	}

	inline void mul_avx_4(double *dst, double val)
	{
		vec4d vec = init_avx256(val);
		mul_avx256(dst, vec);
	}

	inline void mul_avx_8(float *dst, float val)
	{
		vec8f vec = init_avx256(val);
		mul_avx256(dst, vec);
	}

	inline void mul_avx_4(float *dst, float val)
	{
		vec4f vec = init_avx128(val);
		mul_avx128(dst, vec);
	}


	inline void sum_avx256(vec4d &val, const double *src)
	{
		vec4d vec_a = _mm256_loadu_pd(src);
		val = _mm256_add_pd(val, vec_a);
	}

	inline void sum_avx256(vec8f &val, const float *src)
	{
		vec8f vec_a = _mm256_loadu_ps(src);
		val = _mm256_add_ps(val, vec_a);
	}

	inline void sum_avx128(vec4f &val, const float *src)
	{
		vec4f vec_a = _mm_loadu_ps(src);
		val = _mm_add_ps(val, vec_a);
	}


	// To initlized exp vec
	inline vec4d exp_avx256(const double *src)
	{
		return _mm256_set_pd(std::exp(src[0]), std::exp(src[1]), std::exp(src[2]), std::exp(src[3]));
	}

	inline vec8f exp_avx256(const float *src)
	{
		return _mm256_set_ps(std::exp(src[0]), std::exp(src[1]), std::exp(src[2]), std::exp(src[3]),
				     std::exp(src[4]), std::exp(src[5]), std::exp(src[6]), std::exp(src[7]));
	}

	inline vec4f exp_avx128(const float *src)
	{
		return _mm_set_ps(std::exp(src[0]), std::exp(src[1]), std::exp(src[2]), std::exp(src[3]));
	}

	// To compute  quickly the sigmoid function
	// S(x) = 1 / (1 + exp(- x))
	inline void sigmoid_avx_4(double *dst, const double *src)
	{
		vec4d ones = init_avx256((double) 1.0);
		vec4d neg_ones = init_avx256((double) -1.0);
		vec4d vec = _mm256_loadu_pd(src);
		vec4d e = _mm256_mul_pd(vec, neg_ones);
		e = _mm256_set_pd(std::exp(e[0]), std::exp(e[1]), std::exp(e[2]), std::exp(e[3]));
		vec4d res = _mm256_add_pd(ones, e);
		res = _mm256_div_pd(ones, res);
		_mm256_storeu_pd(dst, _mm256_set_pd(res[0], res[1], res[2], res[3]));
	}

	inline void sigmoid_avx_8(double *dst, const double *src)
	{
		sigmoid_avx_4(dst, src);
		sigmoid_avx_4(dst + 4, src + 4);
	}

	inline void sigmoid_avx_8(float *dst, const float *src)
	{
		vec8f ones = init_avx256((float) 1.0);
		vec8f neg_ones = init_avx256((float) -1.0);
		vec8f vec = _mm256_loadu_ps(src);
		vec8f e = _mm256_mul_ps(vec, neg_ones);
		e = _mm256_set_ps(std::exp(e[0]), std::exp(e[1]), std::exp(e[2]), std::exp(e[3]),
				  std::exp(e[4]), std::exp(e[5]), std::exp(e[6]), std::exp(e[7]));
		vec8f res = _mm256_add_ps(ones, e);
		res = _mm256_div_ps(ones, res);
		_mm256_storeu_ps(dst, _mm256_set_ps(res[0], res[1], res[2], res[3],
						    res[4], res[5], res[6], res[7]));
	}

	inline void sigmoid_avx_4(float *dst, const float *src)
	{
		vec4f ones = init_avx128((float) 1.0);
		vec4f neg_ones = init_avx128((float) -1.0);
		vec4f vec = _mm_loadu_ps(src);
		vec4f e = _mm_mul_ps(vec, neg_ones);
		e = _mm_set_ps(std::exp(e[0]), std::exp(e[1]), std::exp(e[2]), std::exp(e[3]));
		vec4f res = _mm_add_ps(ones, e);
		res = _mm_div_ps(ones, res);
		_mm_storeu_ps(dst, _mm_set_ps(res[0], res[1], res[2], res[3]));
	}

	// To compute quickly the derivate of sigmoid function
	// S(x) = 1 / (2 + exp(- x) + exp(x))
	inline void derivate_sigmoid_avx_4(double *dst, const double *src)
	{
		vec4d ones = init_avx256((double) 1.0);
		vec4d twos = init_avx256((double) 2.0);
		vec4d neg_ones = init_avx256((double) -1.0);
		vec4d vec = _mm256_loadu_pd(src);
		vec4d e1 = _mm256_mul_pd(vec, neg_ones);
		e1 = _mm256_set_pd(std::exp(e1[0]), std::exp(e1[1]), std::exp(e1[2]), std::exp(e1[3]));
		vec4d e2 = _mm256_set_pd(std::exp(vec[0]), std::exp(vec[1]),
					 std::exp(vec[2]), std::exp(vec[3]));
		vec4d res = _mm256_add_pd(twos, _mm256_add_pd(e1, e2));
		res = _mm256_div_pd(ones, res);
		_mm256_storeu_pd(dst, _mm256_set_pd(res[0], res[1], res[2], res[3]));
	}

	inline void derivate_sigmoid_avx_8(double *dst, const double *src)
	{
		derivate_sigmoid_avx_4(dst, src);
		derivate_sigmoid_avx_4(dst + 4, src + 4);
	}

	inline void derivate_sigmoid_avx_8(float *dst, const float *src)
	{
		vec8f ones = init_avx256((float) 1.0);
		vec8f twos = init_avx256((float) 2.0);
		vec8f neg_ones = init_avx256((float) -1.0);
		vec8f vec = _mm256_loadu_ps(src);
		vec8f e1 = _mm256_mul_ps(vec, neg_ones);
		e1 = _mm256_set_ps(std::exp(e1[0]), std::exp(e1[1]), std::exp(e1[2]), std::exp(e1[3]),
				   std::exp(e1[4]), std::exp(e1[5]), std::exp(e1[6]), std::exp(e1[7]));
		vec8f e2 = _mm256_set_ps(std::exp(vec[0]), std::exp(vec[1]), std::exp(vec[2]), std::exp(vec[3]),
					 std::exp(vec[4]), std::exp(vec[5]),
					 std::exp(vec[6]), std::exp(vec[7]));
		vec8f res = _mm256_add_ps(twos, _mm256_add_ps(e1, e2));
		res = _mm256_div_ps(ones, res);
		_mm256_storeu_ps(dst, _mm256_set_ps(res[0], res[1], res[2], res[3],
						    res[4], res[5], res[6], res[7]));
	}

	inline void derivate_sigmoid_avx_4(float *dst, const float *src)
	{
		vec4f ones = init_avx128((float) 1.0);
		vec4f twos = init_avx128((float) 2.0);
		vec4f neg_ones = init_avx128((float) -1.0);
		vec4f vec = _mm_loadu_ps(src);
		vec4f e1 = _mm_mul_ps(vec, neg_ones);
		e1 = _mm_set_ps(std::exp(e1[0]), std::exp(e1[1]), std::exp(e1[2]), std::exp(e1[3]));
		vec4f e2 = _mm_set_ps(std::exp(vec[0]), std::exp(vec[1]), std::exp(vec[2]), std::exp(vec[3]));
		vec4f res = _mm_add_ps(twos, _mm_add_ps(e1, e2));
		res = _mm_div_ps(ones, res);
		_mm_storeu_ps(dst, _mm_set_ps(res[0], res[1], res[2], res[3]));
	}

	// To compute  quickly the relu function
	// R(x) = max(0, x)
	inline void relu_avx_4(double *dst, const double *src)
	{
		vec4d zeros = init_avx256((double) 0.0);
		vec4d vec = _mm256_loadu_pd(src);
		vec = _mm256_max_pd(zeros, vec);
		_mm256_storeu_pd(dst, vec);
	}

	inline void relu_avx_8(double *dst, const double *src)
	{
		relu_avx_4(dst, src);
		relu_avx_4(dst + 4, src + 4);
	}

	inline void relu_avx_8(float *dst, const float *src)
	{
		vec8f zeros = init_avx256((float) 0.0);
		vec8f vec = _mm256_loadu_ps(src);
		vec = _mm256_max_ps(zeros, vec);
		_mm256_storeu_ps(dst, vec);
	}

	inline void relu_avx_4(float *dst, const float *src)
	{
		vec4f zeros = init_avx128((float) 0.0);
		vec4f vec = _mm_loadu_ps(src);
		vec = _mm_max_ps(zeros, vec);
		_mm_storeu_ps(dst, vec);
	}
}

#endif
