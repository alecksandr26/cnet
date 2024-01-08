#ifndef UTILS_AVX
#define UTILS_AVX

#include <immintrin.h>

namespace cnet::utils {
	typedef __m256d vec4d;
	typedef __m128d vec2d;
	typedef __m256 vec8f;
	typedef __m128 vec4f;

	inline void mov_avx256(double *dst, double *src)
	{
		vec4d vec = _mm256_loadu_pd(src);
		_mm256_storeu_pd(dst, vec);
	}

	inline void mov_avx256(float *dst, float *src)
	{
		vec8f vec = _mm256_loadu_ps(src);
		_mm256_storeu_ps(dst, vec);
	}

	inline void mov_avx128(double *dst, double *src)
	{
		vec2d vec = _mm_loadu_pd(src);
		_mm_storeu_pd(dst, vec);
	}

	inline void mov_avx128(float *dst, float *src)
	{
		vec4f vec = _mm_loadu_ps(src);
		_mm_storeu_ps(dst, vec);
	}
	
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


	inline void add_avx256(double *dst, double *src)
	{
		vec4d vec_a = _mm256_loadu_pd(dst);
		vec4d vec_b = _mm256_loadu_pd(src);
		vec_a = _mm256_add_pd(vec_a, vec_b);
		_mm256_storeu_pd(dst, vec_a);
	}


	inline void add_avx256(float *dst, float *src)
	{
		vec8f vec_a = _mm256_loadu_ps(dst);
		vec8f vec_b = _mm256_loadu_ps(src);
		vec_a = _mm256_add_ps(vec_a, vec_b);
		_mm256_storeu_ps(dst, vec_a);
	}
	
	inline void add_avx128(float *dst, float *src)
	{
		vec4f vec_a = _mm_loadu_ps(dst);
		vec4f vec_b = _mm_loadu_ps(src);
		vec_a = _mm_add_ps(vec_a, vec_b);
		_mm_storeu_ps(dst, vec_a);
	}


	inline void sub_avx256(double *dst, double *src)
	{
		vec4d vec_a = _mm256_loadu_pd(dst);
		vec4d vec_b = _mm256_loadu_pd(src);
		vec_a = _mm256_sub_pd(vec_a, vec_b);
		_mm256_storeu_pd(dst, vec_a);
	}


	inline void sub_avx256(float *dst, float *src)
	{
		vec8f vec_a = _mm256_loadu_ps(dst);
		vec8f vec_b = _mm256_loadu_ps(src);
		vec_a = _mm256_sub_ps(vec_a, vec_b);
		_mm256_storeu_ps(dst, vec_a);
	}
	
	inline void sub_avx128(float *dst, float *src)
	{
		vec4f vec_a = _mm_loadu_ps(dst);
		vec4f vec_b = _mm_loadu_ps(src);
		vec_a = _mm_sub_ps(vec_a, vec_b);
		_mm_storeu_ps(dst, vec_a);
	}


	inline void mul_avx256(double *dst, double *src)
	{
		vec4d vec_a = _mm256_loadu_pd(dst);
		vec4d vec_b = _mm256_loadu_pd(src);
		vec_a = _mm256_mul_pd(vec_a, vec_b);
		_mm256_storeu_pd(dst, vec_a);
	}


	inline void mul_avx256(float *dst, float *src)
	{
		vec8f vec_a = _mm256_loadu_ps(dst);
		vec8f vec_b = _mm256_loadu_ps(src);
		vec_a = _mm256_mul_ps(vec_a, vec_b);
		_mm256_storeu_ps(dst, vec_a);
	}
	
	inline void mul_avx128(float *dst, float *src)
	{
		vec4f vec_a = _mm_loadu_ps(dst);
		vec4f vec_b = _mm_loadu_ps(src);
		vec_a = _mm_mul_ps(vec_a, vec_b);
		_mm_storeu_ps(dst, vec_a);
	}


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


	inline void sum_avx256(vec4d &val, double *src)
	{
		vec4d vec_a = _mm256_loadu_pd(src);
		val = _mm256_add_pd(val, vec_a);
	}

	inline void sum_avx256(vec8f &val, float *src)
	{
		vec8f vec_a = _mm256_loadu_ps(src);
		val = _mm256_add_ps(val, vec_a);
	}

	inline void sum_avx128(vec4f &val, float *src)
	{
		vec4f vec_a = _mm_loadu_ps(src);
		val = _mm_add_ps(val, vec_a);
	}
}

#endif


