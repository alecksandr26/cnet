#ifndef STRASSEN_INCLUDED
#define STRASSEN_INCLUDED

#include <cstddef>

namespace cnet::mathops::strassen {
	std::size_t pad_size(std::size_t a_rows, std::size_t a_cols,
			     std::size_t b_rows, std::size_t b_cols);
	
	template<typename T>
	extern T *pad_mat(const T *A, std::size_t rows, std::size_t cols, std::size_t n);
	
	template<typename T>
	extern void mat_mul(const T *A, const T *B, T *C, std::size_t n, std::size_t N);
}

#endif

