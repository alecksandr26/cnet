#ifndef UTILS_MAT_INCLUDED
#define UTILS_MAT_INCLUDED

#include <cstdint>

namespace cnet::utils {
	extern void *alloc_mem_matrix(std::size_t n, std::size_t item_size);
	extern void free_mem_matrix(void *ptr);
	
	template<typename T>
	extern void cp_raw_mat(T *dst_mat, T *src_mat,
			       std::size_t rows, std::size_t cols,
			       std::size_t dst_cols, std::size_t src_cols);
	template<typename T>
	extern void mul_sqr_raw_mat(T *A, T *B, T *C, std::size_t n);
	template<typename T>
	extern void init_raw_mat(T *A, std::size_t rows, std::size_t cols, T init_val);
	
	// The result are allocated in A
	template<typename T>
	extern void add_raw_mat(T *A, T *B, std::size_t rows, std::size_t cols);
	template<typename T>
	extern void sub_raw_mat(T *A, T *B, std::size_t rows, std::size_t cols);
	template<typename T>
	extern void hardmard_mul_raw_mat(T *A, T *B, std::size_t rows, std::size_t cols);
	template<typename T>
	extern void scalar_mul_raw_mat(T *A, T b, std::size_t rows, std::size_t cols);
	template<typename T>
	extern T grand_sum_raw_mat(T *A, std::size_t rows, std::size_t cols);
}

#endif


