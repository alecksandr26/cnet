#ifndef DTYPES_INCLUDED
#define DTYPES_INCLUDED

#include <ostream>

#define CNET_ENABLE_SWITCH_CNET_DTYPE 1  // Set to 0 to disable switch(CnetDtype) use case

namespace cnet::dtypes {
	enum CNET_DTYPE {
		NONE_DTYPE = 0,
		FLOAT_32_DTYPE,
		FLOAT_64_DTYPE,
	};
	
	class CnetDtype {
	public:
		CnetDtype() = default;
		constexpr CnetDtype(CNET_DTYPE val) : val_(val) {}
		constexpr void operator=(CNET_DTYPE val) { val_ = val; }
		
#if CNET_ENABLE_SWITCH_CNET_DTYPE
		// Allow switch and comparisons.
		constexpr  operator CNET_DTYPE() const { return val_; }
		
		// Prevent usage: if(fruit)
		explicit  operator bool() const = delete;
#else
		constexpr bool operator==(CnetDtype a) const { return val_ == a.val_; }
		constexpr bool operator!=(CnetDtype a) const { return val_ != a.val_; }
#endif
		constexpr bool is_float32() const { return val_ == FLOAT_32_DTYPE; }
		constexpr bool is_float64() const { return val_ == FLOAT_64_DTYPE; }

		friend std::ostream &operator<<(std::ostream &os, const CnetDtype &dtype)
		{
			os << "dtype=";
			switch (dtype.val_) {
			case FLOAT_32_DTYPE:
				os << "float32";
				break;
			case FLOAT_64_DTYPE:
				os << "float64";
				break;
			default:
				throw std::runtime_error("runtime error: Invalid datatype");
				break;
			}
			return os;
		}
		
		CNET_DTYPE val_;
	};
	
	
	
	typedef float float32;
	typedef double float64;
}

#endif


