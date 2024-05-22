#include <iostream>
#include <cassert>

enum {
	FLOAT_32,
	FLOAT_64
};

// Forward declaration
class Float32Matrix;
class Float64Matrix;

// Abstract class for matrix operations
class Matrix {
public:
	virtual Matrix& multiply(const Matrix& matrix) const = 0;
	virtual Matrix& multiply(const Float32Matrix& matrix) const = 0;
	virtual Matrix& multiply(const Float64Matrix& matrix) const = 0;
	virtual void assign(const Matrix& matrix) = 0;
	virtual void print() const = 0;
	virtual ~Matrix() {} // Virtual destructor

	// Factory method to create matrices
	static Matrix& createMatrix(int type, double value);
};

// Concrete class for float32 matrices
class Float32Matrix : public Matrix {
public:
	Float32Matrix(float value) : value_(value) {}

	Matrix& multiply(const Matrix& matrix) const override {
		return matrix.multiply(*this);
	}

	Matrix& multiply(const Float32Matrix& matrix) const override {
		// Perform multiplication with a Float32Matrix
		// Example implementation
		std::cout << "Multiplying Float32Matrix with another Float32Matrix\n";
		return *new Float32Matrix(value_ * matrix.value_); // Returning a reference to a temporary object
	}

	Matrix& multiply(const Float64Matrix& matrix) const override;

	void assign(const Matrix& matrix) override {
		// Perform assignment with a Float32Matrix
		// Example implementation
		std::cout << "Assigning Float32Matrix with another matrix\n";
	}

	void print() const override {
		// Print the Float32Matrix
		std::cout << "Float32Matrix value: " << value_ << std::endl;
	}

	float value_;
};

// Concrete class for float64 matrices
class Float64Matrix : public Matrix {
public:
	Float64Matrix(double value) : value_(value) {}

	Matrix& multiply(const Matrix& matrix) const override {
		return matrix.multiply(*this);
	}

	Matrix& multiply(const Float32Matrix& matrix) const override;

	Matrix& multiply(const Float64Matrix& matrix) const override {
		// Perform multiplication with a Float64Matrix
		// Example implementation
		std::cout << "Multiplying Float64Matrix with another Float64Matrix\n";
		return *new Float64Matrix(value_ * matrix.value_); // Returning a reference to a temporary object
	}

	void assign(const Matrix& matrix) override {
		// Perform assignment with a Float64Matrix
		// Example implementation
		std::cout << "Assigning Float64Matrix with another matrix\n";
	}

	void print() const override {
		// Print the Float64Matrix
		std::cout << "Float64Matrix value: " << value_ << std::endl;
	}
	double value_;
};

// Factory function to create matrices based on type
Matrix& Matrix::createMatrix(int type, double value) {
	switch (type) {
        case FLOAT_32:
		return *new Float32Matrix(static_cast<float>(value)); // Returning a reference to a temporary object
        case FLOAT_64:
		return *new Float64Matrix(value); // Returning a reference to a temporary object
        default:
		assert(false && "Invalid matrix type");
		// Returning a reference to a temporary object to satisfy the return type
		return *new Float32Matrix(0);
	}
}

// Implementation of multiplication for Float32Matrix with Float64Matrix
Matrix& Float32Matrix::multiply(const Float64Matrix& matrix) const {
	// Perform multiplication between Float32Matrix and Float64Matrix
	// Example implementation
	std::cout << "Multiplying Float32Matrix with Float64Matrix\n";
	return *new Float64Matrix(value_ * matrix.value_); // Returning a reference to a temporary object
}

// Implementation of multiplication for Float64Matrix with Float32Matrix
Matrix& Float64Matrix::multiply(const Float32Matrix& matrix) const {
	// Perform multiplication between Float64Matrix and Float32Matrix
	// Example implementation
	std::cout << "Multiplying Float64Matrix with Float32Matrix\n";
	return *new Float64Matrix(value_ * matrix.value_); // Returning a reference to a temporary object
}

int main() {
	Matrix& matrix1 = Matrix::createMatrix(FLOAT_32, 10);
	Matrix& matrix2 = Matrix::createMatrix(FLOAT_64, 5);

	matrix1.print();
	matrix2.print();

	// Perform matrix operations
	matrix1.multiply(matrix2);
	matrix1.assign(matrix2);

	return 0;
}
