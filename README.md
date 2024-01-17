# CNet: ComplexNet - A Complex-Valued Neural Network Framework
## Overview
ComplexNet is an advanced and flexible artificial neural network framework designed to handle complex-valued parameters. Unlike traditional real-valued neural networks, CNet explores the idea of utilizing complex numbers in the field of machine learning. This library is currently a work in progress, with ongoing development to harness the efficiency

## Features
* Complex-Valued Parameters: CNet supports complex numbers as parameters, allowing for more expressive and nuanced representations in neural networks.

* Strassen Algorithm: The library leverages the Strassen algorithm for matrix multiplication, optimizing the computational efficiency of complex-valued operations.

* SIMD Instructions: CNet takes advantage of SIMD (Single Instruction, Multiple Data) instructions to accelerate matrix operations and enhance overall performance.

* OpenMP (OMP) Optimization: The library employs OpenMP to parallelize matrix multiplication, further optimizing computation on multi-core systems.

## References
* Devansh. (2022, November 17). Improve neural networks by using complex numbers - geek culture - medium. Geek Culture. https://medium.com/geekculture/improve-neural-networks-by-using-complex-numbers-5e142b8931e6
* Preprint, A., Ko, M., Panchal, U. K., Andrade-Loarca, H., & Mendez-Vazquez, A. (n.d.). Coshnet: A hybrid complex valued neural network using shearlets. Arxiv.org. Retrieved January 17, 2024, from http://arxiv.org/abs/2208.06882
