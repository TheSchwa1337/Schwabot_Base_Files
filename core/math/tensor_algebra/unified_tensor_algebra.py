# -*- coding: utf-8 -*-
"""Unified Tensor Algebra for Schwabot Trading
===========================================

Provides core tensor operations and abstractions for multi-dimensional
mathematical analysis within the Schwabot trading framework.
"""

import logging
import numpy as np
from typing import Optional, Union, Tuple, List

logger = logging.getLogger(__name__)


class UnifiedTensorAlgebra:
    """Manages tensor operations and maintains tensor state for the system."""

    def __init__(self, precision: int = 64):
        self.precision = precision
        logger.info(
            f"Unified Tensor Algebra initialized with float{precision} precision")

    def create_tensor(self, data: list, dtype=np.float64) -> np.ndarray:
        """Creates a new tensor from input data.

        Args:
            data (list): List of lists representing tensor data.
            dtype (numpy.dtype): Data type for the tensor elements. Defaults to np.float64.

        Returns:
            np.ndarray: The created tensor.
        """
        return np.array(data, dtype=dtype)

    def tensor_multiply(
            self,
            tensor1: np.ndarray,
            tensor2: np.ndarray) -> np.ndarray:
        """Performs element-wise multiplication of two tensors.

        Args:
            tensor1 (np.ndarray): First tensor.
            tensor2 (np.ndarray): Second tensor.

        Returns:
            np.ndarray: Resulting tensor after multiplication.
        """
        if tensor1.shape != tensor2.shape:
            raise ValueError(
                "Tensors must have the same shape for element-wise multiplication.")
        return tensor1 * tensor2

    def tensor_dot_product(
            self,
            tensor1: np.ndarray,
            tensor2: np.ndarray) -> np.ndarray:
        """Computes the dot product of two tensors.

        Args:
            tensor1 (np.ndarray): First tensor.
            tensor2 (np.ndarray): Second tensor.

        Returns:
            np.ndarray: Resulting tensor after dot product.
        """
        return np.dot(tensor1, tensor2)

    def get_tensor_shape(self, tensor: np.ndarray) -> tuple:
        """Returns the shape of the tensor."""
        return tensor.shape

    def get_tensor_rank(self, tensor: np.ndarray) -> int:
        """Returns the rank (number of dimensions) of the tensor."""
        return tensor.ndim

    def apply_activation(
            self,
            tensor: np.ndarray,
            activation_type: str = "relu") -> np.ndarray:
        """Applies an activation function to the tensor.

        Args:
            tensor (np.ndarray): Input tensor.
            activation_type (str): Type of activation function ('relu', 'sigmoid', 'tanh').

        Returns:
            np.ndarray: Tensor after applying activation.
        """
        if activation_type == "relu":
            return np.maximum(0, tensor)
        elif activation_type == "sigmoid":
            return 1 / (1 + np.exp(-tensor))
        elif activation_type == "tanh":
            return np.tanh(tensor)
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")

    def reduce_tensor(self,
                      tensor: np.ndarray,
                      axis: Optional[Union[int,
                                           Tuple[int,
                                                 ...]]] = None,
                      operation: str = "sum") -> Union[np.ndarray,
                                                       float]:
        """Reduces the tensor along a specified axis.

        Args:
            tensor (np.ndarray): Input tensor.
            axis (Optional[Union[int, Tuple[int, ...]]]): Axis or axes along which to reduce.
            operation (str): Reduction operation ('sum', 'mean', 'max', 'min').

        Returns:
            Union[np.ndarray, float]: Reduced tensor or scalar.
        """
        if operation == "sum":
            return np.sum(tensor, axis=axis)
        elif operation == "mean":
            return np.mean(tensor, axis=axis)
        elif operation == "max":
            return np.max(tensor, axis=axis)
        elif operation == "min":
            return np.min(tensor, axis=axis)
        else:
            raise ValueError(f"Unsupported reduction operation: {operation}")

    def reshape_tensor(self, tensor: np.ndarray,
                       new_shape: Tuple[int, ...]) -> np.ndarray:
        """Reshapes the tensor to a new shape."""
        return tensor.reshape(new_shape)

    def transpose_tensor(self, tensor: np.ndarray,
                         axes: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        """Transposes the tensor."""
        return np.transpose(tensor, axes=axes)

    def concatenate_tensors(self,
                            tensors: List[np.ndarray],
                            axis: int = 0) -> np.ndarray:
        """Concatenates a list of tensors along a specified axis."""
        return np.concatenate(tensors, axis=axis)


if __name__ == '__main__':
    # Basic demonstration of UnifiedTensorAlgebra
    logging.basicConfig(level=logging.INFO)
    algebra = UnifiedTensorAlgebra()

    # Create tensors
    tensor_a = algebra.create_tensor([[1, 2], [3, 4]])
    tensor_b = algebra.create_tensor([[5, 6], [7, 8]])

    print("Tensor A:\n", tensor_a)
    print("Tensor B:\n", tensor_b)

    # Element-wise multiplication
    element_wise_product = algebra.tensor_multiply(tensor_a, tensor_b)
    print("\nElement-wise Product:\n", element_wise_product)

    # Dot product
    dot_product_result = algebra.tensor_dot_product(tensor_a, tensor_b)
    print("\nDot Product:\n", dot_product_result)

    # Apply activation
    relu_tensor = algebra.apply_activation(tensor_a, "relu")
    print("\nTensor A after ReLU:\n", relu_tensor)

    # Reduce tensor
    sum_tensor = algebra.reduce_tensor(tensor_a, operation="sum")
    print("\nSum of Tensor A elements:", sum_tensor)

    # Reshape tensor
    reshaped_tensor = algebra.reshape_tensor(tensor_a, (4, 1))
    print("\nReshaped Tensor A:\n", reshaped_tensor)

    # Transpose tensor
    transposed_tensor = algebra.transpose_tensor(tensor_a)
    print("\nTransposed Tensor A:\n", transposed_tensor)

    # Concatenate tensors
    tensor_c = algebra.create_tensor([[9, 10]])
    concatenated_tensors = algebra.concatenate_tensors(
        [tensor_a, tensor_c], axis=0)
    print("\nConcatenated Tensors:\n", concatenated_tensors)
