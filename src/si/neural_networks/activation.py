from abc import abstractmethod
from typing import Union

import numpy as np

from si.neural_networks.layers import Layer


class ActivationLayer(Layer):
    """
    Base class for activation layers.
    """

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation on the given input.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        self.input = input
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_error: float) -> Union[float, np.ndarray]:
        """
        Perform backward propagation on the given output error.

        Parameters
        ----------
        output_error: float
            The output error of the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The output error of the layer.
        """
        return self.derivative(self.input) * output_error

    @abstractmethod
    def activation_function(self, input: np.ndarray) -> Union[float, np.ndarray]:
        """
        Activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The output of the layer.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input: np.ndarray) -> Union[float, np.ndarray]:
        """
        Derivative of the activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The derivative of the activation function.
        """
        raise NotImplementedError

    def output_shape(self) -> tuple:
        """
        Returns the output shape of the layer.

        Returns
        -------
        tuple
            The output shape of the layer.
        """
        return self._input_shape

    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return 0

class ReLUActivation(ActivationLayer):
    """
    ReLU activation function.
    """

    def activation_function(self, input: np.ndarray):
        """
        ReLU activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        return np.maximum(0, input)

    def derivative(self, input: np.ndarray):
        """
        Derivative of the ReLU activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return np.where(input >= 0, 1, 0)

class SigmoidActivation(ActivationLayer):
    """
    Sigmoid activation function.
    """

    def activation_function(self, input: np.ndarray):
        """
        Sigmoid activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        return 1 / (1 + np.exp(-input))

    def derivative(self, input: np.ndarray):
        """
        Derivative of the sigmoid activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return self.activation_function(input) * (1 - self.activation_function(input))

class TanhActivation(ActivationLayer):
    def activation_function(self, input: np.ndarray):
        """
        Tanh activation function that applies the hyperbolic 
        tangent function to the output of neurons, squashing 
        the values to the range of-1 to 1.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        return (np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input))
    
    def derivative(self, input: np.ndarray):
        """
        Derivative of the Tanh activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        tanh_output= self.activation_function(input)
        return 1 - tanh_output**2

class SoftmaxActivation(ActivationLayer):
    """
    Softmax activation funciton that  transforms the raw output scores 
    into a probability distribution (that sums to 1), making it suitable
    for multi-class classification problems.
    """

    def activation_function(self, input: np.ndarray):
        """
        Softmax activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer, that represents the probability of each class. The sum of all probabilities is equal to 1.

        Note
        -----
        To ensure numerical stability, the maximum value of the input array is subtracted from each 
        element before applying the exponential function.
        
        """

        #Using keepdims=True ensures the result retains the same number of dimensions as the input
        exp_shifted = np.exp(input - np.max(input, axis=1, keepdims=True))  
        return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
    
    def derivative(self, input: np.ndarray):
        """
        Return the full Softmax Jacobian for each sample.

        Parameters
        ----------
        input: numpy.ndarray (n_samples, n_classes)
            The input to the layer.

        Returns
        -------
        numpy.ndarray (n_samples, n_classes, n_classes)
            Entry [sample, output, input] is the derivative of one output
            probability with respect to one input logit. Off-diagonal terms
            express the dependence between classes.
        """
        probabilities = self.activation_function(input)
        identity = np.eye(probabilities.shape[1])
        return probabilities[:, :, None] * (identity - probabilities[:, None, :])

    def backward_propagation(self, output_error: np.ndarray) -> np.ndarray:
        """Apply the Softmax Jacobian without materializing it for each sample."""
        weighted_error = np.sum(output_error * self.output, axis=1, keepdims=True)
        return self.output * (output_error - weighted_error)
