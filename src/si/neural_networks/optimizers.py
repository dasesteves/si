from abc import abstractmethod

import numpy as np


class Optimizer:

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
        
    @abstractmethod
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.
        
        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.
            
        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """
        Initialize the optimizer.
        
        Parameters
        ----------
        learning_rate: float
            The learning rate to use for updating the weights.
        momentum:
            The momentum to use for updating the weights.
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.retained_gradient = None

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.
        
        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.
            
        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w
        return w - self.learning_rate * self.retained_gradient
    
class Adam(Optimizer):
    """
    Adam optimizer.
    
    Combines RMSprop and SGD best advantages.
    Uses moving averages of gradients and squared gradients to
    adaptively scale the learning rate and improve convergence.
    
    Parameters
    ----------
    learning_rate : float
        The learning rate to use for updating the weights.
    beta_1 : float
        The exponential decay rate for the 1st moment estimates.
        Defaults to 0.9.
    beta_2 : float
        The exponential decay rate for the 2nd moment estimates.
        Defaults to 0.999.
    epsilon : float
        A small constant for numerical stability.
        Defaults to 1e-8.
    
    """
    
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = None  # Moving average of the gradients
        self.v = None  # Moving average of the squared gradients
        self.t = 0     # Epoch (iterations)

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Computes and updates the weights using the Adam optimization algorithm.
        
        Parameters
        ----------
        w : numpy.ndarray
            Current weights.
        grad_loss_w : numpy.ndarray
            Gradient of the loss function with respect to the weights.
            
        Returns
        -------
        numpy.ndarray
            Updated weights.
        """
        # Initialize moving averages (m and v) as zeros if not already initialized
        #np.zeros_like create a new array with the same shape and data type as "w", but filled with zeros ( = np.zeros(np.shape(w)))
        if self.m is None:
            self.m = np.zeros_like(w)
        if self.v is None:
            self.v = np.zeros_like(w)
            
        # Update epochs
        self.t += 1
        
        # Compute moving averages of the gradients (m) and squared gradients (v)
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad_loss_w
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (grad_loss_w ** 2)
        
        # Compute bias-corrected moving averages
        m_hat = self.m / (1 - self.beta_1 ** self.t)
        v_hat = self.v / (1 - self.beta_2 ** self.t)
        
        # Update the weights
        w_update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        w = w - w_update
        
        return w