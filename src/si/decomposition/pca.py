import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset

class PCA(Transformer):
    """
    Principal Component Analysis (PCA) implementation using eigenvalue decomposition.
    
    Parameters
    ----------
    n_components : int
        Number of components to keep.
        
    Attributes
    ----------
    mean : np.ndarray
        Mean of the training data
    components : np.ndarray
        Principal components (eigenvectors)
    explained_variance : np.ndarray
        Amount of variance explained by each principal component
    """
    
    def __init__(self, n_components: int):
        super().__init__()
        self.n_components = n_components
        self.is_fitted = False
        self.mean = None
        self.covariance = None
        self.e_values = None
        self.components = None
        self.explained_variance = None
        self.e_vectores = None
        
    def _fit(self, dataset: Dataset) -> 'PCA':
        """
        Fit the PCA model.
        
        Parameters
        ----------
        dataset : Dataset
            Input dataset
            
        Returns
        -------
        self : PCA
            Fitted PCA instance
        """
        if self.n_components == 0 or self.n_components> dataset.shape()[1]:
            raise ValueError("n_components must be a positive integer less than or equal to the number of features.")

        # centering the data
        self.mean = dataset.get_mean()
        dataset.X = dataset.X - self.mean


        # computing the covariance matrix of the centered data and eigenvalue decomposition on the covariance matrix
        # rowvar = False ensures that the columns of the dataset are intrepreted as variables
        self.covariance = np.cov(dataset.X, rowvar= False)
        self.e_values, self.e_vectores = np.linalg.eig(self.covariance)
        # garantees real eigenvalues since numerical approximations or rounding errors can lead to complex eigenvalues on a real valued covariance matrix
        self.e_values = np.real(self.e_values)


        # infer the principal components, sorting them by descending order of eigenvalues
        principal_components_idx = np.argsort(self.e_values) [-self.n_components:][::-1]


        # Infer the explained variance
        self.explained_variance = self.e_values[principal_components_idx] / np.sum(self.e_values)
        self.components = self.e_vectores[:, principal_components_idx].T

        self.is_fitted = True

        return self
        
    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Transform the data using the principal components.
        
        Parameters
        ----------
        dataset : Dataset
            Input dataset
            
        Returns
        -------
        Dataset
            Transformed dataset features. 
        """
        # 1. Center the data
        X_centered = dataset.X - self.mean
        
        # 2. Project data onto principal components
        X_reduced = np.dot(X_centered, self.components.T)
        
        # reducing the dataset to the principal components
        X_reduced = np.dot(X_centered, self.components.T)

        return Dataset(X_reduced, y= dataset.y, features=[f"PC{i+1}" for i in range(self.n_components)], label= dataset.label)
    
    def get_covariance(self)-> np.ndarray:
        """
        Returns the covariance matrix of the centered data.
        -------
        np.ndarray
            - Covariance matrix
        Raises
        -------
        ValueError
            - If PCA has not been fitted to your data.
            """
        if not self.is_fitted:
            raise ValueError("PCA has not been fitted to your data.")

        else:
            return self.covariance