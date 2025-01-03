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
        self.mean = None
        self.components = None
        self.explained_variance = None
        
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
        X = dataset.X
        
        # 1. Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # 2. Calculate covariance matrix and perform eigendecomposition
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort eigenvectors by eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 3. Select top n_components
        self.components = eigenvectors[:, :self.n_components]
        
        # 4. Calculate explained variance
        total_variance = np.sum(eigenvalues)
        self.explained_variance = eigenvalues[:self.n_components] / total_variance
        
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
            Transformed dataset
        """
        # 1. Center the data
        X_centered = dataset.X - self.mean
        
        # 2. Project data onto principal components
        X_transformed = np.dot(X_centered, self.components)
        
        # Create new feature names
        new_features = [f'PC{i+1}' for i in range(self.n_components)]
        
        return Dataset(X_transformed, dataset.y, features=new_features, label=dataset.label)