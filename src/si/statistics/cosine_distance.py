import numpy as np

def cosine_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculates the cosine distance between a sample x and multiple samples y.
    
    The cosine distance is calculated as: 1 - cosine_similarity
    where cosine_similarity = (x·y)/(||x|| ||y||)

    Parameters
    ----------
    x : np.ndarray
        A single sample (vector)
    y : np.ndarray
        Multiple samples (matrix)

    Returns
    -------
    np.ndarray
        Array with cosine distances between x and each sample in y

    Raises
    ------
    ValueError
        If vector dimensions are not compatible
    """
    if x.shape[-1] != y.shape[-1]:
        raise ValueError(f"Last dimension of arrays must be equal. Got: {x.shape[-1]} != {y.shape[-1]}")

    # Vector normalization
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y, axis=1)
    
    # Avoid division by zero
    mask = (x_norm != 0) & (y_norm != 0)
    
    # Calculate cosine similarity
    similarity = np.zeros(len(y))
    if np.any(mask):
        dot_product = np.dot(y[mask], x)
        similarity[mask] = dot_product / (x_norm * y_norm[mask])
    
    # Convert similarity to distance
    distance = 1 - similarity
    
    return distance