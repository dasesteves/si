import numpy as np

def cosine_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the cosine distance between a single sample and multiple samples.

    The cosine distance is calculated as:
        distance = 1 - cosine_similarity,
    where cosine_similarity measures the cosine of the angle between two vectors
    and ranges from -1 (opposite direction) to 1 (same direction).

    Parameters:
    ----------
    x : np.ndarray
        A single sample represented as a 1D array.
    y : np.ndarray
        Multiple samples represented as a 2D array, where each row is a sample.

    Returns:
    -------
    distances : np.ndarray
        A 1D array containing the cosine distances between the single sample `x`
        and each sample in `y`.
    """
    # Compute the dot product between x and each row in y
    dot_product = np.dot(x, y.T)
    
    # Compute the norm of x and each row in y
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y, axis=1)
    
    # Calculate cosine similarity
    similarity = dot_product / (norm_x * norm_y)
    distance = 1 - similarity
    
    # Return cosine distance
    return distance
