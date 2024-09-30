def euclidean_distance:
    x = np.array([1, 2, 3])
    y = np.array([[4, 5, 6], [7, 8, 9]])
    return np.sqrt(np.sum((x - y) ** 2, axis=1))