import numpy as np

def kmeans(data: np.ndarray, k: int, initialization: str = 'random', n_init: int = 1, max_iter: int = 20) -> tuple:
    """Applies the k-means algorithm to a dataset.

    Args:
        data (np.ndarray): Dataset to cluster.
        k (int): Number of clusters.
        initialization (str, optional): Initialization mode. Defaults to 'random'.
        n_init (int, optional): Number of initializations. Defaults to 1.
        max_iter (int, optional): Maximum number of iterations. Defaults to 20.

    Returns:
        tuple: (clustered data, centroids, inertia)
    """
    clusterings: list[np.ndarray] = []
    inertias: list[int] = []
    dics: list[dict] = []

    for _ in range(n_init):
        dic_centroids = initialize_centroids(data, k=k, mode=initialization)

        last_inertia = -1

        for j in range(max_iter):
            if j != 0:
                dic_centroids = recalculate_centroids(data_labels, dic_centroids, data.shape[1])

            labels = np.apply_along_axis(apply_get_labels, axis=1, arr=data, dic_centroids=dic_centroids)

            data_labels = data.copy()
            data_labels = np.append(data_labels, np.expand_dims(labels, axis=1), axis=1)

            inertia_it = inertia(data_labels, dic_centroids=dic_centroids)

            if inertia_it == last_inertia:
                break

            last_inertia = inertia_it

        clusterings.append(data_labels)
        inertias.append(inertia_it)
        dics.append(dic_centroids)

    min_inertia_index = np.argmin(inertias)

    return clusterings[min_inertia_index], dics[min_inertia_index], inertias[min_inertia_index]

def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculates the euclidean distance between two points.
    Args:
        p1 (np.ndarray): First point.
        p2 (np.ndarray): Second point.
    Returns:
        float: Euclidean distance.
    """
    temp = np.array(p1) - np.array(p2)
    return np.sqrt(np.dot(temp.T, temp))

def initialize_centroids(data: np.ndarray, k: int, mode: str) -> dict:
    """Initializes the centroids of the clusters.
    
    Args: 
        data (np.ndarray): Dataset to cluster.
        k (int): Number of clusters.
        mode (str): Initialization mode.

    Returns:
            dic_centroids: Dictionary of centroids.
    """
    dic_centroids = {}

    if mode == 'random':
        random_samples = np.random.choice(data.shape[0], k, replace=False)
        centroids = [tuple(data[i, :]) for i in random_samples]
        dic_centroids = {i: centroids[i] for i in range(len(centroids))}
    elif mode == 'kmeans++':
        first_centroid = np.random.choice(data.shape[0], 1)[0]
        dic_centroids[0] = data[first_centroid, :]

        for i in range(k - 1):
            distances = np.array([np.min([euclidean_distance(point, centroid) for centroid in dic_centroids.values()]) for point in data])
            probabilities = distances / sum(distances)
            indice = np.random.choice(range(len(data)), p=probabilities)
            dic_centroids[i + 1] = data[indice, :]
    else:
        raise ValueError("'" + mode + "' is not a valid mode.")

    return dic_centroids

def get_labels(point: np.ndarray, dic_centroids: dict) -> int:
    """Gets the label of a point.
    
    Args:
        point (np.ndarray): Point to label.
        dic_centroids (dict): Dictionary of centroids.

    Returns:
            int: Label of the point.
    """
    distances = [euclidean_distance(point, centroid) for centroid in dic_centroids.values()]
    closest_centroid_index = np.argmin(distances)
    return closest_centroid_index

def apply_get_labels(row: np.ndarray, dic_centroids: dict) -> int:
    """Applies the get_labels function to a row.
    
    Args:
        row (np.ndarray): Row to label.
        dic_centroids (dict): Dictionary of centroids.

    Returns:
            int: Label of the row.
    """
    return get_labels(row, dic_centroids)

def recalculate_centroids(data: np.ndarray, dic_centroids: dict, dims: int) -> dict:
    """Recalculates the centroids of the clusters.

    Args:
        data (np.ndarray): Dataset to cluster.
        dic_centroids (dict): Dictionary of centroids.
        dims (int): Number of dimensions.

    Returns:
            dict: Dictionary of new centroids.
    """
    new_centroids = {}
    for c, _ in dic_centroids.items():
        data_cluster = data[data[:, dims] == c][:, :-1]
        new_centroids[c] = data_cluster.mean(axis=0)
    return new_centroids

def inertia(data: np.ndarray, dic_centroids: dict) -> float:
    """Calculates the inertia of the model.
    
    Args:
        data (np.ndarray): Dataset to cluster.
        dic_centroids (dict): Dictionary of centroids.

    Returns:
            float: Inertia of the model.
    """
    total_inertia = 0
    for i in range(len(dic_centroids)):
        data_cluster = data[data[:, data.shape[1] - 1] == i][:, :data.shape[1] - 1]
        inertia_val = sum([euclidean_distance(point, dic_centroids[i]) for point in data_cluster])
        total_inertia += inertia_val
    return total_inertia