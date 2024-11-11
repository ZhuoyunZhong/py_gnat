import random
import numpy as np
import time

from py_gnat.gnat import NearestNeighborsGNAT

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Test performance

    # Define a simple distance function for SE3 points
    def euclidean_distance(p1, p2, w=1.0):
        d_position = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
        d_rotation = 1 - np.abs(np.dot(p1[3:7], p2[3:7]))
        return d_position + w * d_rotation

    # Rotation sampling
    def sample_rotation():
        quat = np.random.uniform(-1, 1, 4)
        quat /= np.linalg.norm(quat)
        return quat

    # Create a set of random 2D points
    data_points = [
        [
            random.uniform(0, 100),
            random.uniform(0, 100),
            random.uniform(0, 100),
            *sample_rotation(),
        ]
        for _ in range(10000)
    ]

    # Initialize GNAT
    start_time = time.time()
    nn = NearestNeighborsGNAT()
    nn.set_distance_function(euclidean_distance)
    # Add points to GNAT
    nn.add_list(data_points)
    build_time = time.time() - start_time
    print(f"Build time: {build_time:.6f} seconds")

    # Test nearest neighbor search
    query_point = [50, 50, 50, 0, 0, 0, 1]
    nearest = nn.nearest(query_point)
    print(f"Nearest neighbor to {query_point} is {nearest}")

    # Test k-nearest neighbors search
    start_time = time.time()
    k = 5
    nearest_k = nn.nearest_k(query_point, k)
    search_time = time.time() - start_time
    print(f"Nearest neighbor search time: {search_time:.6f} seconds")
    print(f"{k} nearest neighbors to {query_point} are:")
    for neighbor in nearest_k:
        print(neighbor)

    # Test range search
    radius = 2
    nearest_r = nn.nearest_r(query_point, radius)
    print(f"Neighbors within radius {radius} of {query_point} are:")
    for neighbor in nearest_r:
        print(neighbor)
