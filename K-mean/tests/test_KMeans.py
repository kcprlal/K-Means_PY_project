# tests/test.py

import random
import numpy as np
import pytest
from sklearn.cluster import KMeans as SklearnKMeans

from kmeanproject.Kmeans import KMeansAlgorithm


@pytest.fixture
def algo():
    # Provide a fresh instance of the KMeansAlgorithm for each test
    return KMeansAlgorithm()


def test_calculate_squared_distance_zero(algo):
    # Distance between identical points should be zero
    x1 = [0.0, 0.0]
    x2 = [0.0, 0.0]
    result = algo.CalculateSquaredDistance(x1, x2)
    assert result == pytest.approx(0.0)


def test_calculate_squared_distance_positive_values(algo):
    # Test squared distance on basic positive coordinates
    x1 = [1.0, 2.0]
    x2 = [4.0, 6.0]
    result = algo.CalculateSquaredDistance(x1, x2)
    assert result == pytest.approx(25.0)


def test_calculate_squared_distance_negative_values(algo):
    # Distance should work regardless of sign of coordinates
    x1 = [-1.0, -2.0, -3.0]
    x2 = [1.0, 2.0, 3.0]
    result = algo.CalculateSquaredDistance(x1, x2)
    assert result == pytest.approx(56.0)


def test_calculate_centorids_single_cluster_mean(algo):
    # Centroid should be arithmetic mean of points in cluster
    clusters = [
        [[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]],
    ]
    centroids = [[0.0, 0.0]]

    new_centroids = algo.CalculateCentorids(clusters, centroids)

    assert len(new_centroids) == 1
    assert new_centroids[0] == pytest.approx([2.0, 2.0])


def test_calculate_centorids_multiple_clusters_and_empty_cluster(algo):
    # Test multiple clusters including one empty cluster
    clusters = [
        [
            [1.0, 0.0, 2.0],
            [3.0, 2.0, 4.0],
        ],
        [],  # Empty cluster should retain old centroid
        [
            [4.0, -2.0, 9.0],
            [6.0, 0.0, 11.0],
        ],
    ]

    centroids = [
        [0.0, 0.0, 0.0],
        [100.0, 100.0, 100.0],  # Kept because cluster is empty
        [5.0, -1.0, 10.0],
    ]

    new_centroids = algo.CalculateCentorids(clusters, centroids)

    assert len(new_centroids) == 3

    # Check computed means and preserved centroid
    assert new_centroids[0] == pytest.approx([2.0, 1.0, 3.0])
    assert new_centroids[1] == pytest.approx([100.0, 100.0, 100.0])
    assert new_centroids[2] == pytest.approx([5.0, -1.0, 10.0])


def test_assign_to_clusters_basic_two_centroids(algo):
    # Samples should group around nearest centroid
    samples = [
        [0.0, 0.0],
        [1.0, 1.0],
        [9.0, 9.0],
        [10.0, 10.0],
    ]
    centroids = [
        [0.0, 0.0],
        [10.0, 10.0],
    ]

    clusters = algo.AssignToClusters(samples, centroids)

    assert len(clusters) == 2
    assert sorted(clusters[0]) == sorted([[0.0, 0.0], [1.0, 1.0]])
    assert sorted(clusters[1]) == sorted([[9.0, 9.0], [10.0, 10.0]])


def test_assign_to_clusters_all_to_one_centroid(algo):
    # All samples are much closer to centroid 0
    samples = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    centroids = [
        [0.0, 0.0],
        [100.0, 100.0],
    ]

    clusters = algo.AssignToClusters(samples, centroids)

    assert len(clusters) == 2
    assert sorted(clusters[0]) == sorted(samples)
    assert clusters[1] == []


def test_kmeans_returns_correct_shapes_and_assignments(algo):
    # KMeans should return exactly k centroids and well-formed clusters
    random.seed(0)

    samples = [
        [1.0, 2.0],
        [1.5, 1.8],
        [5.0, 8.0],
        [8.0, 8.0],
        [1.0, 0.6],
        [9.0, 11.0],
    ]
    k = 2

    centroids, clusters = algo.KMeans(samples, k)

    # Correct number of centroids
    assert len(centroids) == k

    # Correct number of cluster lists
    assert len(clusters) == k

    # Every sample must appear exactly once across clusters
    flat_clusters = [tuple(p) for cluster in clusters for p in cluster]
    assert sorted(flat_clusters) == sorted(tuple(p) for p in samples)


def test_kmeans_roughly_finds_two_clusters(algo):
    # KMeans should find two well-separated natural clusters
    random.seed(1)

    cluster1 = [[0.0, 0.0], [0.5, -0.2], [-0.3, 0.4]]
    cluster2 = [[10.0, 10.0], [9.5, 10.2], [10.3, 9.7]]
    samples = cluster1 + cluster2
    k = 2

    centroids, _ = algo.KMeans(samples, k, threshold=1e-6, maxIterations=100)

    # Expected cluster centers; exact values not required
    expected_centers = [[0.0, 0.0], [10.0, 10.0]]
    tolerance = 0.5

    # Ensure each expected center has a nearby found centroid
    for expected in expected_centers:
        found_close = False
        for c in centroids:
            distance = ((c[0] - expected[0]) ** 2 + (c[1] - expected[1]) ** 2) ** 0.5
            if distance <= tolerance:
                found_close = True
                break
        assert found_close, f"No centroid close to expected {expected}"


def test_kmeans_matches_sklearn_on_simple_dataset(algo):
    # Compare our implementation vs sklearn on an easy dataset
    random.seed(123)

    cluster1 = [
        [0.0, 0.0],
        [0.2, -0.1],
        [-0.2, 0.1],
        [0.1, 0.2],
    ]
    cluster2 = [
        [5.0, 5.0],
        [5.2, 5.1],
        [4.8, 4.9],
        [5.1, 4.8],
    ]
    samples = cluster1 + cluster2
    k = 2

    # Run our KMeans
    our_centroids, _ = algo.KMeans(samples, k, threshold=1e-6, maxIterations=100)

    # Run sklearn KMeans for comparison
    sk_model = SklearnKMeans(n_clusters=k, random_state=123, n_init=10)
    sk_model.fit(samples)
    sk_centroids = sk_model.cluster_centers_.tolist()

    # Check each centroid has a close match in sklearn results
    for c in our_centroids:
        distances = []
        for sc in sk_centroids:
            dist = ((sc[0] - c[0]) ** 2 + (sc[1] - c[1]) ** 2) ** 0.5
            distances.append(dist)

        assert min(distances) < 0.5, f"Our centroid {c} too far from any sklearn centroid"


def test_kmeans_stops_early_when_centroids_do_not_change(algo, monkeypatch):
    # Test convergence: algorithm should stop as soon as centroids stop moving
    samples = [[0.0, 0.0], [1.0, 1.0]]
    k = 1

    # Track how many times CalculateCentorids is called
    call_counter = {"count": 0}
    original = algo.CalculateCentorids

    # Patch centroid calculation so it always returns same centroids
    def fake_calculate_centorids(clusters, centroids):
        call_counter["count"] += 1
        return centroids

    monkeypatch.setattr(algo, "CalculateCentorids", fake_calculate_centorids)

    # Run KMeans; should stop after 1 iteration
    centroids, clusters = algo.KMeans(samples, k, threshold=1e-4, maxIterations=100)

    assert call_counter["count"] == 1

    # Restore original method
    algo.CalculateCentorids = original
