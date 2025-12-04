import random
from sklearn.cluster import KMeans as SklearnKMeans

from kmeanproject.Kmeans import KMeansAlgorithm

def main():
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
    algo = KMeansAlgorithm()
    our_centroids, our_clusters = algo.KMeans(samples, k, threshold=1e-6, maxIterations=100)

    # Run sklearn KMeans for comparison
    sk_model = SklearnKMeans(n_clusters=k, random_state=123, n_init=10)
    sk_model.fit(samples)
    sk_centroids = sk_model.cluster_centers_.tolist()
    labels = sk_model.labels_.tolist()
    sk_clusters = [[] for _ in range(k)]

    for i in range(len(samples)):
        label = labels[i]
        sk_clusters[label].append(samples[i])

    print("\n========== K-MEANS COMPARISON ==========\n")

    print("CENTROIDS:")
    for i in range(len(our_centroids)):
        print(f"   Cluster {i}:")
        print(f"      Our:     {our_centroids[i]}")
        print(f"      Sklearn: {sk_centroids[i]}")
    print()

    print("CLUSTERS:")
    for i in range(len(our_clusters)):
        print(f"   Cluster {i}:")
        print(f"      Our:     {our_clusters[i]}")
        print(f"      Sklearn: {sk_clusters[i]}")
    print()

    print("=========================================\n")

if __name__ == "__main__":
    main()