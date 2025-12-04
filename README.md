# K-Means Algorithm (Pure Python Implementation)

This project provides a minimal, dependency-free implementation of the K-Means clustering algorithm in Python.  
The implementation is contained in the `KMeansAlgorithm` class, which performs clustering using only core Python features and the standard library.

The algorithm partitions data points into `k` clusters by minimizing the within-cluster sum of squared distances.

## Features

- Squared Euclidean distance metric  
- Random initialization of centroids  
- Iterative update of clusters and centroids  
- Configurable convergence threshold and iteration limit  
- Handles empty clusters by preserving previous centroids  

## Algorithm Overview

1. **Initialization**  
   Randomly select `k` samples from the dataset as the initial centroids.

2. **Assignment Step**  
   Assign each sample to the cluster whose centroid has the smallest squared Euclidean distance to that sample.

3. **Update Step**  
   Recalculate each centroid as the mean of samples assigned to its cluster.  
   If a cluster is empty, its centroid remains unchanged.

4. **Convergence Check**  
   If the movement of all centroids (sum of squared differences) is below the threshold, or if the maximum number of iterations is reached, the algorithm stops.

## Class and Method Descriptions

### `CalculateSquaredDistance(x1, x2)`
Computes the squared Euclidean distance between two feature vectors.

### `AssignToClusters(samples, centroids)`
Assigns each sample to the closest centroid.  
Returns a list of clusters, where each cluster contains a list of samples.

### `CalculateCentorids(clusters, centroids)`
Computes new centroids by averaging the samples in each cluster.  
Empty clusters keep their previous centroids.

### `KMeans(samples, centoridsNumber, threshold=1e-4, maxIterations=100)`
Runs the K-Means clustering procedure.  
Returns:
- a list of final centroids  
- a list of clusters containing the assigned samples  

## Example Usage

```python
from kmeans import KMeansAlgorithm

samples = [
    [1.0, 2.0],
    [1.5, 1.8],
    [5.0, 8.0],
    [8.0, 8.0],
    [1.0, 0.6],
    [9.0, 11.0]
]

algo = KMeansAlgorithm()
centroids, clusters = algo.KMeans(samples, centoridsNumber=2)

print("Final centroids:", centroids)
print("Clusters:", clusters)
