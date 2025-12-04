import random

class KMeansAlgorithm:
    def __init__(self):
        # Constructor — nothing to initialize explicitly
        return

    def CalculateSquaredDistance(self, x1, x2):
        # Computes squared Euclidean distance between two vectors x1 and x2
        result = 0.0
        for i in range (0, len(x1)):
            # Add squared difference for each dimension
            result += (x2[i] - x1[i])**2
        return result

    def CalculateCentorids(self, clusters, centroids):
        # Computes new centroid positions based on assigned clusters
        newCentroids = []
        for i in range(0, len(centroids)):
            samples = clusters[i]

            # If cluster has no samples, keep previous centroid
            if not samples:
                newCentroids.append(centroids[i])
                continue

            # Number of features per sample
            features = len(samples[0])
            # Accumulators for sum of each feature dimension
            sums = [0.0] * features

            # Sum all points in this cluster dimension-wise
            for s in samples:
                for j in range(0, features):
                    sums[j] += s[j]

            # Convert sums to means → new centroid
            for j in range(features):
                sums[j] /= len(samples)

            newCentroids.append(sums)

        return newCentroids

    def AssignToClusters(self, samples, centroids):
        # Assigns each sample to the nearest centroid
        centroidsNumber = len(centroids)

        # Prepare empty cluster lists
        clusters = [[] for _ in range(centroidsNumber)]

        for sample in samples:
            # Start by assuming centroid 0 is best
            bestCentroid = 0
            closest = self.CalculateSquaredDistance(sample, centroids[0])

            # Check all other centroids
            for centroid in range(1, centroidsNumber):
                distance = self.CalculateSquaredDistance(sample, centroids[centroid])

                # Update best centroid if a closer one is found
                if distance < closest:
                    closest = distance
                    bestCentroid = centroid

            # Assign sample to the closest centroid
            clusters[bestCentroid].append(sample)

        return clusters

    def KMeans(self, samples, centoridsNumber, threshold=1e-4, maxIterations=100):
        # Initialize centroids randomly from the samples
        centroids = random.sample(samples, centoridsNumber)

        for _ in range(maxIterations):
            # Step 1 — assign samples to clusters
            clusters = self.AssignToClusters(samples, centroids)

            # Step 2 — recompute centroid positions
            newCentroids = self.CalculateCentorids(clusters, centroids)

            # Measure change between old and new centroids
            diff = 0.0
            for i in range(0, centoridsNumber):
                diff += self.CalculateSquaredDistance(centroids[i], newCentroids[i])

            # Stop if centroids moved less than threshold — algorithm converged
            if diff < threshold:
                break

            # Update centroids for next iteration
            centroids = newCentroids
        
        # Return final centroids and cluster assignments
        return centroids, clusters
