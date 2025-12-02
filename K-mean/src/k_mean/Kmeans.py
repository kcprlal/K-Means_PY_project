import random

class KMeansAlgorithm:
    def __init__(self):
        return

    def CalculateSquaredDistance(self, x1, x2):
        result = 0.0
        for i in range (0, len(x1)):
            result += (x2[i] - x1[i])**2
        return result


    def CalculateCentorids(self, clusters, centroids):
        newCentroids = []
        for i in range(0, len(centroids)):
            samples = clusters[i]
            if not samples:
                newCentroids.append(centroids[i])
                continue

            features = len(samples[0])
            sums = [0.0] * features

            for s in samples:
                for j in range(0, features):
                    sums[j] += s[j]

            for j in range(features):
                sums[j] /= len(samples)

            newCentroids.append(sums)

        return newCentroids


    def AssignToClusters(self, samples, centroids):
        centroidsNumber = len(centroids)
        clusters = [[] for _ in range(centroidsNumber)]

        for sample in samples:
            bestCentroid = 0
            closest = self.CalculateSquaredDistance(sample, centroids[0])

            for centroid in range(1, centroidsNumber):
                distance = self.CalculateSquaredDistance(sample, centroids[centroid])
                if distance < closest:
                    closest = distance
                    bestCentroid = centroid
            clusters[bestCentroid].append(sample)

        return clusters


    def KMeans(self, samples, centoridsNumber, threshold=1e-4, maxIterations=100):
        centroids = random.sample(samples, centoridsNumber)

        for _ in range(maxIterations):
            clusters = self.AssignToClusters(samples, centroids)
            newCentroids = self.CalculateCentorids(clusters, centroids)

            diff = 0.0
            for i in range(0, centoridsNumber):
                diff += self.CalculateSquaredDistance(centroids[i], newCentroids[i])
            if diff < threshold:
                break

            centroids = newCentroids
        
        return centroids, clusters
