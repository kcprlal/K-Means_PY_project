import random

def CalculateSquaredDistance(x1, x2):
    result = 0.0
    for i in range (0, len(x1)):
        result += (x2[i] - x1[i])**2
    return result


def CalculateCentorids(clusters, centroids):
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


def AssignToClusters(samples, centroids):
    centroidsNumber = len(centroids)
    clusters = [[] for _ in range(centroidsNumber)]

    for sample in samples:
        bestCentroid = 0
        closest = CalculateSquaredDistance(sample, centroids[0])

        for centroid in range(1, k):
            distance = CalculateSquaredDistance(sample, centroids[centroid])
            if distance < closest:
                closest = distance
                bestCentroid = centroid
        clusters[bestCentroid].append(sample)

    return clusters


def KMeans(samples, centoridsNumber, threshold=1e-4, maxIterations=100):
    centroids = random.sample(samples, centoridsNumber)

    for _ in range(maxIterations):
        clusters = AssignToClusters(samples, centroids)
        newCentroids = CalculateCentorids(clusters, centroids)

        diff = 0.0
        for i in range(0, centoridsNumber):
            diff += CalculateSquaredDistance(centroids[i], newCentroids[i])
        if diff < threshold:
            break

        centroids = newCentroids
    
    return centroids, clusters
