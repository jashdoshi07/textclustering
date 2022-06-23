import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances


class Kmeans:
    """ K Means Clustering
    Parameters
    -----------
        k: int , number of clusters

        seed: int, will be randomly set if None

        max_iter: int, number of iterations to run algorithm, default: 200"""


    def __init__(self, k, seed=None, value="euclid", max_iter=500):
        self.k = k
        self.seed = seed
        self.value = value
        if self.seed is not None:
            np.random.seed(self.seed)
        self.max_iter = max_iter

    def initialise_centroids(self, data):
        """Randomly Initialise Centroids
        Parameters
        ----------
        data: array or matrix, number_rows, number_features

        Returns
        --------
        centroids: array of k centroids chosen as random data points
        """
        initial_centroids = [0,8,16]
        #print(initial_centroids)
        self.centroids = data[initial_centroids]

        return self.centroids

    def assign_clusters(self, data):
        """Compute distance of data from clusters and assign data point
           to closest cluster.
        Parameters
        ----------
        data: array or matrix, number_rows, number_features

        Returns
        --------
        cluster_labels: index which minmises the distance of data to each
        cluster

        """

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        dist_to_centroid = pairwise_distances(data, self.centroids, metric='euclidean')
        if self.value == "cosine":
            new_distance = 1 - np.square(dist_to_centroid)
            self.cluster_labels = np.argmax(new_distance, axis=1)
        else:
            self.cluster_labels = np.argmin(dist_to_centroid, axis=1)

        return self.cluster_labels

    def update_centroids(self, data):
        """Computes average of all data points in cluster and
           assigns new centroids as average of data points
        Parameters
        -----------
        data: array or matrix, number_rows, number_features

        Returns
        -----------
        centroids: array, k, number_features
        """
        self.centroids = np.array([data[self.cluster_labels == i].mean(axis=0) for i in range(self.k)])

        return self.centroids

    def predict(self, data):
        """Predict which cluster data point belongs to
        Parameters
        ----------
        data: array or matrix, number_rows, number_features

        Returns
        --------
        cluster_labels: index which minmises the distance of data to each
        cluster
        """

        return self.assign_clusters(data)

    def fit_kmeans(self, data):
        """
        This function contains the main loop to fit the algorithm
        Implements initialise centroids and update_centroids
        according to max_iter
        -----------------------
        Returns
        -------
        instance of kmeans class
        """
        self.centroids = self.initialise_centroids(data)

        # Main kmeans loop
        for iter in range(self.max_iter):

            self.cluster_labels = self.assign_clusters(data)
            self.centroids = self.update_centroids(data)
            if iter % 100 == 0:
                print("Running Model Iteration %d " % iter)

        print("Model finished running")
        return self
