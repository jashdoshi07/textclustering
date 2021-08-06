import pandas as pd
import numpy as np

class PCA:
    def __init__(self, matrix):
        self.matrix = matrix

    def calculate(self):
        features = self.matrix.T
        self.cov_matrix = np.cov(features)

        #using numpy to get eigenvector and values
        values, vectors = np.linalg.eig(self.cov_matrix)

        #importance of each feature
        max_num = (-values).argsort()[:2]
        max1 = max_num[0]
        max2 = max_num[1]

        #eigen vectors
        proj1 = self.matrix.dot(vectors.T[max1])
        proj2 = self.matrix.dot(vectors.T[max2])

        proj_data = pd.DataFrame(proj1, columns=['pc1'])
        proj_data['pc2'] = proj2

        #print(proj_data)

        return proj_data
