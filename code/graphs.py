import matplotlib.pyplot as plt


class Plot():
    def __init__(self, pca, centers):
        self.pca = pca
        self.centers = centers

    def graph(self):
        color = ['r', 'y', 'b']

        x_axis = [o for o in self.pca.pc1]
        y_axis = [o for o in self.pca.pc2]
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.scatter(x_axis, y_axis)
        plt.show()
