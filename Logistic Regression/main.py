import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("./datasets/insurance_data.csv")

class LogisticRegression():
    def __init__(self):

        self.w = None
        self.b = 0
        self.learning_rate = 0.01

    def _sigmoid(self, w, X, b):
        """
        w: Vector Parameters for each feature
        X: Vector X Values for each training example
        b: Scalar Parameter Value
        """

        z = np.dot(X, w) + b

        g_z = 1 / (1 + np.exp(-z))
        
        return g_z

    def visualise(self):
        fig, ax = plt.subplots(figsize=(8,5), label="Insurance Data")

        ax.set_xlabel("Age")
        ax.set_ylabel("Has Insurance")
        ax.legend()

        ax.scatter(self.X, self.y)

        plt.show()

    def _cost(self, y, g_z, m):
        loss = np.sum(y * np.log(g_z) + (1 - y) * np.log(1 - g_z))

        total_cost = ((-1) / m) * loss

        return total_cost

    def _gradient_descent(self, g_z, X, y, m, w, b):
        err = g_z - y

        wd = (1 / m) * np.dot(X.T, err)

        bd = (1 / m) * np.sum(err)

        w = w - self.learning_rate * wd
        b = b - self.learning_rate * bd

        return w, b
    
    def fit(self, X, y):

        X = (X - X.mean(axis=0)) / X.std(axis=0)

        m,n = X.shape

        self.w = np.zeros(n)
        self.b = 0

        for i in range(10000):
            # Forward Pass
            g_z = self._sigmoid(self.w, X, self.b)
            
            # Cost Function
            cost = self._cost(y, g_z, m)

            # Backward Pass
            self.w, self.b = self._gradient_descent(g_z, X, y, m, self.w, self.b)

            if i % 100 == 0:
                print(f"Iteration {i}: Cost: {cost:.4f} | Weights: {self.w} | Bias: {self.b:.4f}")

    def predict(self, X):

        X = (X - X.mean(axis=0)) / X.std(axis=0)

        z = np.dot(X, self.w) + self.b
        
        g_z = 1 / (1 + np.exp(-z))
                
        v = (g_z > 0.5).astype(int)
        return v


if __name__ == "__main__":
    X = data[['age']]
    y = data['bought_insurance']

    model = LogisticRegression()
    model.fit(X, y)