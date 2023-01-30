# a Python code for implementing a self-organizing map for data clustering:
import numpy as np
import matplotlib.pyplot as plt

class SOM:
  def __init__(self, X, output_shape, learning_rate=0.5, sigma=None):
    self.X = X
    self.output_shape = output_shape
    self.learning_rate = learning_rate
    self.weights = np.random.rand(output_shape[0], output_shape[1], X.shape[1])
    if sigma is None:
      self.sigma = max(output_shape) / 2.0
    else:
      self.sigma = sigma
  
  def _neuron_distance(self, x, y):
    return np.sqrt(np.sum((x - y) ** 2))

  def _activate(self, x):
    bmu_index = np.array([0, 0])
    min_distance = np.iinfo(np.int32).max
    for i in range(self.weights.shape[0]):
      for j in range(self.weights.shape[1]):
        distance = self._neuron_distance(x, self.weights[i][j])
        if distance < min_distance:
          min_distance = distance
          bmu_index = np.array([i, j])
    return bmu_index
  
  def _neighbourhood_function(self, bmu_index, iteration):
    h = np.exp(-self._neuron_distance(bmu_index, np.arange(self.output_shape[0])) ** 2 / (2 * (self.sigma * iteration) ** 2))
    return h
  
  def fit(self, num_iterations):
    for iteration in range(num_iterations):
      for x in self.X:
        bmu_index = self._activate(x)
        h = self._neighbourhood_function(bmu_index, iteration)
        for i in range(self.weights.shape[0]):
          for j in range(self.weights.shape[1]):
            self.weights[i][j] += self.learning_rate * h[i] * (x - self.weights[i][j])
            self.sigma = self.sigma * 0.95
    
  def predict(self, X):
    y_pred = []
    for x in X:
      bmu_index = self._activate(x)
      y_pred.append(bmu_index)
    return np.array(y_pred)
  
  def visualize(self, X, y_pred):
    plt.scatter(X[:,0], X[:,1], c=y_pred[:,0])
    plt.show()


# an example usage:
import numpy as np

# Generate toy data
np.random.seed(0)
mean1 = [0,0]
cov1 = [[1,0],[0,1]]
data1 = np.random.multivariate_normal(mean1,cov1,100)

mean2 = [3,3]
cov2 = [[1,0],[0,1]]
data2 = np.random.multivariate_normal(mean2,cov2,100)

X = np.concatenate([data1,data2])

# Train SOM
som = SOM(X, output_shape=(10,10))
som.fit(100)

# Predict clusters
y_pred = som.predict(X)

# Visualize the clusters
som.visualize(X, y_pred)
