from sklearn.datasets import make_circles
import tensorflow as tf
from utils import plot_decision_boundary
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# Make 1000 examples
n_samples = 1000

# Create circles
X, y = make_circles(n_samples, noise=0.03, random_state=42)

# Print the shape of X and y
print(X.shape, y.shape)

# 1. Create a model
tf.random.set_seed(42)

# A model with linear and non-linear activation functions
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(10, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
])

# 2. Compile the model
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# 3. Fit the model
model.fit(X, y, epochs=50)

# plot the decision boundary for the model
plot_decision_boundary(model, X, y)
plt.show()