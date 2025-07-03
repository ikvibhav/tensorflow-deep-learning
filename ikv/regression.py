import matplotlib

matplotlib.use("Qt5Agg")

# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])
plt.scatter(X, y)
plt.show()

tf.random.set_seed(42)

model = tf.keras.Sequential([tf.keras.layers.Dense(1)])

model.compile(
    loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.SGD(), metrics=["mae"]
)

model.summary()

# tf.expand_dims(X, axis=-1) adds an extra dimension to the data
# X with shape (8,) becomes (8, 1)
model.fit(tf.expand_dims(X, axis=-1), y, epochs=50)

y_pred = model.predict(tf.expand_dims(17.0, axis=-1))
print(y_pred)
