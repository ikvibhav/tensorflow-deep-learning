# Import required libraries
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib
import os

matplotlib.use('Qt5Agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
Aim - To Predict the cost of medical insurance for individuals based on different parameters. The parameters are -
1. Age
2. Gender
3. BMI
4. Children
5. Smoker
6. Region
'''
 

# Load the dataset
insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")

# Turn Categories into numbers
insurance_one_hot = pd.get_dummies(insurance)

# Create X & y values
X = insurance_one_hot.drop("charges", axis=1)
y = insurance_one_hot["charges"]

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape) # (1070, 11) (268, 11) (1070,) (268,)

# Build a neural network
# 1. Create a model
tf.random.set_seed(42)
insurance_model = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])
#2. Compile the model
insurance_model.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=["mae"])
#3. Fit the model
history = insurance_model.fit(X_train, y_train, epochs=100)

# Evaluate the model
insurance_model.evaluate(X_test, y_test)

# Plot history (also known as a loss curve)
pd.DataFrame(history.history).plot()
plt.title("Model loss curve")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.show()