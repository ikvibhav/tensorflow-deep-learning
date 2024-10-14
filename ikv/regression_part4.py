# Import required libraries
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
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

# Create column transformer
ct = make_column_transformer(
    (MinMaxScaler(), ["age", "bmi", "children"]),
    (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
)

# Create X & y values
X = insurance.drop("charges", axis=1)
y = insurance["charges"]

# Build our train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the column transformer to ONLY our training data
# Doing this on test data would result in data leakage
# data leakage = when information from outside your training dataset is used to create your model
ct.fit(X_train)

# Transform training and test data with normalization (MinMaxScaler) and OneHotEncoder
X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)

# print(X_train.shape, X_test.shape) # (1070, 6) (268, 6)
# print(X_train_normal.shape, X_test_normal.shape) # (1070, 11) (268, 11)

# Neural Networks
# 1. Set random seed
tf.random.set_seed(42)

# 2. Create the model
insurance_model = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

# 3. Compile the model
insurance_model.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=["mae"])

# 4. Fit the model
history = insurance_model.fit(X_train_normal, y_train, epochs=200)

# 5. Evaluate the model
insurance_model.evaluate(X_test_normal, y_test)

# 6. Predict the model
y_preds = insurance_model.predict(X_test_normal)

# # 7. Plot the loss curve
# pd.DataFrame(history.history).plot()
# plt.ylabel("loss")
# plt.xlabel("epochs")
# plt.title("Loss Curve")
# plt.show()

# # 8. Plot the predictions
# plt.figure(figsize=(10, 7))
# plt.scatter(y_test, y_preds)
# plt.xlabel("True Values")
# plt.ylabel("Predicted Values")
# plt.title("True vs Predicted Values")
# plt.show()

# Plot the above together
fig, ax = plt.subplots(1, 2, figsize=(15, 7))

# Plot the loss curve in the first subplot
ax[0].set_ylabel("Loss")
ax[0].set_xlabel("Epochs")
ax[0].set_title("Loss Curve")
pd.DataFrame(history.history).plot(ax=ax[0])

# Plot the true vs predicted values in the second subplot
ax[1].scatter(y_test, y_preds)
ax[1].set_xlabel("True Values")
ax[1].set_ylabel("Predicted Values")
ax[1].set_title("True vs Predicted Values")
plt.tight_layout()
plt.show()