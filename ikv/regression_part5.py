# Import required libraries
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import matplotlib
import os
import numpy as np

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

DATA_SOURCE = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"

def get_baseline_mae(y_test, y_train):
    return np.mean(np.abs(y_test - np.mean(y_train)))


def get_relative_error(mea, y_test):
    return (mea/np.mean(y_test))*100


def get_insurance_training_testing_data(normalise=False, split_percentage=0.2):
    # Load the dataset
    insurance = pd.read_csv(DATA_SOURCE)

    X = insurance.drop("charges", axis=1)
    y = insurance["charges"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_percentage, random_state=42)

    if normalise:
        # Create column transformer
        ct = make_column_transformer(
            (MinMaxScaler(), ["age", "bmi", "children"]),
            (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
        )

        ct.fit(X_train)

        X_train_normal = ct.transform(X_train)
        X_test_normal = ct.transform(X_test)

        return X_train_normal, X_test_normal, y_train, y_test

    return X_train, X_test, y_train, y_test 


X_train_normal, X_test_normal, y_train, y_test = get_insurance_training_testing_data(normalise=True)

# 1. Set the random seed
tf.random.set_seed(42)

# 2. Create the model
insurance_model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(1024, activation="relu"),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(1, activation="linear")
])
insurance_model.summary()

# 3. Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
insurance_model.compile(loss=tf.keras.losses.mae,
                        optimizer=optimizer,
                        metrics=["mae"])

# 4. Fit the model
history = insurance_model.fit(X_train_normal, y_train, epochs=200)

# 5. Evaluate the model
loss, mae = insurance_model.evaluate(X_test_normal, y_test)

# 6. Predict the model
y_preds = insurance_model.predict(X_test_normal)

print("-----------Final Metrics-------------------")
print(f"Loss: {loss}, MAE: {mae}")
print(f"Baseline MAE: {get_baseline_mae(y_test, y_train)}")
print(f"Relative Error: {get_relative_error(mae, y_test)}")
print("-------------------------------------------")

# 7. Plot the loss curve
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