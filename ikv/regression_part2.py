import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

matplotlib.use("Qt5Agg")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.random.set_seed(42)


def linear_data():
    X = np.arange(-100, 100, 4)
    y = np.arange(-90, 110, 4)
    return X, y


def plot_predictions(train_data, train_labels, test_data, test_labels, predictions):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", label="Training data")
    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", label="Testing data")
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", label="Predictions")
    # Show the legend
    plt.legend()


def split_data(X, y, split_fraction=0.8):
    split_index = int(split_fraction * len(X))
    X_train = X[:split_index]
    y_train = y[:split_index]

    X_test = X[split_index:]
    y_test = y[split_index:]

    return X_train, y_train, X_test, y_test


def single_output_perceptron(X_train, y_train, epochs=100):
    model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
    model.compile(
        loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.SGD(), metrics=["mae"]
    )
    model.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=epochs)
    return model


def metrics(y_true, y_pred):
    mae = tf.keras.losses.mae(y_true, y_pred).numpy()
    mse = tf.keras.losses.mse(y_true, y_pred).numpy()
    return mae, mse


def save_model(model, filename):
    model.save(filename)


def load_model(filename):
    try:
        return tf.keras.models.load_model(filename)
    except FileNotFoundError:
        print(f"Model file {filename} not found.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


if __name__ == "__main__":
    X, y = linear_data()
    X_train, y_train, X_test, y_test = split_data(X, y)

    model = single_output_perceptron(X_train, y_train, epochs=50)
    predictions = model.predict(X_test)
    print(metrics(y_test, predictions.squeeze()))
    plot_predictions(
        train_data=X_train,
        train_labels=y_train,
        test_data=X_test,
        test_labels=y_test,
        predictions=predictions,
    )
    plt.show()

    save_model(model, "single_output_perceptron.h5")
    model = load_model("single_output_perceptron.h5")
    model.summary()
