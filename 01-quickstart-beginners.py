# https://www.tensorflow.org/tutorials/quickstart/beginner?hl=en

# Remove TensorFlow CUDA warning and info messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # removes tf informative messages

# Import section for TensorFlow and other frameworks
import tensorflow as tf
print("TensorFlow version:", tf.__version__)


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#print(f"Trining Set: \n", x_train)
#print(f"Test Set: \n", x_test)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
print(f"Model: ", model)

predictions = model(x_train[:1]).numpy()
print(f"Predictions: ", predictions)

