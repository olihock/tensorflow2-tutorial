# https://www.tensorflow.org/tutorials/quickstart/beginner?hl=en

# Remove TensorFlow CUDA warning and info messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # removes tf informative messages


#
# Set up TensorFlow
#

import tensorflow as tf
print("TensorFlow version:", tf.__version__)


#
# Load a dataset 
#

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#print(f"Trining Set: \n", x_train)
#print(f"Test Set: \n", x_test)


#
# Build a machine learning model
# 

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
print(f"Model: ", model)


#
# Training
#

predictions = model(x_train[:1]).numpy()
print(f"Predictions: ", predictions)

# Convert logits to probabilities
probabilities = tf.nn.softmax(predictions).numpy()
print(f"Probabilities: ", probabilities)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss = loss_fn(y_train[:1], predictions).numpy()
print(f"Loss: ", loss)

# Configure the model
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])


#
# Train and evaluate your model
#

# Minimise the loss
model.fit(x_train, y_train, epochs=5)

# Check model performance
performance = model.evaluate(x_test,  y_test, verbose=2)
print(f"Performance: ", performance)

# Return probabilities
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
prob_mod = probability_model(x_test[:5])
print(f"Probability Model: ", prob_mod)