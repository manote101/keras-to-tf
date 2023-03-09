import tensorflow as tf
import numpy as np

# Define the input data (x) and output data (y) for the 2x+1 function
x = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
y = np.array([-1, 1, 3, 5, 7, 9, 11, 13, 15, 17], dtype=float)

# Define the Keras model with a single dense layer and a linear activation function
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile the model with the mean squared error loss and the stochastic gradient descent optimizer
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.SGD())

# Train the model for 1000 epochs with the input and output data
model.fit(x, y, epochs=200)

# Evaluate the trained model on the input and output data
loss = model.evaluate(x, y)
print("Mean squared error:", loss)

# Predict the output for a new input value (e.g. 10) using the trained model
new_x = np.array([10], dtype=float)
prediction = model.predict(new_x)
print("Prediction for x=10:", prediction)

# Save the trained model to disk in the .h5 format
model.save("2xplus_one.h5")
# Export the trained model as a SavedModel
tf.saved_model.save(model, "2xplus_one/1")

print("Model saved to disk.")