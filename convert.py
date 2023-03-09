# We can load .h5 model and save back to Tensorflow format

import tensorflow as tf
model = tf.keras.models.load_model('model.h5')
tf.saved_model.save(model, 'saved_model')