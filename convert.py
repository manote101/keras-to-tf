# We can load .h5 model and save back to Tensorflow format

import tensorflow as tf

# load existing model
model = tf.keras.models.load_model('2xplus_one.h5')

# save to new name
tf.saved_model.save(model, '2xplus_one2/1')
