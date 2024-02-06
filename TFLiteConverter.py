import tensorflow as tf

model = tf.keras.models.load_model('four_leaf_clover_seeker.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)