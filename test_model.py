import tensorflow as tf
import numpy as np

model = tf.saved_model.load('ml_model/fixed_model')
infer = model.signatures['serving_default']

dummy_input = np.random.rand(1, 1, 16).astype(np.float32)  # Must be float32!

output = infer(tf.constant(dummy_input))
print("Prediction shape:", output['output_0'].shape)  # Should be (1, 3)
print("Sample prediction:", output['output_0'].numpy())
