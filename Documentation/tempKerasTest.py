import sys
sys.path.append(r"C:\\Users\\thema\\AppData\\Roaming\Python\\Python312\\site-packages")

import ktrain
print("ktrain successfully imported!")

try:
    import tensorflow as tf
    tf_version = tf.__version__
except ImportError:
    tf_version = "Not Installed"

try:
    from tensorflow import keras
    keras_version = keras.__version__
except ImportError:
    keras_version = "Not Installed"

try:
    import ktrain
    ktrain_version = ktrain.__version__
except ImportError:
    ktrain_version = "Not Installed"

print(f"Python Version: {sys.version}")
print(f"TensorFlow Version: {tf_version}")
print(f"Keras Version: {keras_version}")
print(f"Ktrain Version: {ktrain_version}")
