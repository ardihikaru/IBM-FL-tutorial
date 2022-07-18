"""
Source: https://towardsdatascience.com/introduction-to-ibm-federated-learning-a-collaborative-approach-to-train-ml-models-on-private-data-2b4221c3839
"""

from numpy import load
from keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report

# load data
# TODO: Change this filename
data = load('examples/data/mnist/random/data_party0.npz')

lst = data.files

X_train, y_train, X_test, y_test = data[lst[0]], data[lst[1]], data[lst[2]], data[lst[3]]

# load model
# by default, the model files are generated in the ROOT PROJECT DIRECTORY
# TODO: Change this filename
model = load_model('keras-cnn_1658148102.4956248_party0.h5')

# summarize model
print(model.summary())

# Add one more dimension to X_test to match the input data
X_test = np.expand_dims(X_test, axis=-1)

# Prediction
pred = model.predict_classes(X_test)

# Evaluate the performance
print(classification_report(y_test, pred))
