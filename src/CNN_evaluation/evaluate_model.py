import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load model and data
model = tf.keras.models.load_model("models/lstm_mfcc.keras")
X = np.load("data/processed/X_mfcc.npy")
y = np.load("data/processed/y_mfcc.npy")

# Prepare input shape
X = np.expand_dims(X, axis=-1)

# Predict
y_pred = model.predict(X)
y_pred_classes = np.argmax(y_pred, axis=1)

# Compute confusion matrix
cm = confusion_matrix(y, y_pred_classes)
labels = ["up", "down", "left", "right", "yes"]

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - CNN + BatchNorm")
plt.tight_layout()
plt.show()
