import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from ..CNN_models.CNNmodel_2 import build_cnn

# Load preprocessed features and labels
X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

# Reshape for CNN (expand channel dimension)
X = np.expand_dims(X, axis=-1)  # Shape: (num_samples, 40, time_steps, 1)

# Split into training (80%) and validation (20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")

# Define model
model = build_cnn(input_shape=X.shape[1:], num_classes=len(set(y)))

# Define early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)

# Save trained model
model.save("models/speech_cnn.h5")
print("Model training complete! Saved at '../models/speech_cnn_2.h5'.")

from sklearn.metrics import classification_report

# Predict on validation data
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)

# Print classification report
print(classification_report(y_val, y_pred_classes, target_names=["up", "down", "left", "right", "yes"]))

