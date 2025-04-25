import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from ..CNN_models.CNNmodel_2 import build_cnn  

X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")
X = np.expand_dims(X, axis=-1)  # Add channel dim

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return float(lr * tf.math.exp(-0.1))

lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=5,
    restore_best_weights=True,
    verbose=1
)

model = build_cnn(input_shape=X.shape[1:], num_classes=len(set(y)))

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, lr_callback]
)

model.save("models/speech_cnn_lr_schedule.h5")
print("Model training complete! Saved at 'models/speech_cnn_lr_schedule.h5'.")

y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)

print(classification_report(y_val, y_pred_classes, target_names=["up", "down", "left", "right", "yes"]))
