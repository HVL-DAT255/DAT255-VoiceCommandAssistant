import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# Load data
X = np.load("/Users/sondrerisnes/Documents/GitHub/DAT255-VoiceCommandAssistant/data/processed/X.npy")
y = np.load("/Users/sondrerisnes/Documents/GitHub/DAT255-VoiceCommandAssistant/data/processed/y.npy")


# Expand dims for CNN input
X = np.expand_dims(X, axis=-1)

# Split into train/val again (same as training, to keep it consistent)
_, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model_paths = {
    "Base CNN": "models/speech_cnn.h5",
    "CNN + LR Schedule": "models/speech_cnn_lr_schedule.h5",
    "CNN + Regularization": "models/speech_cnn_regularized.h5",
    "CNN + BatchNorm": "models/speech_cnn_batchnorm.h5",
    "CNN v2": "models/speech_cnn 2.h5"
}

labels = ["up", "down", "left", "right", "yes"]

for name, path in model_paths.items():
    print(f"\n📊 Evaluating: {name}")
    model = tf.keras.models.load_model(path)
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print(classification_report(y_val, y_pred_classes, target_names=labels))
